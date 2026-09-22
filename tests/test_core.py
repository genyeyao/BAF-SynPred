"""Regression tests for the release BAF-SynPred implementation."""

import os
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bafsynpred.batching import collate_drug_pairs
from bafsynpred.chemistry import smiles_to_data_chem_hypergraph
from bafsynpred.data import collate_drug_pairs_reference
from bafsynpred.model import AtomBondCrossTalk, BAFSynPred, ds_combination


def assert_nested_equal(test_case, left, right):
    test_case.assertEqual(type(left), type(right))
    if torch.is_tensor(left):
        test_case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(test_case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right))
        for item_left, item_right in zip(left, right):
            assert_nested_equal(test_case, item_left, item_right)
    else:
        test_case.assertEqual(left, right)


class TestBAFSynPredCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        graphs = [
            smiles_to_data_chem_hypergraph("CCO"),
            smiles_to_data_chem_hypergraph("c1ccccc1N"),
        ]
        cls.samples = [
            {
                "drug_a": graphs[0],
                "drug_b": graphs[1],
                "label": 0.0,
                "depmap_id": "CELL_A",
                "cell_omics": np.zeros(977, dtype=np.float32),
            },
            {
                "drug_a": graphs[1],
                "drug_b": graphs[0],
                "label": 1.0,
                "depmap_id": "CELL_B",
                "cell_omics": np.ones(977, dtype=np.float32),
            },
        ]

    def test_vectorized_collation_is_bit_equivalent(self):
        reference = collate_drug_pairs_reference(self.samples)
        vectorized = collate_drug_pairs(self.samples)
        assert_nested_equal(self, reference, vectorized)

    def test_atom_bond_endpoint_pairing(self):
        layer = AtomBondCrossTalk(1, 1)
        with torch.no_grad():
            layer.bond_to_atom.weight.fill_(1.0)
            layer.bond_to_atom.bias.zero_()
            layer.atom_to_bond.weight.fill_(1.0)
            layer.atom_to_bond.bias.zero_()
            for gate in (layer.gate_atom[0], layer.gate_bond[0]):
                gate.weight.zero_()
                gate.bias.fill_(50.0)
        atoms = torch.tensor([[1.0], [2.0], [4.0]])
        bonds = torch.tensor([[10.0], [20.0], [30.0]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        atom_out, bond_out = layer(atoms, bonds, edge_index)
        expected_atoms = atoms + torch.tensor([[20.0], [15.0], [25.0]])
        expected_bonds = bonds + torch.tensor([[1.5], [3.0], [2.5]])
        self.assertTrue(torch.allclose(atom_out, expected_atoms))
        self.assertTrue(torch.allclose(bond_out, expected_bonds))

    def test_ds_fusion_matches_reference(self):
        evidence_a = torch.tensor([[2.0, 0.5], [0.1, 4.0]])
        evidence_b = torch.tensor([[1.0, 1.5], [3.0, 0.2]])
        fused = ds_combination(evidence_a, evidence_b, num_classes=2)
        k = 2.0
        s_a = evidence_a.sum(1, keepdim=True) + k
        s_b = evidence_b.sum(1, keepdim=True) + k
        b_a, b_b = evidence_a / s_a, evidence_b / s_b
        u_a, u_b = k / s_a, k / s_b
        conflict = (
            b_a.sum(1, keepdim=True) * b_b.sum(1, keepdim=True)
            - (b_a * b_b).sum(1, keepdim=True)
        )
        denominator = 1.0 - conflict
        b_fused = (b_a * b_b + b_a * u_b + b_b * u_a) / denominator
        u_fused = (u_a * u_b) / denominator
        expected = b_fused * (k / (u_fused + 1e-8))
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue((fused >= 0).all())

    def test_full_forward_and_parameter_count(self):
        batch = collate_drug_pairs(self.samples)
        model = BAFSynPred().eval()
        self.assertEqual(model.bond_k, 3)
        self.assertEqual(model.drug_encoder.atom_block.conv.__class__.__name__, "HypergraphConv")
        self.assertEqual(model.drug_encoder.bond_block.conv.__class__.__name__, "HypergraphConv")
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 4705756)
        with torch.no_grad():
            evidence, reconstruction = model(batch)
        self.assertEqual(tuple(evidence.shape), (2, 2))
        self.assertIsNone(reconstruction)
        self.assertTrue(torch.isfinite(evidence).all())
        self.assertTrue((evidence >= 0).all())

    def test_single_layer_hgnn_is_fixed(self):
        batch = collate_drug_pairs(self.samples)
        model = BAFSynPred(bond_k=3).eval()
        self.assertFalse(hasattr(model, "hgnn_depth"))
        self.assertFalse(hasattr(model, "hgnn_hidden_dim"))
        self.assertFalse(hasattr(model.drug_encoder.atom_block, "stack"))
        self.assertFalse(hasattr(model.drug_encoder.bond_block, "stack"))
        self.assertEqual(model.drug_encoder.bond_block.k, 3)
        with torch.no_grad():
            evidence, reconstruction = model(batch)
        self.assertEqual(tuple(evidence.shape), (2, 2))
        self.assertIsNone(reconstruction)
        self.assertTrue(torch.isfinite(evidence).all())

    def test_single_layer_backward_preflight(self):
        batch = collate_drug_pairs(self.samples)
        model = BAFSynPred(bond_k=3).train()
        evidence, reconstruction = model(batch)
        loss = evidence.sum()
        if reconstruction is not None:
            loss = loss + reconstruction.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.requires_grad]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
