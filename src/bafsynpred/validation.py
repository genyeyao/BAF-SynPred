"""Preflight validation for BAF-SynPred data and molecular graphs."""

import hashlib
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from .chemistry import HYPEREDGE_ATTR_DIM, NUM_HYPEREDGE_TYPES, smiles_to_data_chem_hypergraph


REQUIRED_COLUMNS = ("SMILES1", "SMILES2", "DepMapID", "Label")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_graph(graph, smiles):
    if graph is None:
        raise ValueError("Failed to construct graph for SMILES: {}".format(smiles))
    if tuple(graph.x_old.shape[1:]) != (55,):
        raise ValueError("Atom feature dimension is not 55 for {}".format(smiles))
    if tuple(graph.x.shape[1:]) != (117,):
        raise ValueError("Bond-node feature dimension is not 117 for {}".format(smiles))
    if graph.edge_index_old.shape[0] != 2:
        raise ValueError("Atom edge_index is not [2, E] for {}".format(smiles))
    if graph.edge_index_old.shape[1] != graph.x.shape[0]:
        raise ValueError("Directed atom edges and bond-node rows are misaligned for {}".format(smiles))
    source, target = graph.edge_index_old
    expected_bond_nodes = torch.cat(
        [graph.x_old[source], graph.edge_attr_old, graph.x_old[target]], dim=1
    )
    if not torch.equal(graph.x, expected_bond_nodes):
        raise ValueError("Bond-node features do not match source/edge/target order for {}".format(smiles))
    if graph.atom_hyperedge_index.shape[0] != 2:
        raise ValueError("Hypergraph incidence is not [2, M] for {}".format(smiles))
    if graph.atom_hyperedge_attr.shape[1] != HYPEREDGE_ATTR_DIM:
        raise ValueError("Hyperedge attribute dimension mismatch for {}".format(smiles))
    if graph.atom_hyperedge_type.numel():
        low = int(graph.atom_hyperedge_type.min())
        high = int(graph.atom_hyperedge_type.max())
        if low < 0 or high >= NUM_HYPEREDGE_TYPES:
            raise ValueError("Hyperedge type index out of range for {}".format(smiles))
    if graph.atom_hyperedge_index.numel():
        if int(graph.atom_hyperedge_index[0].min()) < 0:
            raise ValueError("Negative atom incidence index for {}".format(smiles))
        if int(graph.atom_hyperedge_index[0].max()) >= graph.x_old.size(0):
            raise ValueError("Atom incidence index out of range for {}".format(smiles))
        if int(graph.atom_hyperedge_index[1].min()) < 0:
            raise ValueError("Negative hyperedge index for {}".format(smiles))
        if int(graph.atom_hyperedge_index[1].max()) >= graph.atom_hyperedge_type.numel():
            raise ValueError("Hyperedge incidence index out of range for {}".format(smiles))
    tensors = (
        graph.x_old, graph.x, graph.edge_attr, graph.edge_attr_old,
        graph.atom_hyperedge_attr,
    )
    if not all(torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("Non-finite molecular features for {}".format(smiles))


def audit_inputs(csv_path, cell_path, split_path, expected_cell_dim=977):
    for path in (csv_path, cell_path, split_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    frame = pd.read_csv(csv_path)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise ValueError("Missing columns: {}".format(missing_columns))
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required data columns contain missing values")
    labels = set(frame["Label"].unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("Labels are not binary: {}".format(sorted(labels)))
    if frame.duplicated(list(REQUIRED_COLUMNS)).any():
        raise ValueError("Exact duplicate interaction rows were found")

    cells = np.load(cell_path, allow_pickle=True).item()
    missing_cells = sorted(set(frame["DepMapID"]) - set(cells))
    if missing_cells:
        raise ValueError("Missing cell features: {}".format(missing_cells[:5]))
    invalid_cells = [
        key for key, value in cells.items()
        if np.asarray(value).shape != (expected_cell_dim,)
        or not np.isfinite(np.asarray(value, dtype=np.float32)).all()
    ]
    if invalid_cells:
        raise ValueError("Invalid cell feature vectors: {}".format(invalid_cells[:5]))

    split_obj = np.load(split_path, allow_pickle=True).item()
    folds = split_obj.get("splits")
    if not isinstance(folds, (list, tuple)) or len(folds) != 5:
        raise ValueError("Exactly five folds are required")
    validation_indices = []
    fold_sizes = []
    all_indices = set(range(len(frame)))
    for fold_id, fold in enumerate(folds):
        if "train_idx" not in fold or "val_idx" not in fold:
            raise ValueError("Fold {} is missing train_idx or val_idx".format(fold_id))
        train_idx = np.asarray(fold["train_idx"], dtype=np.int64)
        val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
        train_set, val_set = set(train_idx.tolist()), set(val_idx.tolist())
        if len(train_set) != len(train_idx) or len(val_set) != len(val_idx):
            raise ValueError("Fold {} contains duplicate indices".format(fold_id))
        if train_set & val_set:
            raise ValueError("Fold {} has train/validation overlap".format(fold_id))
        if train_set | val_set != all_indices:
            raise ValueError("Fold {} does not partition the full dataset".format(fold_id))
        fold_sizes.append({"fold": fold_id, "train": len(train_idx), "validation": len(val_idx)})
        validation_indices.extend(val_idx.tolist())
    validation_counts = Counter(validation_indices)
    if set(validation_counts) != all_indices or set(validation_counts.values()) != {1}:
        raise ValueError("Validation folds do not cover every row exactly once")

    unique_smiles = sorted(set(frame["SMILES1"]) | set(frame["SMILES2"]))
    invalid_smiles = [smiles for smiles in unique_smiles if Chem.MolFromSmiles(smiles) is None]
    if invalid_smiles:
        raise ValueError("Invalid SMILES: {}".format(invalid_smiles[:5]))
    for smiles in unique_smiles:
        _validate_graph(smiles_to_data_chem_hypergraph(smiles), smiles)

    return {
        "status": "PASS",
        "samples": int(len(frame)),
        "label_counts": {str(key): int(value) for key, value in frame["Label"].value_counts().items()},
        "cell_lines": int(len(cells)),
        "cell_dimension": int(expected_cell_dim),
        "unique_molecules": int(len(unique_smiles)),
        "multi_fragment_molecules": int(sum("." in smiles for smiles in unique_smiles)),
        "fold_sizes": fold_sizes,
        "sha256": {
            "interactions": sha256_file(csv_path),
            "cell_features": sha256_file(cell_path),
            "folds": sha256_file(split_path),
        },
    }


def save_audit(report, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
