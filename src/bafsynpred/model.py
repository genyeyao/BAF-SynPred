"""
BAF-SynPred final model: Bond-Atom Fusion for drug-combination synergy prediction.
==================================================================================

Architecture Overview (4 modules):
  1. CellDenoisingEncoder  - Cell omics encoding with denoising regularization
  2. DrugHypergraphEncoder - Dual-view drug encoding (atom hypergraph + bond line-graph)
                             with gated atom-bond cross-talk
  3. CellDrugModulator     - FiLM-based cell context modulation
  4. DualStreamPredictor   - Evidential deep learning with Dempster-Shafer fusion

Key Design Choices:
  - Expand-then-compress cell encoder (977->2048->512->128): DrugCell, Cancer Cell 2020
  - Chemical-aware hyperedge weighting: type embedding + attribute MLP
  - Dynamic KNN line-graph for bond representation
  - Gated cross-talk: Bresson & Laurent, ICLR 2018
  - Canonical FiLM (single Linear, zero-init): Perez et al., AAAI 2018
  - Adaptive KL evidential loss: Sensoy et al., NeurIPS 2018
  - Dempster-Shafer evidence fusion: Shafer 1976

This file is the canonical fixed-depth implementation used by the historical
0.981035 five-fold run.  The atom and bond branches each contain exactly one
``HypergraphConv -> BatchNorm -> ReLU`` block, followed by the historical
residual path.  The module names ``conv1`` and ``bn1`` are intentional: they
match the serialized checkpoint keys from that run, so the archived weights
load strictly without a key-remapping shim.  Depth is intentionally not a
constructor argument: this selected single-layer architecture is the model
definition, not a sweep dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, knn_graph
from torch_scatter import scatter_max, scatter_mean

from .chemistry import HYPEREDGE_ATTR_DIM, NUM_HYPEREDGE_TYPES

MODEL_NAME = "BAF-SynPred-final"
MODEL_VERSION = "final"




def _positive_int(value, name):
    """Return ``value`` as an integer and reject invalid architecture values."""
    integer = int(value)
    if integer != value or integer < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return integer


# ================================================================
# Module 1: Cell Denoising Encoder
# ================================================================
class CellDenoisingEncoder(nn.Module):
    """
    Encodes high-dimensional cell omics into compact embedding with
    denoising autoencoder regularization.

    Architecture: input_dim -> 2048 -> 512 -> embed_dim (expand-then-compress)
    Regularization: Gaussian noise injection + reconstruction loss

    References:
        - Cone architecture: Kuenzi et al., "Predicting Drug Response
          of Tumors from Integrated Genomic Networks", Cancer Cell 2020
        - Denoising AE: Vincent et al., "Extracting and Composing Robust
          Features with Denoising Autoencoders", JMLR 2008
    """
    def __init__(self, input_dim=977, embed_dim=128, noise_std=0.05):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU())
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.ReLU(),
            nn.Linear(512, input_dim))

    def forward(self, cell_omics):
        if self.training and self.noise_std > 0:
            cell_input = cell_omics + torch.randn_like(cell_omics) * self.noise_std
        else:
            cell_input = cell_omics
        cell_emb = self.encoder(cell_input)
        cell_recon = self.recon_head(cell_emb) if self.training else None
        return cell_emb, cell_recon


# ================================================================
# Module 2: Drug Hypergraph Encoder (dual-view + cross-talk)
# ================================================================
class ChemistryAwareHypergraphConv(nn.Module):
    """
    Atom-level hypergraph convolution with learned chemical-aware
    hyperedge weights.

    Weight computation:
        w_e = softplus(Embed(type_e) + MLP(attr_e)) + eps

    Architecture: one HypergraphConv + BN + ReLU, outer residual, and
    LayerNorm.  ``conv1``/``bn1`` preserve the historical checkpoint schema;
    the block is fixed rather than represented by an unused depth abstraction.
    LayerNorm is retained because learned hyperedge weights can have different
    scales across edge types.

    References:
        - Hypergraph convolution: Feng et al., "Hypergraph Neural Networks",
          AAAI 2019
        - Chemical-aware weighting: domain-specific design
    """
    def __init__(self, in_channels, out_channels, *,
                 num_hyperedge_types=NUM_HYPEREDGE_TYPES,
                 hyperedge_attr_dim=HYPEREDGE_ATTR_DIM, drop_edge_p=0.1):
        super().__init__()
        self.drop_edge_p = float(drop_edge_p)
        self.type_weight = nn.Embedding(num_hyperedge_types, 1)
        self.attr_weight = nn.Sequential(
            nn.Linear(hyperedge_attr_dim, out_channels), nn.ReLU(),
            nn.Linear(out_channels, 1))
        self.conv1 = HypergraphConv(in_channels, out_channels, use_attention=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.res_proj = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
        self.out_norm = nn.LayerNorm(out_channels)

    @property
    def conv(self):
        """Backward-compatible read-only alias; not a second registered module."""
        return self.conv1

    @property
    def norm(self):
        """Backward-compatible read-only alias for the historical BatchNorm."""
        return self.bn1

    def _fallback_self_hyperedges(self, x):
        ids = torch.arange(x.size(0), device=x.device)
        return torch.stack([ids, ids], dim=0), None, None

    def _hyperedge_weight(self, ht, ha, device):
        if ht is None or ht.numel() == 0:
            return None
        logits = self.type_weight(ht.long().to(device))
        if ha is not None and ha.numel() > 0:
            logits = logits + self.attr_weight(ha.float().to(device))
        return F.softplus(logits.squeeze(-1)) + 1e-6

    def _drop_hyperedge_incidence(self, hi, ht, ha):
        if not self.training or self.drop_edge_p <= 0 or hi.numel() == 0:
            return hi, ht, ha
        n = int(hi[1].max().item()) + 1
        keep = torch.rand(n, device=hi.device) >= self.drop_edge_p
        ki = keep[hi[1]]
        if ki.sum() == 0:
            return hi, ht, ha
        return hi[:, ki], ht, ha

    def forward(self, x, hyperedge_index, hyperedge_type=None, hyperedge_attr=None):
        if hyperedge_index is None or hyperedge_index.numel() == 0:
            hyperedge_index, hyperedge_type, hyperedge_attr = self._fallback_self_hyperedges(x)
        hyperedge_index = hyperedge_index.long().to(x.device)
        if hyperedge_type is not None:
            hyperedge_type = hyperedge_type.to(x.device)
        if hyperedge_attr is not None:
            hyperedge_attr = hyperedge_attr.to(x.device)
        hyperedge_index, hyperedge_type, hyperedge_attr = self._drop_hyperedge_incidence(
            hyperedge_index, hyperedge_type, hyperedge_attr)
        w = self._hyperedge_weight(hyperedge_type, hyperedge_attr, x.device)
        residual = self.res_proj(x)
        x = self.conv1(x, hyperedge_index, w)
        x = F.relu(self.bn1(x))
        return self.out_norm(x + residual)


class DynamicLineGraphConv(nn.Module):
    """
    Bond-level hypergraph convolution on dynamically constructed KNN line-graph.

    Topology: KNN(k=self.k) built from bond embeddings each forward pass.
    Architecture: one HypergraphConv + BN + ReLU with an outer residual.
    ``conv1``/``bn1`` preserve the historical checkpoint schema.
    No LayerNorm is used here because the KNN topology has uniform incidence
    weights and BatchNorm is sufficient.

    References:
        - Line graph construction: Harary & Norman, "Some Theorems on
          Directed Graphs", 1953
        - Dynamic KNN: PyG knn_graph
    """
    def __init__(self, in_channels, out_channels, *, bond_k=3, drop_edge_p=0.2):
        super().__init__()
        self.k = _positive_int(bond_k, "bond_k")
        self.drop_edge_p = float(drop_edge_p)
        self.conv1 = HypergraphConv(in_channels, out_channels, use_attention=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

    @property
    def conv(self):
        """Backward-compatible read-only alias; not a second registered module."""
        return self.conv1

    @property
    def norm(self):
        """Backward-compatible read-only alias for the historical BatchNorm."""
        return self.bn1

    def forward(self, x, batch):
        if x is None or x.size(0) == 0:
            return x
        k = min(self.k, max(int(x.size(0)) - 1, 1))
        hi = knn_graph(x, k=k, batch=batch, loop=True)
        if self.training and self.drop_edge_p > 0 and hi.numel() > 0:
            n_e = int(hi[1].max().item()) + 1 if hi.dim() == 2 else hi.size(-1)
            keep = torch.rand(n_e, device=hi.device) >= self.drop_edge_p
            if keep.sum() > 0:
                hi = hi[:, keep[hi[1]]]
        residual = x
        x = self.conv1(x, hi)
        x = F.relu(self.bn1(x))
        return x + residual


class AtomBondCrossTalk(nn.Module):
    """
    Parallel atom-bond information exchange with gated residual.

    Canonical incidence assumption:
        Each bond node `i` corresponds to the directed atom edge
        (src_i, dst_i) = edge_index[:, i].  `edge_index` is the atom-graph
        adjacency (atom-space indices); it is NOT a bond incidence tensor.

    Message passing:
        Bond -> atom: bond `i` sends a transformed message to BOTH src_i and dst_i.
        Atom -> bond: bond `i` aggregates transformed representations from BOTH
                      src_i and dst_i.
        Gated residual updates are then applied independently to atom and bond
        states.

    References:
        - Gated residual: Bresson & Laurent, "Residual Gated Graph ConvNets",
          ICLR 2018
        - Atom-bond interaction: Song et al., "Communicative Message Passing
          for Molecular Property Prediction", IJCAI 2020
    """
    def __init__(self, atom_dim, bond_dim):
        super().__init__()
        self.bond_to_atom = nn.Linear(bond_dim, atom_dim)
        self.atom_to_bond = nn.Linear(atom_dim, bond_dim)
        self.gate_atom = nn.Sequential(
            nn.Linear(atom_dim * 2, atom_dim), nn.Sigmoid())
        self.gate_bond = nn.Sequential(
            nn.Linear(bond_dim * 2, bond_dim), nn.Sigmoid())

    def forward(self, h_atom, h_bond, edge_index):
        n_atom, n_bond = h_atom.size(0), h_bond.size(0)
        if edge_index is None or edge_index.numel() == 0 or n_bond == 0:
            return h_atom, h_bond

        edge_index = edge_index.long().to(h_atom.device)
        E = edge_index.size(1)

        # Canonical assumption: bond-node i <-> directed atom edge edge_index[:, i].
        if n_bond != E:
            raise RuntimeError(
                f"CrossTalk incidence mismatch: n_bond={n_bond}, "
                f"but edge_index contains E={E} directed edges. "
                "The implementation assumes one bond node per directed atom edge."
            )

        # bond 0 -> [src_0, dst_0]; bond 1 -> [src_1, dst_1]; ...
        # atom_ids must interleave (src_i, dst_i), NOT the row-major flatten
        # [src_0..src_{E-1}, dst_0..dst_{E-1}] produced by edge_index.reshape(-1).
        bond_ids = torch.arange(E, device=edge_index.device).repeat_interleave(2)
        atom_ids = edge_index.t().contiguous().reshape(-1)

        msg_a = scatter_mean(self.bond_to_atom(h_bond[bond_ids]), atom_ids,
                             dim=0, dim_size=n_atom)
        msg_b = scatter_mean(self.atom_to_bond(h_atom[atom_ids]), bond_ids,
                             dim=0, dim_size=n_bond)
        gate_a = self.gate_atom(torch.cat([h_atom, msg_a], dim=-1))
        gate_b = self.gate_bond(torch.cat([h_bond, msg_b], dim=-1))
        h_atom = h_atom + gate_a * msg_a
        h_bond = h_bond + gate_b * msg_b
        return h_atom, h_bond


class DrugHypergraphEncoder(nn.Module):
    """
    Dual-view drug molecular encoder.

    Encodes a drug molecule through two complementary views:
      - Atom view: chemistry-aware hypergraph (captures functional groups)
      - Bond view: dynamic KNN line-graph (captures bond topology)
    Then performs atom-bond cross-talk and graph-level readout.

    Input:  drug_data dict with keys:
            x_old (atom features), x (bond features),
            atom_hyperedge_index, atom_hyperedge_type, atom_hyperedge_attr,
            edge_index_old, batch, batch_old
    Output: (atom_graph_feat [B, 2*embed_dim], bond_graph_feat [B, 2*embed_dim])
    """
    def __init__(self, num_atom_features=55, num_bond_features=117,
                 embed_dim=128, num_hyperedge_types=NUM_HYPEREDGE_TYPES,
                 hyperedge_attr_dim=HYPEREDGE_ATTR_DIM, *, bond_k=3):
        super().__init__()
        self.embed_dim = _positive_int(embed_dim, "embed_dim")
        self.bond_k = _positive_int(bond_k, "bond_k")

        # Atom view
        self.atom_proj = nn.Linear(num_atom_features, embed_dim)
        self.atom_block = ChemistryAwareHypergraphConv(
            embed_dim, embed_dim,
            num_hyperedge_types=num_hyperedge_types,
            hyperedge_attr_dim=hyperedge_attr_dim)

        # Bond view
        self.bond_proj = nn.Linear(num_bond_features, embed_dim)
        self.bond_block = DynamicLineGraphConv(
            embed_dim, embed_dim, bond_k=bond_k)

        # Cross-talk
        self.cross_talk = AtomBondCrossTalk(embed_dim, embed_dim)

    @staticmethod
    def _readout(h, batch):
        """Graph-level readout: scatter_mean + scatter_max concatenation."""
        return torch.cat([
            scatter_mean(h, batch, dim=0),
            scatter_max(h, batch, dim=0)[0]], dim=1)

    def forward(self, drug_data):
        # Atom view encoding
        x_atom = self.atom_proj(drug_data["x_old"].float())
        h_atom = self.atom_block(
            x_atom, drug_data["atom_hyperedge_index"],
            drug_data["atom_hyperedge_type"], drug_data["atom_hyperedge_attr"])

        # Bond view encoding
        x_bond = self.bond_proj(drug_data["x"].float())
        h_bond = self.bond_block(x_bond, drug_data["batch"])

        # Atom-bond cross-talk
        h_atom, h_bond = self.cross_talk(h_atom, h_bond, drug_data["edge_index_old"])

        # Graph-level readout
        atom_feat = self._readout(h_atom, drug_data["batch_old"])
        bond_feat = self._readout(h_bond, drug_data["batch"])
        return atom_feat, bond_feat


# ================================================================
# Module 3: Cell-Drug Modulator (FiLM)
# ================================================================
class CellDrugModulator(nn.Module):
    """
    Cell context modulation of drug features via FiLM.

    Steps:
      1. Project drug readout (2*embed_dim) to embed_dim
      2. Generate affine parameters from cell embedding (zero-init)
      3. Apply FiLM modulation with gated residual

    Formula:
        gamma, beta = Linear(cell_emb)          # zero-init
        modulated = (1 + gamma) * drug + beta
        gate = sigmoid(W([drug; modulated]))
        output = LN(dropout(drug + gate*(mod-drug)) + drug)

    References:
        - FiLM: Perez et al., "FiLM: Visual Reasoning with a General
          Conditioning Layer", AAAI 2018
        - Zero-init: ensures identity at initialization (no modulation
          until learned)
    """
    def __init__(self, embed_dim, dropout=0.2):
        super().__init__()
        # Dimension projection (readout 2d -> embed d)
        self.dim_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2), nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim))

        # FiLM modulation
        self.film_generator = nn.Linear(embed_dim, embed_dim * 2)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, cell_emb, drug_feat):
        # Project to embed_dim
        drug_feat = self.dim_proj(drug_feat)

        # FiLM modulation
        gamma_beta = self.film_generator(cell_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated = (1 + gamma) * drug_feat + beta
        g = self.gate(torch.cat([drug_feat, modulated], dim=-1))
        out = drug_feat + g * (modulated - drug_feat)
        return self.norm(self.dropout(out) + drug_feat)


# ================================================================
# Module 4: Dual-Stream Predictor
# ================================================================
class DualStreamPredictor(nn.Module):
    """
    Dual-stream evidential predictor with Dempster-Shafer fusion.

    Two independent streams (atom-view and bond-view) each produce
    Dirichlet evidence, which are fused via Dempster's combination rule.

    Architecture per stream:
        [drugA, drugB, cell] (3*embed_dim) -> 512 -> 128 -> 2 (evidence)

    References:
        - Evidential Deep Learning: Sensoy et al., "Evidential Deep Learning
          to Teach Classification Neural Networks Uncertainty", NeurIPS 2018
        - DS combination: Shafer, "A Mathematical Theory of Evidence", 1976
    """
    def __init__(self, embed_dim):
        super().__init__()
        input_dim = embed_dim * 3  # [drugA, drugB, cell]

        self.atom_stream = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2))
        self.bond_stream = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2))

    def forward(self, da_atom, db_atom, da_bond, db_bond, cell_emb):
        atom_input = torch.cat([da_atom, db_atom, cell_emb], dim=1)
        bond_input = torch.cat([da_bond, db_bond, cell_emb], dim=1)

        ev_atom = F.softplus(self.atom_stream(atom_input))
        ev_bond = F.softplus(self.bond_stream(bond_input))

        return ds_combination(ev_atom, ev_bond, num_classes=2)


# ================================================================
# Dempster-Shafer Evidence Combination
# ================================================================
def ds_combination(evidence_a, evidence_b, num_classes=2):
    """
    Fuse two independent evidence sources via Dempster's rule.

    Args:
        evidence_a, evidence_b: [B, num_classes] non-negative evidence
    Returns:
        fused evidence [B, num_classes]
    """
    # FIX (Dempster-Shafer): proper EDL mass normalisation
    #   alpha = e + 1  =>  S = sum(e) + K,  b_k = e_k / S,  u = K / S,
    # so sum(b) + u == 1.  Then Dempster's rule with a single scalar
    # conflict C = sum_{i != j} b_i^a b_j^b (mass on disjoint singletons).
    K = float(num_classes)
    eps = 1e-8
    s_a = evidence_a.sum(dim=1, keepdim=True) + K
    s_b = evidence_b.sum(dim=1, keepdim=True) + K
    b_a = evidence_a / s_a
    b_b = evidence_b / s_b
    u_a = K / s_a
    u_b = K / s_b
    cap_a = b_a.sum(dim=1, keepdim=True)
    cap_b = b_b.sum(dim=1, keepdim=True)
    same = (b_a * b_b).sum(dim=1, keepdim=True)
    conflict = (cap_a * cap_b - same).clamp(min=0.0)
    denom = (1.0 - conflict).clamp(min=eps)
    b_fused = (b_a * b_b + b_a * u_b + b_b * u_a) / denom
    u_fused = (u_a * u_b) / denom
    return b_fused * (K / (u_fused + eps))


# ================================================================
# BAF-SynPred Main Model
# ================================================================
class BAFSynPred(nn.Module):
    """
    BAF-SynPred: Bond-Atom Fusion for Drug Combination Synergy Prediction.

    Architecture:
        Cell omics (977-d) -> CellDenoisingEncoder -> cell_emb (128-d)
        Drug molecule      -> DrugHypergraphEncoder -> (atom_feat, bond_feat)
        Cell + Drug        -> CellDrugModulator     -> modulated features
        All features       -> DualStreamPredictor   -> synergy evidence

    The selected implementation fixes one HypergraphConv block per atom and
    bond branch.  Only the embedding width and bond KNN value remain model
    configuration parameters.
    """
    def __init__(self, num_features_xd=117, num_features_xa=55, num_features_xt=977,
                 embed_dim=128, num_hyperedge_types=NUM_HYPEREDGE_TYPES,
                 hyperedge_attr_dim=HYPEREDGE_ATTR_DIM,
                  cell_noise_std=0.05, cell_drug_dropout=0.2,
                  *, bond_k=3):
        super().__init__()
        self.embed_dim = _positive_int(embed_dim, "embed_dim")
        self.bond_k = _positive_int(bond_k, "bond_k")

        # Module 1: Cell encoder
        self.cell_encoder = CellDenoisingEncoder(num_features_xt, embed_dim, cell_noise_std)

        # Module 2: Drug encoder (shared for drug A and B)
        self.drug_encoder = DrugHypergraphEncoder(
            num_atom_features=num_features_xa,
            num_bond_features=num_features_xd,
             embed_dim=embed_dim,
             num_hyperedge_types=num_hyperedge_types,
             hyperedge_attr_dim=hyperedge_attr_dim,
             bond_k=self.bond_k)

        # Module 3: Cell-drug modulator (shared across drugs, separate per stream)
        self.atom_modulator = CellDrugModulator(embed_dim, dropout=cell_drug_dropout)
        self.bond_modulator = CellDrugModulator(embed_dim, dropout=cell_drug_dropout)

        # Module 4: Dual-stream predictor
        self.predictor = DualStreamPredictor(embed_dim)

    def forward(self, data):
        # 1. Cell encoding
        cell_emb, cell_recon = self.cell_encoder(data["cell_omics"])

        # 2. Drug encoding (shared encoder for both drugs)
        a_atom, a_bond = self.drug_encoder(data["drug_a"])
        b_atom, b_bond = self.drug_encoder(data["drug_b"])

        # 3. Cell-drug modulation
        da_atom = self.atom_modulator(cell_emb, a_atom)
        db_atom = self.atom_modulator(cell_emb, b_atom)
        da_bond = self.bond_modulator(cell_emb, a_bond)
        db_bond = self.bond_modulator(cell_emb, b_bond)

        # 4. Dual-stream prediction + DS fusion
        evidence = self.predictor(da_atom, db_atom, da_bond, db_bond, cell_emb)

        return evidence, cell_recon
