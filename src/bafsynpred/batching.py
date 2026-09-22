# -*- coding: utf-8 -*-
"""Vectorized, bit-equivalent mini-batch assembly used by default."""
import numpy as np
import torch
from .chemistry import HYPEREDGE_ATTR_DIM


def _num_hyperedges(data):
    value = getattr(data, "num_atom_hyperedges", None)
    if value is None:
        h_type = getattr(data, "atom_hyperedge_type", None)
        return int(h_type.numel()) if h_type is not None else 0
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _per_element_offsets(sizes_np, counts_np):
    """每图起始偏移按counts展开; sizes_np: 每图尺寸, counts_np: 每图元素数"""
    starts = np.concatenate(([0], np.cumsum(sizes_np)[:-1]))
    return torch.from_numpy(np.repeat(starts, counts_np))


def _batch_single_drug(data_list):
    n = len(data_list)
    device = data_list[0].x_old.device

    x_old_l, ei_old_l = [], []
    x_l, ei_l, ea_l = [], [], []
    ea_old_l = []
    he_idx_l, he_type_l, he_attr_l = [], [], []
    atom_counts = np.zeros(n, dtype=np.int64)
    bond_counts = np.zeros(n, dtype=np.int64)
    he_counts = np.zeros(n, dtype=np.int64)
    ei_old_counts = np.zeros(n, dtype=np.int64)
    ei_counts = np.zeros(n, dtype=np.int64)

    for i, d in enumerate(data_list):
        x_old_l.append(d.x_old)
        ei_old_l.append(d.edge_index_old)
        x_l.append(d.x)
        ei_l.append(d.edge_index)
        ea_l.append(d.edge_attr)
        ea_old_l.append(d.edge_attr_old)
        he_idx_l.append(d.atom_hyperedge_index)
        he_type_l.append(d.atom_hyperedge_type)
        he_attr_l.append(d.atom_hyperedge_attr)
        atom_counts[i] = d.x_old.size(0)
        bond_counts[i] = d.x.size(0)
        he_counts[i] = _num_hyperedges(d)
        ei_old_counts[i] = d.edge_index_old.size(1)
        ei_counts[i] = d.edge_index.size(1)

    arange_n = torch.arange(n)

    # ---- atom view ----
    x_old = torch.cat(x_old_l, dim=0)
    batch_old = torch.repeat_interleave(arange_n, torch.from_numpy(atom_counts))

    ei_old_cat = torch.cat(ei_old_l, dim=1)
    if ei_old_cat.numel() > 0:
        off = _per_element_offsets(atom_counts, ei_old_counts).to(device)
        ei_old_cat = ei_old_cat + off.unsqueeze(0)

    # ---- hyperedges ----
    nonempty_he = [h for h in he_idx_l if h.numel() > 0]
    if nonempty_he:
        he_idx_cat = torch.cat(nonempty_he, dim=1)
        he_e_counts = np.array([h.size(1) for h in he_idx_l], dtype=np.int64)
        node_off = _per_element_offsets(atom_counts, he_e_counts).to(device)
        he_off = _per_element_offsets(he_counts, he_e_counts).to(device)
        he_idx_cat = he_idx_cat + torch.stack([node_off, he_off], dim=0)
    else:
        he_idx_cat = torch.empty((2, 0), dtype=torch.long, device=device)

    if he_counts.sum() > 0:
        he_type = torch.cat([t for t in he_type_l if t.numel() > 0], dim=0).long()
        he_attr = torch.cat([a for a in he_attr_l if a.numel() > 0], dim=0).float()
    else:
        he_type = torch.empty((0,), dtype=torch.long, device=device)
        he_attr = torch.empty((0, HYPEREDGE_ATTR_DIM), dtype=torch.float32, device=device)

    # ---- bond view ----
    x_bond = torch.cat(x_l, dim=0)
    batch_bond = torch.repeat_interleave(arange_n, torch.from_numpy(bond_counts))
    ei_cat = torch.cat(ei_l, dim=1)
    if ei_cat.numel() > 0:
        off_b = _per_element_offsets(bond_counts, ei_counts).to(device)
        ei_cat = ei_cat + off_b.unsqueeze(0)
    ea_cat = torch.cat(ea_l, dim=0)

    return {
        "x": x_bond,
        "edge_index": ei_cat,
        "edge_attr": ea_cat,
        "batch": batch_bond,
        "x_old": x_old,
        "edge_index_old": ei_old_cat,
        "edge_attr_old": torch.cat(ea_old_l, dim=0),
        "batch_old": batch_old,
        "atom_hyperedge_index": he_idx_cat,
        "atom_hyperedge_type": he_type,
        "atom_hyperedge_attr": he_attr,
    }


def collate_drug_pairs(batch):
    drug_a_list = [item["drug_a"] for item in batch]
    drug_b_list = [item["drug_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    depmap_ids = [item["depmap_id"] for item in batch]

    final_batch = {
        "drug_a": _batch_single_drug(drug_a_list),
        "drug_b": _batch_single_drug(drug_b_list),
        "label": labels,
        "depmap_id": depmap_ids,
    }

    if "cell_omics" in batch[0]:
        cell_tensors = [torch.from_numpy(item["cell_omics"]).float() for item in batch]
        final_batch["cell_omics"] = torch.stack(cell_tensors, dim=0)

    return final_batch
