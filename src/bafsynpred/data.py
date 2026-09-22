import hashlib
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from .chemistry import HYPEREDGE_ATTR_DIM, smiles_to_data_chem_hypergraph


def canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _split_stem(kfold_path: str) -> str:
    stem = os.path.splitext(os.path.basename(kfold_path))[0]
    return stem or "unknown_split"


def _short_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


class DrugPairDataset(TorchDataset):
    def __init__(
        self,
        csv_path: str,
        cell_data_path: Optional[str] = None,
        split: str = "train",
        kfold_path: Optional[str] = None,
        fold: int = 0,
        cache_dir: Optional[str] = None,
        force_reprocess: bool = False,
        expected_cell_dim: int = 977,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if kfold_path is None:
            raise ValueError("kfold_path is required")
        if not os.path.exists(kfold_path):
            raise FileNotFoundError(f"K-fold split file does not exist: {kfold_path}")

        self.csv_path = csv_path
        self.cell_data_path = cell_data_path
        self.split = split
        self.fold = fold
        self.expected_cell_dim = expected_cell_dim

        base_dir = os.path.dirname(csv_path)
        cache_key = "{}-{}-{}".format(
            _split_stem(kfold_path), _short_sha256(csv_path), _short_sha256(kfold_path)
        )
        if cache_dir is None:
            self.cache_root = os.path.join(base_dir, "cache", "bafsynpred-final", cache_key)
        else:
            self.cache_root = os.path.join(cache_dir, cache_key)

        full_data = np.load(kfold_path, allow_pickle=True).item()
        if "splits" not in full_data:
            raise ValueError("K-fold split file must contain a 'splits' key")

        cv_splits = full_data["splits"]
        if fold < 0 or fold >= len(cv_splits):
            raise ValueError(f"fold {fold} is out of range for {len(cv_splits)} folds")

        split_key = f"{split}_idx"
        if split_key not in cv_splits[fold]:
            raise ValueError(f"Fold {fold} is missing {split_key}")
        indices = np.asarray(cv_splits[fold][split_key], dtype=np.int64)
        self.cache_dir = os.path.join(self.cache_root, f"fold_{fold}", split)
        print(f"[Load] Fold {fold} - {split}: {len(indices)} samples | cache: {self.cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        required_cols = ["SMILES1", "SMILES2", "Label", "DepMapID"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(df)):
            raise ValueError(f"Fold {fold} {split} contains empty or out-of-range indices")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Fold {fold} {split} contains duplicate indices")

        self.df = df.iloc[indices].copy()
        self.df["_source_index"] = indices
        self.df = self.df.reset_index(drop=True)
        labels = set(self.df["Label"].dropna().unique().tolist())
        if not labels.issubset({0, 1}) or self.df["Label"].isna().any():
            raise ValueError(f"Labels must be complete and binary; found {sorted(labels)}")
        self._build_graphs(force_reprocess)

        if cell_data_path is not None:
            self.cell_dict = np.load(cell_data_path, allow_pickle=True).item()
            missing_cells = sorted(set(self.depmap_ids) - set(self.cell_dict))
            if missing_cells:
                raise ValueError(f"Missing cell features for {missing_cells[:5]}")
            bad_cells = [
                key for key in set(self.depmap_ids)
                if np.asarray(self.cell_dict[key]).shape != (self.expected_cell_dim,)
                or not np.isfinite(np.asarray(self.cell_dict[key], dtype=np.float32)).all()
            ]
            if bad_cells:
                raise ValueError(
                    f"Cell features must be finite {self.expected_cell_dim}-vectors; invalid: {bad_cells[:5]}"
                )
        else:
            self.cell_dict = None

    def _build_graphs(self, force_reprocess: bool = False):
        cache_files = {
            "drugA": os.path.join(self.cache_dir, "drugA.pt"),
            "drugB": os.path.join(self.cache_dir, "drugB.pt"),
            "labels": os.path.join(self.cache_dir, "labels.pt"),
            "depmap_ids": os.path.join(self.cache_dir, "depmap_ids.pt"),
        }

        if force_reprocess:
            for f_path in cache_files.values():
                if os.path.exists(f_path):
                    os.remove(f_path)

        if not force_reprocess and all(os.path.exists(path) for path in cache_files.values()):
            self.drug_a_list = torch.load(cache_files["drugA"])
            self.drug_b_list = torch.load(cache_files["drugB"])
            self.labels = torch.load(cache_files["labels"])
            self.depmap_ids = torch.load(cache_files["depmap_ids"])
            self._validate_cached_graphs()
            return

        smiles_to_graph = {}

        def get_or_create_graph(smiles: str):
            canon = canonical_smiles(smiles)
            if canon is None:
                return None
            if canon in smiles_to_graph:
                return smiles_to_graph[canon]
            graph = smiles_to_data_chem_hypergraph(canon)
            if graph is None:
                return None
            clean_graph = Data(**{k: v for k, v in graph.items() if v is not None})
            smiles_to_graph[canon] = clean_graph
            return clean_graph

        drug_a_list, drug_b_list, labels_list, depmap_ids = [], [], [], []
        failures = []
        desc_text = f"Building Chemistry Hypergraphs (Fold {self.fold}-{self.split})"

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc=desc_text):
            try:
                g1 = get_or_create_graph(row["SMILES1"])
                g2 = get_or_create_graph(row["SMILES2"])
                if g1 is None or g2 is None:
                    raise ValueError("Invalid SMILES")
                drug_a_list.append(g1)
                drug_b_list.append(g2)
                labels_list.append(float(row["Label"]))
                depmap_ids.append(row["DepMapID"])
            except Exception as exc:
                failures.append((int(row["_source_index"]), repr(exc)))

        if failures:
            raise ValueError(
                "Molecular preprocessing failed; no samples were silently dropped. "
                f"First failures: {failures[:5]}"
            )

        self.drug_a_list = drug_a_list
        self.drug_b_list = drug_b_list
        self.labels = torch.tensor(labels_list, dtype=torch.float32)
        self.depmap_ids = depmap_ids
        self._validate_cached_graphs()

        torch.save(self.drug_a_list, cache_files["drugA"])
        torch.save(self.drug_b_list, cache_files["drugB"])
        torch.save(self.labels, cache_files["labels"])
        torch.save(self.depmap_ids, cache_files["depmap_ids"])
        print(f"[Cache] Chemistry hypergraph data cached to {self.cache_dir}")

    def _validate_cached_graphs(self):
        if len(self.drug_a_list) != len(self.labels) or len(self.drug_b_list) != len(self.labels):
            raise RuntimeError("Cached graph count does not match label count")
        if len(self.depmap_ids) != len(self.labels):
            raise RuntimeError("Cached cell-line identifier count does not match label count")
        checked = set()
        for graph in self.drug_a_list + self.drug_b_list:
            if id(graph) in checked:
                continue
            checked.add(id(graph))
            required = [
                "x", "edge_index", "edge_attr", "x_old", "edge_index_old", "edge_attr_old",
                "atom_hyperedge_index", "atom_hyperedge_type", "atom_hyperedge_attr",
            ]
            missing = [key for key in required if not hasattr(graph, key)]
            if missing:
                raise RuntimeError(
                    f"Cached graph lacks required fields {missing}. "
                    "Use a fresh chemistry-hypergraph cache or set force_reprocess=True."
                )
            if graph.x_old.dim() != 2 or graph.x_old.size(1) != 55:
                raise RuntimeError(f"Expected 55 atom features, found {tuple(graph.x_old.shape)}")
            if graph.x.dim() != 2 or graph.x.size(1) != 117:
                raise RuntimeError(f"Expected 117 bond-node features, found {tuple(graph.x.shape)}")
            if graph.edge_index_old.dim() != 2 or graph.edge_index_old.size(0) != 2:
                raise RuntimeError("Atom edge_index must have shape [2, E]")
            if graph.edge_index_old.size(1) != graph.x.size(0):
                raise RuntimeError(
                    "Bond-node rows must remain aligned with directed atom-edge columns"
                )
            if graph.atom_hyperedge_index.dim() != 2 or graph.atom_hyperedge_index.size(0) != 2:
                raise RuntimeError("Hypergraph incidence must have shape [2, M]")
            if graph.atom_hyperedge_attr.dim() != 2 or graph.atom_hyperedge_attr.size(1) != HYPEREDGE_ATTR_DIM:
                raise RuntimeError("Hyperedge attribute dimension does not match the model")

    def __len__(self):
        return len(self.drug_a_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "drug_a": self.drug_a_list[idx],
            "drug_b": self.drug_b_list[idx],
            "label": self.labels[idx].item(),
            "depmap_id": self.depmap_ids[idx],
        }
        if self.cell_dict is not None:
            sample["cell_omics"] = self.cell_dict[self.depmap_ids[idx]]
        return sample


def _num_hyperedges(data: Data) -> int:
    value = getattr(data, "num_atom_hyperedges", None)
    if value is None:
        h_type = getattr(data, "atom_hyperedge_type", None)
        return int(h_type.numel()) if h_type is not None else 0
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _batch_atom_hypergraphs_reference(data_list):
    incidence_parts = []
    type_parts = []
    attr_parts = []
    node_offset = 0
    hyperedge_offset = 0

    for data in data_list:
        h_index = data.atom_hyperedge_index
        h_type = data.atom_hyperedge_type
        h_attr = data.atom_hyperedge_attr
        n_nodes = int(data.x_old.size(0))
        n_hyperedges = _num_hyperedges(data)

        if h_index.numel() > 0:
            shifted = h_index.clone()
            shifted[0] += node_offset
            shifted[1] += hyperedge_offset
            incidence_parts.append(shifted)

        if n_hyperedges > 0:
            type_parts.append(h_type.long())
            attr_parts.append(h_attr.float())

        node_offset += n_nodes
        hyperedge_offset += n_hyperedges

    device = data_list[0].x_old.device
    if incidence_parts:
        hyperedge_index = torch.cat(incidence_parts, dim=1)
    else:
        hyperedge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    if type_parts:
        hyperedge_type = torch.cat(type_parts, dim=0)
        hyperedge_attr = torch.cat(attr_parts, dim=0)
    else:
        hyperedge_type = torch.empty((0,), dtype=torch.long, device=device)
        hyperedge_attr = torch.empty((0, HYPEREDGE_ATTR_DIM), dtype=torch.float32, device=device)

    return hyperedge_index, hyperedge_type, hyperedge_attr


def _batch_single_drug_reference(data_list):
    exclude_keys = [
        "x_old",
        "edge_index_old",
        "edge_attr_old",
        "atom_hyperedge_index",
        "atom_hyperedge_type",
        "atom_hyperedge_attr",
        "num_atom_hyperedges",
        "num_atom_nodes",
    ]
    batch_line = Batch.from_data_list(data_list, exclude_keys=exclude_keys)
    atom_list = [
        Data(x=data.x_old, edge_index=data.edge_index_old, edge_attr=data.edge_attr_old)
        for data in data_list
    ]
    batch_atom = Batch.from_data_list(atom_list)
    hyperedge_index, hyperedge_type, hyperedge_attr = _batch_atom_hypergraphs_reference(data_list)

    return {
        "x": batch_line.x,
        "edge_index": batch_line.edge_index,
        "edge_attr": batch_line.edge_attr,
        "batch": batch_line.batch,
        "x_old": batch_atom.x,
        "edge_index_old": batch_atom.edge_index,
        "edge_attr_old": batch_atom.edge_attr,
        "batch_old": batch_atom.batch,
        "atom_hyperedge_index": hyperedge_index,
        "atom_hyperedge_type": hyperedge_type,
        "atom_hyperedge_attr": hyperedge_attr,
    }


def collate_drug_pairs_reference(batch):
    drug_a_list = [item["drug_a"] for item in batch]
    drug_b_list = [item["drug_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    depmap_ids = [item["depmap_id"] for item in batch]

    final_batch = {
        "drug_a": _batch_single_drug_reference(drug_a_list),
        "drug_b": _batch_single_drug_reference(drug_b_list),
        "label": labels,
        "depmap_id": depmap_ids,
    }

    if "cell_omics" in batch[0]:
        cell_tensors = [torch.from_numpy(item["cell_omics"]).float() for item in batch]
        final_batch["cell_omics"] = torch.stack(cell_tensors, dim=0)

    return final_batch


# The vectorized implementation is the public/default batch assembler.  The
# reference implementation above is retained only for exact-equivalence tests.
from .batching import collate_drug_pairs


if __name__ == "__main__":
    CSV_PATH = "./data/10183_drug_feature_0_30.csv"
    CELL_PATH = "./data/10183_cell_features_977.npy"
    KFOLD_PATH = "./data/kfold_splits_processed/5_fold_splits.npy"

    print("[Start] Building chemistry-aware hypergraph caches...")
    for fold_id in range(5):
        print(f"\n[Fold {fold_id}] Processing...")
        DrugPairDataset(
            csv_path=CSV_PATH,
            cell_data_path=CELL_PATH,
            split="train",
            kfold_path=KFOLD_PATH,
            fold=fold_id,
            force_reprocess=True,
        )
        DrugPairDataset(
            csv_path=CSV_PATH,
            cell_data_path=CELL_PATH,
            split="val",
            kfold_path=KFOLD_PATH,
            fold=fold_id,
            force_reprocess=True,
        )
    print("\n[Finish] Chemistry-aware hypergraph preprocessing completed.")
