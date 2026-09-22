"""Load the canonical fold-0 checkpoint and run one inference batch."""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bafsynpred.batching import collate_drug_pairs
from bafsynpred.data import DrugPairDataset
from bafsynpred.model import BAFSynPred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    run_dir = os.path.join(PROJECT_ROOT, "weights", "canonical_5fold")
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as handle:
        config = json.load(handle)["configuration"]
    data_root = os.path.join(PROJECT_ROOT, "data")
    dataset = DrugPairDataset(
        csv_path=os.path.join(data_root, "10183_drug_feature_0_30.csv"),
        cell_data_path=os.path.join(data_root, "10183_cell_features_977.npy"),
        split="val",
        kfold_path=os.path.join(data_root, "kfold_splits_processed", "5_fold_splits.npy"),
        fold=0,
        cache_dir=os.path.join(data_root, "cache", "bafsynpred-final"),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_drug_pairs)
    batch = next(iter(loader))
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model = BAFSynPred(
        num_features_xd=config["num_features_xd"],
        num_features_xa=config["num_features_xa"],
        num_features_xt=config["num_features_xt"],
        embed_dim=config["embed_dim"],
        num_hyperedge_types=config["num_hyperedge_types"],
        hyperedge_attr_dim=config["hyperedge_attr_dim"],
        cell_noise_std=config["cell_noise_std"],
        cell_drug_dropout=config["cell_drug_dropout"],
        bond_k=config["bond_k"],
    ).to(device).eval()
    checkpoint = os.path.join(run_dir, "best_model_fold_0.pth")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    with torch.no_grad():
        evidence, reconstruction = model(batch)
    if tuple(evidence.shape) != (2, 2):
        raise RuntimeError("Unexpected evidence shape: {}".format(tuple(evidence.shape)))
    if not torch.isfinite(evidence).all() or (evidence < 0).any():
        raise RuntimeError("Checkpoint produced invalid evidence")
    if reconstruction is not None:
        raise RuntimeError("The reconstruction head must be disabled during inference")
    print("Checkpoint smoke test passed on {}.".format(device))


if __name__ == "__main__":
    main()

