"""Evaluate saved five-fold checkpoints from a final training run."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bafsynpred.batching import collate_drug_pairs
from bafsynpred.data import DrugPairDataset
from bafsynpred.model import BAFSynPred, MODEL_NAME, MODEL_VERSION
from bafsynpred.train import evaluate, file_sha256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the saved best checkpoint for each validation fold."
    )
    parser.add_argument("--run-dir", required=True, help="Completed training run directory.")
    parser.add_argument("--data-root", required=True, help="Directory containing the data files.")
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Evaluation batch size; defaults to the training configuration.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path; defaults to <run-dir>/evaluation_summary.json.")
    return parser.parse_args()


def build_model(config, device):
    return BAFSynPred(
        num_features_xd=config["num_features_xd"],
        num_features_xa=config["num_features_xa"],
        num_features_xt=config["num_features_xt"],
        embed_dim=config["embed_dim"],
        num_hyperedge_types=config["num_hyperedge_types"],
        hyperedge_attr_dim=config["hyperedge_attr_dim"],
        cell_noise_std=config.get("cell_noise_std", 0.05),
        cell_drug_dropout=config.get("cell_drug_dropout", 0.2),
        bond_k=config.get("bond_k", 3),
    ).to(device)


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    data_root = os.path.abspath(args.data_root)
    with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    config = manifest["configuration"]
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    batch_size = int(args.batch_size or config["batch_size"])
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    csv_path = os.path.join(data_root, "10183_drug_feature_0_30.csv")
    cell_path = os.path.join(data_root, "10183_cell_features_977.npy")
    split_path = os.path.join(data_root, "kfold_splits_processed", "5_fold_splits.npy")
    cache_dir = os.path.join(data_root, "cache", "bafsynpred-final")
    num_folds = int(config.get("num_folds", 5))
    fold_results = []
    for fold in range(num_folds):
        checkpoint = os.path.join(run_dir, "best_model_fold_{}.pth".format(fold))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint))
        dataset = DrugPairDataset(
            csv_path=csv_path, cell_data_path=cell_path, split="val",
            kfold_path=split_path, fold=fold, cache_dir=cache_dir,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_drug_pairs, pin_memory=device.type == "cuda",
        )
        model = build_model(config, device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        _, _, metrics = evaluate(model, loader, device)
        fold_results.append({"fold": fold, "checkpoint": checkpoint, **metrics})

    metric_names = sorted({key for row in fold_results for key in row if key not in ("fold", "checkpoint")})
    aggregate = {
        name: {"mean": float(np.mean([row[name] for row in fold_results])),
               "std": float(np.std([row[name] for row in fold_results]))}
        for name in metric_names
    }
    output = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "run_dir": run_dir,
        "configuration": config,
        "device": str(device),
        "checkpoint_sha256": {
            str(row["fold"]): file_sha256(row["checkpoint"]) for row in fold_results
        },
        "per_fold": fold_results,
        "metrics": aggregate,
    }
    output_path = os.path.abspath(args.output or os.path.join(run_dir, "evaluation_summary.json"))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
