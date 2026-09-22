"""Build versioned chemistry-aware graph caches for all five folds."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bafsynpred.data import DrugPairDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    csv_path = os.path.join(args.data_root, "10183_drug_feature_0_30.csv")
    cell_path = os.path.join(args.data_root, "10183_cell_features_977.npy")
    split_path = os.path.join(args.data_root, "kfold_splits_processed", "5_fold_splits.npy")
    cache_dir = os.path.join(args.data_root, "cache", "bafsynpred-final")
    for fold in range(5):
        for split in ("train", "val"):
            DrugPairDataset(
                csv_path=csv_path,
                cell_data_path=cell_path,
                split=split,
                kfold_path=split_path,
                fold=fold,
                cache_dir=cache_dir,
                force_reprocess=args.force,
            )
    print("BAF-SynPred preprocessing completed successfully.")


if __name__ == "__main__":
    main()
