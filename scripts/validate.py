"""Validate data, splits, molecular graphs, and feature dimensions."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bafsynpred.validation import audit_inputs, save_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "runs", "data_audit.json"))
    args = parser.parse_args()
    report = audit_inputs(
        os.path.join(args.data_root, "10183_drug_feature_0_30.csv"),
        os.path.join(args.data_root, "10183_cell_features_977.npy"),
        os.path.join(args.data_root, "kfold_splits_processed", "5_fold_splits.npy"),
    )
    save_audit(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
