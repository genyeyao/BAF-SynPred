"""
BAF-SynPred final training entry point
==================================
5-fold cross-validation with fixed learning rate.

Model: BAFSynPred selected single-layer architecture
  - CellDenoisingEncoder:  977->2048->512->128 (expand-then-compress)
  - DrugHypergraphEncoder: atom hypergraph + bond line-graph + gated cross-talk
  - CellDrugModulator:     FiLM (zero-init)
  - DualStreamPredictor:   EDL + Dempster-Shafer fusion

Output structure:
  experiments/01_training_runs/main_results/<timestamp>_main/
    ├── best_model_fold_{0-4}.pth   (per-fold weights)
    ├── training_<timestamp>.log    (full log with best-score changes)
    ├── per_fold_metrics.csv        (10 metrics x 5 folds)
    └── experiment_summary.json     (complete results)

Training config:
  - Optimizer: Adam, lr=0.0002 (fixed, no scheduler)
  - Loss: Adaptive KL Evidential Deep Learning (EDL)
  - Regularization: gradient clipping (1.0), early stopping (patience=50)
  - Class weighting: capped positive weight (max 1.5)
  - Cell denoising: noise_std=0.05, recon_weight=0.01
  - Seed: 42
"""

import argparse
import csv
import ctypes
import hashlib
import io
import json
import logging
import os
import random
import platform
import time
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------
# Release metadata.
# ---------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_FAMILY = "bafsynpred"
MODEL_FILE = "src/bafsynpred/model.py"
TRAINER_FILE = "src/bafsynpred/train.py"
PIPELINE = "vectorized_collation_v1"
FOLD_SEED_POLICY = "base_seed_plus_fold_v1"

from .chemistry import HYPEREDGE_ATTR_DIM, NUM_HYPEREDGE_TYPES
from .data import DrugPairDataset, collate_drug_pairs
from .losses import adaptive_edl_loss
from .metrics import metric
from .model import BAFSynPred, MODEL_NAME, MODEL_VERSION
from .validation import audit_inputs, save_audit

# ================================================================
# Constants
# ================================================================
METRICS_KEYS = ["ROC_AUC", "PR_AUC", "F1", "MCC", "Accuracy",
                "Precision", "Recall", "Specificity", "Kappa", "Balanced_Accuracy"]

# ================================================================
# Configuration (no side effects on import)
# ================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SCRIPT_DIR = _PROJECT_ROOT

CONFIG = {
    "csv_path": os.path.join(_PROJECT_ROOT, "data", "10183_drug_feature_0_30.csv"),
    "cell_data_path": os.path.join(_PROJECT_ROOT, "data", "10183_cell_features_977.npy"),
    "kfold_path": os.path.join(_PROJECT_ROOT, "data", "kfold_splits_processed", "5_fold_splits.npy"),
    "cache_dir": os.path.join(_PROJECT_ROOT, "data", "cache", "bafsynpred-final"),
    "num_features_xd": 117,
    "num_features_xa": 55,
    "num_features_xt": 977,
    "embed_dim": 128,
    "num_hyperedge_types": NUM_HYPEREDGE_TYPES,
    "hyperedge_attr_dim": HYPEREDGE_ATTR_DIM,
    "bond_k": 3,
    "batch_size": 64,
    "num_epochs": 200,
    "lr": 0.0002,
    "weight_decay": 0.0,
    "early_stop_patience": 50,
    "num_folds": 5,
    "use_class_weighted_edl": True,
    "edl_pos_weight_cap": 1.5,
    "cell_noise_std": 0.05,
    "cell_drug_dropout": 0.2,
    "recon_weight": 0.01,
    "adaptive_kl_beta": 0.5,
    "grad_clip_norm": 1.0,
    "seed": 42,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Utilities
# ================================================================
def move_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)) and obj and all(torch.is_tensor(i) for i in obj):
        return type(obj)(i.to(device, non_blocking=True) for i in obj)
    return obj


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    source_root = os.path.join(_PROJECT_ROOT, "src")
    manifest = {}
    for root, _, files in os.walk(source_root):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            relative = os.path.relpath(path, _PROJECT_ROOT).replace(os.sep, "/")
            manifest[relative] = file_sha256(path)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Train BAF-SynPred with five-fold cross-validation.")
    parser.add_argument(
        "--data-root",
        default=os.path.join(_PROJECT_ROOT, "data"),
        help="Directory containing the interaction CSV, cell features, and split file.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(_PROJECT_ROOT, "runs"),
        help="Directory in which serialized run folders are created.",
    )
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--high-priority",
        action="store_true",
        help="On Windows, set the training Python process to High priority before timing starts.",
    )
    parser.add_argument("--epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--embed-dim", type=int, default=CONFIG["embed_dim"])
    parser.add_argument("--bond-k", type=int, default=CONFIG["bond_k"])
    parser.add_argument("--cell-noise-std", type=float, default=CONFIG["cell_noise_std"])
    parser.add_argument("--cell-drug-dropout", type=float, default=CONFIG["cell_drug_dropout"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=CONFIG["weight_decay"])
    parser.add_argument("--recon-weight", type=float, default=CONFIG["recon_weight"])
    parser.add_argument("--adaptive-kl-beta", type=float, default=CONFIG["adaptive_kl_beta"])
    parser.add_argument("--edl-pos-weight-cap", type=float,
                        default=CONFIG["edl_pos_weight_cap"])
    parser.add_argument("--grad-clip-norm", type=float, default=CONFIG["grad_clip_norm"])
    parser.add_argument("--early-stop-patience", type=int,
                        default=CONFIG["early_stop_patience"])
    return parser.parse_args()


def set_high_priority(enabled):
    if not enabled:
        return "Normal"
    if os.name != "nt":
        raise RuntimeError("--high-priority is currently supported only on Windows")
    high_priority_class = 0x00000080
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    if not kernel32.SetPriorityClass(handle, high_priority_class):
        raise ctypes.WinError()
    return "High"


def synchronize_device(device):
    """Finish queued CUDA work before reading a wall-clock timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_paths(data_root):
    data_root = os.path.abspath(data_root)
    CONFIG["csv_path"] = os.path.join(data_root, "10183_drug_feature_0_30.csv")
    CONFIG["cell_data_path"] = os.path.join(data_root, "10183_cell_features_977.npy")
    CONFIG["kfold_path"] = os.path.join(data_root, "kfold_splits_processed", "5_fold_splits.npy")
    CONFIG["cache_dir"] = os.path.join(data_root, "cache", "bafsynpred-final")
    for key in ("csv_path", "cell_data_path", "kfold_path"):
        if not os.path.isfile(CONFIG[key]):
            raise FileNotFoundError("Required input not found: {}".format(CONFIG[key]))


def build_class_weights(labels, device, max_pos_weight=2.0):
    labels = labels.float()
    pos_count = labels.sum().item()
    neg_count = labels.numel() - pos_count
    raw_pos_weight = neg_count / (pos_count + 1e-8)
    pos_weight = min(raw_pos_weight, max_pos_weight)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    return class_weights, raw_pos_weight, pos_count, neg_count


# ================================================================
# Training and evaluation
# ================================================================
def train_one_epoch(model, loader, optimizer, device, epoch, class_weights=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        evidence, cell_recon = model(batch)
        labels = batch["label"].float()
        loss = adaptive_edl_loss(
            evidence, labels, epoch,
            class_weights=class_weights,
            adaptive_kl_beta=CONFIG["adaptive_kl_beta"])
        if cell_recon is not None and CONFIG["recon_weight"] > 0:
            loss = loss + CONFIG["recon_weight"] * F.mse_loss(cell_recon, batch["cell_omics"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            evidence, _ = model(batch)
            labels = batch["label"].float()
            loss = adaptive_edl_loss(evidence, labels, epoch=999, adaptive_kl_beta=0.0)
            total_loss += loss.item()
            n_batches += 1
            alpha = evidence + 1.0
            probs = alpha / alpha.sum(dim=1, keepdim=True)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    buf = io.StringIO()
    with redirect_stdout(buf):
        metrics_dict = metric(all_labels, all_probs)
    score = metrics_dict.get("ROC_AUC", float("-inf"))
    return total_loss / max(n_batches, 1), score, metrics_dict


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    process_priority = set_high_priority(args.high_priority)
    configure_paths(args.data_root)
    CONFIG["num_epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["embed_dim"] = args.embed_dim
    CONFIG["bond_k"] = args.bond_k
    CONFIG["cell_noise_std"] = args.cell_noise_std
    CONFIG["cell_drug_dropout"] = args.cell_drug_dropout
    CONFIG["lr"] = args.lr
    CONFIG["weight_decay"] = args.weight_decay
    CONFIG["recon_weight"] = args.recon_weight
    CONFIG["adaptive_kl_beta"] = args.adaptive_kl_beta
    CONFIG["edl_pos_weight_cap"] = args.edl_pos_weight_cap
    CONFIG["grad_clip_norm"] = args.grad_clip_norm
    CONFIG["early_stop_patience"] = args.early_stop_patience
    for name in ("num_epochs", "batch_size", "embed_dim", "bond_k",
                 "early_stop_patience"):
        if int(CONFIG[name]) < 1:
            raise ValueError("{} must be a positive integer".format(name))
    if any(CONFIG[name] < 0 for name in (
        "cell_noise_std", "cell_drug_dropout", "weight_decay", "recon_weight",
        "adaptive_kl_beta", "edl_pos_weight_cap", "grad_clip_norm")):
        raise ValueError("regularization and optimization coefficients must be non-negative")
    if CONFIG["cell_drug_dropout"] >= 1 or CONFIG["lr"] <= 0:
        raise ValueError("cell_drug_dropout must be < 1 and lr must be positive")
    if args.device is not None:
        DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    seed_everything(CONFIG["seed"])
    run_start = time.perf_counter()

    # Create output directory (only when run directly, not on import)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_ROOT = os.path.join(os.path.abspath(args.output_root), f"{timestamp}_{RUN_FAMILY}")
    os.makedirs(OUTPUT_ROOT, exist_ok=False)
    preflight = audit_inputs(
        CONFIG["csv_path"], CONFIG["cell_data_path"], CONFIG["kfold_path"],
        expected_cell_dim=CONFIG["num_features_xt"],
    )
    save_audit(preflight, os.path.join(OUTPUT_ROOT, "data_audit.json"))
    initial_manifest = {
        "status": "RUNNING",
        "run_id": f"{timestamp}_{RUN_FAMILY}",
        "model_version": MODEL_VERSION,
        "pipeline": PIPELINE,
        "fold_seed_policy": FOLD_SEED_POLICY,
        "fold_seeds": [CONFIG["seed"] + fold for fold in range(CONFIG["num_folds"])],
        "configuration": dict(CONFIG),
        "source_sha256": source_hashes(),
        "data_sha256": preflight["sha256"],
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(DEVICE),
            "process_priority": process_priority,
        },
    }
    with open(os.path.join(OUTPUT_ROOT, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(initial_manifest, handle, indent=2, ensure_ascii=False)

    # Setup logging
    class _FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    log_file = os.path.join(OUTPUT_ROOT, f"training_{timestamp}.log")
    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[_FlushFileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"  BAF-SynPred v{MODEL_VERSION}: Formal 5-Fold CV")
    logger.info(f"  Model: BAFSynPred ({MODEL_FILE})")
    logger.info(f"  Data pipeline: {PIPELINE}")
    logger.info("  Correct formulation: interleaved atom-bond incidence + standard EDL/DS fusion")
    logger.info(
        "  Cell Encoder: %d->2048->512->%d (expand-then-compress)",
        CONFIG["num_features_xt"], CONFIG["embed_dim"])
    logger.info(
        "  Drug Encoder: one HypergraphConv per atom/bond branch, width=%d, bond_k=%d + Gated-CrossTalk",
        CONFIG["embed_dim"], CONFIG["bond_k"])
    logger.info(f"  Modulator: Canonical FiLM (zero-init)")
    logger.info(f"  Predictor: Dual-stream EDL + DS fusion")
    logger.info(f"  LR: {CONFIG['lr']} (fixed) | Seed: {CONFIG['seed']}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Process priority: {process_priority}")
    logger.info(f"  Output: {OUTPUT_ROOT}")
    logger.info("=" * 60)

    all_fold_metrics = {k: [] for k in METRICS_KEYS}
    per_fold_results = []
    training_start = time.perf_counter()

    for fold in range(CONFIG["num_folds"]):
        # Make every fold reproducible independently. Without this reset, the
        # stopping epoch of an earlier fold changes model initialization,
        # shuffling, dropout, and denoising noise in every later fold.
        fold_seed = CONFIG["seed"] + fold
        seed_everything(fold_seed)
        logger.info(f"\n{'='*60}\n  FOLD {fold}\n{'='*60}")
        logger.info(f"Fold {fold} - deterministic seed: {fold_seed}")

        train_dataset = DrugPairDataset(
            csv_path=CONFIG["csv_path"], cell_data_path=CONFIG["cell_data_path"],
            split="train", kfold_path=CONFIG["kfold_path"], fold=fold,
            cache_dir=CONFIG["cache_dir"])
        val_dataset = DrugPairDataset(
            csv_path=CONFIG["csv_path"], cell_data_path=CONFIG["cell_data_path"],
            split="val", kfold_path=CONFIG["kfold_path"], fold=fold,
            cache_dir=CONFIG["cache_dir"])

        class_weights = None
        if CONFIG["use_class_weighted_edl"]:
            class_weights, raw_pw, pos_c, neg_c = build_class_weights(
                train_dataset.labels, DEVICE, max_pos_weight=CONFIG["edl_pos_weight_cap"])
            logger.info(f"Fold {fold} - pos={pos_c:.0f}, neg={neg_c:.0f}, weight={class_weights[1]:.4f}")

        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                                  shuffle=True, collate_fn=collate_drug_pairs, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                                shuffle=False, collate_fn=collate_drug_pairs, pin_memory=True)

        model = BAFSynPred(
            num_features_xd=CONFIG["num_features_xd"],
            num_features_xa=CONFIG["num_features_xa"],
            num_features_xt=CONFIG["num_features_xt"],
            embed_dim=CONFIG["embed_dim"],
            num_hyperedge_types=CONFIG["num_hyperedge_types"],
            hyperedge_attr_dim=CONFIG["hyperedge_attr_dim"],
            cell_noise_std=CONFIG["cell_noise_std"],
            cell_drug_dropout=CONFIG["cell_drug_dropout"],
            bond_k=CONFIG["bond_k"]).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Fold {fold} - Model params: {n_params:,}")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        best_score = -float("inf")
        best_epoch = 0
        patience_counter = 0
        model_path = os.path.join(OUTPUT_ROOT, f"best_model_fold_{fold}.pth")

        for epoch in range(CONFIG["num_epochs"]):
            synchronize_device(DEVICE)
            epoch_t0 = time.perf_counter()
            train_t0 = epoch_t0
            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE,
                                         epoch + 1, class_weights=class_weights)
            synchronize_device(DEVICE)
            train_seconds = time.perf_counter() - train_t0

            val_t0 = time.perf_counter()
            val_loss, val_score, val_metrics = evaluate(model, val_loader, DEVICE)
            synchronize_device(DEVICE)
            val_seconds = time.perf_counter() - val_t0

            checkpoint_seconds = 0.0
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch + 1
                patience_counter = 0
                checkpoint_t0 = time.perf_counter()
                torch.save(model.state_dict(), model_path)
                checkpoint_seconds = time.perf_counter() - checkpoint_t0
                epoch_seconds = time.perf_counter() - epoch_t0
                logger.info(
                    f"Fold {fold} - Ep {epoch+1} | {epoch_seconds:.1f}s "
                    f"(train {train_seconds:.1f}, val {val_seconds:.1f}, save {checkpoint_seconds:.1f}) | "
                    f"train={train_loss:.4f} val={val_loss:.4f} | "
                    f"AUC={val_score:.6f} *NEW BEST*")
            else:
                patience_counter += 1
                epoch_seconds = time.perf_counter() - epoch_t0
                if (epoch + 1) % 20 == 0:
                    logger.info(
                        f"Fold {fold} - Ep {epoch+1} | {epoch_seconds:.1f}s "
                        f"(train {train_seconds:.1f}, val {val_seconds:.1f}) | "
                        f"train={train_loss:.4f} val={val_loss:.4f} | "
                        f"AUC={val_score:.4f} best={best_score:.4f} (ep{best_epoch})")

            if val_score <= best_score and patience_counter >= CONFIG["early_stop_patience"]:
                logger.info(f"[STOP] Fold {fold} ep {epoch+1} | "
                            f"best_ep={best_epoch} best_AUC={best_score:.6f}")
                break

        # Final evaluation with best weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        _, _, final_metrics = evaluate(model, val_loader, DEVICE)

        fold_result = {"fold": fold, "fold_seed": fold_seed, "best_epoch": best_epoch,
                       "best_auc": best_score, "n_params": n_params}
        for k in METRICS_KEYS:
            v = final_metrics.get(k, np.nan)
            all_fold_metrics[k].append(v)
            fold_result[k] = v
        per_fold_results.append(fold_result)

        logger.info(f"Fold {fold} Final (ep{best_epoch}): "
                    + " | ".join(f"{k}={final_metrics.get(k,0):.6f}" for k in METRICS_KEYS))

    # ================================================================
    # Summary + Save
    # ================================================================
    synchronize_device(DEVICE)
    training_elapsed = (time.perf_counter() - training_start) / 60
    run_elapsed = (time.perf_counter() - run_start) / 60
    timing_summary = {
        "training_wall_minutes": round(training_elapsed, 3),
        "run_wall_minutes": round(run_elapsed, 3),
    }
    logger.info(f"\n{'#'*60}")
    logger.info(f"[DONE] 5-Fold CV | training={training_elapsed:.2f} min | run={run_elapsed:.2f} min")
    logger.info(f"{'#'*60}")

    summary = {
        "experiment": RUN_FAMILY,
        "run_id": f"{timestamp}_{RUN_FAMILY}",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_file": MODEL_FILE,
        "trainer_file": TRAINER_FILE,
        "pipeline": PIPELINE,
        "fold_seed_policy": FOLD_SEED_POLICY,
        "fold_seeds": [CONFIG["seed"] + fold for fold in range(CONFIG["num_folds"])],
        "model_sha256": file_sha256(os.path.join(_PROJECT_ROOT, MODEL_FILE)),
        "trainer_sha256": file_sha256(os.path.join(_PROJECT_ROOT, TRAINER_FILE)),
        "batching_sha256": file_sha256(
            os.path.join(_PROJECT_ROOT, "src", "bafsynpred", "batching.py")
        ),
        "data_files": {
            "interactions": {"path": CONFIG["csv_path"], "sha256": file_sha256(CONFIG["csv_path"])},
            "cell_features": {"path": CONFIG["cell_data_path"], "sha256": file_sha256(CONFIG["cell_data_path"])},
            "folds": {"path": CONFIG["kfold_path"], "sha256": file_sha256(CONFIG["kfold_path"])},
        },
        "configuration": dict(CONFIG),
        "data_audit": preflight,
        "source_sha256": source_hashes(),
        "n_params": n_params,
        "seed": CONFIG["seed"],
        "lr": CONFIG["lr"],
        "elapsed_min": round(training_elapsed, 3),
        "timing": timing_summary,
        "timestamp": timestamp,
        "metrics": {},
        "per_fold": per_fold_results,
    }

    logger.info("Final Results (Mean +/- Std):")
    for k in METRICS_KEYS:
        valid = [v for v in all_fold_metrics[k] if not np.isnan(v)]
        if valid:
            m, s = np.mean(valid), np.std(valid)
            logger.info(f"  {k}: {m:.6f} +/- {s:.6f}")
            summary["metrics"][k] = {"mean": round(float(m), 6), "std": round(float(s), 6)}

    # Save per-fold CSV
    csv_path = os.path.join(OUTPUT_ROOT, "per_fold_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fold_results[0].keys()))
        w.writeheader()
        for row in per_fold_results:
            w.writerow(row)
    logger.info(f"Per-fold CSV: {csv_path}")

    # Save experiment summary JSON
    json_path = os.path.join(OUTPUT_ROOT, "experiment_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary JSON: {json_path}")
    initial_manifest["status"] = "COMPLETE"
    initial_manifest["elapsed_min"] = round(training_elapsed, 3)
    initial_manifest["timing"] = timing_summary
    initial_manifest["summary_file"] = "experiment_summary.json"
    with open(os.path.join(OUTPUT_ROOT, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(initial_manifest, handle, indent=2, ensure_ascii=False)
