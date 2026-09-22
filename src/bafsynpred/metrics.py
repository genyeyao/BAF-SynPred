import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef,cohen_kappa_score
)


def metric(true_labels, predictions, threshold=0.5):
    """
    Calculates comprehensive binary classification metrics.

    Args:
        true_labels (array-like): True binary labels (0 or log).
        predictions (array-like):
            - If values are in [0, 1] (probabilities), applies the threshold.
            - If values are already 0/1 (hard predictions), treats them as classes.
        threshold (float): Decision threshold for converting probabilities to classes. Default: 0.5.

    Returns:
        dict: Dictionary containing classification metrics.
    """
    y_true = np.asarray(true_labels).flatten()
    y_pred_raw = np.asarray(predictions).flatten()

    # Input validation
    if len(y_true) == 0 or len(y_pred_raw) == 0:
        print("Warning: Empty input arrays provided to metric function.")
        return _nan_metrics_dict()

    if len(y_true) != len(y_pred_raw):
        raise ValueError("Length of true_labels and predictions must be equal.")

    if not np.isfinite(y_pred_raw).all() or np.any(y_pred_raw < 0) or np.any(y_pred_raw > 1):
        raise ValueError("predictions must be finite scores in [0, 1].")
    y_pred_proba = y_pred_raw
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Ensure labels are binary
    unique_labels = np.unique(y_true)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError("true_labels must contain only 0 and 1 for binary classification.")

    # Basic metrics
    try:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)  # = Sensitivity
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bac = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

        # Ranking metrics use the supplied scores, including the valid edge case
        # in which all scores happen to be exactly zero or one.
        auc_roc = float('nan')
        auc_pr = float('nan')
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            auc_pr = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            # A single-class evaluation split has undefined ranking metrics.
            auc_roc = float('nan')
            auc_pr = float('nan')

        # Print results
        print(f"Binary Classification Metrics (threshold={threshold}):")
        print(f"  Accuracy:          {acc:.6f}")
        print(f"  Balanced Acc:      {bac:.6f}")
        print(f"  Precision:         {prec:.6f}")
        print(f"  Recall (Sens):     {rec:.6f}")
        print(f"  Specificity:       {specificity:.6f}")
        print(f"  F1 Score:          {f1:.6f}")
        print(f"  MCC:               {mcc:.6f}")
        print(f"  Cohen's Kappa:     {kappa:.6f}")
        print(f"  ROC-AUC:           {auc_roc:.6f}")
        print(f"  PR-AUC:            {auc_pr:.6f}")
        print(f"  Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        return {
            'Accuracy': acc,
            'Balanced_Accuracy': bac,
            'Precision': prec,
            'Recall': rec,
            'Specificity': specificity,
            'F1': f1,
            'MCC': mcc,
            'Kappa': kappa,
            'ROC_AUC': auc_roc,
            'PR_AUC': auc_pr,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }

    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
        return _nan_metrics_dict()


def _nan_metrics_dict():
    """Return a dict with all metrics as NaN."""
    return {
        'Accuracy': float('nan'),
        'Balanced_Accuracy': float('nan'),
        'Precision': float('nan'),
        'Recall': float('nan'),
        'Specificity': float('nan'),
        'F1': float('nan'),
        'MCC': float('nan'),
        'Kappa': float('nan'),
        'ROC_AUC': float('nan'),
        'PR_AUC': float('nan'),
        'TP': float('nan'),
        'TN': float('nan'),
        'FP': float('nan'),
        'FN': float('nan')
    }

# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
#
#
# def metric(true_labels, predictions):
#     """
#     更详细的回归任务评价指标
#     """
#     y_true = np.asarray(true_labels).flatten()
#     y_pred = np.asarray(predictions).flatten()
#
#     # 基础误差指标
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#
#     # 决定系数
#     r2 = r2_score(y_true, y_pred)  # 新增指标
#
#     # 相关系数指标
#     try:
#         r_pearson, _ = pearsonr(y_true, y_pred)
#         r_spearman, _ = spearmanr(y_true, y_pred)
#     except:
#         r_pearson, r_spearman = 0.0, 0.0
#
#     return {
#         'MSE': mse,
#         'RMSE': rmse,
#         'MAE': mae,
#         'R2': r2,
#         'Pearson': r_pearson,
#         'Spearman': r_spearman
#     }
