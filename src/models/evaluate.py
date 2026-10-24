"""
Metrics and evaluation plots.

Regression (ETA):   MAE  -> "on average we are off by X hours" (easy to explain)
                    RMSE -> punishes big misses more (a 20-hr miss hurts more than 4x 5-hr misses)
                    R²   -> share of the variation in transit time we explain
Classification:     ROC-AUC -> how well we RANK risky above safe shipments (threshold-free)
                    Precision -> of the shipments we flag, how many really were late
                    Recall    -> of the late shipments, how many we caught
                    F1        -> balance of the two
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,  # noqa: E402
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ape = np.abs(y_pred - y_true) / np.clip(y_true, 1, None)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
            "MAPE_%": float(np.mean(ape) * 100),
            "within_10pct_%": float(np.mean(ape <= 0.10) * 100)}


def best_f1_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that maximises F1 on VALIDATION data."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.nanargmax(f1[:-1])])


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true); pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"ROC_AUC": roc_auc_score(y_true, proba), "PR_AUC": average_precision_score(y_true, proba),
            "F1": f1_score(y_true, pred, zero_division=0),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "Accuracy": accuracy_score(y_true, pred), "threshold": threshold,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_eta(y_true, y_pred, path, title="ETA model – test set"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].scatter(y_true, y_pred, s=6, alpha=0.35, color="#2b6cb0")
    lim = [0, max(y_true.max(), y_pred.max()) * 1.02]
    ax[0].plot(lim, lim, "--", color="#c53030", lw=1.2, label="perfect prediction")
    ax[0].set(xlabel="Actual transit time (hrs)", ylabel="Predicted (hrs)", title="Predicted vs actual")
    ax[0].legend()
    res = y_pred - y_true
    ax[1].hist(res, bins=60, color="#2f855a", alpha=0.85)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set(xlabel="Prediction error (hrs)  [negative = we were optimistic]", ylabel="Shipments",
              title="Error distribution")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_classifier(y_true, proba, threshold, path, title="Delay model – test set"):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax[0].plot(fpr, tpr, color="#2b6cb0", lw=2, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].legend()
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    p, r, _ = precision_recall_curve(y_true, proba)
    ax[1].plot(r, p, color="#805ad5", lw=2)
    ax[1].axhline(np.mean(y_true), ls="--", color="grey", label="random guess")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall"); ax[1].legend()
    cm = confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int), labels=[0, 1])
    ax[2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[2].text(j, i, f"{v}", ha="center", va="center", fontsize=13,
                   color="white" if v > cm.max() / 2 else "black")
    ax[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["On time", "Late"], yticklabels=["On time", "Late"],
              xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {threshold:.2f})")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
