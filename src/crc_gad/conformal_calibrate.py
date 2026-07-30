"""Split conformal p-values and metrics — Eq. (9) in paper."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score


def conformal_pvalues(scores: np.ndarray, cal_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    """
    p_i = (1 + #{j in C : s_j >= s_i}) / (1 + |C|)
    Labels are NOT used.
    """
    cal_scores = scores[cal_idx]
    n_cal = len(cal_idx)
    pvals = np.empty(len(test_idx), dtype=np.float64)
    for k, i in enumerate(test_idx):
        s_i = scores[i]
        count = np.sum(cal_scores >= s_i)
        pvals[k] = (1.0 + count) / (1.0 + n_cal)
    return pvals


def decisions(pvals: np.ndarray, alpha: float) -> np.ndarray:
    return (pvals <= alpha).astype(np.int32)


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: List[float],
) -> Dict:
    pvals = conformal_pvalues(scores, cal_idx, test_idx)
    y_test = labels[test_idx]
    normals = y_test == 0
    anoms = y_test == 1

    raw_auc_all = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    raw_auc_test = roc_auc_score(y_test, scores[test_idx]) if len(np.unique(y_test)) > 1 else 0.5
    conf_scores = 1.0 - pvals
    conf_auc = roc_auc_score(y_test, conf_scores) if len(np.unique(y_test)) > 1 else 0.5

    n_cal = len(cal_idx)
    results = {
        "raw_auc_all": float(raw_auc_all),
        "raw_auc_test": float(raw_auc_test),
        "conformal_auc": float(conf_auc),
        "n_cal": n_cal,
        "n_test": len(test_idx),
        "pi_test": float(y_test.mean()),
        "by_alpha": {},
    }

    for alpha in alphas:
        flagged = pvals <= alpha
        flagged_rate = float(flagged.mean())
        fpr = float(flagged[normals].mean()) if normals.any() else float("nan")
        tpr = float(flagged[anoms].mean()) if anoms.any() else float("nan")
        theory_max_flagged = math.floor(alpha * (n_cal + 1)) / (n_cal + 1)
        # Identity: (1-pi)*FPR + pi*TPR ≈ flagged_rate (Reviewer #6)
        identity_lhs = (1.0 - results["pi_test"]) * fpr + results["pi_test"] * tpr
        results["by_alpha"][alpha] = {
            "fpr": fpr,
            "tpr": tpr,
            "flagged_rate": flagged_rate,
            "theory_max_flagged": theory_max_flagged,
            "identity_lhs": identity_lhs,
            "identity_gap": abs(identity_lhs - flagged_rate),
        }

    return results


def inject_calibration_contamination(
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    target_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Move anomalies from T into C until C has target_frac anomalies.
    Does NOT re-score; measures calibration-set composition effect only.
    """
    cal = list(cal_idx)
    test = list(test_idx)
    cal_anom = sum(labels[i] for i in cal)
    target_anom = int(len(cal) * target_frac)
    test_anoms = [i for i in test if labels[i] == 1]
    rng.shuffle(test_anoms)
    while cal_anom < target_anom and test_anoms:
        node = test_anoms.pop()
        test.remove(node)
        cal.append(node)
        cal_anom += 1
    return np.array(cal, dtype=np.int64), np.array(test, dtype=np.int64)
