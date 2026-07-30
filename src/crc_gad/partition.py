"""Train/val/eval and calibration/test partitions — NO label leakage."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PartitionInfo:
    train_idx: np.ndarray
    val_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray
    labels: np.ndarray  # for evaluation metrics only

    @property
    def n_cal(self) -> int:
        return len(self.cal_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def audit_log(self) -> dict:
        """Log |C|, |T|, anomaly counts — required by Reviewer #6."""
        y = self.labels
        return {
            "|C|": self.n_cal,
            "|T|": self.n_test,
            "anomalies_in_C": int(y[self.cal_idx].sum()),
            "anomalies_in_T": int(y[self.test_idx].sum()),
            "normals_in_C": int((1 - y[self.cal_idx]).sum()),
            "normals_in_T": int((1 - y[self.test_idx]).sum()),
            "pi_C": float(y[self.cal_idx].mean()) if self.n_cal else 0.0,
            "pi_T": float(y[self.test_idx].mean()) if self.n_test else 0.0,
            "label_clean_C": False,
        }


def make_partition(
    n_nodes: int,
    labels: np.ndarray,
    train_frac: float,
    val_frac_of_remain: float,
    rho: float,
    rng: np.random.Generator,
    use_label_clean_calibration: bool = False,
) -> PartitionInfo:
    """
    Protocol (Section 5.5):
      1. Shuffle all node indices (no scores/labels used for split order)
      2. train_frac -> train, val_frac_of_remain of rest -> val
      3. Remaining -> eval; split eval into C (rho) and T (1-rho)

    IMPORTANT: use_label_clean_calibration=False by default (label-free).
    If True, only normal nodes enter C (Bates clean-inlier — must be stated in paper).
    """
    idx = np.arange(n_nodes)
    rng.shuffle(idx)

    n_train = int(n_nodes * train_frac)
    train_idx = idx[:n_train]
    remain = idx[n_train:]
    n_val = max(1, int(len(remain) * val_frac_of_remain))
    val_idx = remain[:n_val]
    eval_idx = remain[n_val:]

    if use_label_clean_calibration:
        normal_eval = eval_idx[labels[eval_idx] == 0]
        anom_eval = eval_idx[labels[eval_idx] == 1]
        rng.shuffle(normal_eval)
        n_cal = max(1, int(len(normal_eval) * rho))
        cal_idx = normal_eval[:n_cal]
        test_idx = np.concatenate([normal_eval[n_cal:], anom_eval])
        rng.shuffle(test_idx)
    else:
        rng.shuffle(eval_idx)
        n_cal = max(1, int(len(eval_idx) * rho))
        cal_idx = eval_idx[:n_cal]
        test_idx = eval_idx[n_cal:]

    return PartitionInfo(
        train_idx=train_idx,
        val_idx=val_idx,
        cal_idx=cal_idx,
        test_idx=test_idx,
        labels=labels,
    )


def random_cal_test_split(
    eval_idx: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Single random C/T split of eval nodes (for dependence study)."""
    idx = eval_idx.copy()
    rng.shuffle(idx)
    n_cal = max(1, int(len(idx) * rho))
    return idx[:n_cal], idx[n_cal:]
