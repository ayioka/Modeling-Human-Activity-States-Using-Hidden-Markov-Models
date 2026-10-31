"""Evaluation helpers: confusion matrix, sensitivity/specificity, state mapping."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    """Counts C[i, j] = windows with true class i predicted as class j."""
    C = np.zeros((n, n), dtype=int)
    np.add.at(C, (y_true, y_pred), 1)
    return C


def map_states(states: np.ndarray, y_true: np.ndarray, n: int) -> np.ndarray:
    """Hungarian mapping from learned HMM states to activity labels.

    After Baum-Welch a state is not guaranteed to still mean the activity it
    was initialised with. We choose the one-to-one assignment state -> label
    that maximises agreement with the training labels. Returns `mapping`
    such that mapping[state] = label.
    """
    C = confusion_matrix(states, y_true, n)  # rows: states, cols: labels
    rows, cols = linear_sum_assignment(-C)
    mapping = np.arange(n)
    mapping[rows] = cols
    return mapping


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, float]:
    """Per-activity windows, sensitivity (recall) and specificity, plus overall accuracy.

    sensitivity = TP / (TP + FN): share of the activity's windows recognised
    specificity = TN / (TN + FP): share of other windows not mislabelled as it
    accuracy    = (TP + TN) / all windows, one-vs-rest for that activity
    """
    C = confusion_matrix(y_true, y_pred, len(names))
    total = C.sum()
    rows = []
    for k, name in enumerate(names):
        tp = C[k, k]
        fn = C[k].sum() - tp
        fp = C[:, k].sum() - tp
        tn = total - tp - fn - fp
        rows.append({
            "activity": name,
            "windows": int(C[k].sum()),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "accuracy": (tp + tn) / total if total else np.nan,
        })
    return pd.DataFrame(rows), float(np.trace(C) / total)


def to_markdown(df: pd.DataFrame, accuracy: float, title: str) -> str:
    """Render a per-class table plus overall accuracy as a Markdown section."""
    lines = [f"### {title}", "", "| Activity | Windows | Sensitivity | Specificity |",
             "|---|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.activity} | {r.windows} | {r.sensitivity:.3f} | {r.specificity:.3f} |")
    lines += ["", f"**Overall accuracy: {accuracy:.3f}** ({int(df.windows.sum())} windows)", ""]
    return "\n".join(lines)
