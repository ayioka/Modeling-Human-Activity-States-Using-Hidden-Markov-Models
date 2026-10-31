"""Train/test split, sequence construction and the full train-evaluate loop.

Training sequences
------------------
Every recording contains a single activity, so sequences built from one file
only ever show self-transitions and the learned A would be purely diagonal.
We therefore also build synthetic sequences: the training files are shuffled
and their window sequences concatenated, so the HMM sees activity changes.
Caveat (honest): the off-diagonal entries of A then reflect how we
concatenated files (one switch per ~8 s file, uniformly random next activity),
not how often a real person switches activity. Windows never straddle two
files; only the sequence does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from features import ACTIVITIES, ZScore, feature_columns
from hmm import GaussianHMM
from metrics import map_states, per_class_table

HOLDOUT_FRACTION = 0.3
SEED = 42


def temporary_holdout(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """TEMPORARY split until data/test/ is labelled: hold out whole recordings.

    Per activity, max(1, round(HOLDOUT_FRACTION * n_recordings)) source
    recordings go to the test set, so chunks of one recording never end up on
    both sides. Returns (train, test, {activity: [held-out recordings]}).
    """
    rng = np.random.default_rng(seed)
    held = {}
    for act in ACTIVITIES:
        recs = sorted(df.loc[df.activity == act, "recording"].unique())
        n = max(1, int(round(HOLDOUT_FRACTION * len(recs))))
        held[act] = sorted(rng.choice(recs, size=n, replace=False).tolist())
    test_recs = {r for rs in held.values() for r in rs}
    is_test = df.recording.isin(test_recs)
    return df[~is_test].copy(), df[is_test].copy(), held


def file_sequences(df: pd.DataFrame, cols: list[str]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """One (X, y, file) triple per processed file, windows in time order."""
    out = []
    for f, g in df.groupby("file", sort=True):
        g = g.sort_values("t_start")
        y = g["label"].map(ACTIVITIES.index).to_numpy()
        out.append((g[cols].to_numpy(float), y, f))
    return out


def concat_shuffled(files: list, n_sequences: int, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    """Concatenate all file sequences in `n_sequences` independent random orders."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sequences):
        order = rng.permutation(len(files))
        out.append((np.vstack([files[i][0] for i in order]),
                    np.concatenate([files[i][1] for i in order]),
                    [files[i][2] for i in order]))
    return out


@dataclass
class Result:
    """Everything the notebook needs to plot and report one experiment."""
    name: str
    cols: list[str]
    scaler: ZScore
    model_init: GaussianHMM
    model_bw: GaussianHMM
    mapping: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    test_files: list[str]
    pred_init: np.ndarray
    pred_bw: np.ndarray
    tables: dict = field(default_factory=dict)


def fit(train: pd.DataFrame, drop_gravity_mean: bool = False, n_train_sequences: int = 10,
        var_floor: float = 1e-2, tol: float = 1e-4, max_iter: int = 500):
    """Fit the z-scorer, the supervised-init HMM and the Baum-Welch HMM on `train`.

    Returns (cols, scaler, init_model, bw_model, mapping) where mapping[state] = label
    index for the Baum-Welch model (Hungarian match on the training sequences).
    """
    cols = feature_columns(train, drop_gravity_mean)
    scaler = ZScore().fit(train[cols].to_numpy(float))
    tr = train.copy()
    tr[cols] = scaler.transform(tr[cols].to_numpy(float))

    train_seqs = concat_shuffled(file_sequences(tr, cols), n_train_sequences, SEED)
    Xs, ys = [s[0] for s in train_seqs], [s[1] for s in train_seqs]
    K = len(ACTIVITIES)

    init = GaussianHMM(K, var_floor=var_floor, tol=tol, max_iter=max_iter).init_supervised(Xs, ys)
    bw = GaussianHMM(K, var_floor=var_floor, tol=tol, max_iter=max_iter).init_supervised(Xs, ys)
    bw.fit_baum_welch(Xs)
    # Map learned states back to labels using the training sequences.
    train_states = np.concatenate([bw.viterbi(X)[0] for X in Xs])
    mapping = map_states(train_states, np.concatenate(ys), K)
    return cols, scaler, init, bw, mapping


def run(train: pd.DataFrame, test: pd.DataFrame, drop_gravity_mean: bool = False,
        n_train_sequences: int = 10, var_floor: float = 1e-2, tol: float = 1e-4,
        max_iter: int = 500, name: str = "") -> Result:
    """Fit supervised-init HMM, refine with Baum-Welch, Viterbi-decode the test set.

    The test set is decoded as one sequence: the held-out files concatenated
    in a fixed random order (so every test window is scored exactly once).
    """
    cols, scaler, init, bw, mapping = fit(train, drop_gravity_mean, n_train_sequences,
                                          var_floor, tol, max_iter)
    te = test.copy()
    te[cols] = scaler.transform(te[cols].to_numpy(float))
    test_X, test_y, test_files = concat_shuffled(file_sequences(te, cols), 1, SEED + 1)[0]
    pred_init = init.viterbi(test_X)[0]
    pred_bw = mapping[bw.viterbi(test_X)[0]]

    res = Result(name, cols, scaler, init, bw, mapping, test_X, test_y, test_files, pred_init, pred_bw)
    res.tables["before Baum-Welch"] = per_class_table(test_y, pred_init, ACTIVITIES)
    res.tables["after Baum-Welch"] = per_class_table(test_y, pred_bw, ACTIVITIES)
    return res


def evaluate_recordings(df: pd.DataFrame, train_recordings: list[str], test_recordings: list[str],
                        drop_gravity_mean: bool = False, **fit_kwargs) -> tuple[pd.DataFrame, dict, dict]:
    """Train on exactly `train_recordings`, decode each of `test_recordings` separately.

    Recording names are the `recording` column of the window table (zip stem
    made filename-safe). Every name must be present and the two lists must not
    overlap. Each test recording is decoded as one sequence: its chunks in
    part order (windows never straddle chunks). Windows labelled "transition"
    (or anything that is not an activity) are decoded with the rest, so the
    sequence stays continuous, but get y = -1 and are left out of the metrics.

    Returns (windows, tables, fitted): one row per test window with
    `recording`, `activity`, `label`, `t_s` (window start, seconds from the
    recording start), `y`, `pred_init`, `pred_bw` and the raw `rms_acc` /
    `mean_grav_*` features; {"before Baum-Welch" | "after Baum-Welch":
    (per-class table, accuracy)} over the y >= 0 windows; and the fitted
    {cols, scaler, init, bw, mapping}.
    """
    train_recordings, test_recordings = list(train_recordings), list(test_recordings)
    missing = set(train_recordings + test_recordings) - set(df.recording)
    if missing:
        raise ValueError(f"recordings not in the dataset: {sorted(missing)}")
    if set(train_recordings) & set(test_recordings):
        raise ValueError("train and test recordings overlap")

    cols, scaler, init, bw, mapping = fit(df[df.recording.isin(train_recordings)],
                                          drop_gravity_mean, **fit_kwargs)
    parts = []
    for rec in test_recordings:
        g = df[df.recording == rec].copy()
        g["part"] = g.file.str.extract(r"_part(\d+)\.csv$", expand=False).astype(int)
        g = g.sort_values(["part", "t_start"])
        X = scaler.transform(g[cols].to_numpy(float))
        parts.append(pd.DataFrame({
            "recording": rec, "activity": g.activity.values, "file": g.file.values,
            "label": g.label.values, "t_s": g.start_s.values + g.t_start.values,
            "y": g.label.map({a: i for i, a in enumerate(ACTIVITIES)}).fillna(-1).astype(int).values,
            "pred_init": init.viterbi(X)[0], "pred_bw": mapping[bw.viterbi(X)[0]],
            "rms_acc": g.rms_acc.values,
            **{c: g[c].values for c in ("mean_grav_x", "mean_grav_y", "mean_grav_z") if c in g},
        }))
    windows = pd.concat(parts, ignore_index=True)
    scored = windows[windows.y >= 0]
    tables = {"before Baum-Welch": per_class_table(scored.y.values, scored.pred_init.values, ACTIVITIES),
              "after Baum-Welch": per_class_table(scored.y.values, scored.pred_bw.values, ACTIVITIES)}
    fitted = {"cols": cols, "scaler": scaler, "init": init, "bw": bw, "mapping": mapping}
    return windows, tables, fitted


def decode_unlabelled(res: Result, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Viterbi-decode each (unlabelled, e.g. "mixed") test file with the BW model."""
    out = {}
    for f, g in df.groupby("file"):
        X = res.scaler.transform(g.sort_values("t_start")[res.cols].to_numpy(float))
        out[f] = res.mapping[res.model_bw.viterbi(X)[0]]
    return out
