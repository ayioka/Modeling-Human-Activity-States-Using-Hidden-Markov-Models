"""Windowing and feature extraction for the HMM.

Windows: WINDOW_S = 2 s with 50 % overlap (1 s hop) at 50 Hz = 100 samples.
Walking cadence is ~1.5-2 Hz, so a 2 s window covers 3-4 step cycles, and the
FFT bin spacing is 1 / 2 s = 0.5 Hz, which is fine enough to place the cadence
peak. A 1 s hop gives one observation per second for the HMM: dense enough to
model transitions, while neighbouring windows still share half their samples.

Feature justifications (one line each):
  mean_<sensor>_<axis>  average level per axis; for grav_* it encodes phone
                        orientation (flat on table vs upright at waist)
  std_<sensor>_<axis>   how much each axis moves; ~0 for still, large for jumping
  rms_acc / rms_gyro    overall motion intensity, orientation-independent
  sma_acc / sma_gyro    Signal Magnitude Area, the classic activity-level measure
  corr_acc_<ab>         coupling between axes; periodic gait gives structured
                        correlations, sensor noise gives ~0
  domfreq_<m>           cadence (steps / jumps per second) from the FFT peak
  logenergy_<m>         total spectral energy (log, so the Gaussian fits better)
  band_<m>_<lo>_<hi>    log energy per band: separates slow sway, gait cadence
                        (1.5-3 Hz), jumping impacts and high-frequency harmonics
where <m> is the acceleration magnitude (accmag) or the gyroscope magnitude (gyromag).

Energies are log-transformed because they span orders of magnitude between
still and jumping, and the HMM's Gaussian emissions fit log-energies much better.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FS = 50.0
WINDOW_S = 2.0
HOP_S = 1.0
WIN = int(WINDOW_S * FS)
HOP = int(HOP_S * FS)
BANDS = [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0)]
ACTIVITIES = ["jumping", "standing", "still", "walking"]
MIN_LABEL_SHARE = 0.75  # a window takes a label only if >= 75 % of it lies in one segment

# Features that encode phone orientation rather than motion (used in the ablation).
GRAVITY_MEAN_FEATURES = ["mean_grav_x", "mean_grav_y", "mean_grav_z"]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, defined as 0 when either signal is (numerically) constant."""
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _spectral(sig: np.ndarray, name: str) -> dict[str, float]:
    """Dominant frequency, log total energy and log band energies of one window."""
    x = (sig - sig.mean()) * np.hanning(len(sig))
    power = np.abs(np.fft.rfft(x)) ** 2 / len(sig)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / FS)
    valid = freqs > 0
    out = {
        f"domfreq_{name}": float(freqs[valid][np.argmax(power[valid])]),
        f"logenergy_{name}": float(np.log(power[valid].sum() + 1e-8)),
    }
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out[f"band_{name}_{lo:g}_{hi:g}"] = float(np.log(power[m].sum() + 1e-8))
    return out


def window_features(w: pd.DataFrame) -> dict[str, float]:
    """Compute the feature vector of one window (a slice of a processed CSV)."""
    f: dict[str, float] = {}
    for sensor in ("acc", "gyro", "grav"):
        if f"{sensor}_x" not in w:
            continue
        for ax in "xyz":
            f[f"mean_{sensor}_{ax}"] = float(w[f"{sensor}_{ax}"].mean())
    for sensor in ("acc", "gyro"):
        for ax in "xyz":
            f[f"std_{sensor}_{ax}"] = float(w[f"{sensor}_{ax}"].std())
    acc = w[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = w[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    accmag, gyromag = np.linalg.norm(acc, axis=1), np.linalg.norm(gyro, axis=1)
    f["rms_acc"] = float(np.sqrt(np.mean(accmag ** 2)))
    f["rms_gyro"] = float(np.sqrt(np.mean(gyromag ** 2)))
    f["sma_acc"] = float(np.abs(acc).sum(axis=1).mean())
    f["sma_gyro"] = float(np.abs(gyro).sum(axis=1).mean())
    for a, b in ("xy", "xz", "yz"):
        f[f"corr_acc_{a}{b}"] = _corr(acc[:, "xyz".index(a)], acc[:, "xyz".index(b)])
    f.update(_spectral(accmag, "accmag"))
    f.update(_spectral(gyromag, "gyromag"))
    return f


def file_features(df: pd.DataFrame) -> pd.DataFrame:
    """Slide WIN-sample windows with HOP-sample hop over one processed file.

    Each row gets the window features plus `t_start` (s) and `label`: the
    per-sample label that covers at least MIN_LABEL_SHARE of the window, or
    "transition" when no label does (a window straddling a boundary).
    Single-activity files are uniform, so this only matters for data/test/.
    """
    rows = []
    for s in range(0, len(df) - WIN + 1, HOP):
        w = df.iloc[s:s + WIN]
        r = window_features(w)
        r["t_start"] = s / FS
        share = w["label"].value_counts(normalize=True)
        r["label"] = share.index[0] if share.iloc[0] >= MIN_LABEL_SHARE else "transition"
        rows.append(r)
    return pd.DataFrame(rows)


def load_dataset(processed_dir: Path, include_excluded: bool = False) -> pd.DataFrame:
    """Window features for every processed file listed in manifest.csv.

    Adds `file`, `recording`, `activity` and `start_s` (chunk start in seconds
    from the recording start) so that splits can be made per source recording.
    Files flagged `excluded` are skipped by default.
    """
    manifest = pd.read_csv(processed_dir / "manifest.csv")
    if not include_excluded:
        manifest = manifest[~manifest["excluded"]]
    parts = []
    for _, m in manifest.iterrows():
        feats = file_features(pd.read_csv(processed_dir / m["file"]))
        if feats.empty:
            continue
        feats.insert(0, "file", m["file"])
        feats.insert(1, "recording", m["recording"])
        feats.insert(2, "activity", m["activity"])
        feats.insert(3, "start_s", m["start_s"])
        parts.append(feats)
    return pd.concat(parts, ignore_index=True)


def feature_columns(df: pd.DataFrame, drop_gravity_mean: bool = False) -> list[str]:
    """Names of the numeric feature columns, optionally without the gravity means."""
    meta = {"file", "recording", "activity", "start_s", "t_start", "label"}
    cols = [c for c in df.columns if c not in meta]
    if drop_gravity_mean:
        cols = [c for c in cols if c not in GRAVITY_MEAN_FEATURES]
    return cols


class ZScore:
    """Z-score normaliser fitted on the training set only.

    Using training statistics only keeps the test set truly unseen: computing
    the mean/std on all data would leak test-set information into training.
    Normalising also puts features measured in m/s^2, rad/s, Hz and log-energy
    on a common scale, so the variance floor in the HMM means the same for each.
    """

    def fit(self, X: np.ndarray) -> "ZScore":
        """Store per-column mean and std (std floored to avoid division by 0)."""
        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the stored normalisation."""
        return (X - self.mean_) / self.std_
