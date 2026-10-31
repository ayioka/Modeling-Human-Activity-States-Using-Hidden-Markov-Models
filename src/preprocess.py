"""Preprocess Sensor Logger recordings into fixed-rate, trimmed, chunked CSVs.

Pipeline per recording (one zip = one recording):
  1. extract the zip into data/extracted/<activity>/<zip_stem>/ (originals untouched)
  2. measure the sampling rate of every sensor from its timestamps (median 1/dt,
     jitter = std of dt, largest gap) and read the requested rate from Metadata.csv
  3. resample Accelerometer, Gyroscope, Gravity and TotalAcceleration onto one
     common TARGET_HZ grid with linear interpolation
  4. trim TRIM_S seconds from both ends (screen-tap noise), or the per-recording
     values in data/trim_overrides.csv (columns recording, trim_start_s,
     trim_end_s; recording = zip stem; blank = default)
  5. split into CHUNK_S-second chunks; a remainder >= MIN_CHUNK_S becomes its
     own chunk, otherwise it is dropped and logged
  6. write data/processed/<activity>/<activity>_<stamp>_partNN.csv plus
     data/processed/manifest.csv and data/processed/recordings.csv

The script is idempotent: data/processed/ is rebuilt from scratch on every run,
so new zips dropped into data/<activity>/ or data/test/ are picked up
automatically. Run from the project root:  python src/preprocess.py
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXTRACTED = DATA / "extracted"
PROCESSED = DATA / "processed"

ACTIVITIES = ["jumping", "standing", "still", "walking"]
TEST_DIR = "test"  # continuous multi-activity recordings, label "mixed"

# Target rate. Every sensor on this phone (Samsung SM-S916U) delivers 53.1 Hz
# (dt = 18.83 ms), although Metadata.csv says 100 Hz (sampleRateMs=10) was
# requested. Resampling must stay at or below the lowest real rate so that we
# only interpolate, never invent detail; 50 Hz is the nearest round number
# below 53.1 Hz. Its Nyquist frequency (25 Hz) is far above human-motion
# content (walking/jumping cadence 1.5-3 Hz, harmonics < 10 Hz).
TARGET_HZ = 50.0
TRIM_S = 1.0
CHUNK_S = 8.0
MIN_CHUNK_S = 5.0

# sensor file -> output column prefix
SENSORS = {
    "Accelerometer": "acc",        # linear acceleration, gravity already removed
    "Gyroscope": "gyro",           # rad/s
    "Gravity": "grav",             # gravity vector
    "TotalAcceleration": "tacc",   # acceleration including gravity
}

# Recordings kept on disk but excluded from training/evaluation.
EXCLUDE = {
    ("standing", "2026-10-26_18-52-06"): (
        "phone lay flat during recording (mean gravity ~ (-2.6, 0.1, 9.4) vs "
        "~(-6.5, -3.5, 6.5) for every other standing file); likely mislabeled"
    ),
    ("walking", "2026-10-26_19-01-46"): (
        "phone held in the opposite orientation (mean grav_x ~ +6.2 vs ~ -8 for the "
        "five other walking recordings)"
    ),
}

# Per-recording trim overrides (zip stem -> seconds). Blank cells keep TRIM_S.
TRIM_OVERRIDES_FILE = DATA / "trim_overrides.csv"
# Segment labels for continuous test recordings (recording, start_s, end_s, label).
TEST_LABELS_FILE = DATA / TEST_DIR / "labels.csv"


def load_trim_overrides(path: Path = TRIM_OVERRIDES_FILE) -> dict[str, tuple[float, float]]:
    """Read {zip stem: (trim_start_s, trim_end_s)} from the overrides CSV, if present."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"recording": str})
    out = {}
    for _, r in df.iterrows():
        start = TRIM_S if pd.isna(r.get("trim_start_s")) else float(r["trim_start_s"])
        end = TRIM_S if pd.isna(r.get("trim_end_s")) else float(r["trim_end_s"])
        out[r["recording"].strip()] = (start, end)
    return out


def safe_stamp(stem: str) -> str:
    """Turn a zip stem like '2026-10-26_18-23-01 (1)' into '2026-10-26_18-23-01_1'."""
    s = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return re.sub(r"_+", "_", s)


def extract(zip_path: Path, group: str) -> Path:
    """Extract a zip into data/extracted/<group>/<stem>/ unless already up to date."""
    out = EXTRACTED / group / zip_path.stem
    marker = out / ".extracted"
    if not marker.exists() or marker.stat().st_mtime < zip_path.stat().st_mtime:
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
        marker.touch()
    return out


def read_metadata_rates(folder: Path) -> dict[str, float]:
    """Return {sensor: requested Hz} from Metadata.csv (empty dict if missing)."""
    path = folder / "Metadata.csv"
    if not path.exists():
        return {}
    meta = pd.read_csv(path).iloc[0]
    sensors = str(meta.get("sensors", "")).split("|")
    rates = str(meta.get("sampleRateMs", "")).split("|")
    out = {}
    for s, r in zip(sensors, rates):
        try:
            out[s] = 1000.0 / float(r)
        except ValueError:
            pass
    return out


def read_sensor(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one sensor CSV and return (t seconds, xyz array) in x,y,z order.

    Sensor Logger writes axes as z,y,x; we reorder by name. Rows with duplicate
    timestamps are dropped, and leading all-zero rows (the Accelerometer's
    first sample is always 0,0,0) are removed.
    """
    df = pd.read_csv(path).sort_values("time").drop_duplicates("time")
    xyz = df[["x", "y", "z"]].to_numpy(float)
    t = (df["time"].to_numpy(np.int64) - df["time"].iloc[0]) / 1e9 + df["seconds_elapsed"].iloc[0]
    nz = np.flatnonzero(np.any(xyz != 0, axis=1))
    start = nz[0] if len(nz) else 0
    return t[start:], xyz[start:]


def measure_rate(t: np.ndarray) -> dict[str, float]:
    """Sampling-rate statistics from timestamps: median Hz, jitter and max gap (ms)."""
    dt = np.diff(t)
    dt = dt[dt > 0]
    return {
        "hz": float(np.median(1.0 / dt)),
        "jitter_ms": float(np.std(dt) * 1e3),
        "max_gap_ms": float(np.max(dt) * 1e3),
    }


def split_points(n: int, fs: float) -> tuple[list[tuple[int, int]], float]:
    """Chunk boundaries (sample indices) for a trimmed signal of n samples.

    k = floor(D / CHUNK_S) full chunks; the remainder becomes its own chunk if
    it is >= MIN_CHUNK_S. A file shorter than CHUNK_S therefore stays whole when
    it is >= MIN_CHUNK_S. Returns (chunks, dropped_seconds).
    """
    size = int(round(CHUNK_S * fs))
    k = n // size
    chunks = [(i * size, (i + 1) * size) for i in range(k)]
    rem = n - k * size
    if rem >= int(round(MIN_CHUNK_S * fs)):
        chunks.append((k * size, n))
        return chunks, 0.0
    return chunks, rem / fs


def load_test_labels(path: Path = TEST_LABELS_FILE) -> dict[str, list[tuple[float, float, str]]]:
    """Read {zip stem: [(start_s, end_s, label), ...]} from data/test/labels.csv, if present.

    Times are seconds from the recording start (Sensor Logger seconds_elapsed).
    The boundaries come from raw signals only (see src/segment.py).
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"recording": str})
    return {rec: [(float(r.start_s), float(r.end_s), str(r.label)) for r in g.itertuples()]
            for rec, g in df.groupby("recording")}


def label_samples(t: np.ndarray, segments: list[tuple[float, float, str]]) -> np.ndarray:
    """Per-sample label from [start, end) segments; samples outside every segment -> transition."""
    out = np.full(len(t), "transition", dtype=object)
    for start, end, name in segments:
        out[(t >= start) & (t < end)] = name
    return out


def process_recording(zip_path: Path, group: str, label: str, chunk: bool = True,
                      out_root: Path = PROCESSED, trims: tuple[float, float] = (TRIM_S, TRIM_S),
                      segments: list[tuple[float, float, str]] | None = None):
    """Process one zip into out_root/<group>/. Returns (manifest rows, recording-level log row).

    `trims` is (seconds cut from the start, seconds cut from the end). If
    `segments` is given (continuous test recordings), every sample gets the
    label of the segment containing it instead of `label`.
    """
    folder = extract(zip_path, group)
    meta = read_metadata_rates(folder)
    stamp = safe_stamp(zip_path.stem)
    excluded_reason = EXCLUDE.get((group, zip_path.stem), "")
    trim_start, trim_end = trims

    signals, rec = {}, {
        "group": group, "source_zip": str(zip_path.relative_to(ROOT)), "stamp": stamp,
        "excluded": bool(excluded_reason), "exclude_reason": excluded_reason,
        "trim_start_s": trim_start, "trim_end_s": trim_end,
    }
    for sensor, prefix in SENSORS.items():
        path = folder / f"{sensor}.csv"
        if not path.exists():
            if sensor in ("Accelerometer", "Gyroscope"):
                raise FileNotFoundError(path)
            continue  # gravity / total acceleration are optional
        t, xyz = read_sensor(path)
        signals[prefix] = (t, xyz)
        stats = measure_rate(t)
        rec[f"{prefix}_hz"] = round(stats["hz"], 2)
        rec[f"{prefix}_jitter_ms"] = round(stats["jitter_ms"], 3)
        rec[f"{prefix}_max_gap_ms"] = round(stats["max_gap_ms"], 1)
        rec[f"{prefix}_meta_hz"] = meta.get(sensor, np.nan)

    # Common grid over the time span where every sensor has data, then trim.
    t0 = max(t[0] for t, _ in signals.values())
    t1 = min(t[-1] for t, _ in signals.values())
    rec["raw_duration_s"] = round(t1 - t0, 3)
    a, b = t0 + trim_start, t1 - trim_end
    n = int(np.floor((b - a) * TARGET_HZ + 1e-9)) + 1 if b > a else 0
    rec["trimmed_duration_s"] = round(n / TARGET_HZ, 3)
    grid = a + np.arange(n) / TARGET_HZ

    cols = {}
    for prefix, (t, xyz) in signals.items():
        for j, ax in enumerate("xyz"):
            cols[f"{prefix}_{ax}"] = np.interp(grid, t, xyz[:, j])
    full = pd.DataFrame(cols)

    if chunk:
        chunks, dropped = split_points(n, TARGET_HZ)
    else:  # continuous test recordings are kept whole for HMM decoding
        chunks, dropped = ([(0, n)] if n else []), 0.0
    rec["n_chunks"] = len(chunks)
    rec["dropped_s"] = round(dropped if chunks else n / TARGET_HZ, 3)
    if dropped and chunks:
        print(f"  [drop] {group}/{zip_path.name}: remainder {dropped:.2f} s < {MIN_CHUNK_S} s dropped")
    if not chunks:
        print(f"  [drop] {group}/{zip_path.name}: only {n / TARGET_HZ:.2f} s after trim, file dropped")

    out_dir = out_root / group
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (s, e) in enumerate(chunks, start=1):
        part = full.iloc[s:e].reset_index(drop=True)
        part.insert(0, "timestamp", np.arange(e - s) / TARGET_HZ)
        part["label"] = label
        if segments is not None:
            part["label"] = label_samples(grid[s:e], segments)
        name = f"{group}_{stamp}_part{i:02d}.csv"
        part.to_csv(out_dir / name, index=False, float_format="%.6f")
        rows.append({
            "file": f"{group}/{name}", "source_zip": rec["source_zip"], "activity": label,
            "recording": stamp, "part": i,
            "start_s": round(trim_start + s / TARGET_HZ, 3),
            "duration_s": round((e - s) / TARGET_HZ, 3), "n_samples": e - s,
            **{f"{p}_orig_hz": rec.get(f"{p}_hz", np.nan) for p in SENSORS.values()},
            "resampled_hz": TARGET_HZ,
            "trim_start_s": trim_start, "trim_end_s": trim_end,
            "excluded": rec["excluded"], "exclude_reason": excluded_reason,
        })
    return rows, rec


def main(out_root: Path = PROCESSED, use_trim_overrides: bool = True) -> None:
    """Rebuild out_root (default data/processed/) from every zip under data/<activity>/ and data/test/.

    With use_trim_overrides=False every recording gets the default TRIM_S on
    both ends (used to reproduce the pre-override cross-session experiment).
    """
    out_root = Path(out_root)
    if out_root.exists():
        if any(out_root.iterdir()) and not (out_root / "manifest.csv").exists():
            raise SystemExit(f"Refusing to clear {out_root}: it is not a previous preprocess output.")
        shutil.rmtree(out_root)  # generated output only; originals are never touched
    out_root.mkdir(parents=True)

    overrides = load_trim_overrides() if use_trim_overrides else {}
    test_labels = load_test_labels()
    manifest, recordings = [], []
    jobs = [(z, act, act, True) for act in ACTIVITIES for z in sorted((DATA / act).glob("*.zip"))]
    jobs += [(z, TEST_DIR, "mixed", False) for z in sorted((DATA / TEST_DIR).glob("*.zip"))]
    unused = set(overrides) - {z.stem for z, *_ in jobs}
    if unused:
        print(f"  [warn] trim overrides for unknown recordings: {sorted(unused)}")
    for zip_path, group, label, chunk in jobs:
        trims = overrides.get(zip_path.stem, (TRIM_S, TRIM_S))
        segments = test_labels.get(zip_path.stem) if group == TEST_DIR else None
        if group == TEST_DIR and segments is None:
            print(f"  [warn] no segments for {zip_path.name} in {TEST_LABELS_FILE.name}: labelled 'mixed'")
        rows, rec = process_recording(zip_path, group, label, chunk, out_root, trims, segments)
        manifest += rows
        recordings.append(rec)
        flag = "  EXCLUDED" if rec["excluded"] else ""
        if trims != (TRIM_S, TRIM_S):
            flag += f"  trim {trims[0]:g}/{trims[1]:g} s"
        print(f"{group:9s} {zip_path.name:34s} D={rec['trimmed_duration_s']:6.2f}s "
              f"-> {rec['n_chunks']} file(s){flag}")

    pd.DataFrame(manifest).to_csv(out_root / "manifest.csv", index=False)
    pd.DataFrame(recordings).to_csv(out_root / "recordings.csv", index=False)
    m = pd.DataFrame(manifest)
    used = m[~m.excluded]
    print(f"\nResampled to {TARGET_HZ:g} Hz. Output files per label (excluded files not counted):")
    print(used.groupby("activity").agg(files=("file", "count"), seconds=("duration_s", "sum")).to_string())
    if not (DATA / TEST_DIR).exists() or not any((DATA / TEST_DIR).glob("*.zip")):
        print("\nNo zips in data/test/ yet; the notebook will use a temporary hold-out.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=PROCESSED, help="output directory (default data/processed)")
    p.add_argument("--no-trim-overrides", action="store_true",
                   help=f"ignore {TRIM_OVERRIDES_FILE.name} and trim {TRIM_S:g} s from both ends everywhere")
    args = p.parse_args()
    main(args.out, not args.no_trim_overrides)
