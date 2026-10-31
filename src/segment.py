"""Raw-signal segmentation of the continuous test recordings (no model involved).

The test labels must not come from the HMM (that would be circular), so the
boundaries in data/test/labels.csv are set from three raw signals only:

  motion   std of |total acceleration| over 1 s       -> motion intensity
  flat     angle between gravity and the phone z-axis -> gravity tilt; < FLAT_DEG
           means the phone lies screen-up (on a table, or held flat in the hand)
  domfreq  FFT peak of |total acceleration| over 2 s, 0.5-8 Hz -> cadence

Thresholds are NOT tuned on the test files: each one is the geometric midpoint
between the medians of two neighbouring activities in the single-activity
training recordings (data/processed, excluded files left out).

Rule, evaluated every STEP_S seconds:
  1. flat and motion <  T_still  -> still
  2. flat and motion >= T_still  -> transition (phone handled flat)
  3. motion < T_move             -> standing
  4. motion >= T_jump            -> jumping
  5. otherwise                   -> walking
Then any run shorter than MIN_RUN_S becomes transition, adjacent transition
runs merge, and each boundary sits halfway between two steps.

Why the dominant frequency is plotted but not used in the rule: in training,
walking peaks at ~1.5 Hz and jumping at ~2-2.5 Hz (reference line F_jump), but
in the "unsure" test recording walking peaks at 2-2.5 Hz too, so cadence does
not separate them there. Intensity does, by a factor of ~5 (walking 1-3,
jumping 8-13 m/s², T_jump ~ 5.3), consistently in training and test.

Run from the project root:  python src/segment.py [--write]
It plots every data/test/*.zip to reports/figures/segment_<stem>.png and
prints the segments; with --write it also writes data/test/labels.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from preprocess import DATA, PROCESSED, ROOT, TARGET_HZ, TEST_DIR, extract, read_sensor  # noqa: E402

FS = TARGET_HZ
STEP_S = 0.5
MOTION_WIN_S = 1.0
FREQ_WIN_S = 2.0
FLAT_DEG = 30.0
MIN_RUN_S = 2.0
LABELS_FILE = DATA / TEST_DIR / "labels.csv"
FIGS = ROOT / "reports" / "figures"
COLORS = {"still": "#4c72b0", "standing": "#dd8452", "walking": "#55a868",
          "jumping": "#c44e52", "transition": "#bbbbbb"}


def load_raw(folder: Path) -> pd.DataFrame:
    """Total acceleration and gravity resampled to FS Hz over the whole recording (no trim).

    Time `t` is seconds from the recording start (Sensor Logger seconds_elapsed).
    """
    sig = {p: read_sensor(folder / f"{s}.csv") for p, s in
           (("tacc", "TotalAcceleration"), ("grav", "Gravity"))}
    t0 = max(t[0] for t, _ in sig.values())
    t1 = min(t[-1] for t, _ in sig.values())
    grid = t0 + np.arange(int(np.floor((t1 - t0) * FS)) + 1) / FS
    cols = {"t": grid}
    for p, (t, xyz) in sig.items():
        for j, ax in enumerate("xyz"):
            cols[f"{p}_{ax}"] = np.interp(grid, t, xyz[:, j])
    return pd.DataFrame(cols)


def signal_steps(df: pd.DataFrame, t: np.ndarray | None = None) -> pd.DataFrame:
    """Motion, flatness and dominant frequency every STEP_S s (window centres).

    `df` needs tacc_* and grav_* at FS Hz; `t` (seconds) defaults to sample index / FS.
    """
    t = np.arange(len(df)) / FS if t is None else np.asarray(t)
    mag = np.linalg.norm(df[["tacc_x", "tacc_y", "tacc_z"]].to_numpy(), axis=1)
    g = df[["grav_x", "grav_y", "grav_z"]].to_numpy()
    hm, hf = int(MOTION_WIN_S * FS / 2), int(FREQ_WIN_S * FS / 2)
    hann = np.hanning(2 * hf)
    freqs = np.fft.rfftfreq(2 * hf, 1 / FS)
    band = (freqs >= 0.5) & (freqs <= 8.0)
    rows = []
    for i in range(hf, len(mag) - hf + 1, int(STEP_S * FS)):
        gm = g[i - hm:i + hm].mean(axis=0)
        seg = mag[i - hf:i + hf]
        spec = np.abs(np.fft.rfft((seg - seg.mean()) * hann))
        rows.append({
            "t": t[i] if i < len(t) else t[-1] + (i - len(t) + 1) / FS,
            "motion": mag[i - hm:i + hm].std(),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(gm[2] / np.linalg.norm(gm), -1, 1)))),
            "domfreq": float(freqs[band][np.argmax(spec[band])]),
        })
    return pd.DataFrame(rows)


def training_thresholds(processed: Path = PROCESSED) -> dict[str, float]:
    """Geometric midpoints between neighbouring activity medians in the training recordings."""
    m = pd.read_csv(processed / "manifest.csv")
    m = m[~m.excluded & m.activity.isin(["still", "standing", "walking", "jumping"])]
    parts = []
    for r in m.itertuples():
        s = signal_steps(pd.read_csv(processed / r.file))
        s["activity"] = r.activity
        parts.append(s)
    med = pd.concat(parts).groupby("activity")[["motion", "domfreq"]].median()
    gm = lambda a, b: float(np.sqrt(a * b))  # noqa: E731
    return {
        "T_still": gm(med.motion["still"], med.motion["standing"]),
        "T_move": gm(med.motion["standing"], med.motion["walking"]),
        "T_jump": gm(med.motion["walking"], med.motion["jumping"]),
        "F_jump": gm(med.domfreq["walking"], med.domfreq["jumping"]),
        **{f"median_motion_{a}": float(v) for a, v in med.motion.items()},
        **{f"median_domfreq_{a}": float(v) for a, v in med.domfreq.items()},
    }


def classify(steps: pd.DataFrame, thr: dict[str, float]) -> np.ndarray:
    """Apply rules 1-5 of the module docstring to every step."""
    flat = (steps.tilt_deg < FLAT_DEG).to_numpy()
    mot = steps.motion.to_numpy()
    out = np.full(len(steps), "transition", dtype=object)
    out[flat & (mot < thr["T_still"])] = "still"
    nf = ~flat
    out[nf & (mot < thr["T_move"])] = "standing"
    out[nf & (mot >= thr["T_move"]) & (mot < thr["T_jump"])] = "walking"
    out[nf & (mot >= thr["T_jump"])] = "jumping"
    return out


def to_segments(t: np.ndarray, labels: np.ndarray, t_end: float) -> list[tuple[float, float, str]]:
    """Run-length encode step labels; runs < MIN_RUN_S become transition; merge neighbours."""
    def runs(lab):
        out, start = [], 0
        for i in range(1, len(lab) + 1):
            if i == len(lab) or lab[i] != lab[start]:
                out.append([start, i - 1, lab[start]])
                start = i
        return out

    lab = labels.copy()
    for s, e, name in runs(lab):
        if name != "transition" and (e - s + 1) * STEP_S < MIN_RUN_S:
            lab[s:e + 1] = "transition"
    segs = []
    for s, e, name in runs(lab):
        start = 0.0 if s == 0 else (t[s - 1] + t[s]) / 2
        end = t_end if e == len(t) - 1 else (t[e] + t[e + 1]) / 2
        segs.append((round(start, 2), round(end, 2), name))
    return segs


def plot(stem: str, steps: pd.DataFrame, segs, thr: dict[str, float], user: list[float] | None) -> Path:
    """Raw signals with the rule thresholds and the resulting segments."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [0.6, 1.3, 1, 1]})
    for s, e, name in segs:
        for ax in axes:
            ax.axvspan(s, e, color=COLORS[name], alpha=0.18 if ax is not axes[0] else 0.9, lw=0)
        if e - s >= 2.5:
            axes[0].text((s + e) / 2, 0.5, name, ha="center", va="center", fontsize=8)
    axes[0].set_yticks([])
    axes[0].set_title(f"{stem}: segments from raw signals (rule in src/segment.py)", fontsize=10)
    axes[1].semilogy(steps.t, steps.motion, color="k", lw=1)
    for k, ls in (("T_still", ":"), ("T_move", "--"), ("T_jump", "-.")):
        axes[1].axhline(thr[k], color="gray", ls=ls, lw=1, label=f"{k} = {thr[k]:.3g}")
    axes[1].set_ylabel("std |total acc|\n(m/s², 1 s)")
    axes[1].legend(fontsize=7, loc="upper left")
    axes[2].plot(steps.t, steps.tilt_deg, color="C0", lw=1)
    axes[2].axhline(FLAT_DEG, color="gray", ls="--", lw=1, label=f"flat < {FLAT_DEG:g}°")
    axes[2].set_ylabel("tilt from\nscreen-up (°)")
    axes[2].legend(fontsize=7, loc="upper left")
    axes[3].plot(steps.t, steps.domfreq, "o", ms=2.5, color="C3")
    axes[3].axhline(thr["F_jump"], color="gray", ls="--", lw=1,
                    label=f"training walking/jumping midpoint {thr['F_jump']:.2f} Hz (reference only)")
    axes[3].set_ylabel("dominant freq\n(Hz, 2 s)")
    axes[3].legend(fontsize=7, loc="upper left")
    axes[3].set_xlabel("time from recording start (s)")
    for b in user or []:
        for ax in axes[1:]:
            ax.axvline(b, color="purple", lw=1.2, alpha=0.7)
    if user:
        axes[1].plot([], [], color="purple", label="your note")
        axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"segment_{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def segment_recording(zip_path: Path, thr: dict[str, float]):
    """Steps and segments for one continuous recording."""
    raw = load_raw(extract(zip_path, TEST_DIR))
    steps = signal_steps(raw, raw.t.to_numpy())
    return steps, to_segments(steps.t.to_numpy(), classify(steps, thr), float(raw.t.iloc[-1]))


def main() -> None:
    """Segment every data/test zip, plot, print; write labels.csv with --write."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true", help=f"write {LABELS_FILE.relative_to(ROOT)}")
    args = p.parse_args()
    thr = training_thresholds()
    print("Thresholds from training medians:", {k: round(v, 3) for k, v in thr.items()})
    rows = []
    for z in sorted((DATA / TEST_DIR).glob("*.zip")):
        steps, segs = segment_recording(z, thr)
        print(f"\n{z.stem}  -> {plot(z.stem, steps, segs, thr, None).relative_to(ROOT)}")
        for s, e, name in segs:
            print(f"  {s:6.2f} - {e:6.2f}  {name}")
            rows.append({"recording": z.stem, "start_s": s, "end_s": e, "label": name})
    if args.write:
        pd.DataFrame(rows).to_csv(LABELS_FILE, index=False)
        print(f"\nwrote {LABELS_FILE}")


if __name__ == "__main__":
    main()
