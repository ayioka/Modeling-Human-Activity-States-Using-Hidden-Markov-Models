"""Exploration figures and report numbers.

Reads data/processed/ (run preprocess.py first) and writes to reports/:
  figures/*.png       150 dpi figures (raw signals, magnitudes, spectra,
                      feature boxplots, gravity orientation)
  summary_table.md    per-activity file counts, durations, sampling rates
  report_facts.md     every number in one place, plus data-quality warnings
Run from the project root:  python src/explore.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from features import (ACTIVITIES, BANDS, FS, GRAVITY_MEAN_FEATURES, HOP_S,  # noqa: E402
                      WINDOW_S, feature_columns, load_dataset)
from preprocess import (CHUNK_S, EXCLUDE, MIN_CHUNK_S, PROCESSED, ROOT, TARGET_HZ,  # noqa: E402
                        TEST_DIR, TEST_LABELS_FILE, TRIM_S, safe_stamp)
from segment import FLAT_DEG, MIN_RUN_S, training_thresholds  # noqa: E402

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
DPI = 150
SPEC_WIN_S = 4.0  # longer than the 2 s feature window -> 0.25 Hz resolution for the plot
PALETTE = dict(zip(ACTIVITIES, sns.color_palette("colorblind", len(ACTIVITIES))))
BOX_FEATURES = ["rms_acc", "sma_gyro", "std_acc_z", "corr_acc_xy",
                "domfreq_accmag", "logenergy_accmag", "band_accmag_1.5_3", "logenergy_gyromag",
                "mean_grav_x", "mean_grav_y", "mean_grav_z", "domfreq_gyromag"]


def session(stamp: str) -> str:
    """Recording session = the date in the zip name, e.g. '2026-10-27'."""
    m = re.search(r"\d{4}-\d{2}-\d{2}", stamp)
    return m.group(0) if m else "unknown"


def cross_session_summary() -> str:
    """Summary block of reports/cross_session_test.md (written by cross_session.py), if present."""
    path = REPORTS / "cross_session_test.md"
    if not path.exists():
        return "_Not run yet: `python src/cross_session.py`._"
    m = re.search(r"<!-- summary -->\n(.*?)\n<!-- /summary -->", path.read_text(), re.S)
    return m.group(1) if m else "_Summary block missing in cross_session_test.md._"


def test_labels_section(test_feats: pd.DataFrame) -> str:
    """How data/test/labels.csv was made, its segments and the resulting window counts."""
    if not TEST_LABELS_FILE.exists():
        return "_No `data/test/labels.csv` yet: test recordings are labelled `mixed`._"
    thr = training_thresholds()
    lab = pd.read_csv(TEST_LABELS_FILE)
    out = [
        "Boundaries come from **raw signals only** (never from the model, which would be circular), "
        "using the rule in `src/segment.py`, evaluated every 0.5 s:",
        "",
        f"1. phone flat (gravity within {FLAT_DEG:g}° of screen-up) and motion < T_still -> still",
        "2. phone flat and motion >= T_still -> transition (phone handled flat)",
        "3. motion < T_move -> standing; 4. motion >= T_jump -> jumping; 5. otherwise walking",
        f"6. runs shorter than {MIN_RUN_S:g} s become transition.",
        "",
        "Motion = std of |total acceleration| over 1 s. Thresholds are geometric midpoints between "
        "the medians of neighbouring activities in the single-activity training recordings "
        f"(not tuned on the test files): T_still = {thr['T_still']:.3f}, T_move = {thr['T_move']:.3f}, "
        f"T_jump = {thr['T_jump']:.2f} m/s².",
        "",
        "The dominant frequency of |total acceleration| is plotted but not used in the rule: in training "
        f"walking peaks at {thr['median_domfreq_walking']:g} Hz and jumping at {thr['median_domfreq_jumping']:g} Hz, "
        "but in the \"unsure\" recording walking also peaks at 2-2.5 Hz, so cadence does not separate "
        "walking from jumping there; intensity does (walking 1-3, jumping 8-13 m/s²).",
        "",
        "A 2 s window takes a segment's label only if >= 75 % of it lies inside that segment; "
        "otherwise it is `transition`. Transition windows are decoded (the sequence stays "
        "continuous) but excluded from the metrics.",
        "",
    ]
    for rec, g in lab.groupby("recording", sort=False):
        out.append(f"**{rec}** ([raw signals](figures/segment_{rec}.png))")
        out.append("")
        out.append(md_table(g[["start_s", "end_s", "label"]].reset_index(drop=True)))
        w = test_feats[test_feats.recording == safe_stamp(rec)].label.value_counts()
        out.append("")
        out.append("Windows: " + ", ".join(f"{k} {v}" for k, v in w.items()))
        out.append("")
    out.append(
        "**Revised note.** For the \"unsure\" recording the original note put walking -> jumping at 49 s. "
        "The raw signals put it at about 57 s: walking-level motion continues until ~54 s, the phone is "
        "flat (handled) from ~54 to ~57 s, and jumping-level motion starts at ~57 s. At 48-50 s the phone "
        "shifts in the pocket (tilt drops from ~90° to ~45° for 2 s while motion stays at walking level), "
        "which explains the earlier recollection.")
    return "\n".join(out)


def session_differences(feats: pd.DataFrame, test_feats: pd.DataFrame) -> str:
    """Still noise level per session and standing orientation per session / test file."""
    f = feats.assign(session=feats.recording.map(session))
    still = f[f.activity == "still"].groupby("session").rms_acc.median()
    lines = ["- **Still noise level** (median RMS of linear acceleration per window): "
             + ", ".join(f"{s} {v:.3f} m/s²" for s, v in still.items())]
    if len(still) > 1:
        lines[-1] += (f" -> the new still recording is about {still.iloc[-1] / still.iloc[0]:.0f}x noisier "
                      "(different surface or a vibration source nearby). Still is still far below standing, "
                      "but the margin to standing shrinks, which matters most without gravity features.")
    rows = []
    for s, g in f[f.activity == "standing"].groupby("session"):
        rows.append((f"training {s} ({g.recording.nunique()} rec.)", *g[GRAVITY_MEAN_FEATURES].mean()))
    for rec, g in test_feats[test_feats.label == "standing"].groupby("recording"):
        rows.append((f"test {rec}", *g[GRAVITY_MEAN_FEATURES].mean()))
    grips = pd.DataFrame(rows, columns=["standing in", "grav_x", "grav_y", "grav_z"])
    lines += ["- **Standing appears in three grips** across sessions (mean gravity, m/s²):", "",
              md_table(grips, "+.1f"), "",
              "  The model has to recognise standing from low, body-sway-level motion rather than "
              "from one fixed phone orientation."]
    return "\n".join(lines)


def save(fig, name: str) -> str:
    """Save a figure to reports/figures/ and return its relative path."""
    path = FIGS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def example_files(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """First non-excluded processed file of each activity."""
    out = {}
    for act in ACTIVITIES:
        rows = manifest[(manifest.activity == act) & ~manifest.excluded]
        if len(rows):
            out[act] = pd.read_csv(PROCESSED / rows.iloc[0]["file"])
    return out


def plot_raw(examples: dict[str, pd.DataFrame]) -> str:
    """Acc and gyro x,y,z vs time for one example per activity."""
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 2.4 * len(examples)), sharex=True)
    for i, (act, df) in enumerate(examples.items()):
        for j, (sensor, unit) in enumerate([("acc", "m/s²"), ("gyro", "rad/s")]):
            ax = axes[i, j]
            for axis in "xyz":
                ax.plot(df.timestamp, df[f"{sensor}_{axis}"], lw=0.8, label=axis)
            ax.set_title(f"{act}: {'linear acceleration' if sensor == 'acc' else 'gyroscope'}")
            ax.set_ylabel(unit)
            if i == 0:
                ax.legend(loc="upper right", ncol=3, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Raw signals after resampling to 50 Hz and trimming (one example per activity)")
    fig.tight_layout()
    return save(fig, "raw_signals.png")


def plot_magnitude(examples: dict[str, pd.DataFrame]) -> str:
    """Linear-acceleration magnitude and gyro magnitude per activity, shared y axes."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for act, df in examples.items():
        acc = np.linalg.norm(df[["acc_x", "acc_y", "acc_z"]], axis=1)
        gyr = np.linalg.norm(df[["gyro_x", "gyro_y", "gyro_z"]], axis=1)
        axes[0].plot(df.timestamp, acc, lw=0.9, color=PALETTE[act], label=act)
        axes[1].plot(df.timestamp, gyr, lw=0.9, color=PALETTE[act], label=act)
    axes[0].set_ylabel("|linear acc| (m/s²)")
    axes[0].set_title("Acceleration magnitude (gravity removed)")
    axes[1].set_ylabel("|gyro| (rad/s)")
    axes[1].set_title("Angular-velocity magnitude")
    axes[1].set_xlabel("time (s)")
    axes[0].legend(ncol=4)
    fig.tight_layout()
    return save(fig, "magnitude.png")


def average_spectra(manifest: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mean FFT amplitude of the acc magnitude per activity over 4 s Hann windows."""
    n = int(SPEC_WIN_S * FS)
    freqs = np.fft.rfftfreq(n, 1 / FS)
    spectra = {}
    for act in ACTIVITIES:
        rows = manifest[(manifest.activity == act) & ~manifest.excluded]
        acc = []
        for f in rows.file:
            df = pd.read_csv(PROCESSED / f)
            mag = np.linalg.norm(df[["acc_x", "acc_y", "acc_z"]], axis=1)
            for s in range(0, len(mag) - n + 1, n // 2):
                seg = mag[s:s + n]
                acc.append(np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(n))) / n)
        if acc:
            spectra[act] = np.mean(acc, axis=0)
    return freqs, spectra


def plot_spectra(freqs: np.ndarray, spectra: dict[str, np.ndarray]) -> str:
    """Average spectrum per activity on a log y axis, with peak annotated."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for act, sp in spectra.items():
        ax.semilogy(freqs, sp, color=PALETTE[act], label=act, lw=1.4)
        k = np.argmax(sp[1:]) + 1
        ax.annotate(f"{freqs[k]:.2f} Hz", (freqs[k], sp[k]), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color=PALETTE[act])
    ax.set_xlim(0, 15)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean |FFT| of acc magnitude")
    ax.set_title(f"Average spectrum per activity ({SPEC_WIN_S:g} s Hann windows, 50 % overlap)")
    ax.legend()
    fig.tight_layout()
    return save(fig, "spectra.png")


def plot_boxplots(feats: pd.DataFrame) -> str:
    """Per-activity boxplots of the main window features."""
    cols = [c for c in BOX_FEATURES if c in feats]
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3 * nrow))
    for ax, c in zip(axes.flat, cols):
        sns.boxplot(data=feats, x="activity", y=c, hue="activity", order=ACTIVITIES,
                    palette=PALETTE, ax=ax, fliersize=2, legend=False)
        ax.set_title(c)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"Window features per activity ({WINDOW_S:g} s windows, {HOP_S:g} s hop)")
    fig.tight_layout()
    return save(fig, "feature_boxplots.png")


def plot_gravity(recs: pd.DataFrame, manifest: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Mean gravity vector per recording: shows orientation differs per activity."""
    rows = []
    for _, r in recs[recs.group.isin(ACTIVITIES)].iterrows():
        files = manifest[manifest.recording == r.stamp].file
        g = pd.concat([pd.read_csv(PROCESSED / f)[["grav_x", "grav_y", "grav_z"]] for f in files])
        rows.append({"activity": r.group, "recording": r.stamp, "session": session(r.stamp),
                     "excluded": r.excluded, **g.mean().round(2).to_dict()})
    gdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (a, b) in zip(axes, [("grav_x", "grav_z"), ("grav_y", "grav_z")]):
        for act in ACTIVITIES:
            d = gdf[(gdf.activity == act) & ~gdf.excluded]
            ax.scatter(d[a], d[b], color=PALETTE[act], label=act, s=40)
        ex = gdf[gdf.excluded]
        ax.scatter(ex[a], ex[b], marker="x", color="black", s=60, label="excluded")
        ax.set_xlabel(f"mean {a} (m/s²)")
        ax.set_ylabel(f"mean {b} (m/s²)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean gravity vector per recording (phone orientation)")
    fig.tight_layout()
    return save(fig, "gravity_orientation.png"), gdf


def summary_table(recs: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-activity counts, durations, measured rates and jitter."""
    rows = []
    for act in ACTIVITIES:
        r = recs[(recs.group == act) & ~recs.excluded]
        m = manifest[(manifest.activity == act) & ~manifest.excluded]
        rows.append({
            "activity": act,
            "recordings": len(r),
            "excluded": int(recs[(recs.group == act) & recs.excluded].shape[0]),
            "processed files": len(m),
            "raw total (s)": r.raw_duration_s.sum(),
            "raw mean (s)": r.raw_duration_s.mean(),
            "raw min (s)": r.raw_duration_s.min(),
            "raw max (s)": r.raw_duration_s.max(),
            "kept (s)": m.duration_s.sum(),
            "dropped remainder (s)": r.dropped_s.sum(),
            "acc Hz": r.acc_hz.median(),
            "gyro Hz": r.gyro_hz.median(),
            "grav Hz": r.grav_hz.median(),
            "acc jitter mean (ms)": r.acc_jitter_ms.mean(),
            "acc jitter max (ms)": r.acc_jitter_ms.max(),
            "max gap (ms)": r[["acc_max_gap_ms", "gyro_max_gap_ms", "grav_max_gap_ms", "tacc_max_gap_ms"]].max().max(),
            "Metadata requested Hz": r.acc_meta_hz.median(),
        })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Minimal DataFrame -> Markdown table (no extra dependency)."""
    def fmt(v):
        return f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def warnings(recs: pd.DataFrame, gdf: pd.DataFrame) -> list[str]:
    """Data-quality warnings: exclusions, gaps, short files, dropped data, orientation."""
    w = []
    for (act, stem), why in EXCLUDE.items():
        w.append(f"**Excluded** `{act}/{stem}`: {why}. Kept on disk and in the manifest (`excluded=True`).")
    gap_cols = ["acc_max_gap_ms", "gyro_max_gap_ms", "grav_max_gap_ms", "tacc_max_gap_ms"]
    gappy = recs[recs[gap_cols].max(axis=1) > 30]
    for _, r in gappy.iterrows():
        gap = r[gap_cols].max()
        missing = max(1, int(round(gap / (1000 / r.acc_hz))) - 1)
        w.append(f"Timestamp gap of {gap:.1f} ms in `{r.group}/{r.stamp}` "
                 f"(nominal {1000 / r.acc_hz:.1f} ms; ~{missing} sample(s) missing) - filled by linear interpolation.")
    short = recs[recs.trimmed_duration_s < CHUNK_S]
    for _, r in short.iterrows():
        w.append(f"Short file `{r.group}/{r.stamp}`: {r.trimmed_duration_s:.2f} s after trimming.")
    dropped = recs[(recs.dropped_s > 0) & (recs.n_chunks > 0)]
    w.append(f"Dropped remainders (< {MIN_CHUNK_S:g} s) in {len(dropped)} of {len(recs)} recordings, "
             f"{dropped.dropped_s.sum():.1f} s in total: "
             + ", ".join(f"{a} {s:.1f} s" for a, s in dropped.groupby("group").dropped_s.sum().items())
             + ". Recording each clip >= 13 s would avoid most of this (8 s chunk + 5 s remainder).")
    for act in ACTIVITIES:
        n = (recs.group == act).sum() - recs[(recs.group == act)].excluded.sum()
        if n < 3:
            w.append(f"Only {n} usable `{act}` recording(s): a whole-recording hold-out leaves "
                     f"{max(n - 1, 0)} for training.")
    walk = gdf[(gdf.activity == "walking") & ~gdf.excluded]
    if len(walk) > 1 and (np.sign(walk.grav_x).nunique() > 1):
        w.append("Walking recordings were made with the phone in opposite orientations "
                 "(mean grav_x " + ", ".join(f"{v:+.1f}" for v in walk.grav_x) + "). "
                 "Re-record with a consistent grip.")
    return w


def main() -> None:
    """Build every figure, the summary table and report_facts.md."""
    FIGS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(PROCESSED / "manifest.csv")
    recs = pd.read_csv(PROCESSED / "recordings.csv")
    sns.set_theme(style="whitegrid", context="notebook")

    examples = example_files(manifest)
    figs = {"raw": plot_raw(examples), "mag": plot_magnitude(examples)}
    freqs, spectra = average_spectra(manifest)
    figs["spec"] = plot_spectra(freqs, spectra)
    all_feats = load_dataset(PROCESSED)
    feats = all_feats[all_feats.activity.isin(ACTIVITIES)]           # single-activity training files
    test_feats = all_feats[all_feats.file.str.startswith(f"{TEST_DIR}/")]
    figs["box"] = plot_boxplots(feats)
    figs["grav"], gdf = plot_gravity(recs, manifest)

    summ = summary_table(recs, manifest)
    (REPORTS / "summary_table.md").write_text(md_table(summ) + "\n")
    peaks = {a: freqs[np.argmax(sp[1:]) + 1] for a, sp in spectra.items()}
    feat_means = feats.groupby("activity")[["rms_acc", "sma_gyro", "domfreq_accmag"]].median()
    grav_sess = (gdf[~gdf.excluded].groupby(["activity", "session"])
                 .agg(recordings=("recording", "count"), grav_x=("grav_x", "mean"),
                      grav_y=("grav_y", "mean"), grav_z=("grav_z", "mean")).reset_index())
    trimmed = recs[(recs.trim_start_s != TRIM_S) | (recs.trim_end_s != TRIM_S)]
    trim_lines = "\n".join(f"  - `{r.group}/{r.stamp}`: start {r.trim_start_s:g} s, end {r.trim_end_s:g} s"
                           for r in trimmed.itertuples()) or "  - none"
    n_feat = len(feature_columns(feats))
    win_counts = feats.groupby("activity").size()
    measured = recs[recs.group.isin(ACTIVITIES)]

    facts = f"""# Report facts

All numbers below are generated by `src/explore.py` from `data/processed/`.

## Data
{md_table(summ)}

Figures: [raw signals]({figs['raw']}), [magnitude]({figs['mag']}), [spectra]({figs['spec']}),
[feature boxplots]({figs['box']}), [gravity orientation]({figs['grav']}).

## Sampling rate and resampling
- Measured rate (median of 1/dt from timestamps): **{measured.acc_hz.median():.1f} Hz** for every sensor
  (accelerometer, gyroscope, gravity, total acceleration) in every recording (range
  {measured[['acc_hz', 'gyro_hz', 'grav_hz', 'tacc_hz']].min().min():.2f}-{measured[['acc_hz', 'gyro_hz', 'grav_hz', 'tacc_hz']].max().max():.2f} Hz),
  i.e. one sample every {1000 / measured.acc_hz.median():.2f} ms.
- `Metadata.csv` reports `sampleRateMs = 10`, i.e. **{measured.acc_meta_hz.median():.0f} Hz requested**.
  The phone (Samsung SM-S916U, Android) delivered only ~53 Hz, so the measured rate is the real one.
- Timestamp jitter (std of dt): mean {measured.acc_jitter_ms.mean():.2f} ms, max {measured.acc_jitter_ms.max():.2f} ms.
- Largest gaps: {measured[['acc_max_gap_ms', 'gyro_max_gap_ms', 'grav_max_gap_ms', 'tacc_max_gap_ms']].max(axis=1).nlargest(4).round(1).tolist()} ms
  in the long recordings (1-3 missing samples). These were **filled by linear interpolation** when
  resampling; gaps this short are well below one motion cycle (>= 300 ms).
- **Resampled to {TARGET_HZ:g} Hz** with linear interpolation onto one common grid for all sensors.
  Reason: it must be at or below the lowest measured rate (53.1 Hz) so we only interpolate between real
  samples; 50 Hz is the nearest round rate below it, and its 25 Hz Nyquist limit is far above
  human-motion content (cadence 1.5-3 Hz, harmonics < 10 Hz).
- Trimmed {TRIM_S:g} s from each end (screen-tap noise), then split into {CHUNK_S:g} s chunks;
  a remainder >= {MIN_CHUNK_S:g} s becomes its own chunk, otherwise it is dropped.
- Per-recording trim overrides from `data/trim_overrides.csv` (phone settling / being picked up;
  also recorded in `manifest.csv` as `trim_start_s`, `trim_end_s`):
{trim_lines}
- Accelerometer.csv is linear acceleration (gravity removed: mean |acc| ~ 0.02 m/s² when still);
  gravity is kept from Gravity.csv and TotalAcceleration.csv.

## Windows and features
- **Window: {WINDOW_S:g} s ({int(WINDOW_S * FS)} samples) with 50 % overlap ({HOP_S:g} s hop).**
  Walking cadence is ~1.5-2 Hz, so a window spans 3-4 cycles; FFT resolution is 1/{WINDOW_S:g} s =
  {1 / WINDOW_S:g} Hz. A 1 s hop gives the HMM one observation per second.
- {n_feat} features per window (time domain: per-axis means incl. gravity, per-axis std, RMS, SMA,
  axis correlations; frequency domain from |acc| and |gyro|: dominant frequency, log spectral energy,
  log band energy in {', '.join(f'{lo:g}-{hi:g} Hz' for lo, hi in BANDS)}).
- Windows per activity: {', '.join(f'{a} {n}' for a, n in win_counts.items())} (total {len(feats)}).

## Key numbers
- Spectral peak of |acc| per activity (average spectrum, {SPEC_WIN_S:g} s windows):
  {', '.join(f'{a} **{f:.2f} Hz**' for a, f in peaks.items())}.
  (Still/standing have no real cadence; their "peak" is noise.)
- Median RMS |acc| (m/s²): {', '.join(f'{a} {v:.3f}' for a, v in feat_means.rms_acc.items())}.
- Median SMA gyro (rad/s): {', '.join(f'{a} {v:.3f}' for a, v in feat_means.sma_gyro.items())}.
- Mean gravity vector (m/s²) per activity and recording session, excluded files left out:

{md_table(grav_sess, '+.1f')}

## Cross-session test
{cross_session_summary()}

## Test labels (data/test/)
{test_labels_section(test_feats)}

## Session differences (for the discussion)
{session_differences(feats, test_feats)}

## Data-quality warnings
{chr(10).join('- ' + s for s in warnings(recs, gdf))}

## Limitations
- **Orientation confound.** The gravity-mean features ({', '.join(GRAVITY_MEAN_FEATURES)}) differ
  between activities partly because the phone was *held differently*: in the 2026-10-26 session
  standing was ~(-6.5, -3.5, +6.5), jumping ~(-7, -5.5, -1.5) and still flat (0, 0, 9.8). In the
  2026-10-27 session one consistent grip ~(-8, -3, +1) was used for walking, jumping and standing,
  which weakens the confound but means standing now appears with two different orientations.
  A model may learn "how the phone was held" rather than "what the person did", so the notebook
  trains and evaluates with and without these three features (ablation); both metric tables are in
  `metrics_table.md`. The walking recording with the opposite orientation (19-01-46) is excluded.
- Other per-axis features (per-axis std/mean of linear acc and gyro, axis correlations) are also
  orientation-dependent, to a lesser degree; magnitude-based features are not.
- The main test set is two continuous recordings (one session, one person, one phone), so the
  metrics rest on a few hundred windows; the temporary hold-out and cross-session test are
  supplementary checks with even fewer recordings per activity.
- Test labels come from a raw-signal rule, not from a stopwatch; boundaries are uncertain by
  about +-1 s, which is why windows straddling a boundary are excluded as `transition`.
- Training sequences are concatenations of single-activity files, so the learned transition
  probabilities describe the concatenation scheme, not real activity switching.
"""
    (REPORTS / "report_facts.md").write_text(facts)
    print(md_table(summ))
    print(f"\nWrote {REPORTS / 'report_facts.md'} and {len(figs)} figures to {FIGS}")


if __name__ == "__main__":
    main()
