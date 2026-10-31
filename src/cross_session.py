"""Cross-session test: model trained on the 2026-10-26 session, tested on 2026-10-27.

The 7 recordings from 2026-10-27 were made in a different session with a
different, consistent grip (mean gravity ~ (-8, -3, +1) for every activity).
This experiment was run BEFORE they were folded into the training data, so it
reproduces that state exactly:
  * preprocessing with the default 1 s trims (no data/trim_overrides.csv),
  * training on the 31 usable recordings of 2026-10-26 as they were then
    (walking 19-01-46 included; it was excluded only afterwards),
  * testing on the 7 new recordings, each decoded as its own sequence.

Writes reports/cross_session_test.md and three figures in reports/figures/.
Run from the project root:  python src/cross_session.py
(by default it preprocesses into a temporary directory; --processed DIR reuses
an existing output of `preprocess.py --no-trim-overrides --out DIR`).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import preprocess  # noqa: E402
from experiment import evaluate_recordings  # noqa: E402
from features import ACTIVITIES, load_dataset  # noqa: E402
from metrics import confusion_matrix  # noqa: E402

REPORTS = preprocess.ROOT / "reports"
FIGS = REPORTS / "figures"

TRAIN_RECORDINGS = [
    # jumping (10)
    "2026-10-26_18-44-27", "2026-10-26_18-44-41", "2026-10-26_18-44-56", "2026-10-26_18-45-12",
    "2026-10-26_18-45-38", "2026-10-26_18-45-58", "2026-10-26_18-46-17", "2026-10-26_18-46-33",
    "2026-10-26_18-46-49", "2026-10-26_18-47-06",
    # standing (10; 18-52-06 excluded: phone flat)
    "2026-10-26_18-38-31", "2026-10-26_18-38-44", "2026-10-26_18-38-58", "2026-10-26_18-39-11",
    "2026-10-26_18-39-23", "2026-10-26_18-39-37", "2026-10-26_18-39-51", "2026-10-26_18-40-03",
    "2026-10-26_18-40-19", "2026-10-26_18-40-34",
    # still (9)
    "2026-10-26_18-23-01_1", "2026-10-26_18-23-16", "2026-10-26_18-23-31", "2026-10-26_18-23-45",
    "2026-10-26_18-23-58", "2026-10-26_18-24-12", "2026-10-26_18-24-26", "2026-10-26_18-24-41",
    "2026-10-26_18-50-18",
    # walking (2)
    "2026-10-26_18-54-44", "2026-10-26_19-01-46",
]
TEST_RECORDINGS = [
    "2026-10-27_08-03-12", "2026-10-27_08-03-33",                      # jumping
    "Standing-2026-10-27_07-51-23",                                    # standing
    "2026-10-27_07-55-32", "2026-10-27_07-56-24", "2026-10-27_07-57-13",
    "2026-10-27_07-58-00",                                             # walking
]
STANDING = "Standing-2026-10-27_07-51-23"
VARIANTS = {"all features": False, "without gravity-mean features": True}


def fmt(v: float) -> str:
    """3-decimal number, or an en dash when undefined (no windows of that class)."""
    return "–" if pd.isna(v) else f"{v:.3f}"


def table_md(tab: pd.DataFrame, acc: float) -> str:
    """Per-activity table with one-vs-rest accuracy, plus overall accuracy."""
    lines = ["| Activity | Windows | Sensitivity | Specificity | Accuracy |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r.activity} | {r.windows} | {fmt(r.sensitivity)} | {fmt(r.specificity)} | {fmt(r.accuracy)} |"
              for r in tab.itertuples()]
    return "\n".join(lines + ["", f"**Overall accuracy: {acc:.3f}** ({int(tab.windows.sum())} windows)"])


def plot_confusion(results: dict) -> str:
    """Confusion matrices (after Baum-Welch) for both feature variants."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (name, (w, _)) in zip(axes, results.items()):
        C = confusion_matrix(w.y.values, w.pred_bw.values, len(ACTIVITIES))
        sns.heatmap(C, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=ACTIVITIES, yticklabels=ACTIVITIES)
        ax.set_title(f"{name}, after Baum-Welch", fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle("Cross-session test: trained on 2026-10-26, tested on 2026-10-27")
    fig.tight_layout()
    fig.savefig(FIGS / "cross_session_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "figures/cross_session_confusion.png"


def plot_standing(results: dict) -> str:
    """Decoded vs true over time for the new standing file, with motion and tilt."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    w0 = results["all features"][0]
    w0 = w0[w0.recording == STANDING]
    ax = axes[0]
    ax.step(w0.t_s, w0.y, where="post", lw=4, alpha=0.35, color="k", label="true (standing)")
    for (name, (w, _)), style in zip(results.items(), [dict(color="C0"), dict(color="C3", ls="--")]):
        w = w[w.recording == STANDING]
        ax.step(w.t_s, w.pred_bw, where="post", lw=1.4, label=f"Viterbi, {name}", **style)
        err = w[w.pred_bw != w.y]
        for t in err.t_s:
            ax.axvspan(t, t + 2.0, color=style["color"], alpha=0.12)
    ax.set_yticks(range(len(ACTIVITIES)))
    ax.set_yticklabels(ACTIVITIES)
    ax.set_title(f"{STANDING}: decoded activity (shaded = misclassified 2 s windows)")
    ax.legend(loc="center left", fontsize=8)
    axes[1].plot(w0.t_s + 1.0, w0.rms_acc, "o-", ms=3)
    axes[1].set_ylabel("RMS |acc| (m/s²)")
    axes[1].set_title("Motion level per window", fontsize=10)
    axes[2].plot(w0.t_s + 1.0, w0.mean_grav_z, "o-", ms=3, color="C2")
    axes[2].set_ylabel("mean grav_z (m/s²)")
    axes[2].set_title("Phone tilt per window (9.8 = lying flat)", fontsize=10)
    axes[2].set_xlabel("time from recording start (s); window centres in the lower panels")
    fig.tight_layout()
    fig.savefig(FIGS / "cross_session_standing_decoded.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "figures/cross_session_standing_decoded.png"


def run_all(processed: Path) -> dict:
    """Evaluate both feature variants on the fixed train/test recording lists."""
    df = load_dataset(processed, include_excluded=True)
    return {name: evaluate_recordings(df, TRAIN_RECORDINGS, TEST_RECORDINGS, drop_gravity_mean=drop)[:2]
            for name, drop in VARIANTS.items()}


def write_report(results: dict) -> None:
    """Write reports/cross_session_test.md (with a summary block for report_facts.md)."""
    FIGS.mkdir(parents=True, exist_ok=True)
    conf, stand = plot_confusion(results), plot_standing(results)

    per_rec = []
    for name, (w, _) in results.items():
        for rec, g in w.groupby("recording", sort=False):
            per_rec.append(f"| {name} | {g.activity.iloc[0]} | `{rec}` | {len(g)} | "
                           f"{(g.pred_bw == g.y).sum()} | "
                           + ", ".join(f"{ACTIVITIES[k]} {n}" for k, n in
                                       g.pred_bw[g.pred_bw != g.y].value_counts().sort_index().items())
                           + " |")
    ws = results["all features"][0]
    ws = ws[ws.recording == STANDING]
    err_t = {name: w[(w.recording == STANDING) & (w.pred_bw != w.y)].t_s.tolist()
             for name, (w, _) in results.items()}
    dur = ws.t_s.max() + 2.0 + 1.0  # last window end + default 1 s end trim

    summary_lines = []
    for name, (w, tables) in results.items():
        acc = tables["after Baum-Welch"][1]
        t = tables["after Baum-Welch"][0].set_index("activity")
        summary_lines.append(
            f"- **{name}**: overall accuracy {acc:.3f} ({len(w)} windows); sensitivity "
            + ", ".join(f"{a} {fmt(t.loc[a, 'sensitivity'])}" for a in ACTIVITIES if t.loc[a, "windows"]))
    errs = err_t["all features"]
    err_recs = sorted({r for w, _ in results.values() for r in w[w.pred_bw != w.y].recording})
    if err_recs == [STANDING]:
        where = (f"- All errors are in the new standing file, at t = {', '.join(f'{t:.0f}' for t in errs)} s "
                 f"(2 s windows; file is ~{dur:.0f} s long): the phone settling at the start and being "
                 "picked up at the end (motion and tilt jump in those windows), not the standing itself.")
    else:
        where = f"- Recordings with errors: {', '.join(f'`{r}`' for r in err_recs) or 'none'}."
    summary = "\n".join([
        f"Model trained only on the 2026-10-26 session ({len(TRAIN_RECORDINGS)} recordings), tested on the "
        f"{len(TEST_RECORDINGS)} recordings of 2026-10-27: **a different session and a different grip** "
        "(mean gravity ~ (-8, -3, +1) for every activity; old standing was ~ (-6.5, -3.5, +6.5)). "
        "Default 1 s trims, before any trim overrides.",
        *summary_lines, where,
        "- Full details: `reports/cross_session_test.md`.",
    ])

    sections = []
    for name, (w, tables) in results.items():
        for stage, (tab, acc) in tables.items():
            sections += [f"### {name}, {stage}", "", table_md(tab, acc), ""]

    md = f"""# Cross-session test (unseen session and grip)

**Setup.** The HMM was trained only on the 2026-10-26 session and tested on the 7 recordings made on
2026-10-27, before those were folded into the dataset. The new recordings are a **different session
with a different grip**: every activity has mean gravity ~ (-8, -3, +1), while the training standing
files had ~ (-6.5, -3.5, +6.5). This tests whether the model recognises the activity rather than the
way the phone was held.

* Preprocessing: default 1 s trim at both ends (no trim overrides), 50 Hz, 8 s chunks.
* Train: {len(TRAIN_RECORDINGS)} recordings of 2026-10-26 (standing 18-52-06 excluded; walking
  19-01-46 still included, as it was at the time).
* Test: {len(TEST_RECORDINGS)} recordings, each decoded as its own Viterbi sequence (chunks in order).
  Still has no test windows, so its sensitivity is undefined (–).
* Reproduce: `python src/cross_session.py` (recording lists are fixed in the script).

## Summary
<!-- summary -->
{summary}
<!-- /summary -->

![confusion]({conf})

![standing decoded]({stand})

## Per recording (after Baum-Welch)

| Features | Activity | Recording | Windows | Correct | Errors predicted as |
|---|---|---|---:|---:|---|
{chr(10).join(per_rec)}

## Per-activity metrics

Sensitivity = TP/(TP+FN); specificity = TN/(TN+FP); accuracy = (TP+TN)/all, one-vs-rest.

{chr(10).join(sections)}
"""
    (REPORTS / "cross_session_test.md").write_text(md)
    print(summary)


def main() -> None:
    """Preprocess with default trims into a temp dir (unless --processed) and write the report."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, help="existing --no-trim-overrides preprocess output")
    args = p.parse_args()
    if args.processed:
        write_report(run_all(args.processed))
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "processed"
        preprocess.main(out, use_trim_overrides=False)
        write_report(run_all(out))


if __name__ == "__main__":
    main()
