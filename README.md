# Human activity recognition with a from-scratch HMM

This project classifies four activities (jumping, standing, still, walking) from phone sensor recordings made with the Sensor Logger app. It uses a Gaussian Hidden Markov Model that is implemented with numpy only, in [src/hmm.py](src/hmm.py). Everything runs on the CPU.

- **Report:** [reports/report.pdf](reports/report.pdf)
- **Videos:** <add link>

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Run all commands from the project root. The notebook and the tests run on the included data:
```bash
python -m unittest discover tests   # toy 2-state HMM: forward / Viterbi vs brute force
jupyter nbconvert --to notebook --execute --inplace notebooks/hmm_activity.ipynb   # train, evaluate, plot; writes reports/metrics_table.md
```
The notebook is saved with its outputs, so the results can be read without running it.

The original Sensor Logger zips are not included. To rebuild everything from them, put them in `data/<activity>/` and `data/test/`, then run:
```bash
python src/preprocess.py            # zips -> data/processed/ (+ manifest.csv, recordings.csv)
python src/segment.py --write       # only when data/test/ changes: raw-signal labels -> data/test/labels.csv
python src/preprocess.py            # again, so the test files pick up those labels
python src/cross_session.py         # 2026-10-26 -> 2026-10-27 test, reports/cross_session_test.md
python src/explore.py               # figures + summary_table.md + report_facts.md in reports/
```

## Layout
| Path | What |
|---|---|
| `data/processed/<activity>/` | Labelled 50 Hz CSVs: trimmed and split into 8 s chunks |
| `data/processed/manifest.csv` | One row per CSV: source zip, duration, measured sensor rates, trims, and whether it was excluded and why |
| `data/processed/recordings.csv` | One row per source recording |
| `data/test/` | The 2 continuous multi-activity test recordings as CSV (the main evaluation set; the same files are in `data/processed/test/`, which the notebook reads) |
| `data/test/labels.csv` | Test segments (`recording`, `start_s`, `end_s`, `label`), set from raw signals by `src/segment.py`, never from the model. A 2 s window takes a label only if at least 75% of it falls in one segment; otherwise it is `transition` and excluded from the metrics. |
| `data/trim_overrides.csv` | Per-recording trims (`recording` = zip stem, `trim_start_s`, `trim_end_s`; blank = 1 s default) |
| `src/preprocess.py` | Extract, measure rates, resample, trim, split, save |
| `src/segment.py` | Raw-signal segmentation of the test recordings (motion, tilt; thresholds from the training data), with plots |
| `src/cross_session.py` | Fixed train/test recording lists: model trained on the first session, tested on the second |
| `src/explore.py` | Exploration figures and report numbers |
| `src/features.py` | 2 s / 1 s-hop windows, features, z-score |
| `src/hmm.py` | Gaussian HMM: supervised init, forward–backward, Baum–Welch, Viterbi |
| `src/metrics.py` | Confusion matrix, sensitivity/specificity, Hungarian state mapping |
| `src/experiment.py` | Hold-out split, training-sequence construction, train/evaluate loop |
| `tests/test_hmm.py` | HMM unit tests |
| `notebooks/hmm_activity.ipynb` | Runs everything and shows the results |
| `reports/` | The report, `metrics_table.md`, `report_facts.md`, `cross_session_test.md`, `summary_table.md` and `figures/` |
