# HMM metrics

## Main evaluation: data/test/ (2 continuous recordings)

Labels from raw signals only (`src/segment.py`, `data/test/labels.csv`). Windows: 2 s, 50 % overlap (1 s hop) at 50 Hz; a window is labelled only if >= 75 % of it lies in one segment, otherwise it is `transition` and excluded from the metrics. Each recording is Viterbi-decoded as one continuous sequence. Sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), one-vs-rest. Model trained on all 38 single-activity recordings.

| Features | Stage | Accuracy | Macro sensitivity | Macro specificity |
|---|---|---:|---:|---:|
| all features | before Baum-Welch | 1.000 | 1.000 | 1.000 |
| all features | after Baum-Welch | 1.000 | 1.000 | 1.000 |
| without gravity-mean features | before Baum-Welch | 1.000 | 1.000 | 1.000 |
| without gravity-mean features | after Baum-Welch | 0.979 | 0.976 | 0.993 |

### Per-activity tables

### all features, before Baum-Welch

| Activity | Windows | Sensitivity | Specificity |
|---|---:|---:|---:|
| jumping | 20 | 1.000 | 1.000 |
| standing | 25 | 1.000 | 1.000 |
| still | 21 | 1.000 | 1.000 |
| walking | 31 | 1.000 | 1.000 |

**Overall accuracy: 1.000** (97 windows)

### all features, after Baum-Welch

| Activity | Windows | Sensitivity | Specificity |
|---|---:|---:|---:|
| jumping | 20 | 1.000 | 1.000 |
| standing | 25 | 1.000 | 1.000 |
| still | 21 | 1.000 | 1.000 |
| walking | 31 | 1.000 | 1.000 |

**Overall accuracy: 1.000** (97 windows)

### without gravity-mean features, before Baum-Welch

| Activity | Windows | Sensitivity | Specificity |
|---|---:|---:|---:|
| jumping | 20 | 1.000 | 1.000 |
| standing | 25 | 1.000 | 1.000 |
| still | 21 | 1.000 | 1.000 |
| walking | 31 | 1.000 | 1.000 |

**Overall accuracy: 1.000** (97 windows)

### without gravity-mean features, after Baum-Welch

| Activity | Windows | Sensitivity | Specificity |
|---|---:|---:|---:|
| jumping | 20 | 1.000 | 1.000 |
| standing | 25 | 1.000 | 0.972 |
| still | 21 | 0.905 | 1.000 |
| walking | 31 | 1.000 | 1.000 |

**Overall accuracy: 0.979** (97 windows)

### Transition windows (excluded from the metrics)

| Recording | Windows | Transition (excluded) |
|---|---:|---:|
| mostly sure | 60 | 12 |
| unsure | 68 | 19 |

How the model decoded the transition windows (after Baum-Welch):

| Decoded as | All features | Without gravity-mean |
|---|---:|---:|
| jumping | 1 | 1 |
| standing | 1 | 3 |
| still | 0 | 0 |
| walking | 29 | 27 |

Figures: `figures/test_confusion.png`, `figures/test_decoded_*.png`, `figures/hmm_transition_matrix.png`, `figures/hmm_emissions.png`, `figures/hmm_convergence.png`.

## Supplementary

### Temporary hold-out on the training recordings (whole recordings held out)

| Features | Stage | Accuracy | Windows |
|---|---|---:|---:|
| all features | before Baum-Welch | 0.988 | 163 |
| all features | after Baum-Welch | 1.000 | 163 |
| without gravity-mean features | before Baum-Welch | 0.982 | 163 |
| without gravity-mean features | after Baum-Welch | 0.982 | 163 |

10 random hold-outs (accuracy):

| Features | Stage | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| all features | after Baum-Welch | 0.999 | 0.002 | 0.994 | 1.000 |
| all features | before Baum-Welch | 0.999 | 0.002 | 0.994 | 1.000 |
| without gravity-mean features | after Baum-Welch | 0.993 | 0.008 | 0.982 | 1.000 |
| without gravity-mean features | before Baum-Welch | 0.998 | 0.005 | 0.983 | 1.000 |

### Standing grip check (all standing of one session held out)

| Held-out standing | Features | Stage | Windows | Correct | Sensitivity | Errors predicted as |
|---|---|---|---:|---:|---:|---|
| 2026-10-26 (10 rec.) | all features | before Baum-Welch | 70 | 70 | 1.000 | – |
| 2026-10-26 (10 rec.) | all features | after Baum-Welch | 70 | 70 | 1.000 | – |
| 2026-10-26 (10 rec.) | without gravity-mean features | before Baum-Welch | 70 | 69 | 0.986 | still 1 |
| 2026-10-26 (10 rec.) | without gravity-mean features | after Baum-Welch | 70 | 69 | 0.986 | still 1 |
| 2026-10-27 (1 rec.) | all features | before Baum-Welch | 39 | 38 | 0.974 | walking 1 |
| 2026-10-27 (1 rec.) | all features | after Baum-Welch | 39 | 38 | 0.974 | walking 1 |
| 2026-10-27 (1 rec.) | without gravity-mean features | before Baum-Welch | 39 | 38 | 0.974 | walking 1 |
| 2026-10-27 (1 rec.) | without gravity-mean features | after Baum-Welch | 39 | 38 | 0.974 | walking 1 |

### Cross-session test

Model trained only on the 2026-10-26 session (31 recordings), tested on the 7 recordings of 2026-10-27: **a different session and a different grip** (mean gravity ~ (-8, -3, +1) for every activity; old standing was ~ (-6.5, -3.5, +6.5)). Default 1 s trims, before any trim overrides.
- **all features**: overall accuracy 0.980 (200 windows); sensitivity jumping 1.000, standing 0.905, walking 1.000
- **without gravity-mean features**: overall accuracy 0.980 (200 windows); sensitivity jumping 1.000, standing 0.905, walking 1.000
- All errors are in the new standing file, at t = 1, 45, 46, 47 s (2 s windows; file is ~50 s long): the phone settling at the start and being picked up at the end (motion and tilt jump in those windows), not the standing itself.
- Full details: `reports/cross_session_test.md`.

