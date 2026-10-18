# EEG Motor Imagery Classification (Python, XGBoost, Riemannian log-covariance)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

A small, dependency-light pipeline for **cross-subject EEG motor imagery classification** — the core problem behind non-invasive brain-computer interfaces (BCI). Given a 64-channel EEG epoch, it predicts which movement the person was imagining: **left hand, right hand, both hands, or both feet**.

Built for the Rice NeuroWave Hackathon 2026, then reproduced on the public **PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)** so anyone can check the number and use it as a baseline: two commands, no GPU. The pipeline runs on any epoched motor-imagery dataset in `.npy` form (BCI Competition IV 2a, OpenBMI, your own recordings).

**What you get**

- Leakage-safe validation: `GroupKFold` split by participant, so scores reflect real cross-subject generalisation
- A public, reproducible baseline on EEGMMIDB with a fixed held-out set of subjects
- Two feature families: classic bandpower statistics and Riemannian log-covariance (log-Euclidean SPD) features, auto-selected by CV
- An XGBoost fold ensemble with per-fold scaling
- Cached features, so iterating on the model takes seconds instead of minutes
- Confusion matrix, classification report and feature-importance dumps on every run
- Tests on every push (GitHub Actions)

## Results

4-class accuracy (left hand, right hand, both hands, both feet) on people the model never saw during training. No per-subject calibration.

| Data | Protocol | Subjects | Epochs | Accuracy |
|---|---|---|---|---|
| — | Chance (4 balanced classes) | — | — | 25.0% |
| Hackathon data (private) | 5-fold CV grouped by participant | — | 7,339 | 43.4% |
| PhysioNet EEGMMIDB | 5-fold CV grouped by subject, subjects 1–89 | 88 | 7,883 | 43.2% ± 1.8% |
| PhysioNet EEGMMIDB | Held out: train on subjects 1–89, test once on 90–109 | 18 | 1,612 | 45.8% |

CV accuracy is the mean ± standard deviation over the five folds. Held-out accuracy is pooled over all 1,612 held-out epochs; per subject it is 45.7% ± 10.1% (range 34.4% to 72.2%). Sources: [`results/physionet/run.json`](results/physionet/run.json) and the reports next to it for PhysioNet; [`cv_confusion_matrix.txt`](cv_confusion_matrix.txt) for the hackathon run.

On both datasets the log-covariance features won the `--feature_set auto` comparison; bandpower-only features scored lower under the same split (33.2% ± 0.7% CV on PhysioNet). The hackathon run also produced predictions for 20 evaluation participants, submitted as `predictions.zip` (not in this repo).

Per-class F1 from each run's report (precision and recall are in the same files):

| Class | Hackathon CV | PhysioNet CV | PhysioNet held-out |
|---|---|---|---|
| both_feet | 0.44 | 0.41 | 0.41 |
| both_hands | 0.40 | 0.41 | 0.41 |
| left_hand | 0.46 | 0.47 | 0.49 |
| right_hand | 0.43 | 0.43 | 0.52 |

All of this is the cross-subject, calibration-free setting, on raw, unfiltered EEG. Within-subject models, calibrated on each person, solve a different problem and are not comparable to these numbers.

## Reproduce on public data

```bash
pip install -r requirements.txt -r requirements-physionet.txt

python prepare_physionet.py

python run_pipeline.py \
  --train_dir data/physionet/Training \
  --eval_dir data/physionet/Evaluation \
  --eval_labels_dir data/physionet/Evaluation_labels \
  --out_dir data/physionet/predictions \
  --model_out data/physionet/model.joblib \
  --cache_dir data/physionet/cache \
  --report_dir results/physionet \
  --feature_set auto
```

On an Apple M3 Pro (11 cores, macOS, Python 3.14) the first command took 6 minutes, most of it downloading 1.7 GB, and the second took 32 minutes. The reports land in `results/physionet/`; compare them with the committed ones. Library versions and the machine are recorded in `run.json`; other versions or hardware may move the last digits. `python scripts/reproduce_physionet.py` runs both commands, rebuilding the epochs and the feature cache from the raw files, and also rewrites `run.json`.

**Data.** [EEGMMIDB v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/): 109 volunteers, 64 EEG channels (10-10 system), 160 Hz, recorded with BCI2000. Only the motor *imagery* runs are used; motor execution runs, baselines and T0 rest periods are not.

| Runs | T1 | T2 |
|---|---|---|
| 4, 8, 12 | imagine left fist → `left_hand` | imagine right fist → `right_hand` |
| 6, 10, 14 | imagine both fists → `both_hands` | imagine both feet → `both_feet` |

**Epochs.** Exactly 656 samples (4.1 s) starting at each T1/T2 onset, all 64 EEG channels, in microvolts, float32. The 43 epochs whose window ran past the end of the recording are dropped.

**Preprocessing: none.** No filtering, no re-referencing, no baseline correction, no artifact rejection, no resampling. The classifier sees the raw signal in µV.

**Subjects.** 1–89 for training and 90–109 held out, a fixed split by subject number. Subjects 88, 92 and 100 were recorded at 128 Hz and are skipped, leaving 88 training and 18 held-out subjects. Held-out labels are written to a separate `Evaluation_labels/` folder and are read only after the predictions are saved.

**Protocol.** `--feature_set auto` runs the 5-fold grouped CV on the training subjects for both feature sets and keeps the one with the higher mean; that CV mean is the CV row, so it carries a small optimistic bias from choosing between two. The held-out row has no such bias: the five fold models are averaged and scored once on subjects 90–109. XGBoost settings are the defaults below (800 trees, seed 42).

Downloads come from PhysioNet's AWS open-data copy of the dataset by default, which was about 50 times faster than physionet.org when this was measured; `--mirror physionet` uses physionet.org instead. Each file is checked against the SHA-256 list that ships with MNE as it is downloaded.

### Citing the dataset

If you use the PhysioNet numbers or `prepare_physionet.py`, cite the dataset as PhysioNet asks:

- Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.* IEEE Transactions on Biomedical Engineering 51(6):1034–1043, 2004.
- Goldberger, A.L., Amaral, L.A.N., Glass, L., Hausdorff, J.M., Ivanov, P.Ch., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.-K., Stanley, H.E. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.* Circulation 101(23):e215–e220, 2000.

The [dataset page](https://physionet.org/content/eegmmidb/1.0.0/) also gives a citation for the dataset itself (Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P) and now lists Pollard et al. (2026), *PhysioNet as a global platform for biomedical research*, Nature Health, https://doi.org/10.1038/s44360-026-00096-z, as PhysioNet's standard citation.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/elchibek5/Rice-NeuroWave-Hackathon-2026.git
cd Rice-NeuroWave-Hackathon-2026
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `xgboost`, `joblib`. Nothing else. The PhysioNet preparation also needs `mne` (`requirements-physionet.txt`); the tests need `pytest` (`requirements-dev.txt`).

## Quickstart

```bash
python3 run_pipeline.py \
  --train_dir path/to/Training \
  --eval_dir  path/to/Evaluation \
  --out_dir   predictions \
  --model_out model.joblib
```

This runs 5-fold participant-grouped CV, prints fold accuracies, trains the ensemble, writes one `y_pred_<id>.npy` per evaluation participant and saves the fitted models to `model.joblib`.

To package a submission for the hackathon:

```bash
zip -r predictions.zip predictions
```

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--train_dir` | required | Folder containing `X_train_<id>.npy` and `y_train_<id>.npy` pairs |
| `--eval_dir` | required | Folder containing `X_eval_<id>.npy` (or `X_test_<id>.npy`) |
| `--eval_labels_dir` | none | Optional folder of `y_eval_<id>.npy`. After predicting, scores them and writes `heldout_report.txt`. Must not be `--eval_dir`. |
| `--out_dir` | `predictions` | Where predictions are written |
| `--model_out` | `model.joblib` | Where the fitted model bundle is saved |
| `--report_dir` | `.` | Where `cv_confusion_matrix.txt`, `feature_importance_top20.txt` and `heldout_report.txt` are written |
| `--feature_set` | `auto` | `auto` tries `bandpower` and `covlog` and keeps the better CV score. `combo` concatenates both. |
| `--n_estimators` | `800` | XGBoost trees per model |
| `--cache_dir` | `cache_features` | Per-participant feature cache, tied to the exact source file (path, size, modification time). Delete it after changing feature code. |
| `--seed` | `42` | Random seed for XGBoost |

### Merging several training drops

If your data arrives in more than one folder:

```bash
python3 merge_training.py --out_train_dir Training_merged --train_dirs Drop1/ Drop2/
```

Existing files are never overwritten.

## Using it on your own EEG data

The pipeline only assumes the file layout below. To adapt it:

1. Epoch your recordings into arrays of shape `(n_epochs, n_channels, n_samples)` and save one `X_train_<id>.npy` / `y_train_<id>.npy` pair per participant. Labels can be any strings or integers; a `LabelEncoder` handles them.
2. Set `FS` at the top of `run_pipeline.py` to your sampling rate (default 160 Hz). The bandpower bands are defined in Hz and the Welch window is 256 samples, so other rates work as long as `n_samples >= 256`.
3. Channel count is inferred from the data. Log-covariance features scale as `n_channels * (n_channels + 1) / 2`, so 64 channels gives 2,080 features and 22 channels gives 253.
4. Run with `--feature_set auto` first; check `cv_confusion_matrix.txt` before trusting anything.
5. To score a held-out set, put its labels in a separate folder and pass `--eval_labels_dir`.

Class counts do not need to be balanced, but the CV accuracy reported is plain accuracy, so read the per-class report for imbalanced data.

```
Training/
  X_train_<id>.npy   (N, channels, samples) float
  y_train_<id>.npy   (N,)                   labels
Evaluation/
  X_eval_<id>.npy    (N, channels, samples)   # X_test_<id>.npy also accepted
Evaluation_labels/   (optional, separate folder)
  y_eval_<id>.npy    (N,)                   labels
```

Data is git-ignored. Never commit `.npy` files.

## How it works

```
X_train_<id>.npy ──┐
                   ├─▶ features ─▶ StandardScaler ─▶ XGBoost × 5 folds ─▶ mean probs ─▶ argmax ─▶ y_pred_<id>.npy
X_eval_<id>.npy  ──┘
```

**Input.** Each epoch is 64 channels × 656 samples at 160 Hz (≈ 4.1 s), in both the hackathon data and the PhysioNet preparation. Four imagined movements.

**Features.** Two families, chosen automatically by CV:

- *Bandpower* (14 per channel, 896 total) — mean, std, peak-to-peak, RMS, log power and relative power in the delta (1–4 Hz), theta (4–8), mu (8–12), beta (12–30) and gamma (30–45) bands, from a Welch PSD of each z-scored channel.
- *Log-covariance* (2,080 total) — the channel covariance matrix of each epoch, shrunk 10% toward a scaled identity for numerical stability, mapped through the matrix logarithm via eigendecomposition, upper triangle flattened. This is the log-Euclidean form of the Riemannian approach to motor imagery: it describes the *relationships between channels* rather than any single channel's spectrum.

**Validation.** `GroupKFold(n_splits=5)` with participant ID as the group. Splitting by epoch would let the model memorise each person's EEG signature and inflate accuracy; the numbers above are the honest cross-subject figures.

**Model.** XGBoost `multi:softprob`, 800 trees by default (`--n_estimators`), depth 6, learning rate 0.05, `hist` tree method. One model per fold, each with its own `StandardScaler`, plus one model fit on all data.

**Prediction.** Every fold model scores the evaluation epochs; probabilities are averaged and the argmax taken. Averaging over folds is free variance reduction that needs no extra tuning.

## Output files

| File | What it is |
|---|---|
| `predictions/y_pred_<id>.npy` | One per evaluation participant, shape `(N, 1)`, original label strings |
| `model.joblib` | Dict with `fold_models` (model + scaler each), `full_model`, `full_scaler`, `label_encoder`, `feature_set`, CV stats (`cv_mean_acc`, `cv_std_acc`, `cv_fold_acc`) and the CV report text |
| `cv_confusion_matrix.txt` | Confusion matrix and sklearn classification report from CV, for the selected feature set |
| `feature_importance_top20.txt` | Top XGBoost features by gain, with a key for decoding feature indices back to channel / band |
| `heldout_report.txt` | With `--eval_labels_dir`: pooled accuracy, per-participant accuracy (mean ± std and each participant), confusion matrix, classification report |
| `cache_features/` | Compressed per-participant feature matrices, keyed by feature set and checked against the source file |

Re-scoring new data without retraining:

```python
import joblib, numpy as np
from run_pipeline import featurize_subject

b = joblib.load("model.joblib")
X = np.load("X_eval_95.npy")
F = featurize_subject(X, b["feature_set"])
probs = np.mean([m["model"].predict_proba(m["scaler"].transform(F)) for m in b["fold_models"]], axis=0)
labels = b["label_encoder"].inverse_transform(probs.argmax(1))
```

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

Feature shapes, the feature cache, the PhysioNet label mapping and epoching, and an end-to-end run on synthetic data. A few seconds, no network, no `mne`. The same command runs in GitHub Actions on every push.

## Project layout

```
run_pipeline.py                 feature extraction, CV, training, prediction, held-out scoring (single file)
prepare_physionet.py            download + epoch PhysioNet EEGMMIDB into the layout above
scripts/reproduce_physionet.py  re-run the PhysioNet baseline and rewrite results/physionet/run.json
merge_training.py               merge multiple training folders
results/physionet/              public baseline: CV and held-out reports, feature importances, run.json
tests/                          pytest suite (run in CI)
requirements*.txt               core, PhysioNet (mne) and test dependencies
cv_confusion_matrix.txt         CV diagnostics from the hackathon submission (private data)
feature_importance_top20.txt    feature importances from an earlier bandpower-only hackathon run
AGENTS.md                       notes for AI coding agents (data contract, leakage rules, commands)
CITATION.cff                    citation metadata
```

## Roadmap / ideas for contributors

Pull requests welcome. Ideas that may move the number, each measurable with the PhysioNet reproduction:

- **Tangent-space projection** around the Riemannian mean (as in [pyRiemann](https://pyriemann.readthedocs.io/)) instead of a plain log-map.
- **Per-subject re-centering**: whiten each participant's covariances by their own mean covariance so all subjects share a reference point.
- **Band-pass filtering** (for example 8–30 Hz) before the covariance; the current baseline is deliberately raw.
- **Filter-bank CSP** as a third feature family, selectable through `--feature_set`.
- **Seed × fold ensembling** once features are settled.
- A `--fs` and `--nperseg` flag so no code edit is needed for other sampling rates.

If you open a PR that changes features or the model, re-run `python scripts/reproduce_physionet.py` and include the updated `results/physionet/` so results stay comparable.

## Citation

If this code is useful in your research, please cite it (GitHub's *Cite this repository* button uses `CITATION.cff`), and cite the dataset as described in [Citing the dataset](#citing-the-dataset):

```bibtex
@software{dastanov2026eegmi,
  author = {Dastanov, Elchibek},
  title  = {EEG Motor Imagery Classification with Log-Covariance Features and XGBoost},
  year   = {2026},
  url    = {https://github.com/elchibek5/Rice-NeuroWave-Hackathon-2026}
}
```

## Related

[PaperPilot-Gemini](https://github.com/elchibek5/PaperPilot-Gemini) — a research copilot built alongside this project: summarises PDF papers, extracts claims / method / results, drafts a replication plan and answers questions with citations.

## License

[MIT](LICENSE). Use it, fork it, ship it.

---

*Keywords: EEG, motor imagery, brain-computer interface, BCI, neurotechnology, Riemannian geometry, covariance features, log-Euclidean, XGBoost, scikit-learn, cross-subject, GroupKFold, PhysioNet, EEGMMIDB, Python, hackathon.*
