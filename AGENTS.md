# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

A small cross-subject EEG motor imagery classifier. `run_pipeline.py` (one file) turns each 64-channel epoch into features (bandpower statistics or log-covariance), runs participant-grouped 5-fold cross-validation with XGBoost, picks the better feature set by CV, and writes fold-ensemble predictions for every evaluation participant. `prepare_physionet.py` builds the same input layout from the public PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) so the published number can be checked by anyone.

## Data layout contract

```
Training/
  X_train_<id>.npy        (N, channels, samples) float, one file per participant
  y_train_<id>.npy        (N,) labels: left_hand, right_hand, both_hands, both_feet
Evaluation/
  X_eval_<id>.npy         (N, channels, samples); X_test_<id>.npy is also accepted
Evaluation_labels/        separate folder, optional
  y_eval_<id>.npy         (N,) labels, read only by --eval_labels_dir
predictions/
  y_pred_<id>.npy         (N, 1) object array of label strings
```

`<id>` is an unpadded integer. Sampling rate is `FS = 160` Hz at the top of `run_pipeline.py`; the hackathon and PhysioNet data are both 64 channels x 656 samples (4.1 s). PhysioNet epochs are raw microvolts: no filtering, no re-referencing, no baseline correction.

Data is never committed: `*.npy`, `data/`, models and caches are gitignored. Stage files by explicit path.

## Leakage rules (never break these)

- Split by participant only (`GroupKFold`, group = participant id). Never split, shuffle or stratify by epoch: the same person in train and validation lets the model recognise the person instead of the movement and inflates accuracy.
- Never read evaluation labels during featurization, scaling, training or feature-set selection. They are read only after predictions are written, only from `--eval_labels_dir`, and never from `--eval_dir` (the pipeline refuses the same folder).
- `--feature_set auto` chooses by CV on training participants only.
- Scalers are fit inside each fold.
- The PhysioNet split is fixed: subjects 1-89 train, 90-109 held out. Do not tune anything on the held-out number; it is reported once per change.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                      # a few seconds; no network, no mne
```

PhysioNet reproduction (1.7 GB download, then about 30 minutes on an 11-core laptop):

```bash
pip install -r requirements-physionet.txt
python scripts/reproduce_physionet.py      # rebuilds epochs + features, runs the two README commands, writes run.json
```

The two commands it runs are the ones in the README's "Reproduce on public data" section.

## Where results go

- `results/physionet/`: `cv_confusion_matrix.txt`, `feature_importance_top20.txt`, `heldout_report.txt`, `run.json`. This is the published public baseline.
- The README "Results" table quotes `results/physionet/run.json`.
- `cv_confusion_matrix.txt` and `feature_importance_top20.txt` at the repo root are artifacts of the original hackathon runs on private data. Do not overwrite them.

**If you change features or the model, re-run the PhysioNet reproduction and update `results/physionet/` and the README results table in the same commit.**

## Known traps

- The feature cache is keyed by the source file (path, size, mtime), not by the feature code. After editing a feature function, delete the cache folder (`cache_features/` or `data/physionet/cache/`). `scripts/reproduce_physionet.py` does this itself.
- Running `run_pipeline.py` from the repo root without `--report_dir` writes `cv_confusion_matrix.txt` and `feature_importance_top20.txt` into the cwd and overwrites the committed hackathon files.
- MNE returns volts; `prepare_physionet.py` converts to microvolts. The log-covariance code adds `EPS = 1e-10` to the diagonal, which is negligible in microvolts squared but not in volts squared.
- In EEGMMIDB, T1/T2 mean different things per run (runs 4/8/12: left/right fist; runs 6/10/14: both fists/both feet). The mapping lives in `label_for()` and is tested.
- physionet.org serves these files slowly; `prepare_physionet.py` defaults to PhysioNet's AWS open-data copy (`--mirror aws`). Each file is checked against MNE's SHA-256 registry when it is downloaded; EDF files already in `--raw_dir` are reused without re-hashing.
- `prepare_physionet.py` skips subjects whose output files exist. After changing epoching, labels or units, run it with `--force` (the reproduction script always does).
