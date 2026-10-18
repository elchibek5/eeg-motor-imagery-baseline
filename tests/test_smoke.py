"""End-to-end run of run_pipeline.py on tiny synthetic data (no network, no mne)."""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["left_hand", "right_hand", "both_hands", "both_feet"]
EPOCHS, CHANNELS, SAMPLES = 20, 8, 656


def _make_dataset(root, seed=0):
    """6 participants: 1-4 train, 5-6 eval. Class k boosts the variance of channel k."""
    rng = np.random.default_rng(seed)
    dirs = {name: os.path.join(root, name) for name in ("Training", "Evaluation", "Evaluation_labels")}
    for d in dirs.values():
        os.makedirs(d)
    for pid in range(1, 7):
        y = np.array([CLASSES[i % 4] for i in rng.permutation(EPOCHS)])
        X = rng.standard_normal((EPOCHS, CHANNELS, SAMPLES)).astype(np.float32)
        for i, label in enumerate(y):
            X[i, CLASSES.index(label)] *= 3.0
        if pid <= 4:
            np.save(os.path.join(dirs["Training"], f"X_train_{pid}.npy"), X)
            np.save(os.path.join(dirs["Training"], f"y_train_{pid}.npy"), y)
        else:
            np.save(os.path.join(dirs["Evaluation"], f"X_eval_{pid}.npy"), X)
            np.save(os.path.join(dirs["Evaluation_labels"], f"y_eval_{pid}.npy"), y)
    return dirs


def test_pipeline_end_to_end(tmp_path):
    dirs = _make_dataset(str(tmp_path / "data"))
    out_dir = tmp_path / "predictions"
    report_dir = tmp_path / "reports"
    cmd = [
        sys.executable, os.path.join(ROOT, "run_pipeline.py"),
        "--train_dir", dirs["Training"],
        "--eval_dir", dirs["Evaluation"],
        "--eval_labels_dir", dirs["Evaluation_labels"],
        "--out_dir", str(out_dir),
        "--model_out", str(tmp_path / "model.joblib"),
        "--cache_dir", str(tmp_path / "cache"),
        "--report_dir", str(report_dir),
        "--feature_set", "covlog",
        "--n_estimators", "20",
    ]
    res = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr

    for pid in (5, 6):
        pred = np.load(out_dir / f"y_pred_{pid}.npy", allow_pickle=True)
        assert pred.shape == (EPOCHS, 1)
        assert set(pred.ravel()) <= set(CLASSES)

    for name in ("cv_confusion_matrix.txt", "feature_importance_top20.txt", "heldout_report.txt"):
        assert (report_dir / name).exists(), name
    assert not (tmp_path / "cv_confusion_matrix.txt").exists()  # --report_dir respected

    report = (report_dir / "heldout_report.txt").read_text()
    assert "Participants: 2  Epochs: 40" in report
    acc = float(re.search(r"Accuracy \(all epochs pooled\): ([0-9.]+)", report).group(1))
    assert acc > 0.5  # the class signal is strong; chance is 0.25


def test_eval_labels_next_to_eval_inputs_are_refused(tmp_path):
    dirs = _make_dataset(str(tmp_path / "data"))
    cmd = [
        sys.executable, os.path.join(ROOT, "run_pipeline.py"),
        "--train_dir", dirs["Training"],
        "--eval_dir", dirs["Evaluation"],
        "--eval_labels_dir", dirs["Evaluation"],
    ]
    res = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert res.returncode != 0
    assert "separate folder" in res.stderr
