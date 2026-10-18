"""Re-run the PhysioNet EEGMMIDB baseline and record it in results/physionet/.

Runs the two commands from the README (prepare_physionet.py, then
run_pipeline.py), rebuilding the epochs and the feature cache from the raw
files so nothing made by older code is reused, then writes run.json next to
the three reports: date, code commit, library versions, machine, wall time, CV and
held-out numbers, and the subjects and epochs per split.

    python scripts/reproduce_physionet.py [--raw_dir data/raw]

Use it after any change to the features or the model, and commit
results/physionet/ together with the README results table.
"""
import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from importlib import metadata

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join("data", "physionet")
REPORT_DIR = os.path.join("results", "physionet")
CACHE_DIR = os.path.join(DATA, "cache")

PIPELINE_ARGS = [
    "run_pipeline.py",
    "--train_dir", os.path.join(DATA, "Training"),
    "--eval_dir", os.path.join(DATA, "Evaluation"),
    "--eval_labels_dir", os.path.join(DATA, "Evaluation_labels"),
    "--out_dir", os.path.join(DATA, "predictions"),
    "--model_out", os.path.join(DATA, "model.joblib"),
    "--cache_dir", CACHE_DIR,
    "--report_dir", REPORT_DIR,
    "--feature_set", "auto",
]


def run_tee(cmd):
    """Run a command in the repo root, streaming its output; returns the output."""
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen([sys.executable, *cmd], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    if proc.wait() != 0:
        sys.exit(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return "".join(lines)


def git(*args):
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {res.stderr.strip()} (run.json needs a git checkout)")
    return res.stdout.strip()


def cpu_name():
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        if out.strip():
            return out.strip()
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def split_counts(folder, prefix):
    per_class, subjects = Counter(), 0
    for name in sorted(os.listdir(os.path.join(ROOT, folder))):
        if re.fullmatch(prefix + r"_\d+\.npy", name):
            y = np.load(os.path.join(ROOT, folder, name), allow_pickle=False)
            per_class.update(y.tolist())
            subjects += 1
    return {"subjects": subjects, "epochs": sum(per_class.values()), "per_class": dict(sorted(per_class.items()))}


def parse_cv(log):
    candidates, current = {}, None
    for line in log.splitlines():
        if m := re.match(r"=== Trying feature_set: (\w+) ===", line):
            current = m.group(1)
            candidates[current] = {"folds": []}
        elif current and (m := re.match(r"Fold \d+ accuracy: ([0-9.]+)", line)):
            candidates[current]["folds"].append(float(m.group(1)))
        elif current and (m := re.match(r"Mean CV accuracy: ([0-9.]+) ± ([0-9.]+)", line)):
            candidates[current].update(mean=float(m.group(1)), std=float(m.group(2)))
    selected = re.search(r"=== Selected feature_set: (\w+)", log).group(1)
    return candidates, selected


def parse_heldout(path):
    text = open(path).read()
    acc = float(re.search(r"Accuracy \(all epochs pooled\): ([0-9.]+)", text).group(1))
    m = re.search(r"Per-participant accuracy: mean ([0-9.]+) ± ([0-9.]+)", text)
    return {"accuracy": acc, "per_subject_mean": float(m.group(1)), "per_subject_std": float(m.group(2))}


def parse_skipped(log):
    block = log.split("Skipped", 1)[1] if "Skipped" in log else ""
    return {int(s): reason for s, reason in re.findall(r"^\s+(\d+): (.+)$", block, flags=re.M)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw_dir", default=os.path.join("data", "raw"))
    args = ap.parse_args()

    git("rev-parse", "HEAD")  # fail before the long run, not after it

    # --force: epochs are rebuilt from the raw files, and the feature cache is
    # keyed by source file rather than by feature code, so both start clean and
    # the recorded number always comes from the current code
    prepare_log = run_tee(["prepare_physionet.py", "--raw_dir", args.raw_dir, "--force"])
    shutil.rmtree(os.path.join(ROOT, CACHE_DIR), ignore_errors=True)
    t0 = time.time()
    pipeline_log = run_tee(PIPELINE_ARGS)
    wall = time.time() - t0

    candidates, selected = parse_cv(pipeline_log)
    model = joblib.load(os.path.join(ROOT, DATA, "model.joblib"))["full_model"].get_params()
    code_files = ["run_pipeline.py", "prepare_physionet.py", "requirements.txt", "requirements-physionet.txt"]
    record = {
        "date": datetime.date.today().isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "code_modified_since_commit": bool(git("status", "--porcelain", "--", *code_files)),
        # --raw_dir only moves the download folder, so it is left out (and
        # keeps local paths out of the committed record)
        "commands": [
            "python prepare_physionet.py",
            "python " + " ".join(PIPELINE_ARGS),
        ],
        "python": platform.python_version(),
        "versions": {d: version(d) for d in ("numpy", "scipy", "scikit-learn", "xgboost", "joblib", "mne")},
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "wall_time_s": round(wall, 1),
        "dataset": {
            "name": "PhysioNet EEG Motor Movement/Imagery Database 1.0.0",
            "runs": [4, 6, 8, 10, 12, 14],
            "epoch": "656 samples (4.1 s) from each T1/T2 onset, 64 EEG channels, microvolts, "
                     "no filtering, no re-referencing",
            "train": {"subject_ids": "1-89", **split_counts(os.path.join(DATA, "Training"), "y_train")},
            "heldout": {"subject_ids": "90-109", **split_counts(os.path.join(DATA, "Evaluation_labels"), "y_eval")},
            "skipped_subjects": parse_skipped(prepare_log),
        },
        "model": {"n_estimators": model["n_estimators"], "seed": model["random_state"],
                  "feature_set_requested": "auto"},
        "feature_set_candidates": candidates,
        "selected_feature_set": selected,
        "cv": {"protocol": "GroupKFold(5) by subject over the training subjects", **candidates[selected]},
        "heldout": {"protocol": "fold-ensemble trained on subjects 1-89, scored on subjects 90-109",
                    **parse_heldout(os.path.join(ROOT, REPORT_DIR, "heldout_report.txt"))},
    }
    out = os.path.join(ROOT, REPORT_DIR, "run.json")
    with open(out, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nSaved {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
