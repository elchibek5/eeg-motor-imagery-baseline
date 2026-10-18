"""Build the run_pipeline.py data layout from the public PhysioNet EEGMMIDB.

EEG Motor Movement/Imagery Database (Schalk et al. 2004), 109 subjects,
64 channels at 160 Hz. Only the motor *imagery* runs are used:

    runs 4, 8, 12   T1 = imagine left fist   -> left_hand
                    T2 = imagine right fist  -> right_hand
    runs 6, 10, 14  T1 = imagine both fists  -> both_hands
                    T2 = imagine both feet   -> both_feet

Each epoch is the 656 samples (4.1 s) starting at a T1/T2 annotation onset,
all 64 EEG channels, in microvolts, float32. No filtering, no re-referencing,
no baseline correction. T0 (rest) is not used. An epoch that would run past
the end of the recording is dropped.

Split by subject: 1-89 -> Training, 90-109 -> Evaluation, with evaluation
labels written to a separate Evaluation_labels folder so the pipeline never
sees them. Subjects recorded at a rate other than 160 Hz, or missing a run,
are skipped and listed in the summary.

    pip install -r requirements.txt -r requirements-physionet.txt
    python prepare_physionet.py

Re-running skips subjects whose output files already exist (use --force).
"""
import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FS = 160
N_CHANNELS = 64
N_SAMPLES = 656  # 4.1 s at 160 Hz, the same epoch length as the hackathon data

SUBJECTS = range(1, 110)
TRAIN_SUBJECTS = range(1, 90)    # 1..89
EVAL_SUBJECTS = range(90, 110)   # 90..109

LEFT_RIGHT_RUNS = (4, 8, 12)
HANDS_FEET_RUNS = (6, 10, 14)
RUNS = tuple(sorted(LEFT_RIGHT_RUNS + HANDS_FEET_RUNS))

# Both serve the same PhysioNet files; every download is checked against the
# SHA-256 registry that ships with MNE. The AWS copy is PhysioNet's own
# open-data mirror and is usually much faster than physionet.org.
MIRRORS = {
    "aws": "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/",
    "physionet": "https://physionet.org/files/eegmmidb/1.0.0/",
}


def label_for(run: int, code: str) -> str | None:
    """Class label for one annotation of one imagery run.

    Returns None for T0 (rest). Raises ValueError for a run that is not a
    motor imagery run or for an annotation code other than T0/T1/T2, so a
    surprise in the data stops the build instead of producing wrong labels.
    """
    if run in LEFT_RIGHT_RUNS:
        names = {"T1": "left_hand", "T2": "right_hand"}
    elif run in HANDS_FEET_RUNS:
        names = {"T1": "both_hands", "T2": "both_feet"}
    else:
        raise ValueError(f"run {run} is not a motor imagery run (expected one of {RUNS})")
    if code == "T0":
        return None
    if code not in names:
        raise ValueError(f"unexpected annotation {code!r} in run {run}")
    return names[code]


def epoch_run(data: np.ndarray, sfreq: float, onsets_s, codes, run: int):
    """Cut fixed-length epochs at every T1/T2 onset of one run.

    data: (64, n_times) in microvolts. Returns (X, y, n_dropped) with
    X of shape (n, 64, N_SAMPLES) float32 and y of label strings.
    """
    X, y, dropped = [], [], 0
    n_times = data.shape[1]
    for onset, code in zip(onsets_s, codes):
        label = label_for(run, code)
        if label is None:
            continue
        start = int(round(onset * sfreq))
        if start < 0 or start + N_SAMPLES > n_times:
            dropped += 1
            continue
        X.append(data[:, start:start + N_SAMPLES])
        y.append(label)
    X = np.stack(X).astype(np.float32) if X else np.empty((0, data.shape[0], N_SAMPLES), np.float32)
    return X, np.array(y, dtype=str), dropped


def split_of(subject: int) -> str:
    return "train" if subject in TRAIN_SUBJECTS else "eval"


def output_paths(out_dir: str, subject: int):
    if split_of(subject) == "train":
        d = os.path.join(out_dir, "Training")
        return os.path.join(d, f"X_train_{subject}.npy"), os.path.join(d, f"y_train_{subject}.npy")
    return (os.path.join(out_dir, "Evaluation", f"X_eval_{subject}.npy"),
            os.path.join(out_dir, "Evaluation_labels", f"y_eval_{subject}.npy"))


def _save_atomic(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


# ----------------------------
# Download
# ----------------------------

def _raw_base(raw_dir: str) -> str:
    # the folder layout mne.datasets.eegbci.load_data uses under its `path`
    return os.path.join(raw_dir, "MNE-eegbci-data", "files", "eegmmidb", "1.0.0")


def download(subjects, raw_dir: str, mirror: str, jobs: int):
    """Fetch the imagery runs; returns {subject: [edf paths] or None if a run is missing}."""
    from importlib.resources import files

    import pooch
    from mne.datasets import eegbci

    pooch.get_logger().setLevel("WARNING")
    base = _raw_base(raw_dir)
    fetcher = pooch.create(path=base, base_url=MIRRORS[mirror], registry=None, retry_if_failed=2)
    fetcher.load_registry(files("mne").joinpath("data", "eegbci_checksums.txt"))

    def one(subject):
        parts = [f"S{subject:03d}/S{subject:03d}R{run:02d}.edf" for run in RUNS]
        if any(p not in fetcher.registry for p in parts):
            return subject, None
        for p in parts:
            if not os.path.exists(os.path.join(base, p)):
                fetcher.fetch(p)  # verifies the SHA-256 from MNE's registry
        # every file is local now, so this only resolves paths (no download)
        paths = eegbci.load_data(subject, list(RUNS), path=raw_dir, update_path=False, verbose="ERROR")
        return subject, [str(p) for p in paths]

    out, done, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for subject, paths in ex.map(one, subjects):
            out[subject] = paths
            done += 1
            if done % 10 == 0 or done == len(subjects):
                print(f"  downloaded {done}/{len(subjects)} subjects ({time.time() - t0:.0f}s)", flush=True)
    return out


# ----------------------------
# Epoching
# ----------------------------

def epoch_subject(edf_paths):
    """Returns (X, y, n_dropped, None), or (None, None, 0, reason) when the subject must be skipped."""
    import mne

    Xs, ys, n_dropped = [], [], 0
    for run, path in zip(RUNS, edf_paths):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        if sfreq != FS:
            return None, None, 0, f"sampling rate {sfreq:g} Hz in run {run}"
        picks = mne.pick_types(raw.info, eeg=True)
        if len(picks) != N_CHANNELS:
            return None, None, 0, f"{len(picks)} EEG channels in run {run}"
        if raw.first_samp != 0:
            return None, None, 0, f"unexpected first sample {raw.first_samp} in run {run}"
        data = raw.get_data(picks=picks, units="uV")  # MNE stores volts
        X, y, dropped = epoch_run(data, sfreq, raw.annotations.onset, raw.annotations.description, run)
        Xs.append(X)
        ys.append(y)
        n_dropped += dropped
    return np.concatenate(Xs), np.concatenate(ys), n_dropped, None


# ----------------------------
# Main
# ----------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw_dir", default=os.path.join("data", "raw"), help="Where the EDF files are downloaded")
    ap.add_argument("--out_dir", default=os.path.join("data", "physionet"), help="Where the .npy layout is written")
    ap.add_argument("--mirror", default="aws", choices=sorted(MIRRORS), help="Download source (same files)")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel downloads")
    ap.add_argument("--force", action="store_true", help="Rebuild outputs that already exist")
    args = ap.parse_args(argv)

    import mne
    mne.set_log_level("ERROR")

    t0 = time.time()
    todo = [s for s in SUBJECTS if args.force or not all(map(os.path.exists, output_paths(args.out_dir, s)))]
    print(f"{len(SUBJECTS) - len(todo)} subjects already prepared, {len(todo)} to build")

    paths = {}
    if todo:
        print(f"Downloading imagery runs {', '.join(map(str, RUNS))} from {MIRRORS[args.mirror]} into {args.raw_dir}")
        paths = download(todo, args.raw_dir, args.mirror, args.jobs)

    skipped, dropped = {}, {}
    counts = {"train": Counter(), "eval": Counter()}
    n_subjects = Counter()
    for s in SUBJECTS:
        x_path, y_path = output_paths(args.out_dir, s)
        if s in paths:
            if paths[s] is None:
                skipped[s] = "missing imagery run in the dataset"
                continue
            X, y, n_dropped, reason = epoch_subject(paths[s])
            if reason:
                skipped[s] = reason
                continue
            if n_dropped:
                dropped[s] = n_dropped
            _save_atomic(x_path, X)
            _save_atomic(y_path, y)
        else:
            y = np.load(y_path, allow_pickle=False)
        n_subjects[split_of(s)] += 1
        counts[split_of(s)].update(y.tolist())

    print(f"\nSummary ({time.time() - t0:.0f}s)")
    for split, folder in (("train", "Training"), ("eval", "Evaluation")):
        c = counts[split]
        per_class = ", ".join(f"{k} {c[k]}" for k in sorted(c))
        print(f"  {folder}: {n_subjects[split]} subjects, {sum(c.values())} epochs ({per_class})")
    if dropped:
        detail = ", ".join(f"subject {s}: {n}" for s, n in dropped.items())
        print(f"  Dropped {sum(dropped.values())} epochs that ran past the end of a recording ({detail})")
    if skipped:
        print(f"  Skipped {len(skipped)} subjects:")
        for s, reason in skipped.items():
            print(f"    {s}: {reason}")
    print(f"  Output: {args.out_dir}/Training, {args.out_dir}/Evaluation, {args.out_dir}/Evaluation_labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
