import os

import numpy as np
import pytest

from prepare_physionet import N_SAMPLES, epoch_run, label_for, output_paths


@pytest.mark.parametrize(
    "run, code, label",
    [
        (4, "T1", "left_hand"),
        (8, "T2", "right_hand"),
        (6, "T1", "both_hands"),
        (14, "T2", "both_feet"),
    ],
)
def test_imagery_labels(run, code, label):
    assert label_for(run, code) == label


def test_every_imagery_run_maps_t1_and_t2():
    for run in (4, 8, 12):
        assert (label_for(run, "T1"), label_for(run, "T2")) == ("left_hand", "right_hand")
    for run in (6, 10, 14):
        assert (label_for(run, "T1"), label_for(run, "T2")) == ("both_hands", "both_feet")


def test_rest_is_not_an_epoch():
    assert label_for(4, "T0") is None


@pytest.mark.parametrize("run", [1, 2, 3, 5, 7, 9, 11, 13, 15])
def test_non_imagery_run_is_rejected(run):
    with pytest.raises(ValueError):
        label_for(run, "T1")


def test_unknown_annotation_is_rejected():
    with pytest.raises(ValueError):
        label_for(4, "T3")


def test_epoch_run_cuts_656_samples_from_onset_and_drops_overruns():
    fs = 160.0
    n_times = 3000
    data = np.tile(np.arange(n_times, dtype=np.float64), (64, 1))  # value == sample index
    onsets = [0.0, 1.0, 2.0, 2000 / fs, (n_times - N_SAMPLES) / fs, (n_times - N_SAMPLES + 1) / fs]
    codes = ["T1", "T0", "T2", "T1", "T2", "T1"]

    X, y, dropped = epoch_run(data, fs, onsets, codes, run=6)

    assert X.shape == (4, 64, N_SAMPLES) and X.dtype == np.float32
    assert y.tolist() == ["both_hands", "both_feet", "both_hands", "both_feet"]
    assert [int(x[0, 0]) for x in X] == [0, 320, 2000, n_times - N_SAMPLES]
    assert int(X[-1, 0, -1]) == n_times - 1
    assert dropped == 1


def test_subjects_split_by_id(tmp_path):
    out = str(tmp_path)
    j = os.path.join
    assert output_paths(out, 1) == (j(out, "Training", "X_train_1.npy"), j(out, "Training", "y_train_1.npy"))
    assert output_paths(out, 89)[0] == j(out, "Training", "X_train_89.npy")
    assert output_paths(out, 90) == (j(out, "Evaluation", "X_eval_90.npy"), j(out, "Evaluation_labels", "y_eval_90.npy"))
    assert output_paths(out, 109)[1] == j(out, "Evaluation_labels", "y_eval_109.npy")
