import os

import numpy as np

import run_pipeline
from run_pipeline import featurize_subject, load_training


def _write_participant(folder, pid, seed):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((3, 8, 656)).astype(np.float32)
    np.save(os.path.join(folder, f"X_train_{pid}.npy"), X)
    np.save(os.path.join(folder, f"y_train_{pid}.npy"), np.array(["a", "b", "a"]))
    return X


def test_same_pid_from_another_dataset_is_recomputed(tmp_path):
    cache = str(tmp_path / "cache")
    XA = _write_participant(str(tmp_path / "A"), 1, seed=1)
    XB = _write_participant(str(tmp_path / "B"), 1, seed=2)

    FA, _, _ = load_training(str(tmp_path / "A"), "covlog", cache)
    FB, _, _ = load_training(str(tmp_path / "B"), "covlog", cache)

    np.testing.assert_allclose(FA, featurize_subject(XA, "covlog"))
    np.testing.assert_allclose(FB, featurize_subject(XB, "covlog"))
    assert not np.allclose(FA, FB)


def test_rewritten_source_file_is_recomputed(tmp_path):
    cache = str(tmp_path / "cache")
    folder = str(tmp_path / "A")
    _write_participant(folder, 1, seed=1)
    load_training(folder, "covlog", cache)

    X2 = _write_participant(folder, 1, seed=3)  # same path, same size, new content
    x_path = os.path.join(folder, "X_train_1.npy")
    st = os.stat(x_path)
    os.utime(x_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    F2, _, _ = load_training(folder, "covlog", cache)
    np.testing.assert_allclose(F2, featurize_subject(X2, "covlog"))


def test_unchanged_source_reuses_cache(tmp_path, monkeypatch):
    cache = str(tmp_path / "cache")
    folder = str(tmp_path / "A")
    _write_participant(folder, 1, seed=1)
    F1, _, _ = load_training(folder, "covlog", cache)

    def fail(*args, **kwargs):
        raise AssertionError("features were recomputed although the cache matches")

    monkeypatch.setattr(run_pipeline, "featurize_subject", fail)
    F2, _, _ = load_training(folder, "covlog", cache)
    np.testing.assert_array_equal(F1, F2)


def test_cache_without_source_record_is_recomputed(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    folder = str(tmp_path / "A")
    X = _write_participant(folder, 1, seed=1)
    # a cache file in the old format: features only, here deliberately wrong
    np.savez_compressed(cache / "feats_covlog_1.npz", X=np.zeros((3, 36), np.float32))

    F, _, _ = load_training(folder, "covlog", str(cache))
    np.testing.assert_allclose(F, featurize_subject(X, "covlog"))
