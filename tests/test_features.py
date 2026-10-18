import numpy as np

from run_pipeline import features_bandpower, features_covlog, featurize_subject


def _epoch(seed=0, channels=64, samples=656):
    return np.random.default_rng(seed).standard_normal((channels, samples)).astype(np.float32)


def test_bandpower_has_14_features_per_channel():
    f = features_bandpower(_epoch())
    assert f.shape == (896,)
    assert np.all(np.isfinite(f))


def test_covlog_is_upper_triangle_of_64x64():
    f = features_covlog(_epoch())
    assert f.shape == (2080,)
    assert np.all(np.isfinite(f))


def test_covlog_is_finite_for_rank_deficient_epoch():
    rng = np.random.default_rng(1)
    # 64 channels that are mixtures of only 4 sources: covariance rank 4
    epoch = rng.standard_normal((64, 4)) @ rng.standard_normal((4, 656))
    assert np.linalg.matrix_rank(np.cov(epoch)) == 4
    assert np.all(np.isfinite(features_covlog(epoch)))
    # a flat (all-zero) epoch is the extreme case
    assert np.all(np.isfinite(features_covlog(np.zeros((64, 656)))))


def test_featurize_subject_stacks_epochs():
    X = np.stack([_epoch(i) for i in range(3)])
    assert featurize_subject(X, "bandpower").shape == (3, 896)
    assert featurize_subject(X, "covlog").shape == (3, 2080)
    assert featurize_subject(X, "combo").shape == (3, 896 + 2080)
