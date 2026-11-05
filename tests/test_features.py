import numpy as np

from run_pipeline import FEATS_PER_CHANNEL, HEMI_PAIRS, features_bandpower, features_covlog, featurize_subject


def _epoch(seed=0, channels=64, samples=656):
    return np.random.default_rng(seed).standard_normal((channels, samples)).astype(np.float32)


def test_bandpower_has_14_per_channel_plus_15_hemispheric():
    f = features_bandpower(_epoch())
    assert f.shape == (64 * 14 + 15,)
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
    assert featurize_subject(X, "bandpower").shape == (3, 911)
    assert featurize_subject(X, "covlog").shape == (3, 2080)
    assert featurize_subject(X, "combo").shape == (3, 911 + 2080)


# Issue #1: a per-channel z-score made mean/std/rms constant and every feature
# blind to amplitude, which erased the C3-C4 asymmetry that separates the hands.

def test_bandpower_keeps_channel_amplitude():
    x = _epoch(2)
    f1, f3 = features_bandpower(x), features_bandpower(3 * x)
    per_ch1 = f1[: 64 * FEATS_PER_CHANNEL].reshape(64, FEATS_PER_CHANNEL)
    per_ch3 = f3[: 64 * FEATS_PER_CHANNEL].reshape(64, FEATS_PER_CHANNEL)
    np.testing.assert_allclose(per_ch3[:, 1], 3 * per_ch1[:, 1], rtol=1e-5)  # std
    np.testing.assert_allclose(per_ch3[:, 3], 3 * per_ch1[:, 3], rtol=1e-5)  # rms
    # log bandpower moves by log(9) when the signal is 3x larger
    np.testing.assert_allclose(per_ch3[:, 4:9] - per_ch1[:, 4:9], np.log(9), atol=1e-4)


def test_bandpower_std_rms_vary_across_channels():
    x = _epoch(3) * np.linspace(1, 5, 64)[:, None]
    per_ch = features_bandpower(x)[: 64 * FEATS_PER_CHANNEL].reshape(64, FEATS_PER_CHANNEL)
    assert per_ch[:, 1].std() > 0.5 and per_ch[:, 3].std() > 0.5


def test_hemispheric_features_track_left_right_amplitude():
    x = _epoch(4).astype(np.float64)
    _, c3, _, c4 = HEMI_PAIRS[1]
    x[c3] *= 2.0  # C3 twice the amplitude of C4: 4x the power in every band
    hemi = features_bandpower(x)[64 * FEATS_PER_CHANNEL:].reshape(len(HEMI_PAIRS), 5)
    base = features_bandpower(_epoch(4))[64 * FEATS_PER_CHANNEL:].reshape(len(HEMI_PAIRS), 5)
    np.testing.assert_allclose(hemi[1] - base[1], np.log(4), atol=1e-4)
    np.testing.assert_allclose(hemi[[0, 2]], base[[0, 2]], atol=1e-5)
    # a gain shared by every channel (another subject's amplifier) cancels out
    np.testing.assert_allclose(features_bandpower(7 * x)[64 * FEATS_PER_CHANNEL:].reshape(3, 5), hemi, atol=1e-4)
