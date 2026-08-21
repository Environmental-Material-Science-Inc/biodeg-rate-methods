"""Censoring: the simple-recensoring convention (reference Appendix A)."""

import numpy as np

from biodeg_rates.censoring import recensor, pair_sign, tie_groups, CENSORED_FRACTION_WARN


def _series():
    # 6 events; two non-detects at RL 0.01 and 0.02, one detect below RL* (0.015)
    t = np.array([0, 1, 2, 3, 4, 5], float)
    value = np.array([5.0, 0.01, 0.015, 2.0, 0.02, 1.0])
    detect = np.array([True, False, True, True, False, True])
    rl = np.array([np.nan, 0.01, np.nan, np.nan, 0.02, np.nan])
    return t, value, detect, rl


def test_rl_star_and_censored_class():
    rec = recensor(*_series())
    assert rec.rl_star == 0.02                       # highest ND limit
    # censored = both non-detects + the detect (0.015) reported below RL*
    assert rec.n_censored == 3
    assert rec.censored.sum() == 3


def test_pair_sign_rules():
    rec = recensor(*_series())
    # find indices of two determinate detects (5.0 at t0, 2.0 at t3): later is smaller -> -1
    i = int(np.where(rec.t == 0)[0][0]); j = int(np.where(rec.t == 3)[0][0])
    assert pair_sign(i, j, rec) == -1
    # determinate (t0, value 5.0) vs censored (t1) -> later censored is below RL*, so -1
    a = int(np.where(rec.t == 0)[0][0]); b = int(np.where(rec.t == 1)[0][0])
    assert pair_sign(a, b, rec) == -1
    # two censored points -> tie (0)
    c1 = int(np.where(rec.t == 1)[0][0]); c2 = int(np.where(rec.t == 2)[0][0])
    assert pair_sign(c1, c2, rec) == 0


def test_tie_groups_include_censored_group():
    rec = recensor(*_series())
    groups = tie_groups(rec)
    assert 3 in groups                               # the censored class of size 3


def test_heavy_censoring_flag():
    t = np.arange(10.0)
    value = np.array([0.005] * 8 + [1.0, 2.0])
    detect = np.array([False] * 8 + [True, True])
    rl = np.array([0.005] * 8 + [np.nan, np.nan])
    rec = recensor(t, value, detect, rl)
    assert rec.censored_fraction > CENSORED_FRACTION_WARN
    assert rec.heavily_censored is True
