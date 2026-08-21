"""Module 1: Mann-Kendall + Theil-Sen point decay."""

import numpy as np

from biodeg_rates.contract import WellSeries
from biodeg_rates import method1_mann_kendall as m1


def _well(k_true=0.2, n=10, noise=0.05, c0=10.0, sign=-1, seed=0):
    """A well whose ln C declines at k_true (sign=-1 -> decreasing concentration)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    lnC = np.log(c0) + sign * k_true * t + rng.normal(0, noise, n)
    conc = np.exp(lnC)
    return WellSeries(well_id="W", easting=0.0, northing=0.0, t_years=t,
                      dates=np.array([np.datetime64("2015-01-01")] * n),
                      conc=conc, detect=np.ones(n, bool), rl=np.full(n, 0.005))


def test_recovers_decreasing_rate():
    e = m1.estimate_well(_well(k_true=0.2, noise=0.03))
    assert e.trend == "decreasing"
    assert abs(e.value_per_year - 0.2) < 0.05      # Theil-Sen recovers the slope
    assert e.ci_low <= e.value_per_year <= e.ci_high
    assert e.half_life_years > 0


def test_increasing_trend():
    e = m1.estimate_well(_well(k_true=0.2, sign=+1, noise=0.03))
    assert e.trend == "increasing"
    assert e.value_per_year < 0                    # negative point-decay = rising
    assert e.half_life_years == float("inf")


def test_flat_series_no_trend():
    e = m1.estimate_well(_well(k_true=0.0, noise=0.2, seed=1))
    assert e.trend == "no trend"


def test_too_few_events_na():
    e = m1.estimate_well(_well(n=3))
    assert e.confidence == "N/A"
    assert np.isnan(e.value_per_year)


def test_heavy_censoring_flagged():
    t = np.arange(12.0)
    conc = np.concatenate([np.full(9, 0.005), [2.0, 1.5, 1.0]])
    detect = np.array([False] * 9 + [True, True, True])
    rl = np.concatenate([np.full(9, 0.005), [np.nan, np.nan, np.nan]])
    w = WellSeries("W", 0, 0, t, np.array([np.datetime64("2015-01-01")] * 12), conc, detect, rl)
    e = m1.estimate_well(w)
    # either flagged in notes or routed to N/A for too-few determinate pairs; never high confidence
    assert e.confidence in ("low", "N/A")
    if e.confidence != "N/A":
        assert "censored" in e.notes.lower()


def test_site_summary_is_median_not_cross_method(synth_site):
    per_well, summary = m1.estimate_site(synth_site)
    assert summary.method == m1.METHOD_MANN_KENDALL
    # the synthetic plume declines at K_TRUE=0.15 everywhere; summary should be near it
    if summary.confidence != "N/A" and summary.trend == "decreasing":
        assert 0.05 < summary.value_per_year < 0.30
    assert len(per_well) == len(synth_site.wells)
