"""Module 3: spatio-temporal P-spline spline-centre concentration decay (unit tests).

The estimand is the concentration decay at the plume centre read off the smoothed surface.
``mass_log_series`` (the whole-plume integral) is retained for the validation harness, so the
surface-recovery test below still exercises it. The mgcv cross-validation is an evidence
harness, not a unit test; it lives in research/biodeg_bakeoffs/validate_mgcv.py.
"""

import numpy as np
import pandas as pd
import pytest

from biodeg_rates import method3_spline as m3
from biodeg_rates import dataio


def _separable_plume(k_true=0.2, n_wells=16, n_times=8, noise=0.2, seed=5):
    """C(s,t) = gaussian(s) * exp(-k_true t); both the local centre decay and the whole-plume
    mass-loss rate equal k_true (the plume is separable)."""
    rng = np.random.default_rng(seed)
    E = rng.uniform(0, 200, n_wells); N = rng.uniform(0, 200, n_wells)
    e0, n0 = 100.0, 100.0
    times = np.linspace(0, 9, n_times)
    rows = []
    for e, n in zip(E, N):
        amp = 50.0 * np.exp(-((e - e0) ** 2 + (n - n0) ** 2) / (2 * 50.0 ** 2))
        for t in times:
            rows.append((e, n, t, np.log(max(amp, 1e-9)) - k_true * t + rng.normal(0, noise)))
    df = pd.DataFrame(rows, columns=["e", "n", "t", "y"])
    return df, times


def _kM_from_fit(fit, E, N, times, grid_n=30):
    grid = m3._hull(E, N)[0]
    gx = np.linspace(E.min(), E.max(), grid_n); gy = np.linspace(N.min(), N.max(), grid_n)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    gpts = pts[grid.contains_points(pts)]
    lnM = m3.mass_log_series(fit, gpts, times)
    return -float(np.polyfit(times, lnM, 1)[0])


@pytest.mark.parametrize("k_true", [0.05, 0.2, 0.4])
def test_fit_surface_recovers_kM(k_true):
    df, times = _separable_plume(k_true=k_true)
    fit = m3.fit_surface(df.e.values, df.n.values, df.t.values, df.y.values)
    kM = _kM_from_fit(fit, df.e.values, df.n.values, times)
    assert abs(kM - k_true) < 0.05


def test_concentration_decay_at_recovers_k():
    df, times = _separable_plume(k_true=0.3, noise=0.1)
    fit = m3.fit_surface(df.e.values, df.n.values, df.t.values, df.y.values)
    # the smoothed surface's temporal slope at the plume centre recovers the true rate
    k = m3.concentration_decay_at(fit, {"centre": (df.e.mean(), df.n.mean())}, times)["centre"]
    assert abs(k - 0.3) < 0.06


def test_predict_lnC_tracks_observations():
    df, _ = _separable_plume(k_true=0.2, noise=0.1)
    fit = m3.fit_surface(df.e.values, df.n.values, df.t.values, df.y.values)
    pred = fit.predict_lnC(df.e.values, df.n.values, df.t.values)
    assert np.corrcoef(pred, df.y.values)[0, 1] > 0.9


def test_estimate_site_on_synth(synth_site):
    e = m3.estimate_site(synth_site)
    # synthetic plume declines at K_TRUE=0.15; the spline-centre decay should be positive and
    # in the right range (the same apparent-attenuation estimand as Method 1, smoothed)
    assert np.isfinite(e.value_per_year)
    assert e.trend == "decreasing"
    assert 0.03 < e.value_per_year < 0.4
    from biodeg_rates.contract import Estimand
    assert e.estimand is Estimand.SPLINE_CENTRE_DECAY
    assert e.removes_dilution is False


def test_estimate_site_reports_centre_rate_in_per_day(synth_site):
    # the handoff feeds MODFLOW in 1/day; the centre rate must be carried in days, converted with
    # the 365 day/year convention, consistent with the 1/year value (user requirement)
    e = m3.estimate_site(synth_site)
    d = e.diagnostics
    assert "centre_rate_per_day" in d
    assert abs(d["centre_rate_per_day"] - e.value_per_year / 365.0) < 1e-7
    assert abs(d["centre_rate_per_year"] - e.value_per_year) < 1e-4


def test_insufficient_coverage_na():
    # 3 wells, 3 times -> below thresholds
    df, _ = _separable_plume(n_wells=3, n_times=3)
    from biodeg_rates.contract import SiteObservations
    frame = pd.DataFrame(dict(well_id=[f"w{i%3}" for i in range(len(df))],
                              easting=df.e, northing=df.n, date=pd.Timestamp("2020-01-01"),
                              t_years=df.t, conc=np.exp(df.y), detect=True, rl=np.nan))
    site = SiteObservations("X", "benzene", 32613, 0.005, np.datetime64("2020-01-01"),
                            frame, [], pd.DataFrame(), (100.0, 100.0), 50.0)
    e = m3.estimate_site(site)
    assert e.confidence == "N/A"
