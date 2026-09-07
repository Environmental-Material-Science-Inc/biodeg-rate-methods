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


# ----------------------------------------------------------------- k_mass (plume mass decay)

@pytest.mark.parametrize("k_true", [0.05, 0.15, 0.40])
def test_estimate_site_mass_tracks_the_true_rate(k_true):
    """On a separable plume, ln M(t) is exactly linear with slope -k_true, so k_mass must track
    the truth. It does so with a small POSITIVE offset (see the bias test below), hence the
    asymmetric window: this proves the estimator follows the rate rather than inventing one."""
    df, times = _separable_plume(k_true=k_true)
    fit = m3.fit_surface(df.e.values, df.n.values, df.t.values, df.y.values)
    gpts, _ = m3.mass_grid(df.e.values, df.n.values)
    kM = -m3._theil_sen_slope(times, m3.mass_log_series(fit, gpts, times))
    assert k_true - 0.005 < kM < k_true + 0.02


def test_mass_estimator_carries_a_known_positive_bias():
    """PINS a defect rather than asserting correctness. On separable synthetic plumes the mass
    rate runs high by a near-constant additive offset, independent of the true rate, which the
    credible interval does not cover. If this test starts failing the bias has CHANGED, and the
    documented offset in the notes and diagnostics must be re-measured."""
    errs = []
    for k_true in (0.05, 0.15, 0.40):
        df, times = _separable_plume(k_true=k_true)
        fit = m3.fit_surface(df.e.values, df.n.values, df.t.values, df.y.values)
        gpts, _ = m3.mass_grid(df.e.values, df.n.values)
        errs.append(-m3._theil_sen_slope(times, m3.mass_log_series(fit, gpts, times)) - k_true)
    assert all(e > 0 for e in errs), f"bias is no longer positive: {errs}"
    assert max(errs) - min(errs) < 0.005, f"bias is no longer near-constant: {errs}"
    assert 0.004 < float(np.mean(errs)) < 0.02, f"bias magnitude moved: {errs}"


def test_mass_log_series_survives_a_ballooned_surface():
    """log-sum-exp guard: a surface large enough to overflow exp() must still give a finite
    ln M(t). Proved by inflating the fitted coefficients until plain exp() would overflow."""
    df, times = _separable_plume(k_true=0.2)
    fit = m3.fit_surface(df.e.values, df.n.values, df.t.values, df.y.values)
    gpts, _ = m3.mass_grid(df.e.values, df.n.values)
    huge = fit.theta * 400.0
    with np.errstate(over="ignore"):        # the overflow here is the point of the test
        naive = np.exp(fit.spatial_basis(gpts[:, 0], gpts[:, 1])
                       @ (huge.reshape(fit.K1 * fit.K2, fit.K3) @ fit.temporal_basis(times)[0]))
    assert not np.all(np.isfinite(naive)), "fixture no longer overflows; strengthen it"
    lnM = m3.mass_log_series(fit, gpts, times, theta_vec=huge)
    assert np.all(np.isfinite(lnM))


def test_mass_estimand_is_distinct_from_centre(synth_site):
    from biodeg_rates.contract import Estimand
    centre, mass = m3.estimate_site_both(synth_site)
    assert centre.estimand is Estimand.SPLINE_CENTRE_DECAY
    assert mass.estimand is Estimand.PLUME_MASS_DECAY
    assert centre.estimand != mass.estimand
    assert mass.removes_dilution is False        # boundary export and source dissolution remain
    assert mass.method != centre.method


def test_estimate_site_both_matches_the_separate_calls(synth_site):
    """One fit, two estimands: the shared path must not change either number."""
    centre, mass = m3.estimate_site_both(synth_site)
    assert abs(centre.value_per_year - m3.estimate_site(synth_site).value_per_year) < 1e-9
    assert abs(mass.value_per_year - m3.estimate_site_mass(synth_site).value_per_year) < 1e-9


def test_mass_estimate_reports_per_day_and_the_support_sweep(synth_site):
    e = m3.estimate_site_mass(synth_site)
    d = e.diagnostics
    assert abs(d["mass_rate_per_day"] - e.value_per_year / 365.0) < 1e-7
    assert d["n_grid_supported"] <= d["n_grid_hull"]
    sweep = d["support_radius_sweep"]
    assert len(sweep) >= 2                       # the boundary-sensitivity evidence must be there
    assert d["sweep_spread_per_year"] is not None
    assert "known bias" in e.notes.lower()


def test_mass_estimate_na_on_insufficient_coverage():
    df, _ = _separable_plume(n_wells=3, n_times=3)
    from biodeg_rates.contract import SiteObservations
    frame = pd.DataFrame(dict(well_id=[f"w{i%3}" for i in range(len(df))],
                              easting=df.e, northing=df.n, date=pd.Timestamp("2020-01-01"),
                              t_years=df.t, conc=np.exp(df.y), detect=True, rl=np.nan))
    site = SiteObservations("X", "benzene", 32613, 0.005, np.datetime64("2020-01-01"),
                            frame, [], pd.DataFrame(), (100.0, 100.0), 50.0)
    assert m3.estimate_site_mass(site).confidence == "N/A"
