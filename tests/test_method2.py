"""Module 2: Domenico-normalized flowpath lambda."""

import math

import numpy as np
import pandas as pd

from biodeg_rates.contract import SiteObservations
from biodeg_rates import method2_domenico as m2


def test_lambda_inversion_roundtrip():
    """lambda = -v_c m (1 - alpha_x m): invert the forward Domenico slope and recover lambda."""
    v_c, alpha_x = 0.4, 5.0
    for lam in (8.2e-4, 2e-3, 5e-3):           # per-day rates
        m = (1.0 / (2 * alpha_x)) * (1 - math.sqrt(1 + 4 * lam * alpha_x / v_c))
        assert m < 0
        lam_back = m2._lambda_from_m(m, v_c, alpha_x)
        assert math.isclose(lam_back, lam, rel_tol=1e-9)


def _domenico_site(slope_m=-0.05, c0=20.0, epsg=32613):
    """Centerline plume: C(x) = c0 exp(m x), wells along the flow axis (east), heads sloping
    east so the inferred flow is +E. Source at the hottest (near) well."""
    xs = np.array([4, 8, 12, 18, 24, 30, 38, 46, 55], float)
    ys = np.array([1, -1, 2, 0, -2, 1, -1, 3, -2], float)     # near centerline
    base_e, base_n = 500000.0, 5800000.0
    rows, hrows = [], []
    date = pd.Timestamp("2020-01-01")
    for i, (x, y) in enumerate(zip(xs, ys)):
        e, n = base_e + x, base_n + y
        c = c0 * math.exp(slope_m * x)
        rows.append((f"W{i}", e, n, date, 0.0, c, True, np.nan))
        hrows.append((f"W{i}", e, n, date, 100.0 - 0.01 * x))   # head drops to east -> flow east
    frame = pd.DataFrame(rows, columns=["well_id", "easting", "northing", "date",
                                        "t_years", "conc", "detect", "rl"])
    heads = pd.DataFrame(hrows, columns=["well_id", "easting", "northing", "date", "head_m"])
    src = frame.loc[frame.conc.idxmax()]
    return SiteObservations(site_name="Dom", analyte="benzene", epsg=epsg, threshold=0.005,
                            start_date=np.datetime64("2020-01-01"), frame=frame, wells=[],
                            heads=heads, source_xy=(float(src.easting), float(src.northing)),
                            source_conc=float(src.conc))


def test_recovers_positive_decreasing_lambda():
    site = _domenico_site(slope_m=-0.05)
    e = m2.estimate_site(site, soil_type="sand")
    assert np.isfinite(e.value_per_year)
    assert e.value_per_year > 0                       # declining plume -> positive lambda
    assert e.trend == "decreasing"
    assert e.removes_dilution is True
    # internal consistency: reported lambda equals the inversion of the reported slope/params
    d = e.diagnostics
    lam_day = m2._lambda_from_m(d["slope_m_per_m"], d["contaminant_velocity_m_per_day"], d["alpha_x_m"])
    # diagnostics store rounded params, so check consistency to rounding tolerance
    assert math.isclose(lam_day * m2.DAYS_PER_YEAR, e.value_per_year, rel_tol=0.02)


def test_na_without_heads():
    site = _domenico_site()
    site.heads.drop(site.heads.index, inplace=True)   # remove all heads
    e = m2.estimate_site(site, soil_type="sand")
    assert e.confidence == "N/A"
    assert "flow direction" in e.notes


def _domenico_2d_site(k=0.3, ay=2.0, vc=100.0, Y=10.0, C0=5.0, seed=1):
    """2D Domenico plume: wells on a transverse grid so alpha_y is identifiable."""
    from scipy.special import erf
    rng = np.random.default_rng(seed)
    xs = np.arange(20, 201, 20.0); ys = np.array([-30, -15, 0, 15, 30.0])
    be, bn = 500000.0, 5800000.0; date = pd.Timestamp("2020-01-01"); rows, h = [], []
    for x in xs:
        for y in ys:
            d = 2 * math.sqrt(ay * x)
            phi = 0.5 * (erf((y + Y / 2) / d) - erf((y - Y / 2) / d))
            c = C0 * phi * math.exp(-k * x / vc) * math.exp(rng.normal(0, 0.08))
            e, n = be + x, bn + y
            rows.append((f"W{len(rows)}", e, n, date, 0.0, max(c, 1e-9), True, np.nan))
            h.append((f"H{len(h)}", e, n, date, 100.0 - 0.01 * x))
    frame = pd.DataFrame(rows, columns=["well_id", "easting", "northing", "date",
                                        "t_years", "conc", "detect", "rl"])
    heads = pd.DataFrame(h, columns=["well_id", "easting", "northing", "date", "head_m"])
    src = frame.loc[frame.conc.idxmax()]
    return SiteObservations("D2", "benzene", 32613, 0.005, np.datetime64("2020-01-01"),
                            frame, [], heads, (float(src.easting), float(src.northing)),
                            float(src.conc))


def test_2d_fit_recovers_k_and_estimates_alpha_y():
    site = _domenico_2d_site(k=0.3, ay=2.0)
    e = m2.estimate_site_2d(site, cfg=m2.DomenicoConfig(vc_override=100.0 / m2.DAYS_PER_YEAR,
                                                        source_width_override=10.0))
    assert np.isfinite(e.value_per_year) and e.value_per_year > 0
    assert e.trend == "decreasing" and e.removes_dilution is True
    assert e.diagnostics["r2"] > 0.5
    assert 0.5 < e.diagnostics["alpha_y_fitted_m"] < 12      # right order of the true 2.0


def test_2d_fit_rejects_structureless():
    rng = np.random.default_rng(3)
    be, bn = 500000.0, 5800000.0; date = pd.Timestamp("2020-01-01"); rows, h = [], []
    for i in range(24):
        x = rng.uniform(5, 200); y = rng.uniform(-40, 40); e, n = be + x, bn + y
        rows.append((f"W{i}", e, n, date, 0.0, float(rng.uniform(0.01, 5.0)), True, np.nan))
        h.append((f"H{i}", e, n, date, 100.0 - 0.01 * x))
    frame = pd.DataFrame(rows, columns=["well_id", "easting", "northing", "date",
                                        "t_years", "conc", "detect", "rl"])
    heads = pd.DataFrame(h, columns=["well_id", "easting", "northing", "date", "head_m"])
    src = frame.loc[frame.conc.idxmax()]
    site = SiteObservations("R", "benzene", None, 0.005, np.datetime64("2020-01-01"),
                            frame, [], heads, (float(src.easting), float(src.northing)),
                            float(src.conc))
    e = m2.estimate_site_2d(site)
    assert e.confidence == "N/A"                            # no usable spatial structure


def test_na_transverse_only():
    """Wells spread far transverse to flow -> no centerline transect -> N/A."""
    site = _domenico_site()
    site.frame["northing"] = site.frame["northing"] + np.linspace(-80, 80, len(site.frame))
    e = m2.estimate_site(site, soil_type="sand")
    assert e.confidence == "N/A"
