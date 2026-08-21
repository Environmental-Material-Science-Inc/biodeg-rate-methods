"""Module 1 — Mann-Kendall trend test with Theil-Sen slope (per-well point decay).

Estimand: the per-well first-order point-decay constant k_point (1/yr), the slope of
ln C versus time. This is a STATISTICAL bulk rate at one location. It does not remove
dilution and is not a flowpath reaction coefficient. Method 2 is the only one of the
three that yields a mechanistic rate. Keep the three apart.

The math follows the reference document Section 1 and the committed censoring convention
in Appendix A (simple recensoring). Mann-Kendall is rank based, so the test statistic, its
variance, and the p-value are identical on C or ln C; the log scale matters only for the
Theil-Sen slope, which is the point-decay constant.

Pure module: ``estimate_well`` consumes a WellSeries, ``estimate_site`` consumes a
SiteObservations and returns one RateEstimate per well plus a robust (non-averaged) summary.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .censoring import recensor, pair_sign, tie_groups, RecensoredSeries
from .contract import (RateEstimate, SiteObservations, WellSeries,
                       METHOD_MANN_KENDALL, METHOD_ESTIMAND)

MIN_EVENTS = 4          # minimum sampling events to attempt a trend
MIN_DET_PAIRS = 3       # minimum determinate pairwise slopes for a Theil-Sen estimate
ALPHA = 0.05            # two-sided significance level for the trend call and the slope CI


def _mk_statistic(rec: RecensoredSeries):
    """Mann-Kendall S, tie-corrected Var(S), continuity-corrected Z, two-sided p."""
    n = rec.n
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += pair_sign(i, j, rec)
    groups = tie_groups(rec)
    var = (n * (n - 1) * (2 * n + 5)
           - sum(tp * (tp - 1) * (2 * tp + 5) for tp in groups)) / 18.0
    var = max(var, 0.0)
    if var == 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return int(s), float(var), float(z), float(p)


def _theil_sen(rec: RecensoredSeries):
    """Theil-Sen slope of ln(value) on time over DETERMINATE detect pairs only, the Conover
    intercept, and the count of determinate pairwise slopes. Slope is per year."""
    det = ~rec.censored
    idx = np.where(det)[0]
    slopes = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            dt = rec.t[j] - rec.t[i]
            if dt == 0:
                continue
            yi, yj = math.log(rec.value[i]), math.log(rec.value[j])
            slopes.append((yj - yi) / dt)
    slopes = np.sort(np.asarray(slopes, float))
    if slopes.size == 0:
        return float("nan"), float("nan"), 0, slopes
    beta = float(np.median(slopes))
    y_det = np.log(rec.value[det])
    t_det = rec.t[det]
    intercept = float(np.median(y_det) - beta * np.median(t_det))
    return beta, intercept, int(slopes.size), slopes


def _slope_ci(slopes: np.ndarray, var_s: float, alpha: float = ALPHA):
    """Distribution-free CI on the Theil-Sen slope using Var(S) (reference 1.4 / A.4)."""
    n_prime = slopes.size
    if n_prime < 2 or var_s <= 0:
        return float("nan"), float("nan")
    c_alpha = norm.ppf(1 - alpha / 2) * math.sqrt(var_s)
    m1 = (n_prime - c_alpha) / 2.0
    m2 = (n_prime + c_alpha) / 2.0
    lo_idx = int(math.floor(m1))
    hi_idx = int(math.ceil(m2)) + 1
    lo_idx = max(0, min(lo_idx, n_prime - 1))
    hi_idx = max(0, min(hi_idx, n_prime - 1))
    return float(slopes[lo_idx]), float(slopes[hi_idx])


def estimate_well(w: WellSeries) -> RateEstimate:
    """Point-decay rate for one well. Returns an N/A estimate if the record is too thin."""
    scope = f"well:{w.well_id}"
    if w.n < MIN_EVENTS:
        return RateEstimate.not_applicable(METHOD_MANN_KENDALL, scope,
                                           f"only {w.n} events (<{MIN_EVENTS})")
    rec = recensor(w.t_years, w.conc, w.detect, w.rl)
    s, var_s, z, p = _mk_statistic(rec)
    beta, intercept, n_pairs, slopes = _theil_sen(rec)

    trend = ("decreasing" if (p < ALPHA and s < 0)
             else "increasing" if (p < ALPHA and s > 0) else "no trend")

    if n_pairs < MIN_DET_PAIRS or not np.isfinite(beta):
        est = RateEstimate.not_applicable(METHOD_MANN_KENDALL, scope,
                                          f"only {n_pairs} determinate slope pairs")
        d = dict(mk_S=s, mk_Z=round(z, 3), p_value=round(p, 4), n=w.n, n_detect=w.n_detect,
                 censored_fraction=round(rec.censored_fraction, 3), trend=trend)
        return RateEstimate(**{**est.__dict__, "n": w.n, "trend": trend, "diagnostics": d})

    k_point = -beta                                   # 1/yr
    slo, shi = _slope_ci(slopes, var_s)
    k_lo, k_hi = -shi, -slo                           # negate and swap

    conf = "high" if (w.n >= 8 and not rec.heavily_censored and p < ALPHA) else \
           "medium" if (w.n >= 6 and not rec.heavily_censored) else "low"
    if rec.heavily_censored:
        conf = "low"

    notes = ""
    if rec.heavily_censored:
        notes = (f"censored fraction {rec.censored_fraction:.0%} exceeds the simple-recensoring "
                 f"validity bound (~20%); slope biases toward attenuation, escalate to "
                 f"Akritas-Theil-Sen for a defensible value")

    diagnostics = dict(
        mk_S=s, mk_Z=round(z, 3), p_value=round(p, 4), var_S=round(var_s, 1),
        sen_slope_lnC_per_yr=round(beta, 5), intercept_lnC=round(intercept, 4),
        n=w.n, n_detect=w.n_detect, n_slope_pairs=n_pairs,
        censored_fraction=round(rec.censored_fraction, 3), rl_star=rec.rl_star,
        t_span_years=round(float(w.t_years.max() - w.t_years.min()), 2),
    )
    return RateEstimate(
        method=METHOD_MANN_KENDALL, estimand=METHOD_ESTIMAND[METHOD_MANN_KENDALL], scope=scope,
        value_per_year=float(k_point), ci_low=float(k_lo), ci_high=float(k_hi),
        half_life_years=RateEstimate.half_life(k_point), n=w.n, trend=trend,
        confidence=conf, removes_dilution=False, diagnostics=diagnostics, notes=notes,
    )


def estimate_site(site: SiteObservations) -> tuple[list[RateEstimate], RateEstimate]:
    """Per-well estimates plus a robust SUMMARY across decreasing wells.

    The summary is the median of the per-well point-decay rates among wells with a
    statistically significant decreasing trend (a robust descriptor, not a cross-method
    average; the three methods are never combined). If no well decreases significantly,
    the summary falls back to the source-zone well and is flagged low confidence.
    """
    per_well = [estimate_well(w) for w in site.wells]
    usable = [e for e in per_well if np.isfinite(e.value_per_year)]
    decreasing = [e for e in usable if e.trend == "decreasing"]

    scope = "site"
    if decreasing:
        vals = np.array([e.value_per_year for e in decreasing])
        med = float(np.median(vals))
        lo = float(np.percentile(vals, 25))
        hi = float(np.percentile(vals, 75))
        conf = "high" if len(decreasing) >= 5 else "medium" if len(decreasing) >= 3 else "low"
        notes = (f"median point-decay across {len(decreasing)} significantly decreasing wells "
                 f"(of {len(usable)} usable); IQR shown as the band. A summary statistic, "
                 f"not an average with other methods.")
        diag = dict(n_wells_usable=len(usable), n_wells_decreasing=len(decreasing),
                    n_wells_increasing=sum(e.trend == "increasing" for e in usable),
                    n_wells_no_trend=sum(e.trend == "no trend" for e in usable),
                    per_well_k=[round(v, 4) for v in vals.tolist()])
        summary = RateEstimate(
            method=METHOD_MANN_KENDALL, estimand=METHOD_ESTIMAND[METHOD_MANN_KENDALL], scope=scope,
            value_per_year=med, ci_low=lo, ci_high=hi, half_life_years=RateEstimate.half_life(med),
            n=len(decreasing), trend="decreasing", confidence=conf, removes_dilution=False,
            diagnostics=diag, notes=notes)
    elif usable:
        # fall back to the source-zone well (closest to the hotspot)
        sx, sy = site.source_xy
        wmap = {w.well_id: w for w in site.wells}
        def dist(e):
            w = wmap[e.scope.split(":", 1)[1]]
            return math.hypot(w.easting - sx, w.northing - sy)
        src = min(usable, key=dist)
        summary = RateEstimate(
            method=METHOD_MANN_KENDALL, estimand=METHOD_ESTIMAND[METHOD_MANN_KENDALL], scope=scope,
            value_per_year=src.value_per_year, ci_low=src.ci_low, ci_high=src.ci_high,
            half_life_years=src.half_life_years, n=src.n, trend=src.trend, confidence="low",
            removes_dilution=False,
            diagnostics=dict(fallback="source-zone well", source_well=src.scope,
                             n_wells_usable=len(usable)),
            notes="no well shows a significant decreasing trend; summary is the source-zone well only")
    else:
        summary = RateEstimate.not_applicable(METHOD_MANN_KENDALL, scope,
                                              "no well had enough data for a Theil-Sen slope")
    return per_well, summary
