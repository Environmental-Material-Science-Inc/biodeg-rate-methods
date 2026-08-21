"""Module 2 — Domenico-normalized concentration-versus-distance (flowpath lambda).

Estimand: lambda, the first-order decay coefficient (1/yr) of a steady-state plume along
its flowpath, after analytically removing the concentration decline caused by transverse
(and vertical) dispersion rather than by destruction. This is the ONLY one of the three
methods that yields a coefficient with a mechanistic transport interpretation (it removes
dilution). It is not comparable to the statistical rates of Methods 1 and 3 and must not be
averaged with them.

Math: reference Section 2. The two-step, dispersivity-fixed procedure of 2.6 is used (not a
free nonlinear fit, which is non-identifiable). The dominant uncertainty is dispersivity, so
lambda is reported with a Monte-Carlo band over the joint (alpha_x, v_c, m) and as an
explicit sweep over assumed alpha_x.

Vertical spreading is not removed here (Phi_z = 1): the field inputs carry no vertical
delineation, so the method runs in the 2D horizontal form C = (C0/2) Phi_y exp(m x). This is
stated in the output. Flow direction and gradient come from the measured heads; transport
parameters (K from soil type, porosity, retardation) are literature [ASSUMED].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from .contract import (RateEstimate, SiteObservations,
                       METHOD_DOMENICO, METHOD_DOMENICO_2D, METHOD_ESTIMAND)

DAYS_PER_YEAR = 365.25

# Hydraulic conductivity by soil type (m/d), literature midpoints [ASSUMED].
K_BY_SOIL = {
    "gravel": 50.0, "sand and gravel": 50.0, "sand": 10.0, "fine sand": 3.0,
    "silty sand": 1.0, "sandy silt": 0.5, "silt": 0.1, "clay till": 0.05,
    "silty clay": 0.03, "clay": 0.01,
}
DEFAULT_K = 1.0


@dataclass(frozen=True)
class DomenicoConfig:
    """Transport parameters and procedure knobs. Defaults are literature [ASSUMED]; override
    per site as independent evidence becomes available (reference 2.6)."""
    porosity: float = 0.25
    retardation: float = 1.0            # benzene, low f_oc sand; ~1.0-1.2
    default_gradient: float = 0.01      # used only if heads cannot give a gradient
    alpha_x_frac: float = 0.10          # alpha_x = frac * plume length L
    aniso_xy: float = 10.0              # alpha_x / alpha_y
    representative: str = "max"         # per-well snapshot: "max" | "recent" | "median"
    min_wells: int = 4                  # minimum down-gradient detects to attempt a fit
    n_mc: int = 4000                    # Monte-Carlo draws for the lambda band
    alpha_x_uncertainty_factor: float = 3.0   # log-normal x/ this on alpha_x (Stenback 2004)
    vc_uncertainty_factor: float = 2.0        # log-normal x/ this on v_c
    seed: int = 12345
    # overrides for when transport parameters are KNOWN (a tracer test, or a forward model such
    # as REMChlor) rather than estimated from soil type and the head field:
    vc_override: float | None = None          # contaminant velocity v/R (m/day); skips K,i,n
    alpha_x_override: float | None = None      # longitudinal dispersivity (m); skips 0.1*L
    source_width_override: float | None = None # source width Y (m); skips the data estimate


def _k_from_soil(soil: str | None) -> float:
    if not soil:
        return DEFAULT_K
    s = soil.strip().lower()
    for key, val in K_BY_SOIL.items():        # longest, most specific keys first
        if key in s:
            return val
    return DEFAULT_K


def _flow_from_heads(heads, source_xy):
    """Fit a plane to per-well mean heads -> (unit flow vector, gradient magnitude, azimuth).
    Returns (u, i, azimuth_deg, n_head_wells) or None if heads are unusable."""
    if heads is None or len(heads) == 0:
        return None
    g = heads.groupby("well_id").agg(easting=("easting", "first"),
                                     northing=("northing", "first"),
                                     head_m=("head_m", "mean")).dropna()
    if len(g) < 3:
        return None
    E = g.easting.to_numpy(float); N = g.northing.to_numpy(float); h = g.head_m.to_numpy(float)
    A = np.column_stack([np.ones_like(E), E - E.mean(), N - N.mean()])
    coef, *_ = np.linalg.lstsq(A, h, rcond=None)
    grad = np.array([coef[1], coef[2]])           # dh/dE, dh/dN
    gmag = float(np.hypot(*grad))
    if gmag < 1e-9:
        return None
    u = -grad / gmag                              # flow points down the head gradient
    azimuth = (math.degrees(math.atan2(u[0], u[1]))) % 360.0   # from north, clockwise
    return u, gmag, azimuth, int(len(g))


def _representative_conc(site: SiteObservations, how: str):
    """Per-well representative concentration and a detect flag (steady-state snapshot)."""
    f = site.frame
    rows = []
    for wid, gdf in f.groupby("well_id"):
        det = gdf[gdf.detect]
        if len(det):
            if how == "recent":
                c = float(det.sort_values("date").conc.iloc[-1])
            elif how == "median":
                c = float(det.conc.median())
            else:
                c = float(det.conc.max())
            rows.append((wid, float(gdf.easting.iloc[0]), float(gdf.northing.iloc[0]), c, True))
        else:
            rows.append((wid, float(gdf.easting.iloc[0]), float(gdf.northing.iloc[0]),
                         float(gdf.conc.min()), False))   # non-detect at its RL
    return rows


def estimate_site(site: SiteObservations, cfg: DomenicoConfig = DomenicoConfig(),
                  soil_type: str | None = None) -> RateEstimate:
    scope = "site"
    flow = _flow_from_heads(site.heads, site.source_xy)
    if flow is None:
        return RateEstimate.not_applicable(METHOD_DOMENICO, scope,
                                           "no usable heads to establish a flow direction")
    u, gradient, azimuth, n_head = flow

    sx, sy = site.source_xy
    rep = _representative_conc(site, cfg.representative)
    # project onto source-centred (x along flow, y transverse)
    perp = np.array([-u[1], u[0]])
    wells = []
    for wid, e, n, c, det in rep:
        dvec = np.array([e - sx, n - sy])
        x = float(dvec @ u)               # down-gradient distance (m)
        y = float(dvec @ perp)            # transverse offset (m)
        wells.append((wid, x, y, c, det))

    det_wells = [w for w in wells if w[4] and w[1] > 0]   # detects down-gradient of source
    if len(det_wells) < cfg.min_wells:
        return RateEstimate.not_applicable(
            METHOD_DOMENICO, scope,
            f"only {len(det_wells)} down-gradient detects (<{cfg.min_wells}); cannot fit a transect")

    xs = np.array([w[1] for w in det_wells])
    L = float(xs.max())                                   # plume length along flow
    alpha_x = cfg.alpha_x_override if cfg.alpha_x_override is not None else cfg.alpha_x_frac * L
    alpha_y = alpha_x / cfg.aniso_xy
    # source width: transverse spread of near-source detects, floored
    # source width Y feeds the Phi_y dilution factor. It is the SOURCE zone width (transverse
    # spread of the highest-concentration near-source wells), NOT the full plume spread, so it
    # is capped at a fraction of L. An over-wide Y would otherwise delete the whole transect.
    near = [w for w in det_wells if w[1] < 0.5 * L] or det_wells
    top = sorted(near, key=lambda w: -w[3])[:max(3, len(near) // 4)]
    ys_top = np.array([w[2] for w in top])
    Y = float(min(max(2.0 * np.std(ys_top), 2.0), 0.5 * L)) if L > 5 else 2.0
    if cfg.source_width_override is not None:
        Y = float(cfg.source_width_override)

    # contaminant velocity
    K = _k_from_soil(soil_type)
    v = K * gradient / cfg.porosity
    v_c = v / cfg.retardation                              # m/d
    if cfg.vc_override is not None:                        # known velocity (tracer test / REMChlor)
        v_c = float(cfg.vc_override)
        v = v_c * cfg.retardation

    # normalize: C* = C / Phi_y. Exclude only the immediate near-source band (numeric guard
    # where sqrt(alpha_y x) -> 0 destabilizes Phi), not a full source-length cut, because the field
    # plumes are short. Keep wells whose Phi_y is well conditioned.
    x_min = max(0.1 * L, 2.0)
    X, lnCstar, used = [], [], []
    for wid, x, y, c, det in det_wells:
        if x <= x_min:
            continue
        denom = 2.0 * math.sqrt(alpha_y * x)
        if denom < 1e-6:
            continue
        phi_y = 0.5 * (erf((y + Y / 2) / denom) - erf((y - Y / 2) / denom))
        if phi_y <= 1e-3:
            continue
        X.append(x); lnCstar.append(math.log(c / phi_y)); used.append((wid, x, y, c, phi_y))
    X = np.array(X); lnCstar = np.array(lnCstar)
    if len(X) < cfg.min_wells:
        return RateEstimate.not_applicable(
            METHOD_DOMENICO, scope,
            f"only {len(X)} down-gradient centerline wells with a well-conditioned spreading "
            f"factor (need {cfg.min_wells}); detected wells are spread transverse to the "
            f"measured flow direction, so there is no usable centerline transect")

    # robust slope m (Theil-Sen) of ln C* on x, with a bootstrap SE
    m_hat, se_m = _theil_sen_slope(X, lnCstar)
    if not np.isfinite(m_hat) or m_hat >= 0:
        return RateEstimate(
            method=METHOD_DOMENICO, estimand=METHOD_ESTIMAND[METHOD_DOMENICO], scope=scope,
            value_per_year=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            half_life_years=float("nan"), n=len(X), trend="no trend", confidence="N/A",
            removes_dilution=True,
            diagnostics=dict(slope_m_per_m=round(float(m_hat), 5), reason="non-negative slope",
                             flow_azimuth_deg=round(azimuth, 1), plume_length_m=round(L, 1)),
            notes="normalized concentration does not decline down-gradient; no net flowpath decay")

    lam_day = _lambda_from_m(m_hat, v_c, alpha_x)
    lam_yr = lam_day * DAYS_PER_YEAR

    # Monte-Carlo band over (alpha_x, v_c, m): the map m->lambda is nonlinear (reference 2.6).
    rng = np.random.default_rng(cfg.seed)
    fa = math.log(cfg.alpha_x_uncertainty_factor)
    fv = math.log(cfg.vc_uncertainty_factor)
    ax_s = alpha_x * np.exp(rng.normal(0, fa / 1.645, cfg.n_mc))
    vc_s = v_c * np.exp(rng.normal(0, fv / 1.645, cfg.n_mc))
    m_s = rng.normal(m_hat, se_m if np.isfinite(se_m) and se_m > 0 else abs(m_hat) * 0.3, cfg.n_mc)
    lam_mc = np.array([_lambda_from_m(mm, vc, ax) for mm, vc, ax in zip(m_s, vc_s, ax_s)])
    lam_mc = lam_mc[np.isfinite(lam_mc)] * DAYS_PER_YEAR
    lam_pos = lam_mc[lam_mc > 0]
    ci_lo = float(np.percentile(lam_mc, 5)) if lam_mc.size else float("nan")
    ci_hi = float(np.percentile(lam_mc, 95)) if lam_mc.size else float("nan")
    frac_pos = float(lam_pos.size / lam_mc.size) if lam_mc.size else 0.0

    # sensitivity sweep over assumed alpha_x (reference 2.6: report lambda vs dispersivity)
    sweep = {}
    for frac in (0.05, 0.1, 0.2):
        ax = frac * L
        sweep[f"alpha_x={frac:.2f}L"] = round(_lambda_from_m(m_hat, v_c, ax) * DAYS_PER_YEAR, 4)

    conf = "low (assumed params)"     # dispersivity / K / R are literature values, not measured
    if n_head >= 5 and len(X) >= 6 and frac_pos > 0.9:
        conf = "medium (assumed params)"

    # honesty flags: a CI spanning more than an order of magnitude, or an implausibly fast
    # half-life, means the central value is not trustworthy as a reaction rate.
    poorly_constrained = bool(np.isfinite(ci_hi) and np.isfinite(ci_lo)
                              and (ci_hi <= 0 or ci_lo <= 0
                                   or (ci_hi / max(ci_lo, 1e-9)) > 10.0))
    implausibly_fast = bool(0 < RateEstimate.half_life(lam_yr) < 0.25)
    extra = ""
    if poorly_constrained or implausibly_fast or len(X) < 6:
        conf = "low (assumed params)"
    if poorly_constrained:
        extra += (" CI spans more than an order of magnitude: the slope and transport parameters "
                  "do not constrain lambda; treat the central value as indicative only.")
    if implausibly_fast:
        extra += (" Implied half-life under 3 months is implausibly fast for a flowpath reaction "
                  "rate; the down-gradient drop is likely plume-edge, dilution or heterogeneity, "
                  "not destruction. Do not use as a reaction rate without more centerline wells.")

    diagnostics = dict(
        slope_m_per_m=round(float(m_hat), 5), slope_se=round(float(se_m), 5),
        flow_azimuth_deg=round(azimuth, 1), hydraulic_gradient=round(gradient, 5),
        n_head_wells=n_head, plume_length_m=round(L, 1),
        alpha_x_m=round(alpha_x, 2), alpha_y_m=round(alpha_y, 3), source_width_Y_m=round(Y, 2),
        K_m_per_day=K, porosity=cfg.porosity, retardation=cfg.retardation,
        seepage_velocity_m_per_day=round(v, 4), contaminant_velocity_m_per_day=round(v_c, 4),
        n_wells_in_fit=len(X), mc_fraction_positive=round(frac_pos, 3),
        lambda_vs_assumed_alpha_x=sweep, vertical_spreading_removed=False,
        soil_type=soil_type, poorly_constrained=poorly_constrained,
        implausibly_fast=implausibly_fast,
        qaqc=dict(                                  # compact arrays for the QA/QC renderer
            transect_x=[round(float(x), 2) for x in X],
            transect_lnCstar=[round(float(v), 4) for v in lnCstar],
            fit_slope=round(float(m_hat), 5),
            fit_intercept=round(float(np.median(lnCstar) - m_hat * np.median(X)), 4),
            mc_lambda_per_year=[round(float(v), 5) for v in lam_mc[:600].tolist()],
        ),
    )
    notes = ("flowpath reaction coefficient with transverse dilution removed (Phi_z=1, 2D form). "
             "Dominant uncertainty is dispersivity; see lambda_vs_assumed_alpha_x and the MC band. "
             "K, porosity and retardation are literature [ASSUMED]." + extra)
    return RateEstimate(
        method=METHOD_DOMENICO, estimand=METHOD_ESTIMAND[METHOD_DOMENICO], scope=scope,
        value_per_year=float(lam_yr), ci_low=ci_lo, ci_high=ci_hi,
        half_life_years=RateEstimate.half_life(lam_yr), n=len(X), trend="decreasing",
        confidence=conf, removes_dilution=True, diagnostics=diagnostics, notes=notes,
    )


def _lambda_from_m(m: float, v_c: float, alpha_x: float) -> float:
    """lambda = -v_c m (1 - alpha_x m)  (reference 2.5). Positive when m < 0."""
    return -v_c * m * (1.0 - alpha_x * m)


def _theil_sen_slope(x: np.ndarray, y: np.ndarray):
    """Theil-Sen slope plus a bootstrap standard error."""
    n = len(x)
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(n) for j in range(i + 1, n) if x[j] != x[i]]
    if not slopes:
        return float("nan"), float("nan")
    m = float(np.median(slopes))
    rng = np.random.default_rng(7)
    boot = []
    for _ in range(300):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        sl = [(yb[j] - yb[i]) / (xb[j] - xb[i])
              for i in range(n) for j in range(i + 1, n) if xb[j] != xb[i]]
        if sl:
            boot.append(np.median(sl))
    se = float(np.std(boot)) if boot else float("nan")
    return m, se


# ============================================================ 2D Domenico fit (all wells)
def estimate_site_2d(site: SiteObservations, cfg: DomenicoConfig = DomenicoConfig(),
                     soil_type: str | None = None) -> RateEstimate:
    """Method 2, fitted in 2D to ALL down-gradient detected wells at once.

    Instead of normalizing each well and regressing along a centerline corridor (the 1D path,
    which discards far off-axis wells where Phi_y -> 0 and 1/Phi_y blows up), this fits

        ln C(x, y) = b0 + m x + ln Phi_y(x, y; alpha_y, Y)

    jointly for (b0, m, alpha_y) by nonlinear least squares over every down-gradient detect. Two
    advantages on clustered networks without a clean transect: it uses all wells (off-axis wells
    inform the fit through ln, with no division by a tiny Phi_y), and it ESTIMATES the transverse
    dispersivity alpha_y from the data rather than assuming alpha_y = alpha_x / 10. lambda is then
    recovered from m exactly as in the 1D method (same inversion, same MC band over alpha_x, v_c, m).
    """
    from scipy.optimize import least_squares
    scope = "site"
    flow = _flow_from_heads(site.heads, site.source_xy)
    if flow is None:
        return RateEstimate.not_applicable(METHOD_DOMENICO_2D, scope,
                                           "no usable heads to establish a flow direction")
    u, gradient, azimuth, n_head = flow
    sx, sy = site.source_xy
    perp = np.array([-u[1], u[0]])
    rep = _representative_conc(site, cfg.representative)
    wells = []
    for wid, e, n, c, det in rep:
        d = np.array([e - sx, n - sy])
        wells.append((wid, float(d @ u), float(d @ perp), c, det))
    det_wells = [w for w in wells if w[4] and w[1] > 0]
    if len(det_wells) < 6:                      # 3 params -> want a few extra wells
        return RateEstimate.not_applicable(
            METHOD_DOMENICO_2D, scope,
            f"only {len(det_wells)} down-gradient detects (<6 needed for the 2D fit)")

    xs = np.array([w[1] for w in det_wells])
    ys = np.array([w[2] for w in det_wells])
    cs = np.array([w[3] for w in det_wells])
    L = float(xs.max())
    alpha_x = cfg.alpha_x_override if cfg.alpha_x_override is not None else cfg.alpha_x_frac * L
    # source width Y: fixed (estimate as in 1D, or override). alpha_y is FITTED, not Y/aniso.
    near = [w for w in det_wells if w[1] < 0.5 * L] or det_wells
    top = sorted(near, key=lambda w: -w[3])[:max(3, len(near) // 4)]
    Y = float(min(max(2.0 * np.std([w[2] for w in top]), 2.0), 0.5 * L)) if L > 5 else 2.0
    if cfg.source_width_override is not None:
        Y = float(cfg.source_width_override)

    x_min = max(0.1 * L, 2.0)
    keep = xs > x_min
    X, Yv, C = xs[keep], ys[keep], cs[keep]
    if len(X) < 6:
        return RateEstimate.not_applicable(
            METHOD_DOMENICO_2D, scope,
            f"only {len(X)} down-gradient detects past the near-source band (<6 for the 2D fit)")
    lnC = np.log(np.maximum(C, 1e-12))

    def phi_y(x, y, ay):
        denom = 2.0 * np.sqrt(np.maximum(ay, 1e-6) * x)
        return 0.5 * (erf((y + Y / 2) / denom) - erf((y - Y / 2) / denom))

    def resid(p):
        b0, m, lay = p
        pred = b0 + m * X + np.log(np.maximum(phi_y(X, Yv, math.exp(lay)), 1e-9))
        return pred - lnC

    # bound alpha_y physically: 0.01 m to half the plume length (a transverse dispersivity larger
    # than that is unidentifiable on these data and signals no usable transverse structure).
    ay_hi = max(0.5 * L, 5.0)
    p0 = [float(lnC.max()), -1.0 / max(L, 1.0), math.log(min(max(alpha_x / 10.0, 0.1), ay_hi))]
    lo = [-50.0, -10.0, math.log(0.01)]
    hi = [50.0, -1e-6, math.log(ay_hi)]
    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=6000)
    except Exception as e:
        return RateEstimate.not_applicable(METHOD_DOMENICO_2D, scope, f"2D fit failed: {e}")
    b0, m_hat, lay = sol.x
    alpha_y_fit = math.exp(lay)
    r = sol.fun
    ss_res = float(r @ r); ss_tot = float(np.sum((lnC - lnC.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # reject a fit with no usable spatial structure (worse than, or barely better than, the mean):
    # the well layout does not support a 2D Domenico plume.
    if not np.isfinite(r2) or r2 < 0.2:
        return RateEstimate.not_applicable(
            METHOD_DOMENICO_2D, scope,
            f"2D fit has no usable spatial structure (R2={r2:.2f} < 0.2); the well layout does not "
            f"support a Domenico plume")
    # SE on m from the Gauss-Newton covariance
    try:
        dof = max(len(X) - 3, 1)
        cov = (ss_res / dof) * np.linalg.inv(sol.jac.T @ sol.jac)
        se_m = float(math.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        se_m = abs(m_hat) * 0.3

    if not np.isfinite(m_hat) or m_hat >= 0:
        return RateEstimate(
            method=METHOD_DOMENICO_2D, estimand=METHOD_ESTIMAND[METHOD_DOMENICO_2D], scope=scope,
            value_per_year=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            half_life_years=float("nan"), n=len(X), trend="no trend", confidence="N/A",
            removes_dilution=True, diagnostics=dict(reason="non-negative slope",
            flow_azimuth_deg=round(azimuth, 1), alpha_y_fitted_m=round(alpha_y_fit, 3)),
            notes="2D fit found no down-gradient decline; no net flowpath decay")

    K = _k_from_soil(soil_type)
    v = K * gradient / cfg.porosity
    v_c = v / cfg.retardation
    if cfg.vc_override is not None:
        v_c = float(cfg.vc_override); v = v_c * cfg.retardation
    lam_yr = _lambda_from_m(m_hat, v_c, alpha_x) * DAYS_PER_YEAR

    rng = np.random.default_rng(cfg.seed)
    fa = math.log(cfg.alpha_x_uncertainty_factor); fv = math.log(cfg.vc_uncertainty_factor)
    ax_s = alpha_x * np.exp(rng.normal(0, fa / 1.645, cfg.n_mc))
    vc_s = v_c * np.exp(rng.normal(0, fv / 1.645, cfg.n_mc))
    m_s = rng.normal(m_hat, se_m if np.isfinite(se_m) and se_m > 0 else abs(m_hat) * 0.3, cfg.n_mc)
    lam_mc = np.array([_lambda_from_m(mm, vc, ax) for mm, vc, ax in zip(m_s, vc_s, ax_s)]) * DAYS_PER_YEAR
    lam_mc = lam_mc[np.isfinite(lam_mc)]
    ci_lo = float(np.percentile(lam_mc, 5)) if lam_mc.size else float("nan")
    ci_hi = float(np.percentile(lam_mc, 95)) if lam_mc.size else float("nan")
    frac_pos = float((lam_mc > 0).mean()) if lam_mc.size else 0.0

    poorly = bool(np.isfinite(ci_hi) and np.isfinite(ci_lo)
                  and (ci_hi <= 0 or ci_lo <= 0 or ci_hi / max(ci_lo, 1e-9) > 10.0))
    fast = bool(0 < RateEstimate.half_life(lam_yr) < 0.25)
    conf = "low (assumed params)"
    if n_head >= 5 and len(X) >= 8 and frac_pos > 0.9 and r2 >= 0.3 and not poorly and not fast:
        conf = "medium (assumed params)"
    extra = ""
    if poorly:
        extra += " CI spans more than an order of magnitude; treat the central value as indicative."
    if fast:
        extra += (" Implied half-life under 3 months is implausibly fast for a reaction rate; the "
                  "drop is likely plume-edge/heterogeneity, not destruction.")

    diagnostics = dict(
        fit="2D nonlinear (all down-gradient wells; alpha_y fitted)",
        slope_m_per_m=round(float(m_hat), 5), slope_se=round(float(se_m), 5), r2=round(float(r2), 3),
        alpha_y_fitted_m=round(float(alpha_y_fit), 3), source_width_Y_m=round(Y, 2),
        alpha_x_m=round(alpha_x, 2), flow_azimuth_deg=round(azimuth, 1),
        hydraulic_gradient=round(gradient, 5), n_head_wells=n_head, plume_length_m=round(L, 1),
        K_m_per_day=K, contaminant_velocity_m_per_day=round(v_c, 4),
        n_wells_in_fit=int(len(X)), mc_fraction_positive=round(frac_pos, 3),
        poorly_constrained=poorly, implausibly_fast=fast, vertical_spreading_removed=False,
        soil_type=soil_type,
        qaqc=dict(transect_x=[round(float(x), 2) for x in X],
                  transect_y=[round(float(y), 2) for y in Yv],
                  obs_lnC=[round(float(v), 4) for v in lnC],
                  mc_lambda_per_year=[round(float(v), 5) for v in lam_mc[:600].tolist()]),
    )
    notes = ("2D Domenico fit over all down-gradient wells; transverse dispersivity alpha_y is "
             "fitted from the data (not assumed alpha_x/10). lambda recovered from the longitudinal "
             "slope m as in the 1D method. K, porosity, retardation are literature [ASSUMED]." + extra)
    return RateEstimate(
        method=METHOD_DOMENICO_2D, estimand=METHOD_ESTIMAND[METHOD_DOMENICO_2D], scope=scope,
        value_per_year=float(lam_yr), ci_low=ci_lo, ci_high=ci_hi,
        half_life_years=RateEstimate.half_life(lam_yr), n=len(X), trend="decreasing",
        confidence=conf, removes_dilution=True, diagnostics=diagnostics, notes=notes,
    )
