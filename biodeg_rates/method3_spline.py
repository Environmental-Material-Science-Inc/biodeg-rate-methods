"""Module 3 — spatio-temporal penalized-spline smoothing (spline-centre concentration decay).

Estimand: the first-order CONCENTRATION decay coefficient at the plume centre,
k_centre = -d ln C / dt (1/yr), read off the REML-smoothed log-concentration surface at the
source (peak) location (the local rate field -df/dt, reference 3.8). This is a STATISTICAL
apparent-attenuation rate. It does not remove dilution and is not a flowpath reaction
coefficient. It is the SAME estimand as Method 1 (and McHugh's k_c-max), but smoothed across
all wells and times rather than read from one noisy well, so it is the robust, forecasting
rate. It is not interchangeable with Method 2 (the dilution-removed flowpath lambda).

Math: reference Section 3 and Appendix B. One smooth tensor-product cubic B-spline in
(easting, northing, time), an anisotropic second-order P-spline penalty with a SEPARATE
smoothing parameter per axis (space in metres, time in years are different scales), smoothing
selected by REML (Appendix B), with GCV computed only as a cross-check (B.7). The rate is the
robust (Theil-Sen) slope of ln C at the centre over the observed years; uncertainty is
propagated by sampling the posterior of the coefficients. A data-anchored sign check against
the raw mean-concentration trend guards against "ballooning" of the surface at the centre when
the monitoring network changes over time (Section 3.9).

The surface fit is exposed as ``fit_surface`` returning a ``SplineFit`` with a ``predict_lnC``
method, so the same engine drives the rate, the QA/QC, the tests, and the mgcv validation.
``mass_log_series`` (the whole-plume ln M(t) integral) is retained because the validation
harnesses still exercise it, but it no longer defines the production estimand.

Non-detects enter the surface at RL/2 (a documented substitution; the alternative censored
likelihood is out of scope here). The rate uses only the relative surface, so the constant
porosity * retardation * thickness scale never enters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
from matplotlib.path import Path as MplPath

from .contract import (RateEstimate, SiteObservations,
                       METHOD_ST_PSPLINE, METHOD_ESTIMAND)

MIN_WELLS = 6
MIN_TIMES = 4
MIN_OBS = 20


@dataclass(frozen=True)
class SplineConfig:
    degree: int = 3                 # cubic B-splines
    penalty_order: int = 2          # second-order difference penalty (penalizes curvature)
    max_space_bases: int = 8
    max_time_bases: int = 6
    grid_n: int = 36                # quadrature grid per spatial axis (within the hull)
    n_posterior_draws: int = 200    # posterior samples for the centre-rate credible band
    nd_substitution: float = 0.5    # non-detect value = RL * this
    seed: int = 2024
    # ballooning control (reference 3.9): restrict the mass integral to grid cells within
    # mask_support_factor * (median nearest-neighbour well spacing) of a well, so the smoother
    # cannot inflate M(t) in unsupported corners; and take the ln M(t) slope robustly (Theil-Sen)
    # so a single ballooned timestep cannot drive the rate. Set mask_support_factor=None for the full hull.
    mask_support_factor: float | None = 2.0
    robust_slope: bool = True


# --------------------------------------------------------------------------- bases / penalties
def _bspline_basis(x, a, b, nb, k):
    """Dense B-spline basis (len(x) x nb) on a clamped knot vector over [a, b]."""
    n_interior = nb - k - 1
    if n_interior < 0:
        raise ValueError(f"need at least {k + 1} bases")
    interior = np.linspace(a, b, n_interior + 2)[1:-1] if n_interior > 0 else np.array([])
    t = np.concatenate([[a] * (k + 1), interior, [b] * (k + 1)])
    B = BSpline.design_matrix(np.clip(x, a, b), t, k, extrapolate=False).toarray()
    return B, t


def _diff_penalty(nb, order):
    """P = D^T D for the order-th difference operator on nb coefficients."""
    D = np.eye(nb)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D.T @ D


def _kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


@dataclass
class SplineFit:
    """A fitted spatio-temporal log-concentration surface and its diagnostics.

    ``predict_lnC(E, N, T)`` evaluates the fitted ln C surface at absolute coordinates and time
    (years from the site start). The standardization and clamped-knot ranges are stored so the
    surface can be evaluated anywhere, which the mass integral, the posterior band, the QA/QC
    panels, and the mgcv validation all use.
    """
    theta: np.ndarray
    K1: int; K2: int; K3: int; degree: int
    Ec: float; Esd: float; Nc: float; Nsd: float; Tc: float; Tsd: float
    Erange: tuple; Nrange: tuple; Trange: tuple
    Mmat: np.ndarray; sigma2: float
    edf: float; lam: np.ndarray; Mnull: int
    gcv_at_reml: float; reml_vs_gcv_gap: float

    def spatial_basis(self, E, N):
        E = np.atleast_1d(np.asarray(E, float)); N = np.atleast_1d(np.asarray(N, float))
        B1, _ = _bspline_basis((E - self.Ec) / self.Esd, *self.Erange, self.K1, self.degree)
        B2, _ = _bspline_basis((N - self.Nc) / self.Nsd, *self.Nrange, self.K2, self.degree)
        return (B1[:, :, None] * B2[:, None, :]).reshape(len(E), self.K1 * self.K2)

    def temporal_basis(self, T):
        T = np.atleast_1d(np.asarray(T, float))
        B3, _ = _bspline_basis((T - self.Tc) / self.Tsd, *self.Trange, self.K3, self.degree)
        return B3

    def predict_lnC(self, E, N, T):
        """Fitted ln C at absolute (E, N) and time T (years). Arrays broadcast row-wise."""
        Theta = self.theta.reshape(self.K1 * self.K2, self.K3)
        Sb = self.spatial_basis(E, N)
        Tb = self.temporal_basis(T)
        return ((Sb @ Theta) * Tb).sum(axis=1)


def fit_surface(E, N, T, y, cfg: SplineConfig = SplineConfig()) -> SplineFit:
    """Fit the REML-smoothed tensor-product P-spline surface to (E, N, T, y=lnC) observations."""
    E = np.asarray(E, float); N = np.asarray(N, float)
    T = np.asarray(T, float); y = np.asarray(y, float)
    n = len(y)
    times = np.unique(np.round(T, 6))

    def stdz(v):
        c, s = float(np.mean(v)), float(np.std(v)) or 1.0
        return c, s
    Ec, Esd = stdz(E); Nc, Nsd = stdz(N); Tc, Tsd = stdz(T)
    Es, Ns, Ts = (E - Ec) / Esd, (N - Nc) / Nsd, (T - Tc) / Tsd

    k = cfg.degree
    n_wells = len({(round(e, 3), round(nn, 3)) for e, nn in zip(E, N)})
    K1 = int(min(cfg.max_space_bases, max(k + 1, int(round(math.sqrt(n_wells))) + 1)))
    K2 = K1
    K3 = int(min(cfg.max_time_bases, max(k + 1, len(times))))
    while K1 * K2 * K3 > 0.7 * n and (K1 > k + 1 or K3 > k + 1):
        if K3 > k + 1 and K3 >= K1:
            K3 -= 1
        elif K1 > k + 1:
            K1 -= 1; K2 = K1
        else:
            break

    pad = 1e-6
    Erange = (Es.min() - pad, Es.max() + pad)
    Nrange = (Ns.min() - pad, Ns.max() + pad)
    Trange = (Ts.min() - pad, Ts.max() + pad)
    B1, _ = _bspline_basis(Es, *Erange, K1, k)
    B2, _ = _bspline_basis(Ns, *Nrange, K2, k)
    B3, _ = _bspline_basis(Ts, *Trange, K3, k)
    Bdes = (B1[:, :, None, None] * B2[:, None, :, None] * B3[:, None, None, :]
            ).reshape(n, K1 * K2 * K3)

    I1, I2, I3 = np.eye(K1), np.eye(K2), np.eye(K3)
    Ps1, Ps2, Ps3 = (_diff_penalty(K1, cfg.penalty_order),
                     _diff_penalty(K2, cfg.penalty_order),
                     _diff_penalty(K3, cfg.penalty_order))
    P1 = _kron3(Ps1, I2, I3); P2 = _kron3(I1, Ps2, I3); P3 = _kron3(I1, I2, Ps3)

    BtB = Bdes.T @ Bdes
    Bty = Bdes.T @ y
    eig0 = np.linalg.eigvalsh(P1 + P2 + P3)
    Mnull = int(np.sum(eig0 <= max(eig0) * 1e-8))

    def fit(rho):
        lam = np.exp(np.clip(np.asarray(rho, float), -8.0, 14.0))
        S = lam[0] * P1 + lam[1] * P2 + lam[2] * P3
        Mmat = BtB + S
        theta = np.linalg.solve(Mmat, Bty)
        return theta, S, Mmat, lam

    def reml(rho):
        theta, S, Mmat, lam = fit(rho)
        resid = y - Bdes @ theta
        rss = float(resid @ resid + theta @ S @ theta)
        sigma2 = rss / max(n - Mnull, 1)
        sign, logdetM = np.linalg.slogdet(Mmat)
        evS = np.linalg.eigvalsh(S)
        logdetS_plus = float(np.sum(np.log(evS[evS > max(evS) * 1e-8])))
        if sign <= 0 or not np.isfinite(logdetM):
            return 1e12
        return 0.5 * ((n - Mnull) * math.log(max(sigma2, 1e-12)) + logdetM - logdetS_plus)

    def gcv(rho):
        theta, S, Mmat, lam = fit(rho)
        resid = y - Bdes @ theta
        edf = float(np.trace(np.linalg.solve(Mmat, BtB)))
        denom = (n - edf) ** 2
        return n * float(resid @ resid) / denom if denom > 1e-9 else 1e12

    grid = [-2.0, 0.0, 2.0, 4.0]
    best, best_v = None, np.inf
    for r1 in grid:
        for r3 in grid:
            v = reml(np.array([r1, r1, r3]))
            if v < best_v:
                best_v, best = v, np.array([r1, r1, r3])
    res = minimize(reml, best, method="Nelder-Mead",
                   options=dict(maxiter=300, xatol=1e-2, fatol=1e-2))
    rho_hat = res.x
    theta, S, Mmat, lam = fit(rho_hat)
    resid = y - Bdes @ theta
    sigma2 = float((resid @ resid + theta @ S @ theta) / max(n - Mnull, 1))
    edf = float(np.trace(np.linalg.solve(Mmat, BtB)))
    gcv_reml = gcv(rho_hat)
    gcv_best, gcv_bv = None, np.inf
    for r1 in grid:
        for r3 in grid:
            g = gcv(np.array([r1, r1, r3]))
            if g < gcv_bv:
                gcv_bv, gcv_best = g, np.array([r1, r1, r3])
    gap = float(abs(rho_hat[0] - gcv_best[0]) + abs(rho_hat[2] - gcv_best[2]))

    return SplineFit(theta=theta, K1=K1, K2=K2, K3=K3, degree=k,
                     Ec=Ec, Esd=Esd, Nc=Nc, Nsd=Nsd, Tc=Tc, Tsd=Tsd,
                     Erange=Erange, Nrange=Nrange, Trange=Trange,
                     Mmat=Mmat, sigma2=sigma2, edf=edf, lam=lam, Mnull=Mnull,
                     gcv_at_reml=gcv_reml, reml_vs_gcv_gap=gap)


def _hull(E, N):
    locs = np.unique(np.column_stack([E, N]), axis=0)
    hull = ConvexHull(locs)
    return MplPath(locs[hull.vertices]), float(hull.volume)


def _theil_sen_slope(t, y):
    """Robust slope (median of pairwise slopes), so one ballooned timestep cannot drive the rate."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = len(t)
    sl = [(y[j] - y[i]) / (t[j] - t[i])
          for i in range(n) for j in range(i + 1, n) if t[j] != t[i]]
    return float(np.median(sl)) if sl else float("nan")


def _support_mask(gpts, well_xy, factor):
    """Boolean mask of grid points within factor * (median nearest-neighbour well spacing) of a
    well, plus the support radius. This keeps the mass integral on the data-supported region
    rather than the full convex hull, the primary ballooning defence (reference 3.9)."""
    from scipy.spatial import cKDTree
    wells = np.unique(well_xy, axis=0)
    tree = cKDTree(wells)
    if len(wells) > 1:
        dd, _ = tree.query(wells, k=2)
        nn = float(np.median(dd[:, 1]))
    else:
        nn = 0.0
    r = factor * nn if nn > 0 else float("inf")
    dist, _ = tree.query(gpts)
    return dist <= r, r


def mass_log_series(fit: SplineFit, gpts: np.ndarray, times: np.ndarray,
                    theta_vec: np.ndarray | None = None) -> np.ndarray:
    """ln M(t) over the supplied grid points for each time. phi*R*b cancels in k_M, so this is
    the relative plume mass. ``theta_vec`` overrides the fitted coefficients (posterior draws)."""
    theta = fit.theta if theta_vec is None else theta_vec
    Theta = theta.reshape(fit.K1 * fit.K2, fit.K3)
    Sgrid = fit.spatial_basis(gpts[:, 0], gpts[:, 1])
    tb_mat = fit.temporal_basis(times)
    out = []
    for bt in tb_mat:
        fgrid = Sgrid @ (Theta @ bt)
        out.append(math.log(float(np.sum(np.exp(fgrid)))))
    return np.array(out)


def concentration_decay_at(fit: SplineFit, points: dict, times) -> dict:
    """First-order CONCENTRATION decay coefficient (1/yr) at each named (E, N) location, read off
    the fitted surface: k = -slope of ln C(point, t) over time (the local rate field -df/dt,
    reference 3.8). Unlike the plume mass-loss rate k_M, this is a concentration-vs-time rate (the
    same apparent-attenuation estimand as Method 1 and McHugh's k_c-max), but smoothed across all
    wells and times rather than read from one noisy well. Uses a robust (Theil-Sen) slope.
    """
    t = np.asarray(times, float)
    out = {}
    for name, (E, N) in points.items():
        lnc = fit.predict_lnC(np.full(t.shape, float(E)), np.full(t.shape, float(N)), t)
        out[name] = -_theil_sen_slope(t, lnc)
    return out


def _plume_axis_points(site: SiteObservations) -> dict:
    """centre = source (peak); mid & edge sampled down the concentration-weighted principal axis
    of the detected wells. Mid/edge are data-poorer and serve as diagnostics only."""
    cx, cy = site.source_xy
    f = site.frame
    det = f[f.detect]
    g = det.groupby("well_id").agg(e=("easting", "first"), n=("northing", "first"),
                                   c=("conc", "max"))
    if len(g) < 3:
        return {"centre": (cx, cy)}
    pts = g[["e", "n"]].to_numpy(float); w = g["c"].to_numpy(float)
    d = pts - np.array([cx, cy]); W = w / w.sum()
    axis = np.linalg.eigh((d * W[:, None]).T @ d)[1][:, -1]
    proj = d @ axis
    lpos = proj[proj > 0].max() if (proj > 0).any() else 0.0
    lneg = -proj[proj < 0].min() if (proj < 0).any() else 0.0
    if lneg > lpos:
        axis = -axis; L = lneg
    else:
        L = lpos
    return {"centre": (cx, cy),
            "mid": (cx + 0.5 * L * axis[0], cy + 0.5 * L * axis[1]),
            "edge": (cx + 0.9 * L * axis[0], cy + 0.9 * L * axis[1])}


def estimate_site(site: SiteObservations, cfg: SplineConfig = SplineConfig()) -> RateEstimate:
    scope = "site"
    f = site.frame.copy()
    if len(f) == 0:
        return RateEstimate.not_applicable(METHOD_ST_PSPLINE, scope, "no observations")

    val = f.conc.to_numpy(float).copy()
    det = f.detect.to_numpy(bool)
    rl = f.rl.to_numpy(float)
    val[~det] = np.where(np.isfinite(rl[~det]), rl[~det] * cfg.nd_substitution, val[~det])
    val = np.maximum(val, 1e-9)
    y = np.log(val)
    E = f.easting.to_numpy(float); N = f.northing.to_numpy(float); T = f.t_years.to_numpy(float)

    n_wells = f.well_id.nunique()
    times = np.unique(np.round(T, 6))
    if n_wells < MIN_WELLS or len(times) < MIN_TIMES or len(f) < MIN_OBS:
        return RateEstimate.not_applicable(
            METHOD_ST_PSPLINE, scope,
            f"insufficient spatio-temporal coverage (wells={n_wells}, times={len(times)}, "
            f"obs={len(f)}; need {MIN_WELLS}/{MIN_TIMES}/{MIN_OBS})")

    fit = fit_surface(E, N, T, y, cfg)
    eval_times = times.copy()
    tlo, thi = float(eval_times.min()), float(eval_times.max())

    # spline-CENTRE concentration decay: k = -slope of ln C at the plume centre over time (years,
    # so k is 1/yr; the local rate field -df/dt). centre = source/peak; mid & edge are diagnostic.
    apts = _plume_axis_points(site)
    cx, cy = apts["centre"]
    local = concentration_decay_at(fit, apts, eval_times)     # robust Theil-Sen slopes, 1/yr
    k_centre = float(local["centre"])
    lnC_centre = fit.predict_lnC(np.full(eval_times.shape, cx), np.full(eval_times.shape, cy), eval_times)
    slope_ols = float(np.polyfit(eval_times, lnC_centre, 1)[0])

    # credible band on k_centre from posterior draws: theta ~ N(theta_hat, sigma2 Mmat^-1)
    from scipy.linalg import solve_triangular
    Sb_c = fit.spatial_basis([cx], [cy]); Tb = fit.temporal_basis(eval_times)

    def _centre_lnC(theta_vec):
        Theta = theta_vec.reshape(fit.K1 * fit.K2, fit.K3)
        return (Sb_c @ Theta @ Tb.T).ravel()

    rng = np.random.default_rng(cfg.seed)
    kc_draws = []
    try:
        R = np.linalg.cholesky(fit.Mmat).T
        s = math.sqrt(max(fit.sigma2, 0.0))
        for _ in range(cfg.n_posterior_draws):
            td = fit.theta + s * solve_triangular(R, rng.standard_normal(fit.theta.shape[0]), lower=False)
            kc_draws.append(-_theil_sen_slope(eval_times, _centre_lnC(td)))
    except (np.linalg.LinAlgError, ValueError):
        pass
    kc_draws = np.array(kc_draws)
    ci_lo = float(np.percentile(kc_draws, 5)) if kc_draws.size else float("nan")
    ci_hi = float(np.percentile(kc_draws, 95)) if kc_draws.size else float("nan")

    # data-anchored ballooning check: the smoothed centre decay must agree in SIGN with the raw
    # mean-concentration trend; a time-varying network can let the surface balloon at the centre.
    det = f[f.detect]
    prox = det.groupby(det.t_years.round(3)).conc.mean()
    k_data = (-_theil_sen_slope(prox.index.to_numpy(float), np.log(prox.to_numpy(float)))
              if len(prox) >= 3 else float("nan"))
    data_conflict = bool(np.isfinite(k_data) and k_centre * k_data < 0
                         and abs(k_centre) > 0.05 and abs(k_data) > 0.05)
    ballooning = bool(data_conflict
                      or abs(slope_ols - (-k_centre)) > 0.3 * max(abs(k_centre), 0.05))

    trend = "decreasing" if k_centre > 0 else "increasing"
    ci_excludes_zero = bool(kc_draws.size and np.isfinite(ci_lo) and np.isfinite(ci_hi)
                            and ci_lo * ci_hi > 0)
    conf = ("high" if (n_wells >= 12 and len(times) >= 8 and ci_excludes_zero and ci_lo > 0)
            else "medium" if (n_wells >= 8 and len(times) >= 6 and ci_excludes_zero) else "low")
    if ballooning and conf != "low":
        conf = "low"

    notes = ("spline-centre concentration decay: -slope of ln C at the plume centre on the "
             "REML-smoothed surface (an apparent attenuation rate that includes dilution; same "
             "estimand as Method 1, but smoothed across all wells and times). Robust Theil-Sen "
             "slope over years; non-detects at RL/2.")
    if fit.reml_vs_gcv_gap > 4:
        notes += (" REML and GCV smoothing differ; possible unmodeled spatio-temporal correlation "
                  "(B.7), interpret with care.")
    if data_conflict:
        notes += (f" BALLOONING: centre decay ({k_centre:.2f}/yr) disagrees in sign with the raw "
                  f"mean-concentration trend ({k_data:.2f}/yr); engineer review.")
    if not ci_excludes_zero:
        notes += " Credible interval crosses zero: decay not distinguishable from no change."

    DPY = 365.0     # MODFLOW day/year convention; value_per_year is 1/yr, also reported in 1/day
    diagnostics = dict(
        n_wells=int(n_wells), n_times=int(len(times)), n_obs=int(len(f)),
        bases=f"{fit.K1}x{fit.K2}x{fit.K3}", edf=round(fit.edf, 2),
        lambda_space_E=round(float(fit.lam[0]), 4), lambda_space_N=round(float(fit.lam[1]), 4),
        lambda_time=round(float(fit.lam[2]), 4), sigma2=round(fit.sigma2, 4),
        reml_vs_gcv_loglambda_gap=round(fit.reml_vs_gcv_gap, 2),
        centre_rate_per_year=round(k_centre, 5), centre_rate_per_day=round(k_centre / DPY, 8),
        mid_rate_per_year=(round(float(local["mid"]), 5) if "mid" in local else None),
        edge_rate_per_year=(round(float(local["edge"]), 5) if "edge" in local else None),
        centre_xy=[round(cx, 1), round(cy, 1)],
        ballooning_suspected=ballooning, data_conflict=data_conflict,
        k_data_proxy_per_yr=(round(float(k_data), 4) if np.isfinite(k_data) else None),
        ols_slope_per_yr=round(slope_ols, 5),
        time_span_years=round(thi - tlo, 2), nd_substitution="RL/2",
        note_mid_edge="mid/edge are data-poorer extrapolation; diagnostic only",
        qaqc=dict(
            eval_times_years=[round(float(t), 3) for t in eval_times.tolist()],
            lnC_centre=[round(float(v), 4) for v in lnC_centre.tolist()],
            fit_slope=round(-k_centre, 5),
            fit_intercept=round(float(np.median(lnC_centre) + k_centre * np.median(eval_times)), 4),
            k_draws_per_year=[round(float(v), 5) for v in kc_draws[:200].tolist()],
        ),
    )
    return RateEstimate(
        method=METHOD_ST_PSPLINE, estimand=METHOD_ESTIMAND[METHOD_ST_PSPLINE], scope=scope,
        value_per_year=float(k_centre), ci_low=ci_lo, ci_high=ci_hi,
        half_life_years=RateEstimate.half_life(k_centre), n=int(len(f)), trend=trend,
        confidence=conf, removes_dilution=False, diagnostics=diagnostics, notes=notes,
    )
