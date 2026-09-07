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
                       METHOD_ST_PSPLINE, METHOD_ST_PSPLINE_MASS, METHOD_ESTIMAND)

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


def mass_series_bases(fit: SplineFit, gpts: np.ndarray, times: np.ndarray):
    """Precompute the (spatial, temporal) bases ``mass_log_series`` needs, so a loop of posterior
    draws over the same grid and times does not rebuild them on every draw."""
    return fit.spatial_basis(gpts[:, 0], gpts[:, 1]), fit.temporal_basis(times)


def mass_log_series(fit: SplineFit, gpts: np.ndarray, times: np.ndarray,
                    theta_vec: np.ndarray | None = None, bases=None) -> np.ndarray:
    """ln M(t) over the supplied grid points for each time. phi*R*b and the constant cell area
    cancel in the slope, so this is the relative plume mass. ``theta_vec`` overrides the fitted
    coefficients (posterior draws); ``bases`` accepts the output of ``mass_series_bases``."""
    theta = fit.theta if theta_vec is None else theta_vec
    Theta = theta.reshape(fit.K1 * fit.K2, fit.K3)
    Sgrid, tb_mat = bases if bases is not None else mass_series_bases(fit, gpts, times)
    out = []
    for bt in tb_mat:
        fgrid = Sgrid @ (Theta @ bt)
        # exp of a ballooned surface can overflow; log-sum-exp keeps ln M(t) finite so the caller
        # sees a usable number (or a flagged one) rather than an inf that silently poisons a slope.
        mx = float(np.max(fgrid))
        out.append(mx + math.log(float(np.sum(np.exp(fgrid - mx)))))
    return np.array(out)


def mass_grid(E, N, cfg: SplineConfig = SplineConfig()):
    """Quadrature points for the mass integral: a ``cfg.grid_n`` square grid clipped to the convex
    hull of the wells, then (unless ``mask_support_factor`` is None) restricted to points within
    ``mask_support_factor`` nearest-neighbour well spacings of a well.

    The support restriction is the primary ballooning defence for this estimand (reference 3.9):
    the smoother is unconstrained in the corners of the hull where no well has ever been sampled,
    and an integral taken over those corners can rise while every measured concentration falls.

    Returns (gpts, info) where info records what the mask removed.
    """
    E = np.asarray(E, float); N = np.asarray(N, float)
    grid, hull_area = _hull(E, N)
    gx = np.linspace(E.min(), E.max(), cfg.grid_n)
    gy = np.linspace(N.min(), N.max(), cfg.grid_n)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    gpts = pts[grid.contains_points(pts)]
    n_hull = int(gpts.shape[0])
    r_support = None
    if cfg.mask_support_factor is not None and n_hull:
        mask, r_support = _support_mask(gpts, np.column_stack([E, N]), cfg.mask_support_factor)
        gpts = gpts[mask]
    info = dict(n_grid_hull=n_hull, n_grid_supported=int(gpts.shape[0]),
                support_radius_m=(round(float(r_support), 1) if r_support is not None
                                  and np.isfinite(r_support) else None),
                hull_area_m2=round(float(hull_area), 1),
                mask_support_factor=cfg.mask_support_factor)
    return gpts, info


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


@dataclass(frozen=True)
class _Prepared:
    """One fitted surface plus the arrays both module-3 estimands read from it."""
    fit: SplineFit
    f: "object"                 # the observation frame (pandas)
    E: np.ndarray
    N: np.ndarray
    times: np.ndarray
    eval_times: np.ndarray
    n_wells: int


def _prepare(site: SiteObservations, cfg: SplineConfig):
    """Shared preamble: substitute non-detects, check coverage, fit the surface ONCE.

    Both module-3 estimands read the same surface, so this exists to keep a caller that wants
    both from paying for two REML fits. Returns (_Prepared, None) or (None, reason).
    """
    f = site.frame.copy()
    if len(f) == 0:
        return None, "no observations"

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
        return None, (f"insufficient spatio-temporal coverage (wells={n_wells}, "
                      f"times={len(times)}, obs={len(f)}; need "
                      f"{MIN_WELLS}/{MIN_TIMES}/{MIN_OBS})")

    fit = fit_surface(E, N, T, y, cfg)
    return _Prepared(fit=fit, f=f, E=E, N=N, times=times, eval_times=times.copy(),
                     n_wells=int(n_wells)), None


def _data_proxy_rate(f) -> float:
    """Raw mean-concentration trend over detected observations (1/yr), the data anchor that both
    module-3 estimands are sign-checked against. Nothing smoothed enters this number."""
    det = f[f.detect]
    prox = det.groupby(det.t_years.round(3)).conc.mean()
    if len(prox) < 3:
        return float("nan")
    return -_theil_sen_slope(prox.index.to_numpy(float), np.log(prox.to_numpy(float)))


def _wells_per_time(f) -> tuple[int, int]:
    """Smallest and largest number of wells sampled in any one event. A network that grows or
    shrinks over the record is the documented cause of surface ballooning (3.9), and it hits the
    mass integral harder than a point rate, so both estimands report it."""
    g = f.groupby(f.t_years.round(3)).well_id.nunique()
    return (int(g.min()), int(g.max())) if len(g) else (0, 0)


def _posterior_draws(fit: SplineFit, cfg: SplineConfig, statistic) -> np.ndarray:
    """Sample theta ~ N(theta_hat, sigma2 Mmat^-1) and apply ``statistic`` to each draw.
    Non-finite draws are dropped rather than allowed to poison a percentile."""
    from scipy.linalg import solve_triangular
    rng = np.random.default_rng(cfg.seed)
    out = []
    try:
        R = np.linalg.cholesky(fit.Mmat).T
        s = math.sqrt(max(fit.sigma2, 0.0))
        for _ in range(cfg.n_posterior_draws):
            td = fit.theta + s * solve_triangular(
                R, rng.standard_normal(fit.theta.shape[0]), lower=False)
            v = statistic(td)
            if np.isfinite(v):
                out.append(float(v))
    except (np.linalg.LinAlgError, ValueError):
        pass
    return np.array(out)


def estimate_site(site: SiteObservations, cfg: SplineConfig = SplineConfig()) -> RateEstimate:
    """Module 3, estimand 1: the concentration decay at the plume centre (k_centre)."""
    prep, reason = _prepare(site, cfg)
    if prep is None:
        return RateEstimate.not_applicable(METHOD_ST_PSPLINE, "site", reason)
    return _centre_estimate(site, prep, cfg)


def _centre_estimate(site: SiteObservations, prep: _Prepared,
                     cfg: SplineConfig) -> RateEstimate:
    scope = "site"
    fit, f, times = prep.fit, prep.f, prep.times
    n_wells = prep.n_wells
    eval_times = prep.eval_times
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
    Sb_c = fit.spatial_basis([cx], [cy]); Tb = fit.temporal_basis(eval_times)

    def _centre_lnC(theta_vec):
        Theta = theta_vec.reshape(fit.K1 * fit.K2, fit.K3)
        return (Sb_c @ Theta @ Tb.T).ravel()

    kc_draws = _posterior_draws(fit, cfg,
                                lambda td: -_theil_sen_slope(eval_times, _centre_lnC(td)))
    ci_lo = float(np.percentile(kc_draws, 5)) if kc_draws.size else float("nan")
    ci_hi = float(np.percentile(kc_draws, 95)) if kc_draws.size else float("nan")

    # data-anchored ballooning check: the smoothed centre decay must agree in SIGN with the raw
    # mean-concentration trend; a time-varying network can let the surface balloon at the centre.
    k_data = _data_proxy_rate(f)
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


MIN_SUPPORTED_GRID = 25     # quadrature points needed before a mass integral means anything


def estimate_site_mass(site: SiteObservations, cfg: SplineConfig = SplineConfig()) -> RateEstimate:
    """Module 3, estimand 2: the plume MASS-loss decay (k_mass).

    k_mass = -d ln M(t) / dt, where M(t) is the fitted concentration surface integrated over the
    data-supported plume footprint at time t. Because the integral is taken over space, lateral
    spreading inside the footprint does not by itself lower M(t): spreading moves mass, it does
    not destroy it. That makes k_mass less confounded than the point rate k_centre, and it is the
    reason a spatially integrated rate is the more informative of the two when the network
    supports it.

    It is still NOT a dilution-removed reaction coefficient (Method 2). Mass advected out across
    the footprint boundary also lowers M(t), and continuing source dissolution raises it, so
    ``removes_dilution`` is False and the handoff does not offer k_mass as the mechanistic
    MODFLOW seed.
    """
    prep, reason = _prepare(site, cfg)
    if prep is None:
        return RateEstimate.not_applicable(METHOD_ST_PSPLINE_MASS, "site", reason)
    return _mass_estimate(site, prep, cfg)


def _mass_estimate(site: SiteObservations, prep: _Prepared, cfg: SplineConfig) -> RateEstimate:
    scope = "site"
    fit, f, times = prep.fit, prep.f, prep.times
    n_wells = prep.n_wells
    eval_times = prep.eval_times
    tlo, thi = float(eval_times.min()), float(eval_times.max())

    gpts, ginfo = mass_grid(prep.E, prep.N, cfg)
    if gpts.shape[0] < MIN_SUPPORTED_GRID:
        return RateEstimate.not_applicable(
            METHOD_ST_PSPLINE_MASS, scope,
            f"too few data-supported quadrature points for a mass integral "
            f"({gpts.shape[0]} of {ginfo['n_grid_hull']} in-hull points survive the support "
            f"mask; need {MIN_SUPPORTED_GRID})")

    bases = mass_series_bases(fit, gpts, eval_times)
    lnM = mass_log_series(fit, gpts, eval_times, bases=bases)
    if not np.all(np.isfinite(lnM)):
        return RateEstimate.not_applicable(
            METHOD_ST_PSPLINE_MASS, scope, "ln M(t) is not finite over the fitted surface")

    slope_ols = float(np.polyfit(eval_times, lnM, 1)[0])
    slope_robust = _theil_sen_slope(eval_times, lnM)
    k_mass = -(slope_robust if cfg.robust_slope else slope_ols)

    kM_draws = _posterior_draws(
        fit, cfg,
        lambda td: -_theil_sen_slope(
            eval_times, mass_log_series(fit, gpts, eval_times, theta_vec=td, bases=bases)))
    ci_lo = float(np.percentile(kM_draws, 5)) if kM_draws.size else float("nan")
    ci_hi = float(np.percentile(kM_draws, 95)) if kM_draws.size else float("nan")

    # ballooning defences, in the same order as the centre estimand (reference 3.9). The mass
    # integral is the quantity ballooning was first observed on, so the support mask above is the
    # first defence, the robust slope is the second, and these two checks are the third.
    k_data = _data_proxy_rate(f)
    data_conflict = bool(np.isfinite(k_data) and k_mass * k_data < 0
                         and abs(k_mass) > 0.05 and abs(k_data) > 0.05)
    ballooning = bool(data_conflict
                      or abs(slope_ols - (-k_mass)) > 0.3 * max(abs(k_mass), 0.05))

    w_min, w_max = _wells_per_time(f)
    network_drift = bool(w_max > 0 and (w_max - w_min) / w_max > 0.5)

    # Support-radius sweep, the analogue of Method 2's dispersivity sweep: k_mass depends on where
    # the footprint boundary is drawn, and on a real network that dependence can be large. Report
    # it rather than choosing a radius and hiding the choice. No refit is needed, only re-integration.
    sweep = {}
    for fac in (1.0, 2.0, 3.0, None):
        g2, i2 = mass_grid(prep.E, prep.N, SplineConfig(**{**cfg.__dict__,
                                                          "mask_support_factor": fac}))
        if g2.shape[0] < MIN_SUPPORTED_GRID:
            continue
        lnM2 = mass_log_series(fit, g2, eval_times)
        if not np.all(np.isfinite(lnM2)):
            continue
        sweep[("full_hull" if fac is None else f"factor_{fac:g}")] = dict(
            n_grid=int(g2.shape[0]),
            k_mass_per_year=round(-_theil_sen_slope(eval_times, lnM2), 5))
    sweep_vals = [v["k_mass_per_year"] for v in sweep.values()
                  if v["k_mass_per_year"] is not None and np.isfinite(v["k_mass_per_year"])]
    sweep_spread = (round(max(sweep_vals) - min(sweep_vals), 5) if len(sweep_vals) > 1 else None)
    boundary_sensitive = bool(sweep_spread is not None and np.isfinite(k_mass)
                              and abs(k_mass) > 0
                              and sweep_spread > 0.25 * abs(k_mass))

    trend = "decreasing" if k_mass > 0 else "increasing"
    ci_excludes_zero = bool(kM_draws.size and np.isfinite(ci_lo) and np.isfinite(ci_hi)
                            and ci_lo * ci_hi > 0)
    conf = ("high" if (n_wells >= 12 and len(times) >= 8 and ci_excludes_zero and ci_lo > 0)
            else "medium" if (n_wells >= 8 and len(times) >= 6 and ci_excludes_zero) else "low")
    if ballooning and conf != "low":
        conf = "low"
    if cfg.mask_support_factor is None and conf == "high":
        conf = "medium"     # integrating the full hull is the ungirded configuration

    notes = ("plume mass-loss decay: -slope of ln M(t), the REML-smoothed surface integrated over "
             "the data-supported footprint. Spatially integrated, so lateral spreading inside the "
             "footprint does not register as decay; mass advected across the footprint boundary "
             "and continuing source dissolution still do, so this is NOT a dilution-removed "
             "reaction coefficient. Robust Theil-Sen slope over years; non-detects at RL/2.")
    if fit.reml_vs_gcv_gap > 4:
        notes += (" REML and GCV smoothing differ; possible unmodeled spatio-temporal correlation "
                  "(B.7), interpret with care.")
    if data_conflict:
        notes += (f" BALLOONING: mass decay ({k_mass:.2f}/yr) disagrees in sign with the raw "
                  f"mean-concentration trend ({k_data:.2f}/yr); engineer review.")
    if network_drift:
        notes += (f" Monitoring network changes over the record ({w_min} to {w_max} wells per "
                  f"event); a spatial integral is sensitive to that, interpret with care.")
    if boundary_sensitive:
        notes += (f" BOUNDARY SENSITIVE: k_mass moves by {sweep_spread:.3f}/yr across support "
                  f"radii (see support_radius_sweep), more than a quarter of the estimate; the "
                  f"footprint boundary, not the data, is carrying the rate.")
    if not ci_excludes_zero:
        notes += " Credible interval crosses zero: decay not distinguishable from no change."
    notes += (" Known bias: on separable synthetic plumes with a known rate this estimator runs "
              "about +0.009/yr high regardless of the true rate, which is under 3% at 0.4/yr but "
              "close to 20% at 0.05/yr. The credible interval does not cover that offset.")

    DPY = 365.0     # MODFLOW day/year convention; value_per_year is 1/yr, also reported in 1/day
    diagnostics = dict(
        n_wells=int(n_wells), n_times=int(len(times)), n_obs=int(len(f)),
        bases=f"{fit.K1}x{fit.K2}x{fit.K3}", edf=round(fit.edf, 2),
        lambda_space_E=round(float(fit.lam[0]), 4), lambda_space_N=round(float(fit.lam[1]), 4),
        lambda_time=round(float(fit.lam[2]), 4), sigma2=round(fit.sigma2, 4),
        reml_vs_gcv_loglambda_gap=round(fit.reml_vs_gcv_gap, 2),
        mass_rate_per_year=round(k_mass, 5), mass_rate_per_day=round(k_mass / DPY, 8),
        ballooning_suspected=ballooning, data_conflict=data_conflict,
        k_data_proxy_per_yr=(round(float(k_data), 4) if np.isfinite(k_data) else None),
        ols_slope_per_yr=round(slope_ols, 5),
        robust_slope_per_yr=(round(float(slope_robust), 5)
                             if np.isfinite(slope_robust) else None),
        slope_estimator=("theil_sen" if cfg.robust_slope else "ols"),
        wells_per_event_min=w_min, wells_per_event_max=w_max, network_drift=network_drift,
        support_radius_sweep=sweep, sweep_spread_per_year=sweep_spread,
        boundary_sensitive=boundary_sensitive,
        known_additive_bias_per_year=0.009,
        time_span_years=round(thi - tlo, 2), nd_substitution="RL/2",
        **ginfo,
        qaqc=dict(
            eval_times_years=[round(float(t), 3) for t in eval_times.tolist()],
            lnM=[round(float(v), 4) for v in lnM.tolist()],
            fit_slope=round(-k_mass, 5),
            fit_intercept=round(float(np.median(lnM) + k_mass * np.median(eval_times)), 4),
            k_draws_per_year=[round(float(v), 5) for v in kM_draws[:200].tolist()],
        ),
    )
    return RateEstimate(
        method=METHOD_ST_PSPLINE_MASS, estimand=METHOD_ESTIMAND[METHOD_ST_PSPLINE_MASS],
        scope=scope, value_per_year=float(k_mass), ci_low=ci_lo, ci_high=ci_hi,
        half_life_years=RateEstimate.half_life(k_mass), n=int(len(f)), trend=trend,
        confidence=conf, removes_dilution=False, diagnostics=diagnostics, notes=notes,
    )


def estimate_site_both(site: SiteObservations,
                       cfg: SplineConfig = SplineConfig()) -> tuple[RateEstimate, RateEstimate]:
    """Both module-3 estimands from ONE surface fit: (k_centre estimate, k_mass estimate).

    Prefer this over calling ``estimate_site`` and ``estimate_site_mass`` separately, which fits
    the REML surface twice. The two results are different quantities and are never combined.
    """
    prep, reason = _prepare(site, cfg)
    if prep is None:
        return (RateEstimate.not_applicable(METHOD_ST_PSPLINE, "site", reason),
                RateEstimate.not_applicable(METHOD_ST_PSPLINE_MASS, "site", reason))
    return _centre_estimate(site, prep, cfg), _mass_estimate(site, prep, cfg)
