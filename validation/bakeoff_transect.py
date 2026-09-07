"""Bake-off: Bockelmann's two-control-plane rate, computed from a smoothed surface instead of
from pumped wells.

Bockelmann et al. (2001, J. Contam. Hydrol. 53:429-453, eq. 11) estimate a first-order effective
natural-attenuation rate from the change in contaminant mass flux between two control planes:

    k = -(1 / (R * dt)) * ln( MF_CP-II / MF_CP-I )

Their mass fluxes are MEASURED by pumping: the integral pumping test physically integrates over
the capture zone. This harness asks whether the same calculation survives when the flux is instead
read off a P-spline surface fitted to ordinary monitoring wells, i.e. in a NON-PUMPED system.

Assume a uniform Darcy flux across both planes and q, porosity and thickness cancel in the ratio,
leaving the ratio of transverse CONCENTRATION integrals:

    lambda = -v_c * d ln I(x) / dx ,    I(x) = integral over y of C(x, y)

Why this is worth testing at all: for a Domenico plume the transverse integral of Phi_y is
CONSERVED with distance, because transverse dispersion moves mass sideways within the plane
rather than removing it. Integrating across the plane therefore cancels alpha_y exactly, instead
of dividing it out with an assumed value the way Method 2 does. Method 2's dominant uncertainty is
dispersivity (Stenback 2004: one order of magnitude in dispersivity moves the rate threefold), so
an estimator that does not need alpha_y at all would be a real gain.

Three parts, each isolating one error source:

  A. ANALYTICAL field, wide integration span. Integrates the true C(x,y). This must be near-exact
     at every k and every alpha_y, or the formula above is wrong and B and C mean nothing.
  B. ANALYTICAL field, TRUNCATED integration span. Isolates the error from a transect that stops
     before the plume does. Nothing is fitted here, so whatever appears is truncation alone.
  C. WELLS -> P-spline surface -> transects, against Method 2's 1D and 2D variants on identical
     data, sweeping the number of wells per transect. This is the realistic case.

Run:  python validation/bakeoff_transect.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.special import erf

from biodeg_rates import method2_domenico as m2
from biodeg_rates import method3_spline as m3
from biodeg_rates.dataio import site_from_long as _site_from_long, DAYS_PER_YEAR as DAYS

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")

VC = 100.0          # contaminant velocity, m/yr
Y = 10.0            # source width, m
C0 = 5.0            # source concentration
XS = np.arange(20, 201, 20.0)       # control planes, m downgradient
K_TRUE = (0.1, 0.3, 0.6)            # 1/yr
AY_TRUE = (0.5, 3.0)                # transverse dispersivity, m


# ------------------------------------------------------------------ the synthetic plume (truth)
def phi_y(x, y, ay):
    """Domenico transverse spreading factor. Its integral over y is Y for every x, which is the
    property the transect method exploits."""
    d = 2.0 * np.sqrt(ay * np.maximum(x, 1e-9))
    return 0.5 * (erf((y + Y / 2) / d) - erf((y - Y / 2) / d))


def true_C(x, y, k, ay):
    """Steady 2D plume with pure advective decay (no longitudinal dispersion, so alpha_x = 0 and
    lambda = -m * v_c exactly)."""
    return C0 * phi_y(x, y, ay) * np.exp(-k * x / VC)


def plume_halfwidth(ay, x=None):
    """Roughly where Phi_y has died away at the furthest control plane."""
    return 2.0 * np.sqrt(ay * (XS.max() if x is None else x)) + Y / 2


def lam_from_transects(xs, I, vc=VC, alpha_x=0.0):
    """Bockelmann eq. 11 in its concentration-integral form, fitted over all planes at once.

    With longitudinal dispersion the Buscheck-Alcantar relation gives lambda = -m*v*(1 - m*alpha_x);
    alpha_x = 0 is the pure-advection case the synthetic plume is built on. Note that alpha_x
    enters only at SECOND order here, whereas alpha_y enters Method 2's normalization at first
    order. That asymmetry is the point of the method.
    """
    m = np.polyfit(np.asarray(xs, float), np.log(np.asarray(I, float)), 1)[0]
    return -m * vc * (1.0 - m * alpha_x)


# ------------------------------------------------------------------------------------- part A
def run_analytical():
    print("=== PART A: analytical field, wide span (does the formula hold at all?) ===")
    rows = []
    for k in K_TRUE:
        for ay in AY_TRUE:
            yy = np.linspace(-400, 400, 8001)
            I = np.array([np.trapezoid(true_C(x, yy, k, ay), yy) for x in XS])
            lam = lam_from_transects(XS, I)
            rows.append(dict(k_true=k, ay_true=ay, lam=round(lam, 6),
                             err_pct=round(100 * (lam - k) / k, 2)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    worst = float(np.max(np.abs(df.err_pct)))
    print(f"\n  worst |error| = {worst:.2f}%   "
          f"(the transverse integral cancels alpha_y: the two ay columns agree)")
    return df, worst


# ------------------------------------------------------------------------------------- part B
def run_truncation():
    print("\n=== PART B: analytical field, TRUNCATED span (transect narrower than the plume) ===")
    spans = (45, 90, 200, 400)
    rows = []
    for k in K_TRUE:
        for ay in AY_TRUE:
            rec = dict(k_true=k, ay_true=ay, plume_halfwidth_m=round(plume_halfwidth(ay), 0))
            for span in spans:
                yy = np.linspace(-span, span, 4001)
                I = np.array([np.trapezoid(true_C(x, yy, k, ay), yy) for x in XS])
                rec[f"err_span{span}"] = round(100 * (lam_from_transects(XS, I) - k) / k, 1)
            rows.append(rec)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n  Truncation error is always POSITIVE: a transect that stops short of the plume edge")
    print("  OVERSTATES degradation, which is the direction that flatters a site.")
    return df


# ------------------------------------------------------------------------------------- part C
def _wells(k, ay, n_y, n_times=6, noise=0.08, seed=0):
    """Wells on a transverse grid spanning the plume, sampled at several times. The plume is
    steady, so the repeated events carry independent noise and no temporal trend."""
    rng = np.random.default_rng(seed)
    hw = plume_halfwidth(ay)
    ys = np.linspace(-hw, hw, n_y)
    recs = []
    for t in range(n_times):
        for x in XS:
            for y in ys:
                c = true_C(x, y, k, ay) * float(np.exp(rng.normal(0, noise)))
                recs.append((f"W{int(x)}_{y:.0f}", 500000.0 + x, 5800000.0 + y,
                             float(t), max(c, 1e-9), True))
    return pd.DataFrame(recs, columns=["well_id", "easting", "northing", "t_years",
                                       "conc", "detect"]), hw


def _transect_lambda(site, hw):
    """Fit the surface once, then integrate it across each control plane at each observed time.
    Returns the median per-time lambda and the spread across times (a steady-state diagnostic:
    on a steady plume the per-time values should agree)."""
    prep, why = m3._prepare(site, m3.SplineConfig())
    if prep is None:
        return float("nan"), float("nan"), why
    yspan = np.linspace(-hw, hw, 241)
    lams = []
    for t in prep.eval_times:
        I = [np.trapezoid(np.exp(prep.fit.predict_lnC(
                np.full(yspan.shape, 500000.0 + x), 5800000.0 + yspan,
                np.full(yspan.shape, float(t)))), yspan) for x in XS]
        lams.append(lam_from_transects(XS, np.array(I)))
    return float(np.median(lams)), float(np.ptp(lams)), ""


N_SEEDS = 5     # a single draw ranks these estimators unstably; average over noise realisations


def run_wells(n_seeds: int = N_SEEDS):
    """Every configuration is run over several noise realisations. A single seed is NOT enough:
    an earlier one-seed version of this harness ranked the transect estimator ahead of the 2D fit,
    and a different seed reversed it. The per-seed standard deviation is reported so the reader
    can see whether any ranking is real or is noise."""
    print(f"\n=== PART C: wells -> spline -> transects, vs Method 2 "
          f"(identical data, {n_seeds} seeds per case) ===")
    rows = []
    for k in K_TRUE:
        for ay in AY_TRUE:
            for n_y in (5, 9, 15):
                per_seed = {"1d": [], "2d": [], "transect": [], "abs": [], "spread": []}
                for s in range(n_seeds):
                    df_w, hw = _wells(k, ay, n_y, seed=1000 * s + int(k * 100 + ay * 10 + n_y))
                    site = _site_from_long(df_w)
                    cfg2 = m2.DomenicoConfig(vc_override=VC / DAYS, source_width_override=Y)
                    e1 = m2.estimate_site(site, cfg=cfg2)
                    e2 = m2.estimate_site_2d(site, cfg=cfg2)
                    lam_tr, spread, why = _transect_lambda(site, hw)
                    if why:
                        continue
                    for tag, v in (("1d", e1.value_per_year), ("2d", e2.value_per_year),
                                   ("transect", lam_tr)):
                        per_seed[tag].append(100 * (v - k) / k if np.isfinite(v) else np.nan)
                    per_seed["abs"].append(lam_tr - k)
                    per_seed["spread"].append(spread)
                if not per_seed["transect"]:
                    print(f"  k={k} ay={ay} n_y={n_y}: skipped (no usable seed)")
                    continue
                rows.append(dict(
                    k_true=k, ay_true=ay, wells_per_transect=n_y,
                    y_spacing_m=round(2 * hw / (n_y - 1), 1),
                    err_1d_pct=round(float(np.nanmean(per_seed["1d"])), 1),
                    err_2d_pct=round(float(np.nanmean(per_seed["2d"])), 1),
                    err_transect_pct=round(float(np.nanmean(per_seed["transect"])), 1),
                    err_transect_sd=round(float(np.nanstd(per_seed["transect"])), 1),
                    err_transect_abs=round(float(np.mean(per_seed["abs"])), 4),
                    lam_spread_over_time=round(float(np.mean(per_seed["spread"])), 4)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    e = {c: float(np.nanmean(np.abs(pd.to_numeric(df["err_" + c + "_pct"], errors="coerce"))))
         for c in ("1d", "2d", "transect")}
    print(f"\n  mean |error| in k:  1D {e['1d']:.0f}%   2D {e['2d']:.0f}%   "
          f"transect {e['transect']:.0f}%")
    print(f"  mean per-seed SD of the transect error: {np.nanmean(df.err_transect_sd):.0f} "
          f"percentage points")
    return df, e


def main():
    os.makedirs(OUT, exist_ok=True)
    a, worst = run_analytical()
    b = run_truncation()
    c, e = run_wells()
    a.to_csv(os.path.join(OUT, "bakeoff_transect_analytical.csv"), index=False)
    b.to_csv(os.path.join(OUT, "bakeoff_transect_truncation.csv"), index=False)
    c.to_csv(os.path.join(OUT, "bakeoff_transect_wells.csv"), index=False)

    print("\n================= VERDICT =================")
    if worst > 1.0:
        print(f"PART A FAILED: the formula does not hold on the analytical field "
              f"(worst |err| {worst:.2f}%). Parts B and C are meaningless until this passes.")
        print("outputs:", os.path.abspath(OUT))
        return 1

    print(f"A. The formula is exact on the analytical field (worst |err| {worst:.2f}%), at every")
    print("   k and every alpha_y. Transverse integration cancels alpha_y, as intended.")
    print("B. A transect narrower than the plume overstates the rate, badly and always upward.")

    # the ranking is DERIVED, never asserted: an earlier version hardcoded "transect wins" and a
    # change of seed falsified it while the sentence kept printing.
    order = sorted(e, key=lambda kk: e[kk])
    sd = float(np.nanmean(c.err_transect_sd))
    gap = e[order[1]] - e[order[0]]
    names = {"1d": "Method 2 1D", "2d": "Method 2 2D", "transect": "the transect estimate"}
    print(f"C. Accuracy through a fitted surface, best first: "
          + ", ".join(f"{names[o]} {e[o]:.0f}%" for o in order) + ".")
    if gap < sd:
        print(f"   The gap between the best two ({gap:.0f} points) is smaller than the per-seed")
        print(f"   scatter ({sd:.0f} points), so THIS RUN DOES NOT SEPARATE THEM. Do not report")
        print("   a winner from this harness without more seeds or more configurations.")
    else:
        print(f"   The gap between the best two ({gap:.0f} points) exceeds the per-seed scatter")
        print(f"   ({sd:.0f} points), so the ordering is not purely noise.")
    print("   Either way the estimate is far from Part A's exactness, so the SURFACE, not the")
    print("   formula, carries the error.")

    ab = pd.to_numeric(c.err_transect_abs, errors="coerce").to_numpy(float)
    by_k = c.groupby("k_true").err_transect_abs.mean()
    if np.all(ab > 0):
        print(f"\n   The residual error is a POSITIVE, near-constant ADDITIVE offset: mean "
              f"{np.mean(ab):+.3f}/yr")
        print("   across a sixfold range in the true rate "
              + ", ".join(f"k={kk:.1f}: {vv:+.3f}" for kk, vv in by_k.items()) + ".")
        print("   That is the same signature as the bias measured in method3's k_mass, and both")
        print("   quantities integrate exp() of a log-space smoothed surface. Mechanism unproven.")

    print("\n   NOT PRODUCTION-READY. Teutsch et al. (2000) built the integral pumping test")
    print("   precisely to avoid regionalising point concentrations; pumping makes the INTEGRAL")
    print("   robust while leaving the transverse distribution ambiguous, and a spline does the")
    print("   opposite. Bockelmann eq. 11 consumes the integral, which is the half a spline is")
    print("   weakest at. Resolve the additive bias before promoting this to a method.")
    print("outputs:", os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
