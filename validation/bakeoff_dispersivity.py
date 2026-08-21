"""Bake-off: does feeding Method 2 the site-initialization dispersivity improve it?

NOTE: this harness CANNOT RUN in this repository. Both parts depend on the
site-initialization ContinuousDecay model, which is not included here. It is kept
as a record of what was tested and how, not as a runnable check.

Method 2 currently sets its longitudinal dispersivity by the scaling rule alpha_x = 0.1 * L
(and alpha_y = alpha_x / 10). The site-initialization anisotropic decay model (ContinuousDecay,
which is NOT part of this repository) ALSO yields a longitudinal dispersivity, alpha_L = 1/(2 k(theta)
ln10), from the fitted decay slope. This harness asks whether using that alpha_L in Method 2
recovers the true rate better, or whether the difference is trivial.

It does NOT modify production. It only calls method2.estimate_site with the existing
alpha_x_override knob set to the site-init alpha_L, and compares to the default.

Two parts:
  A. Synthetic plumes with a KNOWN k (and known transverse dispersivity). Compares k recovery
     for alpha_x in {0.1*L (current), site-init alpha_L, oracle}. Tells us if it IMPROVES accuracy.
  B. Field sites. Compares the lambda Method 2 reports with 0.1*L vs site-init alpha_L.
     Tells us if the change is MATERIAL or trivial in practice.

Run:  python validation/bakeoff_dispersivity.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.special import erf

from biodeg_rates import dataio
from biodeg_rates import method2_domenico as m2
from biodeg_rates.dataio import site_from_long as _site_from_long, DAYS_PER_YEAR as DAYS

try:
    from site_sizing import ContinuousDecay      # NOT part of this repository
    HAVE_SITEINIT = True
except Exception as e:        # pragma: no cover
    HAVE_SITEINIT = False
    _IMPORT_ERR = str(e)

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
SITE_ROOT_DEFAULT = ""      # set SITE_ROOT to a folder of site subfolders


def _hf_from_site(site):
    """Per-well max-over-time envelope (what the site-init decay model consumes)."""
    f = site.frame
    g = f.groupby("well_id").agg(easting=("easting", "first"), northing=("northing", "first"),
                                 cmax=("conc", "max"), anydet=("detect", "any"))
    x = g.easting.to_numpy(float); y = g.northing.to_numpy(float)
    c = g.cmax.to_numpy(float); det = g.anydet.to_numpy(bool)
    return dict(x=x, y=y, c=c, det=det, hotspot=site.source_xy,
                span=float(max(np.ptp(x), np.ptp(y)) or 1.0))


def site_init_alpha_L(site, flow_azimuth_deg=None):
    """alpha_L from the site-init ContinuousDecay (median, and at the flow bearing if given)."""
    hf = _hf_from_site(site)
    try:
        m = ContinuousDecay(hf, {"guideline": site.threshold or 0.005})
    except Exception:
        return None, None
    aL_med = float(m.alpha_L_median) if np.isfinite(m.alpha_L_median) else None
    aL_az = None
    if flow_azimuth_deg is not None:
        v = m.alpha_L(flow_azimuth_deg)
        aL_az = float(v) if np.isfinite(v) else None
    return aL_med, aL_az


# ------------------------------------------------------------------ synthetic (known truth)
def synth_long(k_true, ay_true, vc=100.0, Y=10.0, C0=1.0, noise=0.05, seed=0):
    """Domenico-equivalent directional plume: C = C0 * Phi_y(x; ay_true) * exp(-k x/vc).
    Centerline + off-axis wells along +easting (flow). vc, k in /yr, distances in m."""
    rng = np.random.default_rng(seed)
    xs = np.arange(20, 261, 20.0)
    ys_off = [0.0, 0.0, 0.0, 15.0, -15.0, 25.0, -25.0]
    rows = []
    i = 0
    for x in xs:
        for y in (ys_off[i % len(ys_off)], 0.0):
            phi = 0.5 * (erf((y + Y / 2) / (2 * math.sqrt(ay_true * x)))
                         - erf((y - Y / 2) / (2 * math.sqrt(ay_true * x))))
            phi = max(phi, 1e-6)
            c = C0 * phi * math.exp(-k_true * x / vc) * float(np.exp(rng.normal(0, noise)))
            rows.append((f"W{i:03d}", 500000.0 + x, 5800000.0 + y, 15.0, c, True))
            i += 1
    return pd.DataFrame(rows, columns=["well_id", "easting", "northing", "t_years", "conc", "detect"])


def _lambda_with_alpha_x(site, alpha_x, vc_yr, Y):
    cfg = m2.DomenicoConfig(vc_override=vc_yr / DAYS, alpha_x_override=alpha_x,
                            source_width_override=Y)
    e = m2.estimate_site(site, cfg=cfg)
    return e.value_per_year if np.isfinite(e.value_per_year) else float("nan")


def run_synth():
    print("\n=== PART A: synthetic plumes, KNOWN k (does site-init alpha_L recover k better?) ===")
    rows = []
    vc, Y = 100.0, 10.0
    for k_true in (0.1, 0.3, 0.6):
        for ay_true in (1.0, 3.0):
            df = synth_long(k_true, ay_true, vc=vc, Y=Y, seed=int(k_true * 100 + ay_true))
            site = _site_from_long(df)
            L = float((df.easting.max() - df.easting.min()))
            aL_med, aL_az = site_init_alpha_L(site, flow_azimuth_deg=90.0)
            ax_current = 0.1 * L
            ax_oracle = 10.0 * ay_true            # makes alpha_y = alpha_x/10 match ay_true
            lam_cur = _lambda_with_alpha_x(site, ax_current, vc, Y)
            lam_si = _lambda_with_alpha_x(site, aL_med, vc, Y) if aL_med else float("nan")
            lam_or = _lambda_with_alpha_x(site, ax_oracle, vc, Y)
            rows.append(dict(k_true=k_true, ay_true=ay_true, L=round(L, 0),
                             ax_current=round(ax_current, 1), alpha_L_siteinit=round(aL_med, 1) if aL_med else None,
                             ax_oracle=round(ax_oracle, 1),
                             lam_current=round(lam_cur, 4), lam_siteinit=round(lam_si, 4),
                             lam_oracle=round(lam_or, 4),
                             err_current=round(100 * (lam_cur - k_true) / k_true, 1),
                             err_siteinit=round(100 * (lam_si - k_true) / k_true, 1) if lam_si == lam_si else None))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    mae_cur = np.mean(np.abs(df.err_current))
    mae_si = np.nanmean(np.abs(df.err_siteinit.astype(float)))
    print(f"\n  mean |error| in k recovery:  current(0.1L) = {mae_cur:.1f}%   "
          f"site-init alpha_L = {mae_si:.1f}%")
    return df, mae_cur, mae_si


# ------------------------------------------------------------------ field sites (materiality)
def run_real(site_root):
    print("\n=== PART B: field sites (is the change to lambda material or trivial?) ===")
    rows = []
    for s in sorted(os.listdir(site_root)):
        sd = os.path.join(site_root, s)
        if not os.path.isdir(sd):
            continue
        try:
            site = dataio.load_site(sd)
        except Exception:
            continue
        soil = dataio.soil_type(sd)
        base = m2.estimate_site(site, soil_type=soil)
        if not np.isfinite(base.value_per_year):
            continue
        az = base.diagnostics.get("flow_azimuth_deg")
        ax_cur = base.diagnostics.get("alpha_x_m")
        aL_med, aL_az = site_init_alpha_L(site, flow_azimuth_deg=az)
        if not aL_med:
            continue
        alt = m2.estimate_site(site, soil_type=soil,
                               cfg=m2.DomenicoConfig(alpha_x_override=aL_med))
        lam_cur, lam_alt = base.value_per_year, alt.value_per_year
        rows.append(dict(site=site.site_name, ax_current=round(ax_cur, 1),
                         alpha_L_siteinit=round(aL_med, 1),
                         lam_current=round(lam_cur, 4),
                         lam_siteinit=round(lam_alt, 4) if np.isfinite(lam_alt) else None,
                         lambda_pct_change=round(100 * (lam_alt - lam_cur) / lam_cur, 1)
                         if np.isfinite(lam_alt) and lam_cur else None))
    df = pd.DataFrame(rows)
    if len(df):
        print(df.to_string(index=False))
        chg = df.lambda_pct_change.dropna().abs()
        print(f"\n  sites compared: {len(df)};  median |change| in lambda = "
              f"{chg.median():.1f}%;  max |change| = {chg.max():.1f}%")
    else:
        print("  no sites where Method 2 is usable for both variants")
    return df


def main():
    if not HAVE_SITEINIT:
        print("This harness cannot run in this repository.")
        print("Both parts compare Method 2 against the site-initialization ContinuousDecay")
        print("dispersivity model, which is not included here. The file is kept as a record")
        print("of what was tested and how, not as a runnable check.")
        return 1
    os.makedirs(OUT, exist_ok=True)
    syn, mae_cur, mae_si = run_synth()
    site_root = os.environ.get("SITE_ROOT", SITE_ROOT_DEFAULT)
    real = run_real(site_root) if os.path.isdir(site_root) else pd.DataFrame()
    syn.to_csv(os.path.join(OUT, "bakeoff_dispersivity_synth.csv"), index=False)
    if len(real):
        real.to_csv(os.path.join(OUT, "bakeoff_dispersivity_real.csv"), index=False)

    print("\n================= VERDICT =================")
    better = mae_si + 2 < mae_cur            # site-init clearly better on synthetic truth
    worse = mae_si > mae_cur + 2
    med_change = real.lambda_pct_change.dropna().abs().median() if len(real) else float("nan")
    if better:
        acc = "site-init alpha_L recovers k MORE accurately on synthetic truth"
    elif worse:
        acc = "site-init alpha_L recovers k LESS accurately on synthetic truth"
    else:
        acc = "site-init alpha_L and 0.1*L recover k about equally on synthetic truth"
    print(acc + f"  (mean |err|: current {mae_cur:.1f}% vs site-init {mae_si:.1f}%).")
    if len(real):
        triv = med_change < 15
        print(f"On real sites the median change in lambda is {med_change:.1f}% -> "
              f"{'TRIVIAL' if triv else 'MATERIAL'}.")
    print("Recommendation:",
          ("adopt site-init alpha_L in Method 2" if better and (len(real) and med_change >= 15)
           else "keep the current 0.1*L rule; bringing in the site-init dispersivity is not a "
                "worthwhile improvement"))
    print("outputs:", os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
