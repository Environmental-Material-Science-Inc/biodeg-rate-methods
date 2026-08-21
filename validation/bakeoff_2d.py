"""Bake-off: 1D Domenico (current Method 2) vs the 2D fit (all wells, alpha_y fitted).

Part A - synthetic plumes with KNOWN k and KNOWN transverse dispersivity alpha_y. The 1D method
assumes alpha_y = alpha_x/10; the 2D fit estimates alpha_y from the data. Compares k recovery
and whether 2D recovers the true alpha_y.

Part B - field sites: coverage (how many usable), reliability (poorly-constrained
count), and the lambda each reports.

Run:  python validation/bakeoff_2d.py
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

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
SITE_ROOT_DEFAULT = ""      # set SITE_ROOT to a folder of site subfolders


def _phi_y(x, y, ay, Y):
    d = 2.0 * np.sqrt(ay * x)
    return 0.5 * (erf((y + Y / 2) / d) - erf((y - Y / 2) / d))


def synth_2d(k_true, ay_true, vc=100.0, Y=10.0, C0=5.0, noise=0.08, seed=0):
    """2D Domenico plume sampled on a transverse grid of wells (flow = +easting)."""
    rng = np.random.default_rng(seed)
    xs = np.arange(20, 201, 20.0)
    ys = np.array([-30, -15, 0, 15, 30], float)
    rows = []
    for x in xs:
        for y in ys:
            c = C0 * _phi_y(x, y, ay_true, Y) * math.exp(-k_true * x / vc) * float(np.exp(rng.normal(0, noise)))
            rows.append((f"W{len(rows):03d}", 500000.0 + x, 5800000.0 + y, 15.0, max(c, 1e-9), True))
    return pd.DataFrame(rows, columns=["well_id", "easting", "northing", "t_years", "conc", "detect"])


def run_synth():
    print("=== PART A: synthetic, KNOWN k and alpha_y (1D assumes ay=ax/10; 2D fits ay) ===")
    rows = []
    vc, Y = 100.0, 10.0
    for k_true in (0.1, 0.3, 0.6):
        for ay_true in (0.5, 3.0):
            df = synth_2d(k_true, ay_true, vc=vc, Y=Y, seed=int(k_true * 100 + ay_true))
            site = _site_from_long(df)
            cfg = m2.DomenicoConfig(vc_override=vc / DAYS, source_width_override=Y)
            e1 = m2.estimate_site(site, cfg=cfg)
            e2 = m2.estimate_site_2d(site, cfg=cfg)
            ay_fit = e2.diagnostics.get("alpha_y_fitted_m")
            rows.append(dict(
                k_true=k_true, ay_true=ay_true,
                lam_1d=round(e1.value_per_year, 4) if np.isfinite(e1.value_per_year) else None,
                lam_2d=round(e2.value_per_year, 4) if np.isfinite(e2.value_per_year) else None,
                ay_fit_2d=ay_fit, r2_2d=e2.diagnostics.get("r2"),
                err_1d=(round(100 * (e1.value_per_year - k_true) / k_true, 0)
                        if np.isfinite(e1.value_per_year) else None),
                err_2d=(round(100 * (e2.value_per_year - k_true) / k_true, 0)
                        if np.isfinite(e2.value_per_year) else None)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    e1 = np.nanmean(np.abs(pd.to_numeric(df.err_1d)))
    e2 = np.nanmean(np.abs(pd.to_numeric(df.err_2d)))
    print(f"\n  mean |error| in k:  1D = {e1:.0f}%   2D = {e2:.0f}%")
    return df, e1, e2


def run_real(site_root):
    print("\n=== PART B: real sites - coverage and reliability (1D vs 2D) ===")
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
        e1 = m2.estimate_site(site, soil_type=soil)
        e2 = m2.estimate_site_2d(site, soil_type=soil)
        u1 = np.isfinite(e1.value_per_year); u2 = np.isfinite(e2.value_per_year)
        if not (u1 or u2):
            continue
        rows.append(dict(
            site=site.site_id or site.site_name,
            lam_1d=round(e1.value_per_year, 3) if u1 else None,
            ok1=bool(u1 and not e1.diagnostics.get("poorly_constrained") and not e1.diagnostics.get("implausibly_fast")),
            lam_2d=round(e2.value_per_year, 3) if u2 else None,
            ay_2d=e2.diagnostics.get("alpha_y_fitted_m"), r2_2d=e2.diagnostics.get("r2"),
            ok2=bool(u2 and not e2.diagnostics.get("poorly_constrained") and not e2.diagnostics.get("implausibly_fast")),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        print(df.to_string(index=False))
        print(f"\n  sites with any Method-2 output: {len(df)}")
        print(f"  usable:   1D {int(df.lam_1d.notna().sum())}   2D {int(df.lam_2d.notna().sum())}")
        print(f"  RELIABLE (not poorly-constrained/fast):   1D {int(df.ok1.sum())}   2D {int(df.ok2.sum())}")
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    syn, e1, e2 = run_synth()
    site_root = os.environ.get("SITE_ROOT", SITE_ROOT_DEFAULT)
    real = run_real(site_root) if os.path.isdir(site_root) else pd.DataFrame()
    syn.to_csv(os.path.join(OUT, "bakeoff_2d_synth.csv"), index=False)
    if len(real):
        real.to_csv(os.path.join(OUT, "bakeoff_2d_real.csv"), index=False)
    print("\n================= VERDICT =================")
    acc = ("2D recovers k more accurately" if e2 + 3 < e1
           else "2D recovers k less accurately" if e2 > e1 + 3
           else "1D and 2D recover k about equally")
    print(f"Accuracy on synthetic truth: {acc} (mean |err| 1D {e1:.0f}% vs 2D {e2:.0f}%).")
    if len(real):
        print(f"Real-site reliability: 1D {int(real.ok1.sum())} vs 2D {int(real.ok2.sum())} "
              f"reliable of {len(real)} sites with output.")
    print("outputs:", os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
