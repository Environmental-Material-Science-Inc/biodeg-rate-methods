"""Bake-off: spline-derived concentration decay (new) vs Method 1 per-well point decay.

Feasibility (the user's insight): the ST P-spline surface is ln C(s,t). Its plume mass M(t)
lumps concentration and plume volume; but the surface itself gives concentration at any point and
time, so the temporal slope at the plume centre/mid/edge is a first-order CONCENTRATION decay
coefficient (the local rate field -df/dt, reference 3.8). That is the same apparent-attenuation
estimand as Method 1 and McHugh's k_c-max, but smoothed across all wells and times.

This harness compares the spline centre rate to Method 1's per-well summary on:
  A. synthetic plumes with a known rate (accuracy and stability under noise), and
  B. Field sites (coverage and agreement).

Run:  python validation/bakeoff_spline_decay.py
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from biodeg_rates import dataio
from biodeg_rates import method1_mann_kendall as m1
from biodeg_rates import method3_spline as m3

SITE_ROOT_DEFAULT = ""      # set SITE_ROOT to a folder of site subfolders


def _fit_from_site(site, cfg=m3.SplineConfig()):
    f = site.frame
    val = f.conc.to_numpy(float).copy(); det = f.detect.to_numpy(bool); rl = f.rl.to_numpy(float)
    val[~det] = np.where(np.isfinite(rl[~det]), rl[~det] * cfg.nd_substitution, val[~det])
    y = np.log(np.maximum(val, 1e-9))
    return m3.fit_surface(f.easting.to_numpy(float), f.northing.to_numpy(float),
                          f.t_years.to_numpy(float), y, cfg)


def _axis_points(site):
    """centre = source (peak); axis = concentration-weighted principal direction of the detected
    wells; mid/edge sampled down that axis at 0.5 and 0.9 of the plume extent."""
    f = site.frame; det = f[f.detect]
    g = det.groupby("well_id").agg(e=("easting", "first"), n=("northing", "first"),
                                   c=("conc", "max"))
    cx, cy = site.source_xy
    pts = g[["e", "n"]].to_numpy(float); w = g["c"].to_numpy(float)
    if len(pts) < 3:
        return {"centre": (cx, cy)}
    d = pts - np.array([cx, cy]); W = w / w.sum()
    cov = (d * W[:, None]).T @ d
    axis = np.linalg.eigh(cov)[1][:, -1]
    proj = d @ axis
    Lpos = proj[proj > 0].max() if (proj > 0).any() else 0.0
    Lneg = -proj[proj < 0].min() if (proj < 0).any() else 0.0
    if Lneg > Lpos:
        axis = -axis; L = Lneg
    else:
        L = Lpos
    return {"centre": (cx, cy),
            "mid": (cx + 0.5 * L * axis[0], cy + 0.5 * L * axis[1]),
            "edge": (cx + 0.9 * L * axis[0], cy + 0.9 * L * axis[1])}


def _spline_centre_rate(site):
    times = np.unique(np.round(site.frame.t_years.to_numpy(float), 6))
    n_wells = site.frame.well_id.nunique()
    if n_wells < m3.MIN_WELLS or len(times) < m3.MIN_TIMES or len(site.frame) < m3.MIN_OBS:
        return None, None
    fit = _fit_from_site(site)
    ks = m3.concentration_decay_at(fit, _axis_points(site), times)
    return ks.get("centre"), ks


# ---------------------------------------------------------------- synthetic (known truth)
def _synth(k_true, n_wells=14, n_times=6, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.uniform(0, 180, n_wells); N = rng.uniform(0, 180, n_wells)
    e0, n0 = 90.0, 90.0; times = np.linspace(0, 9, n_times)
    rows = []
    for wi, (e, n) in enumerate(zip(E, N)):           # one fixed id per WELL (not per row)
        amp = 30.0 * np.exp(-((e - e0) ** 2 + (n - n0) ** 2) / (2 * 45.0 ** 2))
        for t in times:
            c = max(amp * math.exp(-k_true * t) * float(np.exp(rng.normal(0, noise))), 1e-6)
            rows.append((f"W{wi}", 500000 + e, 5800000 + n, t, c, True))
    return pd.DataFrame(rows, columns=["well_id", "easting", "northing", "t_years", "conc", "detect"])


def run_synth():
    print("=== PART A: synthetic, known rate (accuracy + stability over 10 seeds) ===")
    print(f"{'k_true':>6} {'noise':>5} | {'M1 mean':>8} {'M1 sd':>6} | {'spline mean':>11} {'spline sd':>9}")
    rows = []
    for k_true in (0.10, 0.25, 0.50):
        for noise in (0.3, 0.6):
            m1e, spe = [], []
            for seed in range(10):
                df = _synth(k_true, noise=noise, seed=seed)
                site = dataio.site_from_long(df)
                _, summ = m1.estimate_site(site)
                if np.isfinite(summ.value_per_year):
                    m1e.append(summ.value_per_year)
                kc, _ = _spline_centre_rate(site)
                if kc is not None and np.isfinite(kc):
                    spe.append(kc)
            m1e, spe = np.array(m1e), np.array(spe)
            print(f"{k_true:6.2f} {noise:5.1f} | {m1e.mean():8.3f} {m1e.std():6.3f} | "
                  f"{spe.mean():11.3f} {spe.std():9.3f}")
            rows.append(dict(k_true=k_true, noise=noise,
                             m1_mean=round(m1e.mean(), 3), m1_sd=round(m1e.std(), 3),
                             spline_mean=round(spe.mean(), 3), spline_sd=round(spe.std(), 3),
                             m1_abs_err=round(abs(m1e.mean() - k_true), 3),
                             spline_abs_err=round(abs(spe.mean() - k_true), 3)))
    df = pd.DataFrame(rows)
    print(f"\n  mean |bias|:  M1 {np.mean(df.m1_abs_err):.3f}   spline {np.mean(df.spline_abs_err):.3f}")
    print(f"  mean SD (instability): M1 {df.m1_sd.mean():.3f}   spline {df.spline_sd.mean():.3f}")
    return df


def run_real(site_root):
    print("\n=== PART B: real sites (coverage + agreement, 1/yr) ===")
    rows = []
    for s in sorted(os.listdir(site_root)):
        sd = os.path.join(site_root, s)
        if not os.path.isdir(sd):
            continue
        try:
            site = dataio.load_site(sd)
        except Exception:
            continue
        _, summ = m1.estimate_site(site)
        m1v = summ.value_per_year if (summ and np.isfinite(summ.value_per_year)) else None
        try:
            kc, ks = _spline_centre_rate(site)
        except Exception:
            kc, ks = None, None
        if m1v is None and kc is None:
            continue
        rows.append(dict(site=site.site_id or site.site_name,
                         M1_per_yr=round(m1v, 3) if m1v is not None else None,
                         spline_centre=round(kc, 3) if kc is not None else None,
                         spline_mid=round(ks.get("mid"), 3) if ks and "mid" in ks else None,
                         spline_edge=round(ks.get("edge"), 3) if ks and "edge" in ks else None))
    df = pd.DataFrame(rows)
    if len(df):
        print(df.to_string(index=False))
        print(f"\n  usable: M1 {int(df.M1_per_yr.notna().sum())}   "
              f"spline-centre {int(df.spline_centre.notna().sum())}  (of {len(df)} sites)")
        only_spline = df[(df.M1_per_yr.isna()) & (df.spline_centre.notna())]
        if len(only_spline):
            print(f"  spline-centre estimable where M1 summary is N/A: {len(only_spline)} sites "
                  f"({', '.join(only_spline.site)})")
    return df


def main():
    syn = run_synth()
    site_root = os.environ.get("SITE_ROOT", SITE_ROOT_DEFAULT)
    real = run_real(site_root) if os.path.isdir(site_root) else pd.DataFrame()
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "validation")
    os.makedirs(out, exist_ok=True)
    syn.to_csv(os.path.join(out, "bakeoff_spline_decay_synth.csv"), index=False)
    if len(real):
        real.to_csv(os.path.join(out, "bakeoff_spline_decay_real.csv"), index=False)
    print("\n================= VERDICT =================")
    acc = "comparable accuracy" if abs(syn.spline_abs_err.mean() - syn.m1_abs_err.mean()) < 0.03 \
        else ("spline more accurate" if syn.spline_abs_err.mean() < syn.m1_abs_err.mean() else "M1 more accurate")
    stab = "more stable (lower SD)" if syn.spline_sd.mean() + 1e-9 < syn.m1_sd.mean() else "similar/less stable"
    print(f"Synthetic: {acc}; spline is {stab} than M1.")
    if len(real):
        gain = int(((real.M1_per_yr.isna()) & (real.spline_centre.notna())).sum())
        print(f"Real sites: spline-centre adds {gain} sites of coverage beyond M1 summary.")
    print("outputs:", os.path.abspath(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
