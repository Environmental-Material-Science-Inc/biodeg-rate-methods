"""Validate Module 3 (the from-scratch SciPy P-spline) against R mgcv.

Reference doc (Section 3.10, Appendix B.8): the oracle is R ``mgcv`` with a tensor-product
P-spline smooth fit by REML; the SciPy port must be validated against it on shared data before
it is trusted.

This harness fits the SAME data two ways: our ``fit_surface`` and mgcv
``gam(y ~ te(e,n,t, bs="ps", k=...), method="REML")`` (driven via Rscript over CSV). It then:

  1. compares the two fitted log-concentration surfaces at the observation points and on a
     dense grid (Pearson r and RMSE) -- the parameterization-invariant check; and
  2. compares the DERIVED deliverable k_M, computed for BOTH surfaces with the identical Python
     mass integral, against each other and (for the synthetic case) against the known truth.

Run:  python validation/validate_mgcv.py
Output: console report + outputs/biodeg_rates/_validation/*.csv,*.png
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biodeg_rates import method3_spline as m3

RSCRIPT_CANDIDATES = [os.environ.get("RSCRIPT", ""), "Rscript"]   # RSCRIPT overrides
HERE = os.path.dirname(os.path.abspath(__file__))
RFILE = os.path.join(HERE, "_mgcv_fit.R")
OUT = os.path.join(HERE, "..", "outputs", "validation")


def _rscript() -> str:
    """Locate Rscript: the RSCRIPT environment variable first, then PATH."""
    for c in RSCRIPT_CANDIDATES:
        if c and os.path.sep in c and os.path.exists(c):
            return c
    found = shutil.which("Rscript")
    if found:
        return found
    raise SystemExit(
        "Rscript was not found. This harness cross-checks Method 3 against the R mgcv "
        "package, so it needs an R installation with mgcv available. Install R and "
        "mgcv, then put Rscript on PATH or set the RSCRIPT environment variable to it."
    )


def run_mgcv(d: pd.DataFrame, q: pd.DataFrame, k: tuple, tag: str):
    """Fit with mgcv and return q with a 'pred' column, plus the stderr summary line."""
    os.makedirs(OUT, exist_ok=True)
    dpath = os.path.join(OUT, f"{tag}_data.csv")
    qpath = os.path.join(OUT, f"{tag}_query.csv")
    opath = os.path.join(OUT, f"{tag}_mgcv_pred.csv")
    d.to_csv(dpath, index=False)
    q.to_csv(qpath, index=False)
    proc = subprocess.run([_rscript(), RFILE, dpath, qpath, opath, str(k[0]), str(k[1]), str(k[2])],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(opath):
        raise RuntimeError(f"mgcv failed:\n{proc.stderr}\n{proc.stdout}")
    return pd.read_csv(opath), proc.stderr.strip()


def _hull_grid(E, N, grid_n=36):
    path, _ = m3._hull(E, N)
    gx = np.linspace(E.min(), E.max(), grid_n)
    gy = np.linspace(N.min(), N.max(), grid_n)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    return pts[path.contains_points(pts)]


def _kM_from_grid_preds(pred_lnC: np.ndarray, n_grid: int, times: np.ndarray) -> float:
    """k_M = -slope of ln M(t); M(t) = sum exp(lnC) over the grid at time t. Identical for both
    surfaces, so it isolates the surface-fit difference."""
    P = pred_lnC.reshape(len(times), n_grid)
    lnM = np.log(np.sum(np.exp(P), axis=1))
    return -float(np.polyfit(times, lnM, 1)[0])


def synth_plume(k_true=0.2, n_wells=16, n_times=8, noise=0.30, seed=7):
    rng = np.random.default_rng(seed)
    E = rng.uniform(0, 200, n_wells); N = rng.uniform(0, 200, n_wells)
    e0, n0, sx, sy, C0 = 100.0, 100.0, 45.0, 60.0, 50.0
    times = np.linspace(0, 9, n_times)
    rows = []
    for (e, nn) in zip(E, N):
        amp = C0 * np.exp(-((e - e0) ** 2 / (2 * sx ** 2) + (nn - n0) ** 2 / (2 * sy ** 2)))
        for t in times:
            c = amp * np.exp(-k_true * t)
            lnc = np.log(max(c, 1e-9)) + rng.normal(0, noise)
            rows.append((e, nn, t, lnc))
    df = pd.DataFrame(rows, columns=["e", "n", "t", "y"])
    return df, times


def compare(label, d: pd.DataFrame, eval_times: np.ndarray, k_true=None):
    E = d.e.to_numpy(float); N = d.n.to_numpy(float); T = d.t.to_numpy(float); y = d.y.to_numpy(float)
    fit = m3.fit_surface(E, N, T, y)
    k = (fit.K1, fit.K2, fit.K3)

    grid = _hull_grid(E, N)
    # query rows: observations (tag 0) then grid x eval_times (tag 1)
    q_obs = pd.DataFrame(dict(e=E, n=N, t=T))
    gq = pd.DataFrame([(g[0], g[1], t) for t in eval_times for g in grid], columns=["e", "n", "t"])
    q = pd.concat([q_obs, gq], ignore_index=True)

    mg, summary = run_mgcv(d[["e", "n", "t", "y"]], q, k, label)
    n_obs = len(q_obs)
    mg_obs = mg.pred.to_numpy()[:n_obs]
    mg_grid = mg.pred.to_numpy()[n_obs:]

    mine_obs = fit.predict_lnC(E, N, T)
    mine_grid = np.concatenate([fit.predict_lnC(grid[:, 0], grid[:, 1], np.full(len(grid), t))
                                for t in eval_times])

    r_surf = float(np.corrcoef(mine_obs, mg_obs)[0, 1])
    rmse_surf = float(np.sqrt(np.mean((mine_obs - mg_obs) ** 2)))
    r_grid = float(np.corrcoef(mine_grid, mg_grid)[0, 1])

    kM_mine = _kM_from_grid_preds(mine_grid, len(grid), eval_times)
    kM_mgcv = _kM_from_grid_preds(mg_grid, len(grid), eval_times)

    print(f"\n=== {label} ===  bases k={k}  ({summary})")
    if k_true is not None:
        print(f"  k_true = {k_true:.4f}/yr")
    print(f"  k_M  mine = {kM_mine:.4f}/yr   mgcv = {kM_mgcv:.4f}/yr   "
          f"abs diff = {abs(kM_mine - kM_mgcv):.4f}")
    print(f"  surface agreement (mine vs mgcv): obs r={r_surf:.4f} rmse={rmse_surf:.3f} | grid r={r_grid:.4f}")

    # scatter figure
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].scatter(mg_obs, mine_obs, s=14, alpha=0.6)
    lim = [min(mg_obs.min(), mine_obs.min()), max(mg_obs.max(), mine_obs.max())]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("mgcv fitted ln C"); ax[0].set_ylabel("Module 3 fitted ln C")
    ax[0].set_title(f"{label}: surface at obs (r={r_surf:.3f})", fontsize=9)
    P_mine = mine_grid.reshape(len(eval_times), len(grid))
    P_mg = mg_grid.reshape(len(eval_times), len(grid))
    lnM_mine = np.log(np.sum(np.exp(P_mine), axis=1)); lnM_mg = np.log(np.sum(np.exp(P_mg), axis=1))
    ax[1].plot(eval_times, lnM_mine - lnM_mine[0], "o-", label=f"mine (k_M={kM_mine:.3f})")
    ax[1].plot(eval_times, lnM_mg - lnM_mg[0], "s--", label=f"mgcv (k_M={kM_mgcv:.3f})")
    if k_true is not None:
        ax[1].plot(eval_times, -k_true * eval_times, ":", color="k", label=f"truth (k={k_true:.3f})")
    ax[1].set_xlabel("time (yr)"); ax[1].set_ylabel("ln M(t) - ln M(0)")
    ax[1].set_title("plume mass decline", fontsize=9); ax[1].legend(fontsize=8)
    fig.suptitle(f"Module 3 vs mgcv — {label}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, f"{label}_validation.png"), dpi=130)
    plt.close(fig)
    return dict(label=label, k=k, k_true=k_true, kM_mine=kM_mine, kM_mgcv=kM_mgcv,
                r_surf_obs=r_surf, rmse_surf_obs=rmse_surf, r_surf_grid=r_grid)


def main():
    print("Validating Module 3 against R mgcv (te, bs='ps', method='REML')")
    print("Rscript:", _rscript())
    results = []

    # 1) synthetic plumes with known k_M = k_true (clean separable exponential decay)
    for kt, seed in ((0.20, 7), (0.05, 11), (0.40, 23)):
        d, _ = synth_plume(k_true=kt, seed=seed)
        eval_times = np.unique(d.t.to_numpy())
        results.append(compare(f"synthetic_k{kt:.2f}", d, eval_times, k_true=kt))

    # 2) optionally, a data-rich field site (no truth; surface + k_M agreement only)
    # optional no-truth cross-check: point SITE_DIR at one data-rich site folder
    real = os.environ.get("SITE_DIR", "")
    if os.path.isdir(real):
        from biodeg_rates import dataio
        so = dataio.load_site(real)
        f = so.frame
        val = f.conc.to_numpy(float).copy(); det = f.detect.to_numpy(bool); rl = f.rl.to_numpy(float)
        val[~det] = np.where(np.isfinite(rl[~det]), rl[~det] * 0.5, val[~det])
        d = pd.DataFrame(dict(e=f.easting, n=f.northing, t=f.t_years,
                              y=np.log(np.maximum(val, 1e-9))))
        eval_times = np.unique(np.round(f.t_years.to_numpy(), 6))
        results.append(compare("field_site", d, eval_times))

    rep = pd.DataFrame(results)
    os.makedirs(OUT, exist_ok=True)
    rep.to_csv(os.path.join(OUT, "validation_summary.csv"), index=False)
    print("\n================= SUMMARY =================")
    print(rep.to_string(index=False))
    # Verdict. The PORT is validated on the synthetic ground-truth cases: the surface must match
    # mgcv (r > 0.99) and both must recover the known k_M (within 0.02/yr). Real-site k_M spread
    # between two legitimate smoothers is model-form uncertainty, reported but not a failure.
    synth = [r for r in results if r["k_true"] is not None]
    port_ok = all(r["r_surf_obs"] > 0.99
                  and abs(r["kM_mine"] - r["k_true"]) <= 0.02
                  and abs(r["kM_mine"] - r["kM_mgcv"]) <= 0.02 for r in synth)
    print("\nPORT VALIDATION (synthetic, vs known truth + mgcv):",
          "PASS" if port_ok else "FAIL")
    for r in results:
        if r["k_true"] is None:
            spread = abs(r["kM_mine"] - r["kM_mgcv"])
            print(f"REAL-SITE NOTE ({r['label']}): surfaces agree (obs r={r['r_surf_obs']:.3f}); "
                  f"k_M model-form spread mine-vs-mgcv = {spread:.3f}/yr "
                  f"({'within' if spread <= 0.1 else 'beyond'} 0.1/yr) -- carry as uncertainty.")
    print("outputs:", os.path.abspath(OUT))
    return 0 if port_ok else 1


if __name__ == "__main__":
    sys.exit(main())
