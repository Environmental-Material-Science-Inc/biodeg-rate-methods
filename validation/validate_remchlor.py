"""Validate the rate estimators against REMChlor output (filled workbook), plus a self-test.

Two modes:

  --xlsx <REMChlor_validation.xlsx>
      Read the long-format data and known parameters the user pasted after running REMChlor, run
      the three production methods, and compare Method 2 to REMChlor's plume decay rate k.

  --selftest
      Generate a synthetic REMChlor-EQUIVALENT centerline (the same Domenico analytical form
      REMChlor uses) with a known k, run Method 2 with the known transport parameters pinned, and
      assert recovery. This proves the validation loop is correct before any REMChlor run.

Method 2 is the apples-to-apples test (it inverts REMChlor's plume model). Methods 1 and 3 are
reported too, but they measure different quantities (temporal/source decline; whole-plume mass
loss) and are not expected to equal k.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd
from scipy.special import erf

from biodeg_rates.contract import SiteObservations
from biodeg_rates import method1_mann_kendall as m1
from biodeg_rates import method2_domenico as m2
from biodeg_rates import method3_spline as m3

from biodeg_rates.dataio import site_from_long as _site_from_long, DAYS_PER_YEAR as DAYS

def run_methods(df: pd.DataFrame, params: dict):
    site = _site_from_long(df)
    cfg2 = m2.DomenicoConfig(
        vc_override=params["vc_m_per_year"] / DAYS,           # m/day
        alpha_x_override=params.get("alpha_x_m"),
        source_width_override=params.get("source_width_Y_m"),
    )
    est2 = m2.estimate_site(site, cfg=cfg2)
    _, summ1 = m1.estimate_site(site)
    est3 = m3.estimate_site(site)
    return site, summ1, est2, est3


def _report(summ1, est2, est3, k_true):
    print(f"  REMChlor plume k (truth)           = {k_true:.4f} /yr")
    if est2 and np.isfinite(est2.value_per_year):
        err = 100 * (est2.value_per_year - k_true) / k_true
        print(f"  Method 2 (Domenico, dilution removed) = {est2.value_per_year:.4f} /yr  "
              f"[{err:+.1f}% vs k]   <-- apples-to-apples")
        m = est2.diagnostics.get("slope_m_per_m")
        print(f"      fitted normalized slope m = {m} /m, vc = "
              f"{est2.diagnostics.get('contaminant_velocity_m_per_day')} m/day")
    else:
        print(f"  Method 2 = N/A ({est2.notes if est2 else 'none'})")
    s1 = summ1.value_per_year if summ1 else float("nan")
    s3 = est3.value_per_year if est3 else float("nan")
    print(f"  Method 1 (per-well point decay)    = {s1 if np.isnan(s1) else round(s1,4)} /yr  "
          f"(different estimand: source/temporal decline)")
    print(f"  Method 3 (plume mass loss k_M)     = {s3 if np.isnan(s3) else round(s3,4)} /yr  "
          f"(different estimand: whole-plume mass loss)")


def selftest():
    """Synthetic REMChlor-equivalent: centerline C(x) = C0 * Phi_y(x) * exp(-k x / vc)."""
    k_true, C0, vc, Y, ay = 0.5, 1.0, 100.0, 10.0, 1.0      # vc in m/yr
    xs = np.arange(25, 401, 25.0)
    rows = []
    for i, x in enumerate(xs):
        phi = erf((Y / 2) / (2 * math.sqrt(ay * x)))
        c = C0 * phi * math.exp(-k_true * x / vc)
        rows.append((f"CL{i:02d}", 500000.0 + x, 5800000.0, 20.0, c, True))
    df = pd.DataFrame(rows, columns=["well_id", "easting", "northing", "t_years", "conc", "detect"])
    params = dict(lambda_true_per_year=k_true, vc_m_per_year=vc, alpha_x_m=1.0, source_width_Y_m=Y)
    print("SELF-TEST: synthetic REMChlor-equivalent centerline, known k = 0.5 /yr")
    _, s1, e2, e3 = run_methods(df, params)
    _report(s1, e2, e3, k_true)
    ok = bool(e2 and np.isfinite(e2.value_per_year) and abs(e2.value_per_year - k_true) <= 0.1)
    # also show that ignoring dilution (raw slope) over-estimates
    lnC = np.log(df.conc.to_numpy())
    x = (df.easting.to_numpy() - df.easting.min())
    m_raw = np.polyfit(x, lnC, 1)[0]
    lam_raw = -(vc / DAYS) * m_raw * (1 - 1.0 * m_raw) * DAYS
    print(f"  RAW slope (no dilution removal) lambda = {lam_raw:.4f} /yr  "
          f"[{100*(lam_raw-k_true)/k_true:+.1f}% vs k]  <-- biased high, as expected")
    print("\nSELF-TEST VERDICT:", "PASS - Method 2 recovers REMChlor's k within 0.1/yr" if ok
          else "REVIEW")
    return 0 if ok else 1


def from_workbook(path: str):
    wb = pd.ExcelFile(path)
    pr = wb.parse("7_LongFormat_Python", header=None)
    params = {}
    for i in range(5, 12):
        k = pr.iloc[i, 0]
        if isinstance(k, str) and k.strip():
            try:
                params[k.strip()] = float(pr.iloc[i, 1])
            except (TypeError, ValueError):
                pass
    data = wb.parse("7_LongFormat_Python", header=13)
    data = data.rename(columns={data.columns[0]: "well_id", data.columns[1]: "easting",
                                data.columns[2]: "northing", data.columns[3]: "t_years",
                                data.columns[4]: "conc", data.columns[5]: "detect"})
    data = data.dropna(subset=["well_id", "easting", "northing", "t_years", "conc"])
    if len(data) == 0:
        print("No rows pasted into sheet '7_LongFormat_Python'. Fill it after running REMChlor.")
        return 1
    print(f"Loaded {len(data)} well-events from {path}")
    k_true = params.get("lambda_true_per_year", float("nan"))
    if "vc_m_per_year" not in params:
        params["vc_m_per_year"] = 100.0
    _, s1, e2, e3 = run_methods(data, params)
    _report(s1, e2, e3, k_true)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=None, help="filled REMChlor_validation.xlsx")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic recovery demo")
    a = ap.parse_args()
    if a.selftest or not a.xlsx:
        return selftest()
    return from_workbook(a.xlsx)


if __name__ == "__main__":
    sys.exit(main())
