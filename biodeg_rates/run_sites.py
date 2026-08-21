"""Driver: run the three estimators on a folder of sites.

For each site it loads the observations, runs Methods 1-3 (kept separate), and writes the
handoff JSON, then prints a one-line summary.

Usage:
  python -m biodeg_rates.run_sites --root "<site folder>" --out outputs/biodeg_rates
  python -m biodeg_rates.run_sites --root "<...>" --site "Example Site"
"""

from __future__ import annotations

import argparse
import os
import traceback

import numpy as np

from . import dataio
from . import method1_mann_kendall as m1
from . import method2_domenico as m2
from . import method3_spline as m3
from . import handoff


def run_site(site_dir: str, out_root: str, analyte: str = "benzene") -> dict:
    site = dataio.load_site(site_dir, analyte=analyte)
    soil = dataio.soil_type(site_dir)
    per_well, summary = m1.estimate_site(site)
    est2 = m2.estimate_site(site, soil_type=soil)
    est2_2d = m2.estimate_site_2d(site, soil_type=soil)
    est3 = m3.estimate_site(site)
    bundle = handoff.assemble_bundle(site, per_well, summary, est2, est3, method2_2d=est2_2d)

    out_dir = os.path.join(out_root, (site.site_id or site.site_name).replace(" ", "_").replace("/", "_"))
    hpath = handoff.write_handoff(site, bundle, out_dir)
    return dict(site=site.site_name, summary=summary, m2=est2, m2_2d=est2_2d, m3=est3,
                bundle=bundle, handoff=hpath)


def _fmt(est):
    if est is None or not np.isfinite(est.value_per_year):
        return "   N/A   "
    return f"{est.value_per_year:7.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing the site subfolders")
    ap.add_argument("--out", default="outputs/biodeg_rates", help="output root")
    ap.add_argument("--site", default=None, help="run only this one site (folder name)")
    ap.add_argument("--analyte", default="benzene")
    a = ap.parse_args()

    sites = ([a.site] if a.site else
             sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d))))
    print(f"{'site':24s} {'M1 k':>8s} {'M2 lam':>8s} {'M2-2D':>8s} {'M3 ctr':>8s}  consistency")
    print("-" * 86)
    for s in sites:
        sd = os.path.join(a.root, s)
        try:
            r = run_site(sd, a.out, a.analyte)
        except FileNotFoundError as e:
            print(f"{s:24s}  (skip: {e})")
            continue
        except Exception:
            print(f"{s:24s}  ERROR")
            traceback.print_exc()
            continue
        cc = r["bundle"].consistency_check()
        spread = cc.get("spread_orders_of_magnitude")
        flag = "" if spread is None else (f"{spread} oom " + ("OK" if cc["agree_within_1_oom"] else "DISAGREE"))
        print(f"{r['site']:24s} {_fmt(r['summary'])} {_fmt(r['m2'])} {_fmt(r['m2_2d'])} "
              f"{_fmt(r['m3'])}  {flag}")
    print(f"\noutputs under: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
