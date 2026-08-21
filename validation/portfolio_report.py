"""Portfolio QA/QC: read a run's handoff JSONs and evaluate how the estimators did across sites.

Produces a console evaluation, a summary CSV, and a multi-panel portfolio figure (coverage,
rate distributions, cross-method consistency, Method-2 reliability).

Run:  python validation/portfolio_report.py --dir outputs/biodeg_rates_pf
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biodeg_rates.contract import (METHOD_MANN_KENDALL, METHOD_DOMENICO,
                                             METHOD_DOMENICO_2D, METHOD_ST_PSPLINE)
from biodeg_rates.handoff import MODFLOW_DAYS_PER_YEAR as DPY  # 365.0, MODFLOW units

# rates are read from the handoff in 1/year and shown in 1/day (MODFLOW units); thresholds too
IMPLAUSIBLE_PER_DAY = 5.0 / DPY        # ~0.0137/day, an implausibly fast first-order rate
BIG_NEG_PER_DAY = -0.5 / DPY           # a strongly "increasing" rate worth investigating


def _pd(v):
    """1/year -> 1/day (NaN-safe)."""
    return v / DPY

C1, C2, C3 = "#1f4e79", "#2e7d32", "#d2691e"


def _val(est):
    if not est:
        return float("nan"), "N/A", "", {}
    v = est.get("value_per_year")
    return (float(v) if v is not None else float("nan"),
            est.get("confidence", "N/A"), est.get("trend", ""), est.get("diagnostics") or {})


def load_runs(dirpath: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(dirpath, "*", "*_biodeg_handoff.json"))):
        d = json.loads(open(p, encoding="utf-8").read())
        est = d["estimates"]
        m1v, m1c, m1t, _ = _val(est[METHOD_MANN_KENDALL].get("site_summary"))
        m2v, m2c, m2t, m2d = _val(est[METHOD_DOMENICO])
        m22v, m22c, _, m22d = _val(est.get(METHOD_DOMENICO_2D))
        m3v, m3c, m3t, m3d = _val(est[METHOD_ST_PSPLINE])
        cc = d.get("consistency_check", {})
        mf = d.get("modflow_input", {})
        rows.append(dict(
            site=d["site"], folder=os.path.basename(os.path.dirname(p)),
            n_wells=len(est[METHOD_MANN_KENDALL].get("per_well", [])),
            m1=m1v, m1_conf=m1c, m1_trend=m1t,
            m2=m2v, m2_conf=m2c, m2_poorly=bool(m2d.get("poorly_constrained")),
            m2_fast=bool(m2d.get("implausibly_fast")),
            m2_2d=m22v, m2_2d_conf=m22c,
            m2_2d_poorly=bool(m22d.get("poorly_constrained")), m2_2d_fast=bool(m22d.get("implausibly_fast")),
            m3=m3v, m3_conf=m3c, m3_trend=m3t,
            m3_balloon=bool(m3d.get("ballooning_suspected") or m3d.get("mass_vs_data_conflict")),
            spread_oom=cc.get("spread_orders_of_magnitude"),
            agree=cc.get("agree_within_1_oom"),
            modflow_ready=bool(mf.get("ready_for_modflow")),
        ))
    df = pd.DataFrame(rows)
    # per-day (MODFLOW units) versions of every rate, alongside the 1/year values
    for col in ("m1", "m2", "m2_2d", "m3"):
        if col in df:
            df[col + "_pd"] = df[col] / DPY
    return df


def _trust(conf):
    return isinstance(conf, str) and (conf.startswith("high") or conf.startswith("medium"))


def robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-site robustness scorecard. A method is 'solid' when it produced a finite estimate at
    medium+ confidence with no disqualifying flag (M2: poorly/fast; M3: ballooning). The tier
    combines how many estimands are solid with whether they agree across methods."""
    out = []
    for r in df.itertuples():
        m1_solid = bool(np.isfinite(r.m1) and _trust(r.m1_conf) and r.m1_trend != "no trend")
        flow_v = r.m2_2d if np.isfinite(r.m2_2d) else r.m2
        flow_conf = r.m2_2d_conf if np.isfinite(r.m2_2d) else r.m2_conf
        flow_flag = (r.m2_2d_poorly or r.m2_2d_fast) if np.isfinite(r.m2_2d) else (r.m2_poorly or r.m2_fast)
        flow_solid = bool(np.isfinite(flow_v) and _trust(flow_conf) and not flow_flag)
        m3_solid = bool(np.isfinite(r.m3) and _trust(r.m3_conf) and not r.m3_balloon)
        n_usable = int(np.isfinite(r.m1)) + int(np.isfinite(flow_v)) + int(np.isfinite(r.m3))
        n_solid = int(m1_solid) + int(flow_solid) + int(m3_solid)

        if n_solid >= 2 and r.agree:
            tier = "STRONG"
        elif n_solid >= 1 and n_usable >= 2:
            tier = "MODERATE"
        elif n_usable >= 1:
            tier = "WEAK"
        else:
            tier = "NONE"
        bits = []
        bits.append(f"M1 {r.m1_conf}{'*' if m1_solid else ''}")
        bits.append(f"flow {flow_conf}{'*' if flow_solid else ''}{' BALLOON' if False else ''}")
        bits.append(f"M3 {r.m3_conf}{'*' if m3_solid else ''}{' BALLOON' if r.m3_balloon else ''}")
        reason = "; ".join(bits) + (f"; agree({r.spread_oom}oom)" if r.agree else
                                    "; disagree" if r.agree is False else "; n/a")
        out.append(dict(folder=r.folder, n_wells=r.n_wells,
                        M1_k_per_day=round(_pd(r.m1), 6) if np.isfinite(r.m1) else None, M1_conf=r.m1_conf,
                        flow_lambda_per_day=round(_pd(flow_v), 6) if np.isfinite(flow_v) else None,
                        flow_conf=flow_conf,
                        M3_centre_per_day=round(_pd(r.m3), 6) if np.isfinite(r.m3) else None, M3_conf=r.m3_conf,
                        M3_ballooning=r.m3_balloon, agree=r.agree,
                        n_solid=n_solid, n_usable=n_usable, robustness=tier, reason=reason))
    order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "NONE": 3}
    rdf = pd.DataFrame(out)
    return rdf.sort_values(["robustness", "n_solid", "n_wells"],
                           key=lambda s: s.map(order) if s.name == "robustness" else s,
                           ascending=[True, False, False]).reset_index(drop=True)


def evaluate(df: pd.DataFrame):
    n = len(df)
    print(f"\n================= PORTFOLIO EVALUATION ({n} sites with output) =================")
    for m, lab in ((("m1", "m1_conf"), "Method 1 point-decay"),
                   (("m2", "m2_conf"), "Method 2 flowpath lambda"),
                   (("m3", "m3_conf"), "Method 3 spline-centre decay")):
        v, c = df[m[0]], df[m[1]]
        usable = int(np.isfinite(v).sum())
        na = int((c == "N/A").sum())
        dec = int(((v > 0) & np.isfinite(v)).sum())
        inc = int(((v < 0) & np.isfinite(v)).sum())
        print(f"  {lab:26s} usable {usable:2d}/{n}   N/A {na:2d}   "
              f"decreasing {dec:2d}   increasing {inc:2d}")
    # confidence mix
    print("\n  Method 2 reliability flags:")
    print(f"    poorly_constrained: {int(df.m2_poorly.sum())}   implausibly_fast: {int(df.m2_fast.sum())}"
          f"   (of {int(np.isfinite(df.m2).sum())} usable)")
    impl = df[(np.isfinite(df.m2_pd)) & (df.m2_pd > IMPLAUSIBLE_PER_DAY)]
    if len(impl):
        print(f"    Method-2 values > {IMPLAUSIBLE_PER_DAY:.4f}/day (implausible):",
              ", ".join(f"{r.folder}={r.m2_pd:.4f}" for r in impl.itertuples()))
    # consistency
    comp = df.dropna(subset=["agree"])
    if len(comp):
        agree = int(comp.agree.sum())
        print(f"\n  Cross-method consistency: {len(comp)} sites comparable; "
              f"{agree} agree within 1 oom, {len(comp)-agree} DISAGREE")
    print(f"\n  MODFLOW-ready (Method 2 reaction rate available): {int(df.modflow_ready.sum())}/{n}")
    # large-magnitude negatives worth a look (rates in 1/day)
    big_neg = df[(df.m1_pd < BIG_NEG_PER_DAY) | (df.m3_pd < BIG_NEG_PER_DAY)]
    if len(big_neg):
        print("\n  Large 'increasing' rates to investigate, 1/day (data artifact vs real growth):")
        for r in big_neg.itertuples():
            print(f"    {r.folder:30s} M1={r.m1_pd:.5f} M3={r.m3_pd:.5f}")


def figure(df: pd.DataFrame, out_png: str):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    n = len(df)

    # A: coverage
    a = ax[0, 0]
    meths = [("m1", "m1_conf", "Method 1\n(point decay)", C1),
             ("m2", "m2_conf", "Method 2\n(flowpath lambda)", C2),
             ("m3", "m3_conf", "Method 3\n(spline centre)", C3)]
    x = np.arange(3)
    usable = [int(np.isfinite(df[m[0]]).sum()) for m in meths]
    na = [n - u for u in usable]
    a.bar(x, usable, color=[m[3] for m in meths], label="usable")
    a.bar(x, na, bottom=usable, color="#dddddd", label="N/A")
    for i, u in enumerate(usable):
        a.text(i, n + 0.3, f"{u}/{n}", ha="center", fontsize=9)
    a.set_xticks(x); a.set_xticklabels([m[2] for m in meths], fontsize=8)
    a.set_ylabel("sites"); a.set_title("(A) Coverage: how many sites each method estimated", fontsize=10)
    a.legend(fontsize=8)

    # B: rate distributions (signed, symlog) per method, in 1/day (MODFLOW units)
    b = ax[0, 1]
    data = [df.m1_pd.dropna().values, df.m2_pd.dropna().values, df.m3_pd.dropna().values]
    parts = b.boxplot(data, vert=True, showfliers=True, patch_artist=True,
                      tick_labels=["M1", "M2", "M3"])
    for patch, col in zip(parts["boxes"], (C1, C2, C3)):
        patch.set_facecolor(col); patch.set_alpha(0.5)
    b.axhline(0, color="k", lw=0.8, ls=":")
    b.set_yscale("symlog", linthresh=0.1 / DPY)
    b.set_ylabel("rate (1/day, symlog)")
    b.set_title("(B) Rate distributions, 1/day (note M2 outliers = poorly constrained)", fontsize=10)

    # C: consistency outcomes
    c = ax[1, 0]
    comp = df.dropna(subset=["agree"])
    agree = int(comp.agree.sum()); dis = len(comp) - agree
    only = n - len(comp)
    c.bar(["agree\n(<=1 oom)", "DISAGREE\n(>1 oom)", "not comparable\n(<2 positive)"],
          [agree, dis, only], color=["#2e7d32", "#c0392b", "#bbbbbb"])
    for i, v in enumerate([agree, dis, only]):
        c.text(i, v + 0.2, str(v), ha="center", fontsize=10)
    c.set_ylabel("sites"); c.set_title("(C) Cross-method consistency (never averaged)", fontsize=10)

    # D: Method 2 reliability
    d = ax[1, 1]
    m2u = int(np.isfinite(df.m2).sum())
    good = m2u - int((df.m2_poorly | df.m2_fast).sum())
    poorly = int(df.m2_poorly.sum()); fast = int(df.m2_fast.sum())
    d.bar(["reliable", "poorly\nconstrained", "implausibly\nfast"],
          [good, poorly, fast], color=["#2e7d32", "#e08e0b", "#c0392b"])
    for i, v in enumerate([good, poorly, fast]):
        d.text(i, v + 0.1, str(v), ha="center", fontsize=10)
    d.set_ylabel("sites (of usable M2)")
    d.set_title(f"(D) Method 2 reliability ({m2u} usable of {n})", fontsize=10)

    fig.suptitle(f"Biodeg rate estimators — portfolio QA/QC across {n} sites",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/biodeg_rates_pf")
    a = ap.parse_args()
    df = load_runs(a.dir)
    if not len(df):
        print("no handoff JSONs found under", a.dir); return 1
    df = df.sort_values("folder").reset_index(drop=True)
    evaluate(df)

    rob = robustness(df)
    counts = rob.robustness.value_counts().to_dict()
    print("\n================= PER-SITE ROBUSTNESS (ranked) =================")
    print("  tiers:", "  ".join(f"{k}={counts.get(k,0)}" for k in ("STRONG", "MODERATE", "WEAK", "NONE")))
    print("  (a method counts as solid* only at medium+ confidence with no disqualifying flag)\n")
    print("  (all rates in 1/day, MODFLOW units)\n")
    show = rob[["folder", "n_wells", "M1_k_per_day", "M1_conf", "flow_lambda_per_day", "flow_conf",
                "M3_centre_per_day", "M3_conf", "M3_ballooning", "robustness"]]
    print(show.to_string(index=False))

    csv = os.path.join(a.dir, "_portfolio_summary.csv")
    png = os.path.join(a.dir, "_portfolio_qaqc.png")
    robcsv = os.path.join(a.dir, "_portfolio_robustness.csv")
    df.to_csv(csv, index=False)
    rob.to_csv(robcsv, index=False)
    figure(df, png)
    print("\n  summary table:   ", os.path.abspath(csv))
    print("  robustness table:", os.path.abspath(robcsv))
    print("  figure:          ", os.path.abspath(png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
