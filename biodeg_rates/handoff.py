"""Module 5 — collect the three estimates and prepare the modelling-startup handoff.

This module assembles the three method results into one ``SiteRateBundle`` and serializes a
handoff payload for the code that initializes the reactive-transport model. It enforces the
rule that the three estimates are different quantities: they travel in SEPARATE slots, the
order-of-magnitude consistency check never averages them, and ``forbid_average`` exists so
that any attempt to collapse them to a single number fails loudly.

The startup code (or the modeller) selects ONE estimand to seed the first-order rate, with
its eyes open about what that estimand means:

  point_decay_k        statistical per-well decline; includes dilution; good sanity bound
  flowpath_lambda      mechanistic reaction coefficient with dilution removed; the closest to a
                       reaction rate, but parameter-sensitive (dispersivity)
  spline_centre_decay  smoothed concentration decline at the plume centre; includes dilution;
                       the robust apparent-attenuation rate read off the spatio-temporal surface

The selection is deliberately left null in the payload (``selected_for_initialization``); the
handoff records the menu and the caveats, it does not make the modelling decision.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .contract import (RateEstimate, SiteObservations, SiteRateBundle,
                        METHOD_MANN_KENDALL, METHOD_DOMENICO, METHOD_DOMENICO_2D,
                        METHOD_ST_PSPLINE, METHOD_ESTIMAND, _jsonable)
from . import prior as _prior

SCHEMA_VERSION = "biodeg_rates.handoff/1.1"

# The downstream model runs MODFLOW 6 in days (TDIS time_units='days'); its day/year factor is 365.0.
# Every rate in this handoff is reported in BOTH 1/year and 1/day, converted with this factor, so
# a rate is never dropped into a 1/day model field in the wrong unit.
MODFLOW_DAYS_PER_YEAR = 365.0


def _per_day(rate_per_year):
    """Convert a 1/year rate to the 1/day MODFLOW units (None/NaN safe)."""
    if rate_per_year is None:
        return None
    try:
        v = float(rate_per_year)
    except (TypeError, ValueError):
        return None
    if v != v:                       # NaN
        return None
    return round(v / MODFLOW_DAYS_PER_YEAR, 8)


class CombineRefused(Exception):
    """Raised on any attempt to average or merge the three estimands."""


def forbid_average(*_args, **_kwargs):
    """The three estimates are not interchangeable; there is no valid mean of them."""
    raise CombineRefused(
        "the three estimates are different physical quantities (point decay, flowpath lambda, "
        "spline-centre decay) and must never be averaged or merged into one rate")


def assemble_bundle(site: SiteObservations,
                    method1_per_well: list[RateEstimate],
                    method1_summary: RateEstimate | None,
                    method2: RateEstimate | None,
                    method3: RateEstimate | None,
                    method2_2d: RateEstimate | None = None) -> SiteRateBundle:
    return SiteRateBundle(
        site_name=site.site_name, analyte=site.analyte,
        method1_per_well=list(method1_per_well or []),
        method1_summary=method1_summary, method2=method2, method2_2d=method2_2d,
        method3=method3,
    )


def _conservative_reaction_rate(m2_1d, m2_2d):
    """Choose the flowpath reaction rate to offer MODFLOW under the CONSERVATIVE policy: the
    SMALLEST reliable dilution-removed rate. The smallest decay gives the longest plume
    persistence and the largest predicted extent, so it is the protective assumption. Falls back
    to the smallest usable rate (for the not-ready message) if none is reliable."""
    def reliable(e):
        return (e is not None and e.removes_dilution and np.isfinite(e.value_per_year)
                and e.value_per_year > 0
                and not e.diagnostics.get("poorly_constrained")
                and not e.diagnostics.get("implausibly_fast"))
    cands = [e for e in (m2_1d, m2_2d) if reliable(e)]
    if cands:
        return min(cands, key=lambda e: e.value_per_year)     # smallest = most conservative
    usable = [e for e in (m2_1d, m2_2d) if e is not None and np.isfinite(e.value_per_year)]
    if usable:
        return min(usable, key=lambda e: e.value_per_year)
    return m2_2d or m2_1d


def _estimate_block(est: RateEstimate | None) -> dict | None:
    if est is None:
        return None
    d = est.to_dict()
    # add the 1/day equivalents MODFLOW needs, alongside the 1/year scientific values
    d["value_per_day"] = _per_day(est.value_per_year)
    d["ci_low_per_day"] = _per_day(est.ci_low)
    d["ci_high_per_day"] = _per_day(est.ci_high)
    return d


def build_handoff(site: SiteObservations, bundle: SiteRateBundle) -> dict:
    """The full handoff payload (a plain dict, JSON-serializable)."""
    m1 = bundle.method1_summary
    m2 = bundle.method2
    m2_2d = bundle.method2_2d
    m3 = bundle.method3

    # the "menu" the startup code chooses from: one entry per method, never combined
    menu = []
    for est, axis in ((m1, "per-well point decay (statistical, includes dilution)"),
                      (m2, "flowpath reaction coefficient, 1D centerline transect (dilution removed)"),
                      (m2_2d, "flowpath reaction coefficient, 2D fit with alpha_y fitted (dilution removed)"),
                      (m3, "spline-centre concentration decay (smoothed, includes dilution)")):
        if est is None:
            continue
        usable = bool(np.isfinite(est.value_per_year)) and est.confidence != "N/A"
        menu.append({
            "method": est.method,
            "estimand": est.estimand.value,
            "interpretation": axis,
            "first_order_rate_per_day": _per_day(est.value_per_year),   # MODFLOW units
            "ci_per_day": [_per_day(est.ci_low), _per_day(est.ci_high)],
            "first_order_rate_per_year": _jsonable(est.value_per_year),
            "ci_per_year": [_jsonable(est.ci_low), _jsonable(est.ci_high)],
            "half_life_years": _jsonable(est.half_life_years),
            "confidence": est.confidence,
            "removes_dilution": est.removes_dilution,
            "usable_for_init": usable,
            "caveat": est.notes,
        })

    # informed prior: combine the literature default (McHugh 2023 for benzene) with the site's
    # apparent attenuation rate (Method 1, the matching estimand). No site rate -> robust prior;
    # site rate outside the prior CI -> QA flag for engineer review.
    m1_rate = m1.value_per_year if (m1 is not None and np.isfinite(m1.value_per_year)) else None
    m1_ci = ((m1.ci_low, m1.ci_high)
             if (m1 is not None and np.isfinite(m1.ci_low) and np.isfinite(m1.ci_high)) else None)
    informed = _prior.informed_prior(site.analyte, site_rate_per_year=m1_rate, site_ci_per_year=m1_ci)

    # MODFLOW reaction-rate block (mechanistic, Method 2). When no reliable reaction rate exists,
    # offer the informed literature prior's lower bound as the conservative fallback seed. The
    # benzene population band runs negative (18% of LUST sites increase), so clamp the lower bound
    # at zero: a first-order MST decay coefficient cannot be negative (that would GENERATE mass),
    # and zero decay is already the most protective seed (no attenuation credit, longest plume
    # persistence). This mirrors the non-negative clamp on the PEST bound in modflow_decay_block.
    mf = modflow_decay_block(_conservative_reaction_rate(m2, m2_2d))
    if not mf.get("ready_for_modflow") and informed is not None:
        prior_lo_per_day = informed.to_dict()["ci_per_day"][0]
        clamped = max(prior_lo_per_day, 0.0) if prior_lo_per_day is not None else 0.0
        mf["conservative_fallback_per_day"] = clamped
        note = ("No reliable site reaction rate. Conservative fallback: the lower bound of the "
                "informed literature prior (see informed_prior); smallest value = most protective.")
        if prior_lo_per_day is not None and prior_lo_per_day < 0.0:
            note += (f" The prior lower bound ({prior_lo_per_day:.2e}/day) is negative because the "
                     "population includes increasing sites; it is clamped to 0 (no decay credit), "
                     "the most protective physically valid seed.")
        mf["conservative_fallback_note"] = note

    payload = {
        "schema": SCHEMA_VERSION,
        "site": site.site_name,
        "site_id": site.site_id,        # stable folder-based id (site_name may differ from folder)
        "analyte": site.analyte,
        "epsg": site.epsg,
        "regulatory_threshold": site.threshold,
        "rate_units": "first-order constants reported in BOTH 1/year and 1/day",
        "modflow_rate_units": "1/day",
        "days_per_year_for_conversion": MODFLOW_DAYS_PER_YEAR,
        "source_location_xy": list(site.source_xy),
        "source_max_concentration": _jsonable(site.source_conc),

        # the estimates, kept strictly separate (two flowpath variants reported side by side)
        "estimates": {
            METHOD_MANN_KENDALL: {
                "site_summary": _estimate_block(m1),
                "per_well": [e.to_dict() for e in bundle.method1_per_well],
            },
            METHOD_DOMENICO: _estimate_block(m2),
            METHOD_DOMENICO_2D: _estimate_block(m2_2d),
            METHOD_ST_PSPLINE: _estimate_block(m3),
        },

        # the menu and the cross-check; NO averaged value anywhere
        "initialization_menu": menu,
        "selected_for_initialization": None,   # the modeller/startup code picks ONE estimand
        "consistency_check": bundle.consistency_check(),

        # ready-to-consume block for the downstream MODFLOW 6 GWT transport model
        "modflow_input": mf,

        # literature-anchored informed prior + QA/QC verdict (combine + outside-CI flag)
        "informed_prior": informed.to_dict() if informed is not None else None,

        "handling_rules": [
            "LITERATURE DEFAULT: the robust prior for benzene is McHugh et al. 2023 (median "
            "0.14/yr = 0.000383/day, n=1905 LUST sites). See informed_prior. With no reliable site "
            "rate, seed the model from the informed prior; the downstream default_model_priors "
            "benzene value (0.0029/day) is ~7.5x higher and should be reconciled to this.",
            "QA/QC: if informed_prior.qa_review_required is true, the site estimate is outside the "
            "literature population CI. Hold for engineer review before using the site value.",
            "CONSERVATIVE POLICY: always seed the model with the SMALLEST decay rate. The smallest "
            "rate is the most conservative (protective) choice: it gives the longest plume "
            "persistence and the largest predicted extent. modflow_input selects the smallest "
            "reliable reaction rate; if none is reliable, use the informed prior's LOWER bound "
            "(conservative_fallback_per_day), not a central value.",
            "The three estimates are different physical quantities. Do not average or blend them.",
            "flowpath_lambda is the only mechanistic reaction rate (dilution removed); the other "
            "two are statistical bulk rates that include dilution.",
            "For a MODFLOW/MT3D/GWT transport model use ONLY the dilution-removed reaction rate "
            "(flowpath_lambda); see modflow_input. The solver already simulates advection and "
            "dispersion, so a bulk rate (Method 1 or 3) double-counts dilution and over-predicts "
            "destruction.",
            "Carry the chosen estimate's confidence and caveat into the model run log.",
            "If the consistency_check shows more than one order of magnitude of spread, treat all "
            "three as weakly constrained and prefer the literature prior's lower bound.",
        ],
    }
    return payload


def modflow_decay_block(reaction: RateEstimate | None) -> dict:
    """Build the MODFLOW 6 GWT-MST first-order decay input.

    The downstream model sets the MST package with first_order_decay=True, sorption='LINEAR', and
    decay = decay_sorbed = decay_rate (a single first-order constant in 1/day). Only the
    dilution-removed reaction rate (Method 2 flowpath lambda) belongs here, because the GWT
    solver integrates advection + dispersion itself; a bulk statistical rate would double-count
    the dilution and over-predict destruction. Our rates are 1/year, so they are converted to
    1/day. CI bounds become the PEST prior bounds.
    """
    block = {
        "target": "MODFLOW 6 GWT MST (flopy ModflowGwtmst) — downstream decay_rate prior",
        "mst_parameters": "first_order_decay=True, sorption='LINEAR', decay=decay_sorbed=decay_rate",
        "time_units": "days (downstream TDIS, 365.0 day/yr); rate reported in 1/day",
        "sign_convention": "positive value = first-order decay (mass removed at rate decay*C)",
        "selection_policy": "conservative: smallest reliable dilution-removed reaction rate "
                            "(smallest decay = longest plume persistence = protective)",
        "source_method": (reaction.method if reaction else None),
    }
    diag = reaction.diagnostics if reaction else {}
    unreliable = bool(diag.get("poorly_constrained") or diag.get("implausibly_fast"))
    usable = (reaction is not None and reaction.removes_dilution
              and np.isfinite(reaction.value_per_year) and reaction.value_per_year > 0
              and not unreliable)
    if not usable:
        block["decay_rate_per_day"] = None
        block["ready_for_modflow"] = False
        why = ("flagged unreliable (poorly constrained or implausibly fast)" if unreliable
               else "N/A or non-positive at this site")
        block["note"] = (
            f"No trustworthy dilution-removed reaction rate (Method 2 is {why}). Do NOT substitute "
            "a bulk rate (Method 1 or 3) as the MST decay coefficient — that double-counts the "
            "dilution the GWT solver already computes. Conservative fallback: seed from the "
            "literature decay_rate prior's LOWER bound (the smallest decay), not its central value.")
        return block

    per_day = reaction.value_per_year / MODFLOW_DAYS_PER_YEAR
    lo = (max(reaction.ci_low, 0.0) / MODFLOW_DAYS_PER_YEAR
          if np.isfinite(reaction.ci_low) else None)
    hi = (reaction.ci_high / MODFLOW_DAYS_PER_YEAR
          if (np.isfinite(reaction.ci_high) and reaction.ci_high > 0) else None)
    block["ready_for_modflow"] = True
    block["decay_rate_per_day"] = round(per_day, 8)
    block["decay_rate_per_year"] = _jsonable(reaction.value_per_year)
    block["decay"] = round(per_day, 8)
    block["decay_sorbed"] = round(per_day, 8)
    block["pest_prior_settings_per_day"] = {     # drop-in for downstream PriorSettings
        "initial_value": round(per_day, 8),
        "lower_bound": round(lo, 8) if lo is not None else None,
        "upper_bound": round(hi, 8) if hi is not None else None,
    }
    block["confidence"] = reaction.confidence
    block["note"] = (
        "Smallest reliable dilution-removed flowpath lambda, used as the MST first-order decay (a "
        "reaction rate, consistent with the transport equation) under the conservative policy. "
        "decay and decay_sorbed are set equal, matching the downstream model. Reconcile against the "
        "literature decay_rate prior before calibration, preferring the smaller value.")
    return block


def write_handoff(site: SiteObservations, bundle: SiteRateBundle, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = build_handoff(site, bundle)
    safe = site.site_name.replace(" ", "_").replace("/", "_")
    path = os.path.join(out_dir, f"{safe}_biodeg_handoff.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
