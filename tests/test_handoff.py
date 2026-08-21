"""Module 5: handoff assembly and the hard no-averaging guarantee."""

import json

import numpy as np
import pytest

from biodeg_rates import handoff
from biodeg_rates.contract import RateEstimate, Estimand, METHOD_DOMENICO


def test_forbid_average_raises():
    with pytest.raises(handoff.CombineRefused):
        handoff.forbid_average(0.1, 0.2, 0.3)


def test_modflow_block_uses_reaction_rate_in_per_day():
    from biodeg_rates.contract import METHOD_DOMENICO
    rxn = RateEstimate(method=METHOD_DOMENICO, estimand=Estimand.FLOWPATH_LAMBDA, scope="site",
                       value_per_year=0.365, ci_low=0.1825, ci_high=0.73,
                       half_life_years=RateEstimate.half_life(0.365), n=8,
                       confidence="medium (assumed params)", removes_dilution=True)
    blk = handoff.modflow_decay_block(rxn)
    assert blk["ready_for_modflow"] is True
    assert abs(blk["decay_rate_per_day"] - 0.365 / 365.0) < 1e-9    # 1/yr -> 1/day
    assert blk["decay"] == blk["decay_sorbed"]                       # the downstream model sets them equal
    assert blk["pest_prior_settings_per_day"]["lower_bound"] is not None


def test_all_rates_available_in_per_day(synth_site):
    from biodeg_rates.contract import (METHOD_MANN_KENDALL, METHOD_DOMENICO,
                                                 METHOD_ST_PSPLINE)
    m1 = _est(METHOD_MANN_KENDALL, Estimand.POINT_DECAY, 0.365)
    m2 = _est(METHOD_DOMENICO, Estimand.FLOWPATH_LAMBDA, 0.365)
    m3 = _est(METHOD_ST_PSPLINE, Estimand.SPLINE_CENTRE_DECAY, 0.365)
    p = handoff.build_handoff(synth_site, handoff.assemble_bundle(synth_site, [m1], m1, m2, m3))
    assert p["modflow_rate_units"] == "1/day" and p["days_per_year_for_conversion"] == 365.0
    # every menu entry carries a 1/day rate consistent with its 1/year value
    for e in p["initialization_menu"]:
        if e["first_order_rate_per_year"] is not None:
            assert abs(e["first_order_rate_per_day"] - e["first_order_rate_per_year"] / 365.0) < 1e-6
    # site-level estimate blocks carry per-day too
    blk = p["estimates"][METHOD_DOMENICO]
    assert abs(blk["value_per_day"] - blk["value_per_year"] / 365.0) < 1e-6


def test_conservative_policy_picks_smallest_reliable_rate():
    from biodeg_rates.contract import METHOD_DOMENICO, METHOD_DOMENICO_2D
    def rxn(method, val):
        return RateEstimate(method=method, estimand=Estimand.FLOWPATH_LAMBDA, scope="site",
                            value_per_year=val, ci_low=val * 0.6, ci_high=val * 1.4,
                            half_life_years=RateEstimate.half_life(val), n=8, removes_dilution=True,
                            confidence="medium (assumed params)")
    small = rxn(METHOD_DOMENICO, 0.12)        # 1D
    big = rxn(METHOD_DOMENICO_2D, 0.40)       # 2D, more accurate but larger
    chosen = handoff._conservative_reaction_rate(small, big)
    assert chosen.value_per_year == 0.12      # smallest = most conservative, not the "best"
    blk = handoff.modflow_decay_block(chosen)
    assert blk["ready_for_modflow"] is True
    assert abs(blk["decay_rate_per_day"] - 0.12 / 365.0) < 1e-6
    assert "conservative" in blk["selection_policy"].lower()


def test_modflow_block_refuses_unreliable_method2():
    from biodeg_rates.contract import METHOD_DOMENICO
    # a positive, dilution-removing Method 2 value that is flagged poorly constrained must NOT
    # be marked MODFLOW-ready (do not seed the model with an untrustworthy reaction rate)
    rxn = RateEstimate(method=METHOD_DOMENICO, estimand=Estimand.FLOWPATH_LAMBDA, scope="site",
                       value_per_year=10.5, ci_low=-5.0, ci_high=1900.0,
                       half_life_years=RateEstimate.half_life(10.5), n=5, removes_dilution=True,
                       diagnostics={"poorly_constrained": True, "implausibly_fast": True})
    blk = handoff.modflow_decay_block(rxn)
    assert blk["ready_for_modflow"] is False
    assert blk["decay_rate_per_day"] is None


def test_modflow_block_refuses_bulk_or_na():
    from biodeg_rates.contract import METHOD_MANN_KENDALL, METHOD_DOMENICO
    # a bulk rate (removes_dilution False) must not become the MST decay coefficient
    bulk = RateEstimate(method=METHOD_MANN_KENDALL, estimand=Estimand.POINT_DECAY, scope="site",
                        value_per_year=0.5, ci_low=0.3, ci_high=0.7,
                        half_life_years=RateEstimate.half_life(0.5), n=10, removes_dilution=False)
    blk = handoff.modflow_decay_block(bulk)
    assert blk["ready_for_modflow"] is False and blk["decay_rate_per_day"] is None
    # an N/A reaction rate also yields no MODFLOW input
    na = RateEstimate.not_applicable(METHOD_DOMENICO, "site", "no transect")
    assert handoff.modflow_decay_block(na)["ready_for_modflow"] is False


def test_conservative_fallback_is_never_negative(synth_site):
    # When no reliable site reaction rate exists, the handoff offers the informed literature
    # prior's lower bound as the conservative fallback seed. The benzene population band runs
    # negative (18% of LUST sites increase), but a MODFLOW first-order MST decay coefficient cannot
    # be negative (that would generate mass). Confirm the fallback is clamped to zero, the most
    # protective valid seed, not handed through negative.
    from biodeg_rates.contract import (METHOD_MANN_KENDALL, METHOD_DOMENICO,
                                                 METHOD_DOMENICO_2D, METHOD_ST_PSPLINE)
    m1 = RateEstimate.not_applicable(METHOD_MANN_KENDALL, "site", "no usable well")
    m2 = RateEstimate.not_applicable(METHOD_DOMENICO, "site", "no transect")
    m2_2d = RateEstimate.not_applicable(METHOD_DOMENICO_2D, "site", "too few wells")
    m3 = _est(METHOD_ST_PSPLINE, Estimand.SPLINE_CENTRE_DECAY, 0.18)
    payload = handoff.build_handoff(
        synth_site, handoff.assemble_bundle(synth_site, [m1], m1, m2, m3, method2_2d=m2_2d))
    mf = payload["modflow_input"]
    assert mf.get("ready_for_modflow") is False
    fb = mf.get("conservative_fallback_per_day")
    assert fb is not None and fb >= 0.0                       # never a negative decay coefficient
    # the benzene literature lower bound is negative, so the clamp must have engaged with a note
    assert "clamped to 0" in mf.get("conservative_fallback_note", "")


def _est(method, estd, val):
    return RateEstimate(method=method, estimand=estd, scope="site", value_per_year=val,
                        ci_low=val * 0.6, ci_high=val * 1.4,
                        half_life_years=RateEstimate.half_life(val), n=10, confidence="medium")


def test_build_handoff_keeps_three_separate(synth_site):
    from biodeg_rates.contract import (METHOD_MANN_KENDALL, METHOD_ST_PSPLINE)
    m1 = _est(METHOD_MANN_KENDALL, Estimand.POINT_DECAY, 0.15)
    m2 = _est(METHOD_DOMENICO, Estimand.FLOWPATH_LAMBDA, 0.10)
    m3 = _est(METHOD_ST_PSPLINE, Estimand.SPLINE_CENTRE_DECAY, 0.18)
    bundle = handoff.assemble_bundle(synth_site, [m1], m1, m2, m3)
    payload = handoff.build_handoff(synth_site, bundle)

    # estimates live in separate slots (incl. the 2D flowpath variant key, value may be None here)
    from biodeg_rates.contract import METHOD_DOMENICO_2D
    assert {METHOD_MANN_KENDALL, METHOD_DOMENICO, METHOD_DOMENICO_2D,
            METHOD_ST_PSPLINE} == set(payload["estimates"])
    # the menu has one row per non-None method, and nothing pre-selected (the modeller chooses)
    assert len(payload["initialization_menu"]) == 3
    assert payload["selected_for_initialization"] is None
    # no averaged/combined rate key anywhere in the payload
    text = json.dumps(payload).lower()
    assert "average" not in text or "do not average" in text or "never average" in text
    # the consistency check reports a spread but not a merged value
    assert "spread_orders_of_magnitude" in payload["consistency_check"]


def test_handoff_reports_both_flowpath_methods_separately(synth_site):
    from biodeg_rates.contract import (METHOD_MANN_KENDALL, METHOD_DOMENICO, METHOD_DOMENICO_2D,
                                        METHOD_ST_PSPLINE)
    m1 = _est(METHOD_MANN_KENDALL, Estimand.POINT_DECAY, 0.15)
    m2 = _est(METHOD_DOMENICO, Estimand.FLOWPATH_LAMBDA, 0.10)
    m2_2d = _est(METHOD_DOMENICO_2D, Estimand.FLOWPATH_LAMBDA, 0.13)
    m3 = _est(METHOD_ST_PSPLINE, Estimand.SPLINE_CENTRE_DECAY, 0.18)
    bundle = handoff.assemble_bundle(synth_site, [m1], m1, m2, m3, method2_2d=m2_2d)
    payload = handoff.build_handoff(synth_site, bundle)
    # both flowpath variants present as distinct, un-merged entries
    assert METHOD_DOMENICO in payload["estimates"]
    assert METHOD_DOMENICO_2D in payload["estimates"]
    methods_in_menu = {m["method"] for m in payload["initialization_menu"]}
    assert {METHOD_DOMENICO, METHOD_DOMENICO_2D}.issubset(methods_in_menu)
    # the cross-check uses ONE flowpath representative (same estimand), prefers the 2D value
    assert payload["consistency_check"]["values_per_year"][METHOD_DOMENICO] == 0.13


def test_write_handoff_roundtrips(tmp_path, synth_site):
    from biodeg_rates.contract import (METHOD_MANN_KENDALL, METHOD_ST_PSPLINE)
    m1 = _est(METHOD_MANN_KENDALL, Estimand.POINT_DECAY, 0.15)
    bundle = handoff.assemble_bundle(synth_site, [m1], m1,
                                     RateEstimate.not_applicable(METHOD_DOMENICO, "site", "no transect"),
                                     _est(METHOD_ST_PSPLINE, Estimand.SPLINE_CENTRE_DECAY, 0.18))
    path = handoff.write_handoff(synth_site, bundle, str(tmp_path))
    loaded = json.loads(open(path).read())
    assert loaded["schema"] == handoff.SCHEMA_VERSION
    assert loaded["analyte"] == "benzene"
    # the N/A Domenico estimate is carried as not usable, not dropped or averaged away
    dom = loaded["estimates"][METHOD_DOMENICO]
    assert dom["confidence"] == "N/A"
