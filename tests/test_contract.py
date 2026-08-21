"""Contract: rate bookkeeping, JSON safety, and the no-averaging rule."""

import math

import numpy as np

from biodeg_rates.contract import (Estimand, RateEstimate, SiteRateBundle,
                                    METHOD_MANN_KENDALL, METHOD_DOMENICO, METHOD_ST_PSPLINE)


def test_half_life_cases():
    assert math.isclose(RateEstimate.half_life(math.log(2)), 1.0)
    assert RateEstimate.half_life(0.0) == float("inf")
    assert RateEstimate.half_life(-0.1) == float("inf")
    assert math.isnan(RateEstimate.half_life(float("nan")))


def test_not_applicable():
    e = RateEstimate.not_applicable(METHOD_DOMENICO, "site", "no transect")
    assert e.confidence == "N/A"
    assert math.isnan(e.value_per_year)
    assert e.estimand is Estimand.FLOWPATH_LAMBDA
    assert e.removes_dilution is True


def test_to_dict_is_json_safe():
    e = RateEstimate(method=METHOD_MANN_KENDALL, estimand=Estimand.POINT_DECAY, scope="well:A",
                     value_per_year=float("nan"), ci_low=float("-inf"), ci_high=float("inf"),
                     half_life_years=float("inf"), n=5)
    d = e.to_dict()
    assert d["value_per_year"] is None          # NaN -> None
    assert d["ci_high_per_year"] == "inf"        # +inf -> "inf"
    assert d["ci_low_per_year"] == "-inf"
    import json
    json.dumps(d)                                # must not raise


def _site_est(method, val):
    est = {
        METHOD_MANN_KENDALL: Estimand.POINT_DECAY,
        METHOD_DOMENICO: Estimand.FLOWPATH_LAMBDA,
        METHOD_ST_PSPLINE: Estimand.SPLINE_CENTRE_DECAY,
    }[method]
    return RateEstimate(method=method, estimand=est, scope="site", value_per_year=val,
                        ci_low=val * 0.5, ci_high=val * 1.5,
                        half_life_years=RateEstimate.half_life(val), n=10, confidence="medium")


def test_consistency_agree_within_oom():
    b = SiteRateBundle(site_name="S", analyte="benzene",
                       method1_summary=_site_est(METHOD_MANN_KENDALL, 0.30),
                       method2=_site_est(METHOD_DOMENICO, 0.10),
                       method3=_site_est(METHOD_ST_PSPLINE, 0.20))
    cc = b.consistency_check()
    assert cc["agree_within_1_oom"] is True
    assert cc["spread_orders_of_magnitude"] < 1.0


def test_consistency_disagree_flags():
    b = SiteRateBundle(site_name="S", analyte="benzene",
                       method1_summary=_site_est(METHOD_MANN_KENDALL, 0.30),
                       method2=_site_est(METHOD_DOMENICO, 41.0),
                       method3=_site_est(METHOD_ST_PSPLINE, 0.25))
    cc = b.consistency_check()
    assert cc["agree_within_1_oom"] is False
    assert "do not average" in cc["note"].lower()


def test_representative_rates_keeps_methods_separate():
    b = SiteRateBundle(site_name="S", analyte="benzene",
                       method1_summary=_site_est(METHOD_MANN_KENDALL, 0.30),
                       method2=_site_est(METHOD_DOMENICO, 0.10))
    rep = b.representative_rates()
    assert set(rep) == {METHOD_MANN_KENDALL, METHOD_DOMENICO}   # never merged into one
