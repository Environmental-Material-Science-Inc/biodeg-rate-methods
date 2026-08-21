"""Literature prior + informed-prior combine + QA/QC outside-CI flag (McHugh 2023 benzene)."""

import math

from biodeg_rates import prior


def test_benzene_literature_default_is_mchugh():
    lp = prior.LITERATURE_PRIORS["benzene"]
    assert lp.median_per_year == 0.14            # McHugh 2023 median
    assert lp.n_sites == 1905
    assert "McHugh" in lp.source


def test_no_site_estimate_uses_robust_prior():
    ip = prior.informed_prior("benzene", site_rate_per_year=None)
    assert ip.source == "literature"
    assert ip.rate_per_year == 0.14
    assert ip.qa_flag is False
    # per-day equivalents present and consistent (MODFLOW units, 365.0 day/yr)
    d = ip.to_dict()
    assert abs(d["rate_per_day"] - 0.14 / 365.0) < 1e-6


def test_site_within_ci_combines():
    ip = prior.informed_prior("benzene", site_rate_per_year=0.20,
                              site_ci_per_year=(0.10, 0.40))
    assert ip.source.startswith("informed")
    assert ip.qa_flag is False
    # posterior sits between the prior median (0.14) and the site (0.20)
    assert 0.14 <= ip.rate_per_year <= 0.20


def test_site_outside_ci_flags_for_review():
    ip = prior.informed_prior("benzene", site_rate_per_year=10.0,   # implausibly fast
                              site_ci_per_year=(5.0, 20.0))
    assert ip.qa_flag is True
    assert "review" in ip.qa_message.lower()


def test_strong_increase_flags_and_falls_back():
    ip = prior.informed_prior("benzene", site_rate_per_year=-0.4)   # outside population CI
    assert ip.qa_flag is True
    assert ip.rate_per_year == 0.14                                 # safe default = robust prior
    assert "review" in ip.qa_message.lower()


def test_mild_increase_within_population_not_flagged():
    # a small negative rate is within the empirical population (18% of sites increase): not a
    # QA flag, but still cannot seed a negative model decay -> robust prior used
    ip = prior.informed_prior("benzene", site_rate_per_year=-0.10)
    assert ip.qa_flag is False
    assert ip.rate_per_year == 0.14


def test_ci_matches_supplementary_distribution():
    lp = prior.LITERATURE_PRIORS["benzene"]
    assert (lp.ci_low_per_year, lp.ci_high_per_year) == (-0.15, 0.53)   # Data S1 5th-95th pctile


def test_unknown_analyte_returns_none():
    assert prior.informed_prior("trichloroethene", site_rate_per_year=0.1) is None
