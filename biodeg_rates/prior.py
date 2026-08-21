"""Literature priors for biodegradation rates, the informed-prior combine, and the QA/QC check.

The robust literature default for benzene is the McHugh et al. (2023) median first-order
attenuation rate from 1905 California GeoTracker (LUST) petroleum sites: 0.14/yr (half-life
4.9 yr; 82% of sites attenuating). That paper's central finding is that a portfolio median is a
better predictor of a site's future attenuation than the site's own historical rate, which is
exactly why it is our robust prior and the anchor for an informed (prior + site) estimate.

Estimand note (honesty contract): McHugh's k_c-max is an APPARENT first-order
concentration-vs-time attenuation rate (it lumps source decline, dilution, and reaction). It is
the same kind of quantity as our Method 1 point decay, so the informed combine uses the Method 1
site rate. It is NOT a dilution-removed reaction rate (Method 2); when a reliable Method 2 rate
exists, the handoff still prefers it for the mechanistic MODFLOW decay term.

The combine is a conjugate Gaussian update in log space (rates are positive, multiplicative). The
QA/QC rule: if the site estimate falls OUTSIDE the prior's population CI, flag it for engineer
review (per the paper, a lone site estimate that disagrees with the population is suspect).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DAYS_PER_YEAR = 365.0           # MODFLOW day/year convention
_Z95 = 1.96


@dataclass(frozen=True)
class LiteraturePrior:
    """A population prior for a first-order attenuation rate (1/yr).

    ``median_per_year`` is the population central value. ``ci_low/high_per_year`` is the
    population PLAUSIBLE RANGE for an individual site (the empirical across-site spread), used to
    flag site estimates for review; it is NOT the (much tighter) standard error of the median.
    ``log_sd`` is the prior standard deviation in log space (from the positive-rate spread), used
    for the conjugate combine.
    """
    analyte: str
    median_per_year: float
    ci_low_per_year: float
    ci_high_per_year: float
    log_sd: float
    source: str
    n_sites: int | None = None
    half_life_years: float | None = None
    note: str = ""


# Robust literature defaults. Benzene = McHugh et al. (2023). All numbers are computed from the
# paper's Supporting Information (Data S1, full-record first-order k_c-max, n=1905 GeoTracker LUST
# sites): median 0.141/yr; 82% attenuating; empirical 5th-95th percentile [-0.15, 0.53]/yr (the
# population plausible range, including the 18% of increasing sites); positive-rate geometric SD
# 2.79 -> log_sd 1.03. This replaces the earlier guessed CI with the empirical distribution.
LITERATURE_PRIORS: dict[str, LiteraturePrior] = {
    "benzene": LiteraturePrior(
        analyte="benzene", median_per_year=0.14,
        ci_low_per_year=-0.15, ci_high_per_year=0.53, log_sd=1.03,
        source=("McHugh et al. 2023, Groundwater Monitoring & Remediation 43(4):92-103, Data S1; "
                "median first-order k_c-max, n=1905 GeoTracker LUST sites; 5th-95th pctile "
                "[-0.15, 0.53]/yr"),
        n_sites=1905, half_life_years=4.9,
        note=("Apparent concentration-vs-time attenuation rate (bulk; comparable to Method 1, "
              "not a dilution-removed reaction rate). 82% of sites attenuating; the CI is the "
              "empirical population 5th-95th percentile. Population median is a more robust "
              "predictor than a single site's history.")),
}


@dataclass(frozen=True)
class InformedPrior:
    """The prior (or prior+site posterior) used to seed/inform the model, with the QA verdict."""
    analyte: str
    rate_per_year: float
    ci_low_per_year: float
    ci_high_per_year: float
    source: str                     # "literature" | "informed (literature + site Method 1)"
    site_rate_per_year: float | None
    qa_flag: bool                   # True -> site estimate outside the prior CI; engineer review
    qa_message: str
    literature_source: str

    @staticmethod
    def _pd(v):
        return None if v is None else round(v / DAYS_PER_YEAR, 8)

    def to_dict(self) -> dict:
        return {
            "analyte": self.analyte,
            "rate_per_year": round(self.rate_per_year, 5),
            "rate_per_day": self._pd(self.rate_per_year),
            "ci_per_year": [round(self.ci_low_per_year, 5), round(self.ci_high_per_year, 5)],
            "ci_per_day": [self._pd(self.ci_low_per_year), self._pd(self.ci_high_per_year)],
            "source": self.source,
            "site_rate_per_year": (round(self.site_rate_per_year, 5)
                                   if self.site_rate_per_year is not None else None),
            "qa_review_required": self.qa_flag,
            "qa_message": self.qa_message,
            "literature_source": self.literature_source,
        }


def _log_sd_from_ci(lo: float, hi: float) -> float:
    return (math.log(hi) - math.log(lo)) / (2.0 * _Z95)


def informed_prior(analyte: str, site_rate_per_year: float | None = None,
                   site_ci_per_year: tuple[float, float] | None = None) -> InformedPrior | None:
    """Combine the literature prior for ``analyte`` with a site estimate into an informed prior.

    No usable site estimate -> the robust literature prior. A positive site estimate within the
    prior CI -> a conjugate log-space posterior. A site estimate outside the prior CI (including a
    non-positive / increasing rate) -> qa_flag set for engineer review, and the robust prior is
    used as the safe default rather than the suspect site value.
    """
    lp = LITERATURE_PRIORS.get(analyte)
    if lp is None:
        return None

    # no site estimate: use the robust prior as-is
    if site_rate_per_year is None or not math.isfinite(site_rate_per_year):
        return InformedPrior(analyte, lp.median_per_year, lp.ci_low_per_year, lp.ci_high_per_year,
                             source="literature", site_rate_per_year=None, qa_flag=False,
                             qa_message="No site estimate; using the robust literature prior.",
                             literature_source=lp.source)

    # QA/QC plausibility: is the site estimate inside the empirical population 5th-95th band?
    outside = not (lp.ci_low_per_year <= site_rate_per_year <= lp.ci_high_per_year)
    ci_txt = f"[{lp.ci_low_per_year:.3g}, {lp.ci_high_per_year:.3g}]/yr"

    # a non-positive (increasing) site rate cannot be combined in log space and cannot seed a
    # positive model decay: report the robust prior. Flag for review only if also outside the
    # population band (a mild increase is within the population and need not be flagged).
    if site_rate_per_year <= 0:
        msg = (f"Site estimate {site_rate_per_year:.3g}/yr is non-positive. "
               + (f"OUTSIDE the population CI {ci_txt}; engineer review required. "
                  if outside else "Within the population CI (some sites increase). ")
               + "Using the robust prior median as the model decay (cannot seed a negative rate).")
        return InformedPrior(analyte, lp.median_per_year, lp.ci_low_per_year, lp.ci_high_per_year,
                             source="literature (site non-positive)",
                             site_rate_per_year=site_rate_per_year, qa_flag=outside,
                             qa_message=msg, literature_source=lp.source)

    # conjugate Gaussian update in log space: prior log-SD from the population spread
    sig0 = lp.log_sd
    if (site_ci_per_year and site_ci_per_year[0] and site_ci_per_year[1]
            and site_ci_per_year[0] > 0 and site_ci_per_year[1] > site_ci_per_year[0]):
        sigs = _log_sd_from_ci(site_ci_per_year[0], site_ci_per_year[1])
    else:
        sigs = sig0                       # site as uncertain as the prior when no CI is supplied
    sigs = max(sigs, 1e-6)
    mu0, mus = math.log(lp.median_per_year), math.log(site_rate_per_year)
    prec = 1.0 / sig0 ** 2 + 1.0 / sigs ** 2
    mu_post = (mu0 / sig0 ** 2 + mus / sigs ** 2) / prec
    sig_post = math.sqrt(1.0 / prec)
    central = math.exp(mu_post)
    ci_lo, ci_hi = math.exp(mu_post - _Z95 * sig_post), math.exp(mu_post + _Z95 * sig_post)

    if outside:
        msg = (f"Site estimate {site_rate_per_year:.3g}/yr is OUTSIDE the population CI {ci_txt}. "
               f"Engineer review required before use; the combined value is for reference only.")
    else:
        msg = (f"Site estimate within the population CI; combined into an informed prior "
               f"(literature median {lp.median_per_year:.3g}/yr + site {site_rate_per_year:.3g}/yr).")
    return InformedPrior(analyte, central, ci_lo, ci_hi,
                         source="informed (literature + site Method 1)",
                         site_rate_per_year=site_rate_per_year, qa_flag=outside,
                         qa_message=msg, literature_source=lp.source)
