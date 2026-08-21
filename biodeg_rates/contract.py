"""Shared data contract for the three rate-estimation modules.

Two jobs:

1. Define the normalized site observation view (``SiteObservations`` / ``WellSeries``)
   that every estimator consumes, so ingestion is solved once and the estimators stay
   pure functions of data.

2. Encode the rule that the three estimates are different quantities that must never be
   averaged. ``Estimand`` tags each ``RateEstimate`` with what it physically is, and
   ``SiteRateBundle`` stores the three method results in separate slots with no method
   that returns a single combined rate.

Style: active voice, no em dashes, uncertainty always carried (see config.py header).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Estimand(str, Enum):
    """What a rate number physically is. Two estimates with different Estimand values
    are not comparable as the same quantity and cannot be averaged."""

    POINT_DECAY = "point_decay_k"          # module 1: per-well temporal decline, statistical
    FLOWPATH_LAMBDA = "flowpath_lambda"    # module 2: along-flow reaction coefficient, mechanistic
    SPLINE_CENTRE_DECAY = "spline_centre_decay"  # module 3: concentration decay at the plume centre
                                           # read from the spatio-temporal P-spline surface (-df/dt)


# Human-readable method identifiers (stable; used as dict keys in the handoff JSON).
METHOD_MANN_KENDALL = "mann_kendall_theil_sen"
METHOD_DOMENICO = "domenico_normalized"          # the 1D centerline-transect variant
METHOD_DOMENICO_2D = "domenico_2d_fit"           # the 2D fit (all wells; alpha_y fitted)
METHOD_ST_PSPLINE = "st_pspline_centre_decay"    # spline-centre concentration decay (module 3)

# Which Estimand each method produces, and whether it removes physical dilution.
METHOD_ESTIMAND = {
    METHOD_MANN_KENDALL: Estimand.POINT_DECAY,
    METHOD_DOMENICO: Estimand.FLOWPATH_LAMBDA,
    METHOD_DOMENICO_2D: Estimand.FLOWPATH_LAMBDA,
    METHOD_ST_PSPLINE: Estimand.SPLINE_CENTRE_DECAY,
}
METHOD_REMOVES_DILUTION = {
    METHOD_MANN_KENDALL: False,
    METHOD_DOMENICO: True,
    METHOD_DOMENICO_2D: True,
    METHOD_ST_PSPLINE: False,
}


@dataclass(frozen=True)
class RateEstimate:
    """One rate estimate, carrying its own uncertainty and limits (honesty contract).

    All rates are reported in 1/year on the natural-log scale (a first-order constant).
    ``value_per_year`` is NaN when the method does not apply or finds no usable trend;
    ``confidence`` then reads "N/A" and ``notes`` says why.
    """

    method: str                    # one of METHOD_* above
    estimand: Estimand
    scope: str                     # "site" or "well:<id>"
    value_per_year: float          # first-order rate k or lambda (1/yr); NaN if N/A
    ci_low: float                  # lower confidence/credible bound (1/yr); NaN if N/A
    ci_high: float                 # upper bound (1/yr); NaN if N/A
    half_life_years: float         # ln 2 / value; inf if value <= 0; NaN if N/A
    n: int = 0                     # observations the estimate rests on
    trend: str = ""                # "decreasing" | "increasing" | "no trend" | ""
    confidence: str = "low"        # "high" | "medium" | "low" | "low (assumed params)" | "N/A"
    removes_dilution: bool = False # True only for the Domenico flowpath coefficient
    diagnostics: dict = field(default_factory=dict)
    notes: str = ""

    @staticmethod
    def half_life(value_per_year: float) -> float:
        if value_per_year is None or (isinstance(value_per_year, float) and math.isnan(value_per_year)):
            return float("nan")
        if value_per_year <= 0:
            return float("inf")
        return math.log(2.0) / value_per_year

    @classmethod
    def not_applicable(cls, method: str, scope: str, why: str) -> "RateEstimate":
        return cls(method=method, estimand=METHOD_ESTIMAND[method], scope=scope,
                   value_per_year=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                   half_life_years=float("nan"), n=0, confidence="N/A",
                   removes_dilution=METHOD_REMOVES_DILUTION[method], notes=why)

    def to_dict(self) -> dict:
        d = {
            "method": self.method,
            "estimand": self.estimand.value,
            "scope": self.scope,
            "value_per_year": _jsonable(self.value_per_year),
            "ci_low_per_year": _jsonable(self.ci_low),
            "ci_high_per_year": _jsonable(self.ci_high),
            "half_life_years": _jsonable(self.half_life_years),
            "n": int(self.n),
            "trend": self.trend,
            "confidence": self.confidence,
            "removes_dilution": bool(self.removes_dilution),
            "notes": self.notes,
            "diagnostics": _jsonable(self.diagnostics),
        }
        return d


@dataclass
class WellSeries:
    """One well's full time series of the target analyte (the unit Method 1 consumes)."""

    well_id: str
    easting: float
    northing: float
    t_years: np.ndarray            # time since the site start date (years), ascending
    dates: np.ndarray              # datetime64 of each event (same order as t_years)
    conc: np.ndarray               # reported concentration; for non-detects = reporting limit
    detect: np.ndarray             # bool: True = quantified detection, False = non-detect
    rl: np.ndarray                 # reporting limit per event

    @property
    def n(self) -> int:
        return int(len(self.conc))

    @property
    def n_detect(self) -> int:
        return int(np.sum(self.detect))


@dataclass
class SiteObservations:
    """Normalized, method-agnostic view of one site. Built by ``dataio.load_site``.

    ``frame`` is the long-format table (one row per well-event) every estimator can slice:
    columns well_id, easting, northing, date, t_years, conc, detect, rl.
    """

    site_name: str
    analyte: str
    epsg: int | None
    threshold: float | None        # regulatory criterion for the analyte (same unit as conc)
    start_date: np.datetime64
    frame: "object"                # pandas.DataFrame (kept untyped to avoid a hard import here)
    wells: list[WellSeries]
    heads: "object"                # pandas.DataFrame: well_id, easting, northing, date, head_m
    source_xy: tuple[float, float] # hotspot: location of the maximum detected concentration
    source_conc: float             # that maximum concentration
    site_id: str | None = None     # stable identifier (the source folder), for traceability;
                                   # site_name may come from level0 and differ from the folder

    def well_locations(self) -> dict:
        return {w.well_id: (w.easting, w.northing) for w in self.wells}


@dataclass
class SiteRateBundle:
    """The three method results for one site, kept strictly separate.

    There is deliberately NO method that returns a single merged rate. The three live in
    their own slots. ``consistency_check`` compares their orders of magnitude (a diagnostic
    the reference doc recommends) but never averages them.
    """

    site_name: str
    analyte: str
    method1_per_well: list[RateEstimate] = field(default_factory=list)  # module 1
    method1_summary: RateEstimate | None = None     # robust per-well summary (NOT a cross-method avg)
    method2: RateEstimate | None = None             # module 2, 1D centerline transect (site lambda)
    method2_2d: RateEstimate | None = None          # module 2, 2D fit (site lambda; alpha_y fitted)
    method3: RateEstimate | None = None             # module 3 (spline-centre concentration decay)

    def representative_rates(self) -> dict:
        """One representative value per ESTIMAND for the order-of-magnitude check. The two
        flowpath methods (1D, 2D) share an estimand, so a single flowpath value is used (the 2D
        fit when usable, else the 1D). These are separate numbers compared side by side, never
        combined."""
        def _reliable(e):
            return (e is not None and np.isfinite(e.value_per_year)
                    and not e.diagnostics.get("poorly_constrained")
                    and not e.diagnostics.get("implausibly_fast"))

        out = {}
        if self.method1_summary is not None:
            out[METHOD_MANN_KENDALL] = self.method1_summary.value_per_year
        # one flowpath representative: a reliable 2D fit if available, else a usable 1D, else 2D.
        flow = None
        if _reliable(self.method2_2d):
            flow = self.method2_2d.value_per_year
        elif self.method2 is not None and np.isfinite(self.method2.value_per_year):
            flow = self.method2.value_per_year
        elif self.method2_2d is not None and np.isfinite(self.method2_2d.value_per_year):
            flow = self.method2_2d.value_per_year
        if flow is not None:
            out[METHOD_DOMENICO] = flow
        if self.method3 is not None:
            out[METHOD_ST_PSPLINE] = self.method3.value_per_year
        return out

    def consistency_check(self) -> dict:
        """Order-of-magnitude agreement across the three representative rates. Large spread
        is diagnostic (non-steady state, dilution masquerading as decay, or dispersivity
        misspecification), not a reason to pick a middle value."""
        vals = {k: v for k, v in self.representative_rates().items()
                if v is not None and np.isfinite(v) and v > 0}
        if len(vals) < 2:
            return {"comparable": list(vals.keys()), "spread_orders_of_magnitude": None,
                    "agree_within_1_oom": None,
                    "note": "fewer than two positive rates; nothing to cross-check"}
        logs = {k: math.log10(v) for k, v in vals.items()}
        spread = max(logs.values()) - min(logs.values())
        return {"comparable": list(vals.keys()),
                "values_per_year": vals,
                "spread_orders_of_magnitude": round(spread, 2),
                "agree_within_1_oom": bool(spread <= 1.0),
                "note": ("rates agree to within an order of magnitude"
                         if spread <= 1.0 else
                         "rates disagree by more than an order of magnitude; investigate, do not average")}


def _jsonable(x):
    """Recursively coerce numpy / non-finite values into JSON-safe Python types."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        f = float(x)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    return x
