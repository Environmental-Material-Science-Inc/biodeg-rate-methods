"""biodeg_rates — modular bulk-attenuation rate estimation from groundwater PHC data.

Three independent estimators of a bulk attenuation rate, each measuring a DIFFERENT
quantity by a DIFFERENT method on a DIFFERENT data geometry:

  module 1  method1_mann_kendall  — per-well point-decay rate (Mann-Kendall + Theil-Sen)
  module 2  method2_domenico      — flowpath decay coefficient lambda (Domenico); 1D transect
                                     (estimate_site) and 2D fit (estimate_site_2d, alpha_y fitted)
  module 3  method3_spline        — spline-centre concentration decay rate (spatio-temporal P-spline,
                                     -df/dt at the plume centre)

  module 4  handoff               — collect the estimates, keep them SEPARATE, emit the MODFLOW handoff

The numbers are not interchangeable and must never be averaged. The contract (``contract.py``)
encodes that rule in the types; the handoff (``handoff.py``) refuses to aggregate across methods.

See README.md for the estimand table, the validation record, and the known documentation gap.
"""

from .contract import (
    Estimand, RateEstimate, SiteObservations, SiteRateBundle, WellSeries,
)

__all__ = [
    "Estimand", "RateEstimate", "SiteObservations", "SiteRateBundle", "WellSeries",
]
