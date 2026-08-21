# biodeg-rate-methods

Three independent estimators of a bulk attenuation rate for a dissolved groundwater
contaminant, plus a literature-prior combine and a model-handoff writer. The estimators are
pure functions of a `SiteObservations` record, so the whole package runs and is tested
without touching a file of field data.

The organising principle is that the three estimators **measure different quantities** by
different methods on different data geometries. They are not three attempts at one number,
they are not interchangeable, and the code refuses to average them: `contract.py` encodes the
distinction in the types and `handoff.py` raises rather than aggregate across methods.

## Install and run

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
python -m pytest tests/ -q
```

55 tests, all of which run against a synthetic plume built in a temporary directory by
`tests/conftest.py`. The fixture is separable by construction, a fixed Gaussian footprint
times `exp(-K_TRUE * t)` with `K_TRUE = 0.15 / yr`, so the per-well point decay and the
spline-centre decay both have the same known answer. On that fixture Method 1 returns
0.157/yr and Method 3 returns 0.139/yr.

To run the driver over a folder of sites:

```bash
python -m biodeg_rates.run_sites --root "<folder of site folders>" --out outputs/rates
```

Each site folder is expected to hold `site_data/phase_2_obs_<site>.xlsx` (sheets: locations,
gw_depths, gw_contaminants) and `site_data/level0_<site>.json` (thresholds, EPSG, start date,
soil type). `dataio.py` is the only module that knows the spreadsheet layout; swapping in a
different input format means replacing that one module.

## The three estimands

| Method | Module | Estimand | Dilution | Interpretation |
|---|---|---|---|---|
| 1 | `method1_mann_kendall.py` | `point_decay_k`, per well | included | Theil-Sen slope of ln C against time at one well, gated on a Mann-Kendall trend test. Rank based, so a shifting detection limit does not move the test statistic. Local and statistical. |
| 2 | `method2_domenico.py` | `flowpath_lambda` | removed | First-order decay along the flowpath after analytically dividing out transverse spreading. The only one of the three with a mechanistic transport interpretation. 1D centreline (`estimate_site`) and 2D fit with `alpha_y` estimated (`estimate_site_2d`). |
| 3 | `method3_spline.py` | `spline_centre_decay` | included | Decay at the plume centre read off a REML-smoothed tensor-product P-spline surface in (easting, northing, time). Same estimand as Method 1, smoothed across all wells and events rather than read from one well. |

`prior.py` carries the literature prior. For benzene it is the McHugh et al. (2023) median
first-order attenuation rate, 0.14/yr across 1,905 California GeoTracker petroleum sites, with
82% of sites attenuating and an empirical 5th to 95th percentile range of -0.15 to 0.53/yr
(computed from that paper's Data S1). Because McHugh's rate is an apparent
concentration-versus-time rate, the combine uses **Method 1 only**, as a conjugate Gaussian
update in log space. A site estimate falling outside the population band is flagged for
review rather than used.

`handoff.py` collects the estimates into a `SiteRateBundle`, records a menu with one entry per
method and the caveat attached to each, runs an order-of-magnitude consistency check that
never averages, and emits a MODFLOW 6 GWT-MST first-order decay block in 1/day. It leaves
`selected_for_initialization` null: the payload records the menu, it does not make the
modelling decision.

## Honesty contract

Every `RateEstimate` carries a confidence interval, an `n`, and a confidence tier, and a method
that does not apply returns an `N/A` estimate rather than raising. Method 2 flags fits that are
implausibly fast or poorly constrained. Method 3 drops to low confidence when its credible band
crosses zero, and it guards against surface ballooning at the plume centre when the monitoring
network changes over time, using a robust slope, a sign check against the raw mean-concentration
trend, and an OLS-versus-robust agreement test.

## Validation record, and where the evidence lives

These numbers come from harnesses that are **not included in this repository**, so they are
reported here as claims rather than as something a reader can re-run from this tree:

- **Method 2 against EPA REMChlor.** On a REMChlor-equivalent centreline with a known
  `k = 0.5/yr`, Method 2 recovered 0.52/yr while the raw un-normalized slope returned 0.81/yr.
  The 61% overstatement is the transverse dilution the normalization removes.
- **Method 3 against R `mgcv`.** A `te(bs="ps", method="REML")` cross-fit agreed to r = 0.999
  and recovered the known rate on synthetic plumes to better than 2%.
- **Method 2, 1D against 2D.** On synthetic truth the 2D variant recovered `k` roughly three
  times more accurately (mean absolute error 62% against 185%) and also recovered `alpha_y`.

## Known limitations

1. **The mathematical reference document does not exist.** Module docstrings cite it
   repeatedly and specifically, for example "the two-step, dispersivity-fixed procedure of
   2.6", "reference Section 3 and Appendix B", and "the censoring convention, reference
   Appendix A". No such document has been written. The derivations behind the code are
   currently recoverable only from the code itself. This is the largest gap for anyone
   intending to build on the methods.
2. **Method 2 is unreliable on clustered well networks.** Neither the 1D nor the 2D variant
   performs well when wells are clustered rather than distributed along a flowpath. The
   binding constraint is the data geometry, not the estimator.
3. **Vertical spreading is not removed** in Method 2 (`Phi_z = 1`). The input format carries no
   vertical delineation, so the method runs in the 2D horizontal form. The output states this.
4. **Non-detects enter Method 3 at RL/2**, a documented substitution rather than a censored
   likelihood. Methods 1 and 2 handle censoring by rank and by exclusion respectively.
5. **Transport parameters in Method 2 are literature values keyed to soil type**, not measured.
   Dispersivity dominates the uncertainty, which is why the rate is reported with a Monte-Carlo
   band and an explicit sweep over assumed `alpha_x`.
6. **The QA/QC figure modules are not included.** Methods 2 and 3 still emit a `qaqc` payload of
   compact arrays intended for a renderer; no renderer ships here.

## Data

This repository contains **no field data**. The only data is the synthetic plume generated at
test time. Nothing in the input format is site-specific, and no monitoring records,
coordinates, or site identities are included.

## Licence

MIT. See `LICENSE`.
