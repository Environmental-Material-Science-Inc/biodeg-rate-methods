"""End-to-end: run the three estimators on the synthetic site and check the artifacts."""

import json
import os

import numpy as np

from biodeg_rates import (method1_mann_kendall as m1, method2_domenico as m2,
                          method3_spline as m3, handoff, dataio)


def test_full_pipeline(synth_site_dir, synth_site, tmp_path):
    site = synth_site
    soil = dataio.soil_type(synth_site_dir)
    per_well, summary = m1.estimate_site(site)
    est2 = m2.estimate_site(site, soil_type=soil)
    est3 = m3.estimate_site(site)
    bundle = handoff.assemble_bundle(site, per_well, summary, est2, est3)

    # the handoff JSON is written and well-formed
    out = str(tmp_path / "handoff")
    hpath = handoff.write_handoff(site, bundle, out)
    payload = json.loads(open(hpath).read())
    assert payload["site"] == "Synth Site"
    assert len(payload["initialization_menu"]) == 3

    # Methods 1 and 3 apply on this rich synthetic site; Method 2 may be N/A (a radial plume
    # has no down-gradient centerline transect), which is correct, not a failure.
    assert np.isfinite(summary.value_per_year)
    assert np.isfinite(est3.value_per_year)
    # the consistency check compares only the methods that produced a finite rate
    cc = bundle.consistency_check()
    assert "spread_orders_of_magnitude" in cc


def test_methods_report_distinct_estimands(synth_site):
    _, summary = m1.estimate_site(synth_site)
    est3 = m3.estimate_site(synth_site)
    assert summary.estimand != est3.estimand            # never the same quantity
