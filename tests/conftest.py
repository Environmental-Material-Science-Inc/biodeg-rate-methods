"""Shared fixtures for the biodeg_rates test suite.

The key fixture ``synth_site_dir`` writes a synthetic site (the same
phase_2_obs_*.xlsx + level0_*.json schema the field inputs use) to a tmp dir, so the
loader and the full pipeline are tested hermetically with NO dependency on the customer data.

The synthetic plume is separable: a fixed Gaussian footprint times exp(-K_TRUE * t), so the
true point-decay rate at every well, the spline-centre concentration decay, and the whole-plume
mass-loss rate all equal K_TRUE. That gives the rate tests a known answer.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

K_TRUE = 0.15            # true first-order rate (1/yr) baked into the synthetic site
RL = 0.005               # benzene reporting limit (mg/L)


def _build_plume(seed=3):
    """Return (locations, contaminants, depths) DataFrames for a synthetic benzene plume."""
    rng = np.random.default_rng(seed)
    n_wells = 14
    E = rng.uniform(0, 180, n_wells) + 500000.0
    N = rng.uniform(0, 180, n_wells) + 5800000.0
    e0, n0 = E.mean(), N.mean()
    well_ids = [f"MW-{i:02d}" for i in range(n_wells)]
    locations = pd.DataFrame(dict(well_id=well_ids, easting=E, northing=N))

    # the three wells farthest from the source are clean sentinels (all non-detect), which
    # guarantees some censored data to exercise the recensoring path
    dist = np.hypot(E - e0, N - n0)
    clean = set(np.argsort(dist)[-3:].tolist())

    start = pd.Timestamp("2015-01-01")
    dates = [start + pd.DateOffset(years=int(y)) for y in range(9)]   # 9 annual rounds
    C0 = 20.0
    crows, drows = [], []
    for idx, (wid, e, n) in enumerate(zip(well_ids, E, N)):
        dist2 = (e - e0) ** 2 + (n - n0) ** 2
        amp = C0 * np.exp(-dist2 / (2 * 45.0 ** 2))
        for d in dates:
            t = (d - start).days / 365.25
            c = amp * np.exp(-K_TRUE * t) * float(np.exp(rng.normal(0, 0.15)))
            if idx in clean or c < RL:
                val = f"< {RL}"
            else:
                val = round(float(c), 4)
            crows.append((wid, d, val))
            # heads: gradient to the east (flow points +E), gentle slope, small noise
            head = 100.0 - 0.01 * (e - e0) + float(rng.normal(0, 0.02))
            drows.append((wid, d, round(head, 3)))
    contaminants = pd.DataFrame(crows, columns=["well_id", "date", "gw_benzene_mg_L"])
    depths = pd.DataFrame(drows, columns=["well_id", "date", "gw_head_m"])
    return locations, contaminants, depths


@pytest.fixture(scope="session")
def synth_site_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthsite")
    pf = d / "site_data"
    pf.mkdir()
    locations, contaminants, depths = _build_plume()
    xlsx = pf / "phase_2_obs_synth.xlsx"
    with pd.ExcelWriter(xlsx) as xw:
        locations.to_excel(xw, sheet_name="locations", index=False)
        depths.to_excel(xw, sheet_name="gw_depths", index=False)
        contaminants.to_excel(xw, sheet_name="gw_contaminants", index=False)
    level0 = dict(site_name="Synth Site", site_id="synth", soil_type="Sand",
                  EPSG_code="32613", num_years_init=10,
                  contaminant_thresholds={"benzene": [RL]},
                  start_date_string="2015-01-01")
    (pf / "level0_synth.json").write_text(json.dumps(level0))
    return str(d)


@pytest.fixture(scope="session")
def synth_site(synth_site_dir):
    from biodeg_rates import dataio
    return dataio.load_site(synth_site_dir)
