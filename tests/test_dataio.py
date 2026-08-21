"""Data loader: censored-value parsing and the phase_2_obs ingestion (hermetic fixture)."""

import math

import numpy as np

from biodeg_rates import dataio


def test_parse_censored_detect():
    v, det, rl = dataio.parse_censored("0.745")
    assert det is True and math.isclose(v, 0.745) and math.isnan(rl)


def test_parse_censored_nondetect():
    v, det, rl = dataio.parse_censored("< 0.001")
    assert det is False and math.isclose(v, 0.001) and math.isclose(rl, 0.001)


def test_parse_censored_blank():
    v, det, rl = dataio.parse_censored("")
    assert det is None and math.isnan(v)


def test_load_site_schema(synth_site):
    s = synth_site
    assert s.analyte == "benzene"
    assert s.epsg == 32613
    assert math.isclose(s.threshold, 0.005)
    assert len(s.wells) >= 10
    # frame columns
    for col in ("well_id", "easting", "northing", "date", "t_years", "conc", "detect", "rl"):
        assert col in s.frame.columns
    # some non-detects were generated and parsed as detect=False
    assert (~s.frame.detect).any()
    # source is the max detected concentration location
    det = s.frame[s.frame.detect]
    assert math.isclose(s.source_conc, det.conc.max())


def test_load_site_has_heads(synth_site):
    assert len(synth_site.heads) > 0
    assert {"well_id", "easting", "northing", "head_m"}.issubset(synth_site.heads.columns)
