"""Loader / adapter for the ``phase_2_obs_*.xlsx`` site-observation format.

Turns one site folder into a ``SiteObservations`` (the contract every estimator consumes).
This is the only module that knows the spreadsheet layout, so the estimators never touch a
file. It keeps the FULL time series (it does not collapse to a max-over-time envelope the way
an envelope-based ingestion would), because Methods 1 and 3 need every event.

The format (one site):
  <site>/site_data/phase_2_obs_<site>.xlsx   sheets: locations, gw_depths, gw_contaminants, ...
  <site>/site_data/level0_<site>.json        thresholds, EPSG, start date, soil type
Sheet names vary across sites (capitalization, a stray leading space); the loader matches
them fuzzily.
"""

from __future__ import annotations

import glob
import json
import math
import os
import warnings

import numpy as np
import pandas as pd

from .contract import SiteObservations, WellSeries

warnings.simplefilter("ignore")

DAYS_PER_YEAR = 365.25

ANALYTE_COLUMN = {
    "benzene": "benzene",
    "toluene": "toluene",
    "ethylbenzene": "ethylbenzene",
    "xylenes": "xylenes",
}


def parse_censored(val):
    """Return (numeric_value, is_detect, reporting_limit) for one cell.

    "< 0.001" -> (0.001, False, 0.001)  (non-detect at its reporting limit)
    "0.745"   -> (0.745, True,  nan)    (quantified detection)
    blank / NA -> (nan, nan-detect, nan) and the caller drops it.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan, None, np.nan
    s = str(val).strip()
    if s == "" or s.lower() in ("na", "nan", "n/a", "-"):
        return np.nan, None, np.nan
    if s.startswith("<"):
        try:
            rl = float(s[1:].strip())
        except ValueError:
            return np.nan, None, np.nan
        return rl, False, rl
    try:
        return float(s), True, np.nan
    except ValueError:
        return np.nan, None, np.nan


def _find_xlsx(site_dir: str) -> str | None:
    for sub in ("site_data", "."):
        hits = glob.glob(os.path.join(site_dir, sub, "phase_2_obs_*.xlsx"))
        if hits:
            return hits[0]
    return None


def _find_level0(site_dir: str) -> dict:
    for sub in ("site_data", "."):
        hits = glob.glob(os.path.join(site_dir, sub, "level0_*.json"))
        if hits:
            try:
                return json.loads(open(hits[0], "r", encoding="utf-8").read())
            except Exception:
                return {}
    return {}


def _sheet(xl: pd.ExcelFile, *needles) -> str | None:
    for s in xl.sheet_names:
        low = s.strip().lower()
        if all(n in low for n in needles):
            return s
    return None


def _col(df: pd.DataFrame, *needles) -> str | None:
    for c in df.columns:
        low = str(c).strip().lower()
        if all(n in low for n in needles):
            return c
    return None


def load_site(site_dir: str, analyte: str = "benzene") -> SiteObservations:
    """Read one site folder into a SiteObservations. Raises FileNotFoundError if no xlsx."""
    xlsx = _find_xlsx(site_dir)
    if xlsx is None:
        raise FileNotFoundError(f"no phase_2_obs_*.xlsx under {site_dir}")
    level0 = _find_level0(site_dir)
    site_name = level0.get("site_name") or os.path.basename(site_dir.rstrip("/\\"))

    xl = pd.ExcelFile(xlsx)
    loc_sheet = _sheet(xl, "loc")
    con_sheet = _sheet(xl, "gw", "contam") or _sheet(xl, "contam")
    dep_sheet = _sheet(xl, "gw", "depth") or _sheet(xl, "depth")
    if loc_sheet is None or con_sheet is None:
        raise ValueError(f"{xlsx}: could not find location/contaminant sheets in {xl.sheet_names}")

    loc = xl.parse(loc_sheet)
    loc.columns = [str(c).strip() for c in loc.columns]
    wid = _col(loc, "well") or loc.columns[0]
    ecol = _col(loc, "east") or _col(loc, "x")
    ncol = _col(loc, "north") or _col(loc, "y")
    loc = loc.rename(columns={wid: "well_id", ecol: "easting", ncol: "northing"})
    loc["well_id"] = loc["well_id"].astype(str).str.strip()
    coords = {r.well_id: (float(r.easting), float(r.northing)) for r in loc.itertuples()}

    con = xl.parse(con_sheet)
    con.columns = [str(c).strip() for c in con.columns]
    wid = _col(con, "well") or con.columns[0]
    dcol = _col(con, "date")
    acol = _col(con, ANALYTE_COLUMN.get(analyte, analyte))
    if acol is None:
        raise ValueError(f"{xlsx}: no column for analyte '{analyte}' in {list(con.columns)}")
    con = con.rename(columns={wid: "well_id", dcol: "date"})
    con["well_id"] = con["well_id"].astype(str).str.strip()
    con["date"] = pd.to_datetime(con["date"], errors="coerce")

    # start date: prefer level0, else earliest sample
    sds = level0.get("start_date_string")
    start_date = pd.to_datetime(sds) if sds else con["date"].min()

    rows = []
    for r in con.itertuples():
        if pd.isna(r.date) or r.well_id not in coords:
            continue
        v, det, rl = parse_censored(con.at[r.Index, acol])
        if det is None or np.isnan(v):
            continue
        e, n = coords[r.well_id]
        t_years = (r.date - start_date).days / 365.25
        rows.append((r.well_id, e, n, r.date, t_years, v, bool(det),
                     rl if not np.isnan(rl) else (v if not det else np.nan)))

    frame = pd.DataFrame(rows, columns=["well_id", "easting", "northing", "date",
                                        "t_years", "conc", "detect", "rl"])
    frame = frame.sort_values(["well_id", "date"]).reset_index(drop=True)

    # per-well series
    wells = []
    for wid_, g in frame.groupby("well_id"):
        g = g.sort_values("date")
        wells.append(WellSeries(
            well_id=str(wid_), easting=float(g.easting.iloc[0]), northing=float(g.northing.iloc[0]),
            t_years=g.t_years.to_numpy(float), dates=g.date.to_numpy(),
            conc=g.conc.to_numpy(float), detect=g.detect.to_numpy(bool), rl=g.rl.to_numpy(float),
        ))

    # heads
    heads = pd.DataFrame(columns=["well_id", "easting", "northing", "date", "head_m"])
    if dep_sheet is not None:
        dep = xl.parse(dep_sheet)
        dep.columns = [str(c).strip() for c in dep.columns]
        hwid = _col(dep, "well") or dep.columns[0]
        hdate = _col(dep, "date")
        hhead = _col(dep, "head")
        if hhead is not None:
            dep = dep.rename(columns={hwid: "well_id", hdate: "date", hhead: "head_m"})
            dep["well_id"] = dep["well_id"].astype(str).str.strip()
            dep["date"] = pd.to_datetime(dep["date"], errors="coerce")
            dep["head_m"] = pd.to_numeric(dep["head_m"], errors="coerce")
            dep = dep.dropna(subset=["head_m"])
            dep["easting"] = dep["well_id"].map(lambda w: coords.get(w, (np.nan, np.nan))[0])
            dep["northing"] = dep["well_id"].map(lambda w: coords.get(w, (np.nan, np.nan))[1])
            heads = dep[["well_id", "easting", "northing", "date", "head_m"]].dropna(subset=["easting"])

    # source = location of the maximum detected concentration
    det_frame = frame[frame.detect]
    if len(det_frame):
        si = det_frame.conc.idxmax()
        source_xy = (float(frame.at[si, "easting"]), float(frame.at[si, "northing"]))
        source_conc = float(frame.at[si, "conc"])
    else:
        source_xy = (float(frame.easting.mean()), float(frame.northing.mean())) if len(frame) else (0.0, 0.0)
        source_conc = float("nan")

    thresholds = level0.get("contaminant_thresholds", {})
    thr = thresholds.get(analyte)
    threshold = float(thr[0]) if isinstance(thr, (list, tuple)) and thr else (float(thr) if thr else None)
    epsg = level0.get("EPSG_code")
    epsg = int(epsg) if epsg not in (None, "") else None

    return SiteObservations(
        site_name=site_name, analyte=analyte, epsg=epsg, threshold=threshold,
        start_date=np.datetime64(pd.Timestamp(start_date)), frame=frame, wells=wells,
        heads=heads, source_xy=source_xy, source_conc=source_conc,
        site_id=os.path.basename(site_dir.rstrip("/\\")),
    )


def soil_type(site_dir: str) -> str | None:
    return (_find_level0(site_dir) or {}).get("soil_type")


def site_from_long(df: pd.DataFrame, flow_azimuth_deg: float = 90.0,
                   site_name: str = "site", analyte: str = "benzene") -> SiteObservations:
    """Build a SiteObservations from a long-format table (well_id, easting, northing, t_years,
    conc[, detect]). Heads are synthesized consistent with ``flow_azimuth_deg`` so Method 2 can
    establish a flow direction. Used to ingest synthetic / external (e.g. REMChlor) data and by
    the evidence harnesses; the spreadsheet loader ``load_site`` is the path for real site folders.
    """
    df = df.copy()
    if "detect" not in df:
        df["detect"] = True
    df["detect"] = df["detect"].astype(bool)
    df["rl"] = np.where(df["detect"], np.nan, df["conc"])
    df["date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(df["t_years"] * DAYS_PER_YEAR, unit="D")
    frame = df[["well_id", "easting", "northing", "date", "t_years", "conc", "detect", "rl"]] \
        .sort_values(["well_id", "t_years"]).reset_index(drop=True)

    wells = []
    for wid, g in frame.groupby("well_id"):
        wells.append(WellSeries(str(wid), float(g.easting.iloc[0]), float(g.northing.iloc[0]),
                                g.t_years.to_numpy(float), g.date.to_numpy(),
                                g.conc.to_numpy(float), g.detect.to_numpy(bool), g.rl.to_numpy(float)))
    az = math.radians(flow_azimuth_deg)
    ue, un = math.sin(az), math.cos(az)
    locs = frame.groupby("well_id").agg(easting=("easting", "first"),
                                        northing=("northing", "first")).reset_index()
    locs["head_m"] = 100.0 - 0.01 * (locs.easting * ue + locs.northing * un)
    heads = locs[["well_id", "easting", "northing", "head_m"]].copy()
    heads["date"] = pd.Timestamp("2000-01-01")

    det = frame[frame.detect]
    si = det.conc.idxmax()
    return SiteObservations(site_name=site_name, analyte=analyte, epsg=None, threshold=None,
                            start_date=np.datetime64("2000-01-01"), frame=frame, wells=wells,
                            heads=heads, source_xy=(float(frame.at[si, "easting"]),
                                                    float(frame.at[si, "northing"])),
                            source_conc=float(frame.at[si, "conc"]), site_id=site_name)
