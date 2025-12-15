# -*- coding: utf-8 -*-
"""
Figure 2 (prep): Compute station-level performance metrics (RMSE, KGE) for CHELSA vs GHCN-D comparisons.

Expected inputs in:  <data-dir>/raw/fig_2/GHCN_data/
  - comp_data_pr/*.csv
  - comp_data_tavg/*.csv
    (filenames must match: <station_id>_lat<lat>_lon<lon>_elev<elev>.csv)

Each station CSV must contain:
  - PRCP [mm]  (for pr)  or  TAVG [degC] (for tavg)
  - 30arcsec, 90arcsec, 300arcsec, 1800arcsec  (model/predictor columns)

Writes output to:    <data-dir>/processed/fig_2/
  - error_metrics_pr.csv
  - error_metrics_tavg.csv
"""


from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RES_COLS = ["30arcsec", "90arcsec", "300arcsec", "1800arcsec"]

FILENAME_RE = re.compile(
    r"^(?P<station_id>.+?)_lat(?P<lat>-?\d+(?:\.\d+)?)_lon(?P<lon>-?\d+(?:\.\d+)?)_elev(?P<elev>-?\d+(?:\.\d+)?)\.csv$"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Repo data directory")
    return p.parse_args()


def rmse(obs: pd.Series, pred: pd.Series) -> float:
    x = (pred - obs).to_numpy(dtype=float)
    return float(np.sqrt(np.nanmean(x * x)))


def kge(obs: pd.Series, pred: pd.Series) -> float:
    valid = (~obs.isna()) & (~pred.isna())
    o = obs[valid].astype(float)
    p = pred[valid].astype(float)
    if len(o) == 0:
        return float("nan")
    r = o.corr(p)
    mo, mp = o.mean(), p.mean()
    so, sp = o.std(), p.std()
    beta = mp / mo if mo != 0 else np.nan
    gamma = sp / so if so != 0 else np.nan
    return float(1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))


def station_ids(folder: Path) -> set[str]:
    out = set()
    for f in folder.glob("*.csv"):
        m = FILENAME_RE.match(f.name)
        if m:
            out.add(m.group("station_id"))
    return out


def load_case(case_dir: Path, case: str) -> pd.DataFrame:
    obs_col = "PRCP [mm]" if case == "pr" else "TAVG [degC]"
    rows = []

    for f in sorted(case_dir.glob("*.csv")):
        m = FILENAME_RE.match(f.name)
        if not m:
            continue

        sid = m.group("station_id")
        lat = float(m.group("lat"))
        lon = float(m.group("lon"))
        elev = float(m.group("elev"))

        df = pd.read_csv(f, index_col=0)
        if df.empty:
            continue

        if obs_col not in df.columns:
            raise ValueError(f"{f.name}: missing '{obs_col}' column.")
        missing = [c for c in RES_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name}: missing columns {missing}")

        obs = df[obs_col]

        for res in RES_COLS:
            pred = df[res]
            valid = (~obs.isna()) & (~pred.isna())
            if valid.sum() == 0:
                continue

            o = obs[valid]
            p = pred[valid]

            rows.append({
                "station_id": sid,
                "lat": lat,
                "lon": lon,
                "elev": elev,
                "n_obs": int(valid.sum()),
                "resolution": res,
                "RMSE": rmse(o, p),
                "KGE": kge(o, p),
            })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    raw_root = args.data_dir / "raw" / "fig_2" / "GHCN_data"
    pr_dir = raw_root / "comp_data_pr"
    tv_dir = raw_root / "comp_data_tavg"

    if not pr_dir.exists() or not tv_dir.exists():
        raise FileNotFoundError(f"Expected:\n  {pr_dir}\n  {tv_dir}")


    processed = args.data_dir.parent / "processed" / "fig_2"
    processed.mkdir(parents=True, exist_ok=True)

    df_pr = load_case(pr_dir, "pr")
    df_tv = load_case(tv_dir, "tavg")

    df_pr.to_csv(processed / "error_metrics_pr.csv", index=False)
    df_tv.to_csv(processed / "error_metrics_tavg.csv", index=False)

    print(f"Wrote {processed / 'error_metrics_pr.csv'} ({len(df_pr)} rows)")
    print(f"Wrote {processed / 'error_metrics_tavg.csv'} ({len(df_tv)} rows)")


if __name__ == "__main__":
    main()
