# -*- coding: utf-8 -*-
"""
Figure 4: TreeMig example (Larix decidua) — observed vs simulated basal area + difference maps + skill table.

Expected inputs in:  <data-dir>/fig_4/raw/
  - larix_decidua.nc
  - rmse_all.csv, mae_all.csv, kge_all.csv
  - rmse_ensemble.csv, mae_ensemble.csv, kge_ensemble.csv
Writes output to:    <out-dir>/fig_4/larix_decidua_comp.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# NetCDF variable names expected
OBS_VAR = "ba"
SIM_VARS = ["ba_60km", "ba_10km", "ba_3km", "ba_1km"]  # maps to 1800", 300", 90", 30"
RES_LABELS = ['1800"', '300"', '90"', '30"']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def make_white_low_cmap(base_name: str = "viridis_r"):
    base = plt.get_cmap(base_name)
    cols = base(np.linspace(0, 1, 256))
    cols[0, :] = np.array([1, 1, 1, 1])  # lowest -> white
    cmap = mcolors.ListedColormap(cols)
    cmap.set_bad(color="lightgrey")
    return cmap


def pick_resolution_columns(df: pd.DataFrame) -> list[str]:
    """
    Accept either:
      - ['60km','10km','3km','1km']
      - ['0','1','2','3']
    Returns column names in the correct order (60km -> 1km).
    """
    cols = list(df.columns)

    # normalize to strings for matching
    cols_str = [str(c) for c in cols]

    named = ["60km", "10km", "3km", "1km"]
    if all(c in cols_str for c in named):
        return named

    numeric = ["0", "1", "2", "3"]
    if all(c in cols_str for c in numeric):
        return numeric

    raise KeyError(
        f"Stats CSV has too few columns: {cols}. "
        f"Expected 4 resolution columns."
    )


def read_stats_row(csv_path: Path, row_name: str) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0)

    if row_name not in df.index:
        raise KeyError(
            f"Row '{row_name}' not found in {csv_path.name}. "
            f"Available (first 15): {list(df.index)[:15]}"
        )

    res_cols = pick_resolution_columns(df)
    s = df.loc[row_name, res_cols].astype(float)

    # If columns are 0..3, they correspond to 60km..1km in that order
    s.index = RES_LABELS
    return s


def load_summary_table(fig4_dir: Path, species: str = "Larix_decidua") -> pd.DataFrame:
    rmse = read_stats_row(fig4_dir / "rmse_all.csv", species)
    mae = read_stats_row(fig4_dir / "mae_all.csv", species)
    kge = read_stats_row(fig4_dir / "kge_all.csv", species)

    summary = pd.DataFrame(
        {"NRMSE": rmse, "MAE": mae, "KGE": kge},
        index=RES_LABELS,
    ).round(3)

    return summary


def main():
    args = parse_args()

    fig4_dir = args.data_dir / "raw" / "fig_4"
    out_dir = args.out_dir / "fig_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_path = fig4_dir / "larix_decidua.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing NetCDF: {nc_path}")

    for f in ["rmse_all.csv", "mae_all.csv", "kge_all.csv"]:
        if not (fig4_dir / f).exists():
            raise FileNotFoundError(f"Missing stats file: {fig4_dir / f}")

    ds = xr.open_dataset(nc_path)

    required_vars = [OBS_VAR] + SIM_VARS
    missing = [v for v in required_vars if v not in ds.variables]
    if missing:
        raise KeyError(f"NetCDF missing variables: {missing}")

    obs = ds[OBS_VAR]
    sims = [ds[v] for v in SIM_VARS]
    diffs = [s - obs for s in sims]

    summary = load_summary_table(fig4_dir, species="Larix_decidua")

    max_ba = float(np.nanmax([obs.max().values, *[s.max().values for s in sims]]))
    max_diff = float(np.nanmax([np.nanmax(np.abs(d.values)) for d in diffs]))

    cmap_ba = make_white_low_cmap("viridis_r")
    cmap_diff = plt.get_cmap("RdBu_r")
    cmap_diff.set_bad(color="lightgrey")

    plot_crs = ccrs.LambertConformal(central_longitude=8, central_latitude=47)
    swiss_crs = ccrs.TransverseMercator(
        central_longitude=7.43958333,
        central_latitude=46.95240556,
        scale_factor=1,
        false_easting=600000,
        false_northing=200000,
    )

    fig, axes = plt.subplots(
        2, 5,
        figsize=(11, 5),
        subplot_kw={"projection": plot_crs},
        constrained_layout=True,
    )

    # Row 0: observed + simulations
    p_obs = obs.plot(
        ax=axes[0, 0],
        transform=swiss_crs,
        cmap=cmap_ba,
        vmin=0,
        vmax=max_ba,
        add_colorbar=False,
    )
    axes[0, 0].set_title("Observed Basal Area (LFI)", fontsize=10, fontweight="bold", pad=8)

    for i, (da, lab) in enumerate(zip(sims, RES_LABELS), start=1):
        da.plot(
            ax=axes[0, i],
            transform=swiss_crs,
            cmap=cmap_ba,
            vmin=0,
            vmax=max_ba,
            add_colorbar=False,
        )
        axes[0, i].set_title(lab, fontsize=10, fontweight="bold", pad=8)

    # Row 1 col 0: table
    ax_table = axes[1, 0]
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=summary.values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        cellLoc="center",
        rowLoc="center",
        loc="lower center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.7)
    tbl.scale(0.65, 1.7)

    # Row 1: differences
    last_p = None
    for i, (d, lab) in enumerate(zip(diffs, RES_LABELS), start=1):
        last_p = d.plot(
            ax=axes[1, i],
            transform=swiss_crs,
            cmap=cmap_diff,
            center=0,
            vmin=-max_diff,
            vmax=max_diff,
            add_colorbar=False,
        )
        axes[1, i].set_title(f"{lab} - LFI", fontsize=10, fontweight="bold", pad=8)

    for ax in axes.flat:
        if ax is ax_table:
            continue
        ax.coastlines(resolution="10m")
        ax.add_feature(cfeature.BORDERS, edgecolor="dimgrey")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Colorbars under each row
    fig.colorbar(
        p_obs,
        ax=axes[0, :],
        orientation="horizontal",
        shrink=0.7,
        aspect=40,
        pad=0.1,
        label="Basal area (m²/ha)",
    )
    fig.colorbar(
        last_p,
        ax=axes[1, 1:],
        orientation="horizontal",
        shrink=0.7,
        aspect=30,
        pad=0.1,
        label="Difference to LFI (m²/ha)",
    )

    out_path = out_dir / "larix_decidua_comp.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
