# -*- coding: utf-8 -*-
"""
Figure 2a: CHELSA climate fields over the Alps (1800") and differences to finer grids + value distributions.

Expected inputs in:  <data-dir>/fig_2/raw/chelsa_alps_grids/
  - tas_30arcsec_mean.nc
  - tas_90arcsec_to_30arcsec.nc
  - tas_300arcsec_to_30arcsec.nc
  - tas_1800arcsec_to_30arcsec.nc
  - pr_30arcsec_mean.nc
  - pr_90arcsec_to_30arcsec.nc
  - pr_300arcsec_to_30arcsec.nc
  - pr_1800arcsec_to_30arcsec.nc

Writes output to:  <out-dir>/fig_2/chelsa_alps_grids_relative_and_distribution.png
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def _sample_1d(values: np.ndarray, n: int) -> np.ndarray:
    """Randomly subsample 1D array (deterministic seed) if needed."""
    values = values[np.isfinite(values)]
    if n <= 0 or values.size <= n:
        return values
    rng = np.random.default_rng(42)
    idx = rng.choice(values.size, size=n, replace=False)
    return values[idx]


def main():
    args = parse_args()

    chelsa_dir = (
        args.data_dir / "fig_2" / "chelsa_alps_grids"
    )
    outdir = args.out_dir / "fig_2"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- config ---
    variables = ["tas", "pr"]
    units = {"tas": "°C", "pr": "mm"}
    titles_row1 = {
        "tas": 'Annual mean temperature (1800")',
        "pr": 'Annual total precipitation (1800")',
    }
    cmap_abs = {"tas": "viridis", "pr": "Blues"}

    # files:
    fn = {
        "30": "{var}_30arcsec_mean.nc",
        "90": "{var}_90arcsec_to_30arcsec.nc",
        "300": "{var}_300arcsec_to_30arcsec.nc",
        "1800": "{var}_1800arcsec_to_30arcsec.nc",
    }

    # --- figure setup  ---
    fig = plt.figure(figsize=(20, 2 * 3.2))
    gs = gridspec.GridSpec(nrows=2, ncols=5, figure=fig, width_ratios=[1, 1, 1, 1, 0.45])

    row_artists = []

    for row, var in enumerate(variables):
        # --- load datasets ---
        with xr.open_dataset(chelsa_dir / fn["30"].format(var=var)) as ds30, \
             xr.open_dataset(chelsa_dir / fn["90"].format(var=var)) as ds90, \
             xr.open_dataset(chelsa_dir / fn["300"].format(var=var)) as ds300, \
             xr.open_dataset(chelsa_dir / fn["1800"].format(var=var)) as ds1800:

            da_30 = ds30[var]
            da_90 = ds90[var]
            da_300 = ds300[var]
            da_1800 = ds1800[var]

            # apply land mask
            mask = da_30.where(da_30 != 0)
            da_30 = mask
            da_90 = da_90.where(~mask.isnull())
            da_300 = da_300.where(~mask.isnull())
            da_1800 = da_1800.where(~mask.isnull())

            if var == "tas":
                da_30 = da_30 - 273.15
                da_90 = da_90 - 273.15
                da_300 = da_300 - 273.15
                da_1800 = da_1800 - 273.15

            # --- diffs and violin distributions ---
            rel_diff_300 = da_300 - da_1800
            rel_diff_90 = da_90 - da_1800
            rel_diff_30 = da_30 - da_1800
            rel_max = float(np.nanmax([
                np.nanmax(np.abs(rel_diff_300.values)),
                np.nanmax(np.abs(rel_diff_90.values)),
                np.nanmax(np.abs(rel_diff_30.values)),
            ]))

            vals_1800 = _sample_1d(da_1800.values.ravel(), args.violin_sample)
            vals_300 = _sample_1d(da_300.values.ravel(), args.violin_sample)
            vals_90 = _sample_1d(da_90.values.ravel(), args.violin_sample)
            vals_30 = _sample_1d(da_30.values.ravel(), args.violin_sample)

            df_dist = pd.DataFrame({
                '1800"': vals_1800,
                '300"': vals_300,
                '90"': vals_90,
                '30"': vals_30,
            }).melt(var_name="Resolution", value_name="Value").dropna()

            datasets = [da_1800, rel_diff_300, rel_diff_90, rel_diff_30]
            titles = [
                titles_row1[var],
                '300" - 1800"',
                '90" - 1800"',
                '30" - 1800"',
            ]

            # --- maps (cols 0–3) ---
            diff_axes = []
            im_abs = None
            im_rel = None

            for col in range(4):
                ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
                da = datasets[col]

                if col == 0:
                    im_abs = da.plot.pcolormesh(
                        ax=ax, transform=ccrs.PlateCarree(),
                        cmap=cmap_abs[var], add_colorbar=False
                    )
                else:
                    im_rel = da.plot.pcolormesh(
                        ax=ax, transform=ccrs.PlateCarree(),
                        cmap="RdBu_r", vmin=-rel_max, vmax=rel_max, add_colorbar=False
                    )
                    diff_axes.append(ax)

                ax.coastlines()
                ax.set_title(titles[col], fontsize=16, fontweight="bold")

            # --- violin plot (col 4) ---
            box_col_spec = gs[row, 4]
            inner = box_col_spec.subgridspec(5, 1, height_ratios=[1, 0.5, 3, 0.5, 1])
            ax_dist = fig.add_subplot(inner[2, 0])

            sns.violinplot(
                data=df_dist,
                x="Resolution",
                y="Value",
                ax=ax_dist,
                inner="box",
                density_norm="width",
                color="grey",
                saturation=0.8,
                linewidth=0.8,
                order=['1800"', '300"', '90"', '30"'],
            )
            ax_dist.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
            ax_dist.set_ylabel(f"{var} [{units[var]}]", fontsize=14)
            ax_dist.set_xlabel("Resolution", fontsize=14)
            ax_dist.yaxis.set_label_position("right")
            ax_dist.yaxis.tick_right()

            row_artists.append((row, var, fig.axes[-5], im_abs, diff_axes, im_rel))

    plt.tight_layout(pad=1.2, w_pad=0.3, h_pad=2.0)
    fig.subplots_adjust(hspace=0.35)  # helps keep violin/box area from creeping upward

    # --- row-wise horizontal colorbars ---
    for row, var, ax0, im_abs, diff_axes, im_rel in row_artists:
        pos0 = ax0.get_position()
        cb_h = 0.015
        cb_gap = 0.035
        bottom_y = pos0.y0 - cb_gap

        # absolute bar under column 0
        abs_cax = fig.add_axes([pos0.x0, bottom_y, pos0.width, cb_h])
        cb_abs = fig.colorbar(im_abs, cax=abs_cax, orientation="horizontal")
        cb_abs.set_label(units[var], fontsize=14)

        # relative bar spanning columns 1–3
        pos1 = diff_axes[0].get_position()
        pos3 = diff_axes[-1].get_position()
        rel_x0 = pos1.x0
        rel_w = (pos3.x0 + pos3.width) - pos1.x0

        rel_cax = fig.add_axes([rel_x0, bottom_y, rel_w, cb_h])
        cb_rel = fig.colorbar(im_rel, cax=rel_cax, orientation="horizontal")
        cb_rel.set_label(f"Difference [Δ {units[var]}]", fontsize=14)

    out = outdir / "chelsa_alps_grids_relative_and_distribution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
