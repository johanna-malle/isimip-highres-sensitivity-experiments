# -*- coding: utf-8 -*-
"""
Figure 2b: Station-scale performance (RMSE, KGE) of CHELSA-W5-E5 v1 across resolutions (baseline + differences + boxplots).

Expected inputs in:  <data-dir>/fig_2/processed/
  - error_metrics_pr.csv
  - error_metrics_tavg.csv
Writes output to:    <out-dir>/fig_2/error_comparison_relative_pr_rmse_kge.png
                    <out-dir>/fig_2/error_comparison_relative_tavg_rmse_kge.png

Note: Run fig_2/prep.py first if processed files are not yet present.
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature as cfeature


RES_ORDER = ["1800arcsec", "300arcsec", "90arcsec", "30arcsec"]
DIFF_ORDER = ["300arcsec", "90arcsec", "30arcsec"]
METRICS = ["RMSE", "KGE"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--fig-id", type=str, default="fig_2")
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument("--lon-min", type=float, default=3 - 0.05)
    p.add_argument("--lon-max", type=float, default=18 + 0.05)
    p.add_argument("--lat-min", type=float, default=43 - 0.15)
    p.add_argument("--lat-max", type=float, default=49 + 0.15)

    p.add_argument("--scale-pr", type=float, default=50)
    p.add_argument("--scale-tavg", type=float, default=50)
    return p.parse_args()


def colorbar_extend(vals: np.ndarray, vmin: float, vmax: float) -> str:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "neither"
    below = np.nanmin(vals) < vmin
    above = np.nanmax(vals) > vmax
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def main():
    args = parse_args()

    processed = args.data_dir.parent / "processed" / args.fig_id
    outdir = args.out_dir / args.fig_id
    outdir.mkdir(parents=True, exist_ok=True)

    units = {"pr": {"RMSE": "mm", "KGE": "-"}, "tavg": {"RMSE": "°C", "KGE": "-"}}

    for case in ["pr", "tavg"]:
        f_in = processed / f"error_metrics_{case}.csv"
        if not f_in.exists():
            raise FileNotFoundError(f"Missing {f_in}. Run prep.py first.")

        df = pd.read_csv(f_in)
        df = df[["station_id", "lat", "lon", "n_obs", "resolution"] + METRICS].copy()

        scale = args.scale_pr if case == "pr" else args.scale_tavg

        # --- Precompute consistent color limits ---
        fixed_limits = {}
        abs_extends = {}
        rel_vlims = {}
        rel_extends = {}

        df1800_all = df[df["resolution"] == "1800arcsec"].set_index("station_id")

        for metric in METRICS:
            vals_abs = df1800_all[metric].to_numpy(float)

            if metric == "RMSE":
                vmin_abs = np.nanpercentile(vals_abs, 5)
                vmax_abs = np.nanpercentile(vals_abs, 95)
            else:  # KGE
                vmin_abs, vmax_abs = -1, 1

            fixed_limits[metric] = (vmin_abs, vmax_abs)
            abs_extends[metric] = colorbar_extend(vals_abs, vmin_abs, vmax_abs)

            # relative diffs across all stations & resolutions
            diffs_all = []
            for res in DIFF_ORDER:
                df_res = df[df["resolution"] == res].set_index("station_id").reindex(df1800_all.index)
                if metric == "KGE":
                    d = np.abs(1 - df1800_all[metric]) - np.abs(1 - df_res[metric])
                else:
                    d = df_res[metric] - df1800_all[metric]
                diffs_all.append(d.to_numpy(float))

            diffs_all = np.concatenate(diffs_all)
            diffs_all = diffs_all[np.isfinite(diffs_all)]
            absmax = float(np.nanmax(np.abs(diffs_all))) if diffs_all.size else 1.0
            rel_vlims[metric] = (-absmax, absmax)
            # by construction absmax is the max, extend should usually be "neither"
            rel_extends[metric] = "neither"

        # --- Plot ---
        fig = plt.figure(figsize=(20, 3.2 * len(METRICS)))
        gs = gridspec.GridSpec(
            nrows=len(METRICS), ncols=5, figure=fig,
            width_ratios=[1, 1, 1, 1, 0.45]
        )

        # store mappables + axes per row
        row_artists = []

        for row, metric in enumerate(METRICS):
            df1800 = df[df["resolution"] == "1800arcsec"].set_index("station_id")
            lats = df1800["lat"]
            lons = df1800["lon"]
            sizes = (df1800["n_obs"] / scale).clip(lower=5)

            vmin_abs, vmax_abs = fixed_limits[metric]
            vmin_rel, vmax_rel = rel_vlims[metric]

            cmap_abs = "inferno_r" if metric == "RMSE" else "coolwarm"
            if metric == "KGE":
                cmap_rel = "PRGn"  # diverging
            else:
                cmap_rel = "PRGn_r" # diverging reversed

            # --- Column 0: absolute baseline ---
            ax0 = fig.add_subplot(gs[row, 0], projection=ccrs.PlateCarree())
            sc0 = ax0.scatter(
                lons, lats, c=df1800[metric],
                cmap=cmap_abs, vmin=vmin_abs, vmax=vmax_abs,
                s=sizes, edgecolor="grey", alpha=0.85,
                transform=ccrs.PlateCarree()
            )
            ax0.coastlines()
            ax0.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="gray", linewidth=0.8)
            ax0.set_extent([args.lon_min, args.lon_max, args.lat_min, args.lat_max], crs=ccrs.PlateCarree())
            if row == 0:
                ax0.set_title('1800"', fontsize=16, fontweight="bold")

            # --- Columns 1–3: diffs vs baseline ---
            diff_axes = []
            last_sc = None

            for col, res in enumerate(DIFF_ORDER, start=1):
                ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())

                df_res = df[df["resolution"] == res].set_index("station_id").reindex(df1800.index)
                if metric == "KGE":
                    diff = np.abs(1 - df1800[metric]) - np.abs(1 - df_res[metric])
                else:
                    diff = df_res[metric] - df1800[metric]

                sc = ax.scatter(
                    lons, lats, c=diff,
                    cmap=cmap_rel, vmin=vmin_rel, vmax=vmax_rel,
                    s=sizes, edgecolor="grey", alpha=0.85,
                    transform=ccrs.PlateCarree()
                )
                last_sc = sc
                diff_axes.append(ax)

                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="gray", linewidth=0.8)
                ax.set_extent([args.lon_min, args.lon_max, args.lat_min, args.lat_max], crs=ccrs.PlateCarree())
                if row == 0:
                    ax.set_title(f'{res.replace("arcsec","")}″ - 1800"', fontsize=16, fontweight="bold")

            # --- Column 4: boxplot ---
            axb = fig.add_subplot(gs[row, 4])
            sns.boxplot(
                data=df,
                x="resolution",
                y=metric,
                order=RES_ORDER,
                ax=axb,
                notch=True,
                width=0.4,
                showfliers=False,
                boxprops=dict(linewidth=1.5, color="grey"),
                whiskerprops=dict(linewidth=1.1),
                medianprops=dict(linewidth=1.5, color="darkred"),
                capprops=dict(linewidth=0),
            )
            axb.set_xlabel("")
            axb.set_xticklabels(["1800″", "300″", "90″", "30″"], fontsize=12)
            axb.set_ylabel(f"{metric} [{units[case][metric]}]", fontsize=14)
            axb.grid(True, linestyle="--", alpha=0.7)
            axb.yaxis.set_label_position("right")
            axb.yaxis.tick_right()

            # store for colorbars
            row_artists.append((metric, ax0, sc0, diff_axes, last_sc))


        plt.tight_layout(w_pad=0.2, h_pad=1.5)

        # --- Add horizontal colorbars below each row ---
        for metric, ax0, sc0, diff_axes, last_sc in row_artists:
            pos0 = ax0.get_position()

            # Put bars just below the row
            cb_h = 0.015
            cb_gap = 0.03
            bottom_y = pos0.y0 - cb_gap

            # Absolute bar under column 0
            abs_cax = fig.add_axes([pos0.x0, bottom_y, pos0.width, cb_h])
            cb_abs = fig.colorbar(sc0, cax=abs_cax, orientation="horizontal", extend=abs_extends[metric])
            cb_abs.set_label(f"{metric} [{units[case][metric]}]", fontsize=13)

            # Relative bar spanning columns 1–3
            pos1 = diff_axes[0].get_position()
            pos3 = diff_axes[-1].get_position()
            rel_x0 = pos1.x0
            rel_w = (pos3.x0 + pos3.width) - pos1.x0

            rel_cax = fig.add_axes([rel_x0, bottom_y, rel_w, cb_h])
            cb_rel = fig.colorbar(last_sc, cax=rel_cax, orientation="horizontal", extend=rel_extends[metric])

            if metric == "KGE":
                cb_rel.set_label("Improvement towards KGE = 1", fontsize=13)
            else:
                cb_rel.set_label(f"Δ {metric} [{units[case][metric]}]", fontsize=13)

        out = outdir / f"error_comparison_relative_{case}_rmse_kge.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
