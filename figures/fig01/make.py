# -*- coding: utf-8 -*-
"""
Figure 1: Global map of participating ISIMIP sectors/models (overview).

Expected inputs in:  <data-dir>/fig_1/raw/
Writes output to:    <out-dir>/fig_1/overview_models_no_borders.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def scatter_x(ax, df, x, y, color, label):
    ax.scatter(
        df[x], df[y],
        color=color, marker="x", s=50, linewidth=2.5, alpha=0.8,
        transform=ccrs.PlateCarree(), label=label, zorder=5
    )


def main():
    args = parse_args()

    raw = args.data_dir / "fig_1"
    outdir = args.out_dir / "fig_1"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load inputs ---
    reg_forests_collelongo = pd.read_csv(raw / "reg_forests_collelongo.csv")
    reg_forests_caraib = pd.read_csv(raw / "reg_forests_caraib.csv")
    reg_forests_prebas = pd.read_csv(raw / "reg_forests_prebas.csv")

    reg_water_cwatm = pd.read_csv(raw / "reg_water_cwatm_pt.csv")
    reg_water_swat = pd.read_csv(raw / "reg_water_swat_pt.csv")
    reg_water_swim = pd.read_csv(raw / "reg_water_swim_pt.csv")
    reg_water_gr4j = pd.read_csv(raw / "reg_water_gr4j_pt.csv")

    lakes_simstrat = pd.read_csv(raw / "lakes_simstrat_pt.csv")
    lakes_mitgcm = pd.read_csv(raw / "lakes_mitgcm_pt.csv")

    biomes_orchidee = pd.read_csv(raw / "biomes_orchidee_pt.csv")

    # Load country borders for Switzerland highlight
    ne_zip = raw / "ne_110m_admin_0_countries.zip"
    world = gpd.read_file(ne_zip)

    # --- base map ---
    fig = plt.figure(figsize=(11, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent([-120, 170, -58, 90], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="gainsboro", edgecolor="black", linewidth=0.25, zorder=1)
    ax.coastlines(resolution="110m", linewidth=0.3, color="black", zorder=2)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # --- points ---
    scatter_x(ax, lakes_mitgcm, "long", "lat", "darkblue", r"$\mathbf{Lakes}$" "\n" "MITGCM")
    scatter_x(ax, lakes_simstrat, "long", "lat", "dodgerblue", r"$\mathbf{Lakes}$" "\n" "Simstrat")

    scatter_x(ax, reg_water_swat, "lon", "lat", "darkred", r"$\mathbf{Regional\ Water}$" "\n" "SWAT")
    scatter_x(ax, reg_water_swim, "lon", "lat", "red", r"$\mathbf{Regional\ Water}$" "\n" "SWIM")
    scatter_x(ax, reg_water_gr4j, "long", "lat", "orange", r"$\mathbf{Regional\ Water}$" "\n" "GR4J")
    scatter_x(ax, reg_water_cwatm, "long", "lat", "darkgoldenrod", r"$\mathbf{Regional\ Water}$" "\n" "CWATM")

    scatter_x(ax, biomes_orchidee, "long", "lat", "salmon", r"$\mathbf{Biomes}$" "\n" "ORCHIDEE-MICT")

    scatter_x(
        ax, reg_forests_collelongo, "long", "lat", "seagreen",
        r"$\mathbf{Regional\ forest}$ (point-scale)" "\n"
        "4C/i3PGmiX/3PGHydro/BBGCMuSo/3D-CMCC-FEM"
    )
    scatter_x(ax, reg_forests_prebas, "long", "lat", "yellowgreen",
              r"$\mathbf{Regional\ forest}$ (point-scale)" "\n" "Prebas")

    # --- polygons (Treemig + CARAIB) ---
    sw = world.loc[world["ADMIN"] == "Switzerland"].geometry.values[0]
    ax.add_geometries([sw], crs=ccrs.PlateCarree(),
                      facecolor="greenyellow", edgecolor="darkgreen", linewidth=0.8, zorder=3)

    caraib = reg_forests_caraib.iloc[0]
    polygon_coords = [
        (caraib.x_1, caraib.y_1),
        (caraib.x_2, caraib.y_1),
        (caraib.x_2, caraib.y_2),
        (caraib.x_1, caraib.y_2),
    ]
    ax.add_patch(Polygon(polygon_coords, closed=True, ec="darkslategrey", fc="teal",
                         lw=0.5, transform=ccrs.PlateCarree(), zorder=4))

    # extra legend handles
    treemig_h = Line2D([0], [0], marker="s", color="w",
                       markerfacecolor="greenyellow", markeredgecolor="darkgreen",
                       label=r"$\mathbf{Regional\ forest}$ (spatial)" "\n" "Treemig", lw=0)
    caraib_h = Line2D([0], [0], marker="s", color="w",
                      markerfacecolor="teal", markeredgecolor="darkslategrey",
                      label=r"$\mathbf{Regional\ forest}$ (spatial)" "\n" "CARAIB", lw=0)
    labour_h = Line2D([0], [0], marker="s", color="w",
                      markerfacecolor="white", markeredgecolor="white",
                      label=r"$\mathbf{Labour}$" "\n" "(global, not shown)", lw=0)

    handles, labels = ax.get_legend_handles_labels()
    handles += [treemig_h, caraib_h, labour_h]
    labels += [treemig_h.get_label(), caraib_h.get_label(), labour_h.get_label()]

    leg = ax.legend(
        handles, labels,
        title="Participating ISIMIP sectors and models",
        title_fontsize="medium",
        fontsize="small",
        ncol=2,
        loc="lower left",
        labelspacing=0.3,
        borderpad=0.2,
        frameon=True,
    )

    # color legend text by keyword
    for t in leg.get_texts():
        s = t.get_text()
        if "Lakes" in s:
            t.set_color("darkblue")
        elif "Regional\\ Water" in s or "Regional Water" in s:
            t.set_color("darkturquoise")
        elif "Biomes" in s:
            t.set_color("salmon")
        elif "Labour" in s:
            t.set_color("black")
        else:
            t.set_color("darkgreen")

    out = outdir / "overview_models_no_borders.png"
    fig.savefig(out, facecolor="w", transparent=False, bbox_inches="tight", pad_inches=0, dpi=args.dpi)
    plt.show()


if __name__ == "__main__":
    main()
