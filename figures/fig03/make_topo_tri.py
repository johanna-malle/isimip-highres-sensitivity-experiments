# -*- coding: utf-8 -*-
"""
Figure 3b: Terrain Ruggedness Index (TRI) per model.

Expected inputs in:  <data-dir>/fig_3/earthenv/
  - *_vrm_tri_all.csv  (per model; TRI/VRM columns)
Writes output to:    <out-dir>/fig_3/topo_tri.png
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Labour INCLUDED, but will be forced to NaN
MODEL_GROUPS = {
    "Forest (pt)": ["BBGCMuSo", "3D-CMCC", "3PGN-BW", "4C", "3PGHydro", "Prebas"],
    "Forest (sp)": ["TreeMig", "CARAIB"],
    "Biomes": ["ORCHIDEE"],
    "Water": ["CWatM", "GR4J", "SWAT", "SWIM"],
    "Lakes": ["MITgcm", "SIMSTRAT"],
    "Labour": ["Labour"],
}

FOREST_PT_MODELS = {"BBGCMuSo", "3D-CMCC", "3PGN-BW", "4C", "3PGHydro"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def model_key(model: str) -> str:
    # forest point models share one file
    if model in FOREST_PT_MODELS:
        return "for"
    return model.lower()


def read_tri_mean(earthenv_dir: Path, key: str) -> float:
    f = earthenv_dir / f"{key}_vrm_tri_all.csv"
    df = pd.read_csv(f)
    if "tri" not in df.columns:
        raise ValueError(f"Missing 'tri' column in {f}")
    return float(df["tri"].mean())


def main():
    args = parse_args()

    earthenv_dir = args.data_dir / "fig_3" / "earthenv"
    outdir = args.out_dir / "fig_3"
    outdir.mkdir(parents=True, exist_ok=True)

    if not earthenv_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {earthenv_dir}")

    cache: dict[str, float] = {}
    grouped_data: dict[str, pd.DataFrame] = {}
    all_tri_vals: list[float] = []

    for group, models in MODEL_GROUPS.items():
        rows = []
        idx = []
        for m in models:
            # Force Labour to NaN
            if m == "Labour":
                tri = np.nan
            else:
                k = model_key(m)
                if k not in cache:
                    try:
                        cache[k] = read_tri_mean(earthenv_dir, k)
                    except Exception:
                        cache[k] = np.nan
                tri = cache[k]

            rows.append([tri])
            idx.append(m)
            all_tri_vals.append(tri)

        grouped_data[group] = pd.DataFrame(rows, index=idx, columns=["TRI"])

    all_tri = np.array(all_tri_vals, dtype=float)
    finite = all_tri[np.isfinite(all_tri)]
    tri_vmin = float(np.nanmin(finite)) if finite.size else 0.0
    tri_vmax = float(np.nanmax(finite)) if finite.size else 1.0

    sorted_groups = list(MODEL_GROUPS.keys())
    row_counts = [grouped_data[g].shape[0] for g in sorted_groups]

    fig_height = sum(row_counts) * 0.6 + 2.9
    fig, axes = plt.subplots(
        nrows=len(sorted_groups),
        ncols=1,
        figsize=(1.0, fig_height),
        gridspec_kw={"height_ratios": row_counts},
    )
    if len(sorted_groups) == 1:
        axes = [axes]

    plt.subplots_adjust(hspace=0.255, bottom=0.08)

    cmap = sns.color_palette("YlGnBu", as_cmap=True).copy()
    cmap.set_bad(color="grey")  # NaNs show as grey (Labour)

    tri_mappable = None

    for i, group in enumerate(sorted_groups):
        ax = axes[i]
        data = grouped_data[group]

        hm = sns.heatmap(
            data[["TRI"]],
            cmap=cmap,
            ax=ax,
            annot=True,
            fmt=".2f",
            cbar=False,
            linewidths=0.7,
            linecolor="gray",
            square=False,
            vmin=tri_vmin,
            vmax=tri_vmax,
        )
        if tri_mappable is None:
            tri_mappable = hm.get_children()[0]

        # narrow TRI strip
        ax.tick_params(axis="y", labelleft=False)
        ax.set_xticklabels([])

        ax.add_patch(
            patches.Rectangle(
                (0, 0), 1, 1,
                linewidth=2,
                edgecolor="gray",
                facecolor="none",
                transform=ax.transAxes,
                zorder=10,
            )
        )

    cax = fig.add_axes([0.15, 0.02, 0.75, 0.02])
    cb = fig.colorbar(tri_mappable, cax=cax, orientation="horizontal")
    cb.set_label("TRI [-]", fontsize=15)

    out_path = outdir / "topo_tri.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
