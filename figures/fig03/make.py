# -*- coding: utf-8 -*-
"""
Figure 3a: Cross-sector/overall heatmaps for RMSE and KGE.

Expected inputs in:  <data-dir>/fig_3/stats_all/
  - *rmse.csv
  - *kge.csv
Writes output to:    <out-dir>/fig_3/overall_rmse_pct_lab.png
                    <out-dir>/fig_3/overall_kge_lab.png

Notes:
- RMSE plotted as % change vs baseline resolution.
- KGE plotted as absolute change vs baseline resolution.
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


METRICS = ["rmse", "kge"]
METRIC_LABELS = {"rmse": "NRMSE", "kge": "KGE"}

RES_COLS = ["60km", "10km", "3km", "1km"]
RES_TICKLABELS = ['1800"', '300"', '90"', '30"']


MODEL_GROUPS = {
    "Forest (pt)": ["BBGCMuSo", "3D-CMCC", "3PGN-BW", "4C", "3PGHydro", "Prebas"],
    "Forest (sp)": ["TreeMig", "CARAIB"],
    "Biomes": ["ORCHIDEE"],
    "Water": ["CWatM", "GR4J", "SWAT", "SWIM"],
    "Lakes": ["MITgcm", "SIMSTRAT"],
    "Labour": ["Labour"],  # always included
}

GROUP_COLORS = {
    "Forest (pt)": "#4daf4a",
    "Forest (sp)": "darkgreen",
    "Biomes": "darkorange",
    "Water": "darkturquoise",
    "Lakes": "darkblue",
    "Labour": "grey",
}

DISPLAY_RENAME = {
    "3PGN-BW": "i3PGmiX",
    "3D-CMCC": "3D-CMCC-FEM",
    "ORCHIDEE": "ORCHIDEE-MICT",
}

EXCLUDED_KEYWORDS = [
    "mean_per_variable", "mean_per_location",
    "reservoir", "discharge", "evap", "gpp",
    "nrmse", "gr4j_all", "mitgcm_all", "alt", "magt", "stemp",
    "simstrat_all", "swat_all_", "treemig_all_",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def list_metric_files(stats_dir: Path, metric: str) -> list[Path]:
    files = sorted(stats_dir.glob(f"*{metric}.csv"))
    return [f for f in files if not any(kw in f.name for kw in EXCLUDED_KEYWORDS)]


def baseline_index(model_name: str) -> int:
    # special case for MITgcm (different baseline resolution)
    return 1 if model_name == "MITgcm" else 0


def metric_is_percent(metric: str) -> bool:
    return metric == "rmse"


def compute_diff(vals: np.ndarray, base_idx: int, percent: bool) -> np.ndarray:
    base = vals[base_idx]
    if percent:
        return (vals - base) / base * 100 if base != 0 else np.full_like(vals, np.nan, dtype=float)
    return vals - base


def make_annotations(vals: np.ndarray, diff: np.ndarray, base_idx: int, percent: bool) -> list[str]:
    out = []
    for i, (v, d) in enumerate(zip(vals, diff)):
        if i == base_idx:
            out.append(f"{v:.2f}")
        else:
            out.append(f"{v:.2f}\n({d:+.1f}%)" if percent else f"{v:.2f}\n({d:+.3f})")
    return out


def extend_from_limits(all_vals: np.ndarray, vmin: float, vmax: float) -> str:
    x = all_vals[np.isfinite(all_vals)]
    if x.size == 0:
        return "neither"
    below = np.nanmin(x) < vmin
    above = np.nanmax(x) > vmax
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def main():
    args = parse_args()

    stats_dir = args.data_dir / "fig_3" / "stats_all"
    outdir = args.out_dir / "fig_3"
    outdir.mkdir(parents=True, exist_ok=True)

    if not stats_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {stats_dir}")

    for metric in METRICS:
        files = list_metric_files(stats_dir, metric)
        if not files:
            print(f"[warn] No files found for metric='{metric}' in {stats_dir}")
            continue

        percent_mode = metric_is_percent(metric)

        model_tables: dict[str, np.ndarray] = {}
        ann_tables: dict[str, list[str]] = {}
        diffs_for_scaling = []

        for f in files:
            df = pd.read_csv(f, index_col="model")[RES_COLS]

            for model_name in df.index:
                vals = df.loc[model_name].to_numpy(dtype=float)
                bidx = baseline_index(model_name)
                diff = compute_diff(vals, bidx, percent=percent_mode)

                model_tables[model_name] = diff
                ann_tables[model_name] = make_annotations(vals, diff, bidx, percent=percent_mode)

                # ignore baseline cell for scaling
                mask = np.ones_like(diff, dtype=bool)
                mask[bidx] = False
                diffs_for_scaling.append(diff[mask])

        diffs_for_scaling = np.concatenate(diffs_for_scaling) if diffs_for_scaling else np.array([])
        diffs_for_scaling = diffs_for_scaling[np.isfinite(diffs_for_scaling)]

        if diffs_for_scaling.size == 0:
            vmin, vmax = -1, 1
        else:
            vmin = np.nanpercentile(diffs_for_scaling, 5)
            vmax = np.nanpercentile(diffs_for_scaling, 95)

        extend = extend_from_limits(diffs_for_scaling, vmin, vmax)

        grouped_data = {}
        grouped_annot = {}

        for group, models in MODEL_GROUPS.items():
            rows, ann, idx = [], [], []
            for m in models:
                if m in model_tables:
                    rows.append(model_tables[m])
                    ann.append(ann_tables[m])
                    idx.append(m)

            if rows:
                d = pd.DataFrame(rows, index=idx, columns=RES_COLS).rename(index=DISPLAY_RENAME)
                a = pd.DataFrame(ann, index=idx, columns=RES_COLS).rename(index=DISPLAY_RENAME)
                grouped_data[group] = d
                grouped_annot[group] = a

        sorted_groups = [g for g in MODEL_GROUPS.keys() if g in grouped_data]
        row_counts = [grouped_data[g].shape[0] for g in sorted_groups]

        cmap_used = "PiYG" if metric == "kge" else "PiYG_r"
        cmap = sns.color_palette(cmap_used, as_cmap=True).copy()
        cmap.set_bad(color="grey")

        fig_height = sum(row_counts) * 0.6 + 2.7
        fig, axes = plt.subplots(
            nrows=len(sorted_groups),
            figsize=(5.2, fig_height),
            gridspec_kw={"height_ratios": row_counts},
        )
        plt.subplots_adjust(hspace=0.25)
        if len(sorted_groups) == 1:
            axes = [axes]

        heatmaps = []
        for ax, group in zip(axes, sorted_groups):
            data = grouped_data[group]
            annot = grouped_annot[group]

            hm = sns.heatmap(
                data,
                cmap=cmap,
                center=0,
                linewidths=0.7,
                linecolor="gray",
                annot=annot,
                fmt="",
                ax=ax,
                cbar=False,
                vmin=vmin,
                vmax=vmax,
            )
            heatmaps.append(hm)

            ax.set_ylabel(
                group,
                rotation=90,
                labelpad=2,
                color=GROUP_COLORS[group],
                weight="bold",
                fontsize=10.5,
            )
            ax.tick_params(axis="y", labelsize=9)
            ax.set_yticklabels(
                ax.get_yticklabels(),
                color=GROUP_COLORS[group],
                rotation=0,
                ha="right",
                va="center",
            )

            if group != sorted_groups[-1]:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(RES_TICKLABELS)

            ax.add_patch(
                patches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="gray",
                    facecolor="none",
                    transform=ax.transAxes,
                    zorder=10,
                )
            )

        axes[-1].set_xlabel("Resolution", fontsize=11)

        metric_label = METRIC_LABELS[metric]
        if percent_mode:
            cbar_label = f"% change in {metric_label}"
            out_name = f"overall_{metric}_pct_lab.png"
        else:
            cbar_label = f"$\\Delta$ {metric_label}"
            out_name = f"overall_{metric}_lab.png"

        fig.colorbar(
            heatmaps[-1].collections[0],
            ax=axes,
            orientation="horizontal",
            shrink=0.7,
            pad=0.08,
            label=cbar_label,
            extend=extend,
        )

        fig.align_ylabels()
        out_path = outdir / out_name
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
