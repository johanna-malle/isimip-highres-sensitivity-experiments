# -*- coding: utf-8 -*-
"""
Figure 5: Climate accuracy vs impact-model performance (ΔNRMSE).

Expected inputs in:
  - <data-dir>/fig_3/stats_all/                  (model NRMSE/RMSE stats)
  - <data-dir>/fig_5/stats_all_climate/          (climate NRMSE stats; GR4J uses *_shp.csv)
Writes output to:
  - <out-dir>/fig_5/climate_vs_model_nrmse_ta_pr.png
"""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


RES_STEPS = [("60km", "10km"), ("10km", "3km"), ("3km", "1km")]

MODEL_GROUPS = {
    "Forest (pt)": ["BBGCMuSo", "3D-CMCC", "3PGN-BW", "4C", "3PGHydro", "Prebas"],
    "Forest (sp)": ["TreeMig", "CARAIB"],
    "Water": ["CWatM", "GR4J", "SWAT", "SWIM"],
    "Lakes": ["MITgcm", "SIMSTRAT"],
    "Labour": ["Labour"],
}
GROUP_COLORS = {
    "Forest (pt)": "#2e7d32",
    "Forest (sp)": "#a6d854",
    "Water": "darkturquoise",
    "Lakes": "darkblue",
    "Labour": "grey",
}
MODEL_TO_GROUP = {m.lower(): g for g, ms in MODEL_GROUPS.items() for m in ms}

UNIT_OVERRIDE_MAP = {
    "CARAIB": "Vielsalm",
    "BBGCMuSo": "Collelongo",
    "3D-CMCC": "Collelongo",
    "3PGN-BW": "Collelongo",
    "4C": "Collelongo",
    "3PGHydro": "Collelongo",
}

MODEL_PREFIXES = {
    "forest": ["forest_evap_", "forest_gpp_"],
    "CARAIB": ["evap_caraib_", "gpp_caraib_"],
    "Prebas": ["prebas_gpp_", "prebas_evap_"],
    "TreeMig": ["treemig_all_"],
    "CWatM": ["cwatm_discharge_all_", "cwatm_reservoir_all_"],
    "ORCHIDEE": ["orchidee_alt_", "orchidee_magt_", "orchidee_stemp_"],  # excluded
    "GR4J": ["gr4j_all_"],
    "SWAT": ["swat_all_"],
    "SWIM": ["swim_"],
    "MITgcm": ["mitgcm_all_"],
    "SIMSTRAT": ["simstrat_all_"],
}
CLIMATE_PREFIXES = {
    "forest": ["forest_all_"],
    "CARAIB": ["caraib_all_"],
    "Prebas": ["forest_prebas_sod_", "forest_prebas_hyy_"],
    "TreeMig": ["forest_treemig_all_sp_"],
    "CWatM": ["CWatM_all_"],
    "ORCHIDEE": ["orchidee_all_"],  # excluded
    "GR4J": ["GR4J_all_"],
    "SWAT": ["SWAT_all_"],
    "SWIM": ["SWIM_"],
    "MITgcm": ["mitgcm_all_"],
    "SIMSTRAT": ["simstrat_all_"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def _find_case_insensitive(dirpath: Path, prefix: str, suffix: str) -> Path | None:
    if not dirpath.exists():
        return None
    pref = prefix.lower()
    suf = suffix.lower()
    for p in dirpath.iterdir():
        if p.is_file():
            n = p.name.lower()
            if n.startswith(pref) and n.endswith(suf):
                return p
    return None


def _read_stats_csv(path: Path, model_key: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    if model_key.upper() == "GR4J":
        df.index = df.index.to_series().astype(str).str.replace(r"\.0$", "", regex=True)
        df.rename(index={"6.1": "6.10"}, inplace=True)

    if model_key == "MITgcm":
        df.index = df.index.to_series().astype(str).str.lower().apply(lambda x: f"station {x}")

    df.index = df.index.to_series().astype(str).str.lower().str.replace("sodankylae", "sodankyla")
    return df


def _percent_change(df: pd.DataFrame, from_res: str, to_res: str) -> pd.Series:
    a = df[from_res].astype(float)
    b = df[to_res].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (b - a) / a * 100.0
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _rows_from_series(series: pd.Series, default_model: str) -> pd.DataFrame:
    rows = []
    for idx, val in series.items():
        unit = str(idx).strip().lower()

        if unit in MODEL_TO_GROUP:
            model = unit
        else:
            model = str(default_model).strip().lower()

        model_canon = None
        for g, ms in MODEL_GROUPS.items():
            for m in ms:
                if m.lower() == model:
                    model_canon = m
                    break
            if model_canon:
                break
        if model_canon is None:
            model_canon = model

        group = MODEL_TO_GROUP.get(model_canon.lower(), None)
        if group is None:
            # not in included groups -> drop
            continue

        loc = unit
        if model_canon in UNIT_OVERRIDE_MAP:
            loc = UNIT_OVERRIDE_MAP[model_canon].lower()

        rows.append(
            {"model": model_canon, "group": group, "loc": loc, "improvement": float(val) if pd.notna(val) else np.nan}
        )

    return pd.DataFrame(rows)


def load_model_improvements(model_dir: Path, steps: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {f"{a}_{b}": [] for a, b in steps}

    for model_key, prefixes in MODEL_PREFIXES.items():
        for prefix in prefixes:
            f = _find_case_insensitive(model_dir, prefix, "rmse.csv")
            if f is None:
                continue
            df = _read_stats_csv(f, model_key)

            for a, b in steps:
                key = f"{a}_{b}"
                if a not in df.columns or b not in df.columns:
                    continue
                s = _percent_change(df, a, b)
                out[key].append(_rows_from_series(s, default_model=model_key))

    final = {}
    for k, dfs in out.items():
        if not dfs:
            final[k] = pd.DataFrame(columns=["model", "group", "loc", "improvement_model"])
            continue
        d = pd.concat(dfs, ignore_index=True)
        d = d.rename(columns={"improvement": "improvement_model"})
        final[k] = d
    return final


def load_climate_improvements(clim_dir: Path, var: str, steps: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    out: dict[str, list[pd.DataFrame]] = {f"{a}_{b}": [] for a, b in steps}

    for model_key, prefixes in CLIMATE_PREFIXES.items():
        for prefix in prefixes:
            if model_key.upper() == "GR4J":
                f = _find_case_insensitive(clim_dir, prefix, f"nrmse_{var}_shp.csv")
            else:
                f = _find_case_insensitive(clim_dir, prefix, f"nrmse_{var}.csv")
                if f is None:
                    f = _find_case_insensitive(clim_dir, prefix, f"nrmse_{var}_shp.csv")

            if f is None:
                continue

            df = _read_stats_csv(f, model_key)

            for a, b in steps:
                key = f"{a}_{b}"
                if a not in df.columns or b not in df.columns:
                    continue
                s = _percent_change(df, a, b)
                out[key].append(_rows_from_series(s, default_model=model_key))

    final = {}
    for k, dfs in out.items():
        if not dfs:
            final[k] = pd.DataFrame(columns=["model", "group", "loc", "improvement_climate"])
            continue
        d = pd.concat(dfs, ignore_index=True)
        d = d.rename(columns={"improvement": "improvement_climate"})
        final[k] = d
    return final


def merge_pairs(clim: dict[str, pd.DataFrame], mod: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    merged = {}
    for key in clim.keys():
        c = clim[key]
        m = mod.get(key, pd.DataFrame())
        if c.empty or m.empty:
            merged[key] = pd.DataFrame()
            continue

        df = pd.merge(c, m, on=["model", "loc"], suffixes=("_clim", "_mod"))

        df["group"] = df["model"].str.lower().map(MODEL_TO_GROUP)
        df = df[df["group"].notna()].copy()  # drop excluded groups (Biomes)
        df["color"] = df["group"].map(GROUP_COLORS)

        merged[key] = df
    return merged


def step_title(step: tuple[str, str]) -> str:
    mapping = {
        ("60km", "10km"): ('1800"', '300"'),
        ("10km", "3km"): ('300"', '90"'),
        ("3km", "1km"): ('90"', '30"'),
    }
    fr, to = mapping.get(step, (step[0], step[1]))
    return f"{fr} \u2192 {to}"


def draw_quadrants(ax, vmin, vmax):
    ax.axhline(0, color="black", linestyle="--", linewidth=1, zorder=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=1)

    ax.axhspan(0, vmax, xmin=0.5, xmax=1, facecolor="lightcoral", alpha=0.15, zorder=0)
    ax.axhspan(0, vmax, xmin=0, xmax=0.5, facecolor="lightblue", alpha=0.15, zorder=0)
    ax.axhspan(vmin, 0, xmin=0, xmax=0.5, facecolor="lightgreen", alpha=0.15, zorder=0)
    ax.axhspan(vmin, 0, xmin=0.5, xmax=1, facecolor="moccasin", alpha=0.15, zorder=0)

    ax.text(0.5 * (vmin + 0), 0.5 * (vmin + 0), "Better Climate\nBetter Model",
            fontsize=12, color="gray", alpha=0.55, ha="center", va="center", zorder=0)
    ax.text(0.5 * (vmax + 0), 0.5 * (vmin + 0), "Worse Climate\nBetter Model",
            fontsize=12, color="gray", alpha=0.55, ha="center", va="center", zorder=0)
    ax.text(0.5 * (vmax + 0), 0.5 * (vmax + 0), "Worse Climate\nWorse Model",
            fontsize=12, color="gray", alpha=0.55, ha="center", va="center", zorder=0)
    ax.text(0.5 * (vmin + 0), 0.5 * (vmax + 0), "Better Climate\nWorse Model",
            fontsize=12, color="gray", alpha=0.55, ha="center", va="center", zorder=0)


def main():
    args = parse_args()

    clim_dir = args.data_dir / "fig_5" / "stats_all_climate"
    model_dir = args.data_dir / "fig_3" / "stats_all"
    out_dir = args.out_dir / "fig_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clim_dir.exists():
        raise FileNotFoundError(f"Missing climate stats folder: {clim_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model stats folder: {model_dir}")

    sns.set(style="whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 12})

    model_imp = load_model_improvements(model_dir, RES_STEPS)
    clim_ta = load_climate_improvements(clim_dir, "ta", RES_STEPS)
    clim_pr = load_climate_improvements(clim_dir, "pr", RES_STEPS)

    merged_ta = merge_pairs(clim_ta, model_imp)
    merged_pr = merge_pairs(clim_pr, model_imp)

    # Fixed limits as requested
    vmin, vmax = -60.0, 60.0

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 11), sharex=True, sharey=True)

    for col, step in enumerate(RES_STEPS):
        key = f"{step[0]}_{step[1]}"

        for row, merged in enumerate([merged_ta, merged_pr]):
            ax = axs[row, col]
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_aspect("equal", adjustable="box")

            draw_quadrants(ax, vmin, vmax)

            df = merged.get(key, pd.DataFrame())
            if df is not None and not df.empty:
                ax.scatter(
                    df["improvement_climate"],
                    df["improvement_model"],
                    c=df["color"],
                    s=105,
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.9,
                    zorder=3,
                )

            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            ax.set_title(step_title(step), fontsize=13, fontweight="bold")

    axs[0, 0].set_ylabel("Δ Model Performance (% ΔNRMSE)", fontweight="bold", fontsize=15)
    axs[1, 0].set_ylabel("Δ Model Performance (% ΔNRMSE)", fontweight="bold", fontsize=15)
    for ax in axs[1, :]:
        ax.set_xlabel("Δ Climate Accuracy (% ΔNRMSE)", fontweight="bold", fontsize=15)

    fig.text(0.07, 0.91, "(a) Climate Accuracy vs. Model Performance – Temperature",
             ha="left", va="bottom", fontsize=16, fontweight="bold")
    fig.text(0.07, 0.49, "(b) Climate Accuracy vs. Model Performance – Precipitation",
             ha="left", va="bottom", fontsize=16, fontweight="bold")

    legend_groups = ["Forest (pt)", "Forest (sp)", "Water", "Lakes", "Labour"]
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=GROUP_COLORS[g],
               label=g, markeredgecolor="black", markersize=10)
        for g in legend_groups
    ]
    fig.legend(
        handles=handles,
        title="ISIMIP Sectors",
        loc="center left",
        bbox_to_anchor=(0.90, 0.5),
        ncol=1,
        frameon=True,
        fontsize=13,
        title_fontsize=14,
        labelspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(right=0.88, wspace=0.03, hspace=0.20)

    out = out_dir / "climate_vs_model_nrmse_ta_pr.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
