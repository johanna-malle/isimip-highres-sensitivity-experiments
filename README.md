# Reproducible figures for:
## *When and where higher-resolution climate data improve impact model performance*

This repository contains the **code and instructions needed to reproduce the figures** in the paper:

> **When and where higher-resolution climate data improve impact model performance**

Figures are organized as **one folder per figure**, and can be generated via:

- **prep** (optional / where necessary): convert raw inputs into processed tables (CSV) used for plotting
- **make**: generate the final figure file(s)

Scripts can be run either from the command line or directly via “Run” in an IDE.
All scripts accept `--data-dir` and `--out-dir` where relevant. Outputs are written to `outputs/fig_X/`.

---

## Repository layout

- `figures/fig_1/ ... figures/fig_5/`  
  One folder per figure. Each typically contains:
  - `make*.py`: generates the figure(s)
  - `prep.py` (where necessary): creates `data/processed/...` inputs used by `make*.py`

- `data/raw/fig_X/`  
  Raw inputs required to reproduce Figure X (download required; see below).

- `data/processed/fig_X/`  
  Intermediate products created by `prep.py` (not meant to be edited manually).

- `outputs/fig_X/`  
  Generated figures.

---

## Data download (required)

Before you can run this repository, the following data needs to be downloaded from [Zenodo (doi:10.5281/zenodo.17940720)](https://doi.org/10.5281/zenodo.17940720)
and extracted into the **data/raw/** folder.

After extraction, the expected layout should look like:

- `data/raw/fig_1/...`
- `data/raw/fig_2/...`
- `data/raw/fig_3/...`
- `data/raw/fig_4/...`
- `data/raw/fig_5/...`

---

## Get the code

Clone the repository and move into it:

```bash
git clone https://github.com/johanna-malle/isimip-highres-sensitivity-experiments.git
cd isimip-highres-sensitivity-experiments
```

## Requirements

Python is used throughout. The recommended setup is via Conda.

### Install the environment

```bash
conda env create -f environment.yml
conda activate isimip-highres-figures
```

### Run the scripts (from the repo root)

```bash
python figures/fig01/make.py --data-dir data --out-dir outputs

python figures/fig02/prep.py --data-dir data
python figures/fig02/make_station.py --data-dir data --out-dir outputs
python figures/fig02/make_chelsa.py --data-dir data --out-dir outputs

python figures/fig03/make.py --data-dir data --out-dir outputs
python figures/fig03/make_topo_tri.py --data-dir data --out-dir outputs

python figures/fig04/make.py --data-dir data --out-dir outputs

python figures/fig05/make.py --data-dir data --out-dir outputs
```




