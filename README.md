# Reproducible figures for:
## *When and where higher-resolution climate data improve impact model performance*

This repository contains the **code and data needed to reproduce the figures** in the paper:

> **When and where higher-resolution climate data improve impact model performance**

Each figure is generated from a two-step workflow:
1) **prep**: convert raw inputs into processed tables used for plotting  
2) **make**: generate the final figure files

## What’s in this repository

- `figures/fig01/ ... fig05/`  
  One folder per figure, each containing:
  - `prep.py` (optional): prepares processed inputs
  - `make.py`: generates the figure(s)
  - `config.yml`: figure-specific configuration (filenames, settings)
  - `README.md`: figure-specific instructions

- `data/figXX/raw/`  
  Published raw inputs required to reproduce the corresponding figure.

- `data/figXX/processed/`  
  Intermediate products created by `prep.py` (may be regenerated).

- `outputs/figXX/`  
  Generated figures (ignored by git by default).

- `src/common/`  
  Shared helpers (paths, style, metrics) used across figures.

## Data download (required)

Before you can run this repository, the following data needs to be downloaded from **xxx** and extracted into the **yyy** folder.

> **Note:** The expected layout after extraction should match the `data/figXX/raw/` structure described above.

## Requirements

Python is used throughout. The recommended setup is via Conda.

### Install the environment

```bash
conda env create -f environment.yml
conda activate paper-figures
