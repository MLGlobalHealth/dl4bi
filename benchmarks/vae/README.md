# DeepRV

DeepRV is a decoder-only surrogate neural model designed to emulate Gaussian process priors and accelerate Bayesian inference for spatial and spatiotemporal data.

## Environment Setup

For environment setup and installation instructions, refer to the [main README](../../README.md).

Additionally, if not already installed through benchmarks, install the following dependencies:

```bash
uv pip install geopandas seaborn shapely
```

---

## Reproduce Paper

All experiments are runnable from the repository root. Results and processed outputs are saved under `results/<experiment>` or `outputs/<experiment>` as noted.


- **Benchmarking — Matérn-1/2**
  ```bash
  uv run python benchmarks/vae/benchmark_matern_1_2.py
  ```
  - Notes:
    - Ground truth data and INLA outputs are expected under `results/INLA_raw_batch_results`. Due to instability in regenerating the exact ground truth observations and masking (despite seed control), we **recommend downloading the precomputed ground truth data** from the following link: [Drive link](https://drive.google.com/file/d/1OXKwBUC1FYttw_NHK-jFK_zO4u421myB/view) and placing the extracted contents directly into `results/INLA_raw_batch_results`.
    - Once the precomputed data are in place, you may either: run `benchmarks/vae/inla.r`, which will **reuse the existing ground truth observation and will generate INLA outputs** in this directory, or run the Python benchmarking script directly, which will also **detect and use the precomputed results and INLA outputs**.
    - The R script `benchmarks/vae/inla.r` can optionally regenerate the ground truth data by setting the internal `generate` flag to `TRUE` and rerunning the script.
  This enables full end-to-end regeneration for benchmarking purposes, but **may result isn discrepancies in the generated ground truth** relative to the precomputed version used in the paper.
    - Long runtime (machine-dependent; expect days on some setups).
    - Results saved to `results/Benchmark_Matern_1_2_ls_<lengthscale>` for lengthscales 10, 30, 50.
    - Interruption-safe: intermediate results are reloaded automatically.
    - For a more computationally intensive INLA configuration, you can uncomment the lines related to mesh spacing (line 60) in `benchmarks/vae/inla.r`.
   This increases mesh resolution and computation time but did not yield significant improvements in our results. It remains available for full reproducibility.

- **Benchmarking — Matérn-3/2**
  ```bash
  uv run python benchmarks/vae/benchmark_matern_3_2.py
  ```
  - Notes:
    - Same structure and runtime considerations as Matérn-1/2, without INLA.
    - Results saved to `results/Benchmark_Matern_3_2_ls_<lengthscale>`.

- **Spatiotemporal (non-separable kernel)**
  ```bash
  uv run python benchmarks/vae/spatiotemporal_kernel.py
  ```
  - Notes:
    - Results saved under `results/spatiotemporal`.

- **Real-world LSOA / MSOA (London)**
  - Before running, you **must** download the required shapefiles from:
    [Google Drive link](https://drive.google.com/file/d/12oPJGONKqSRLXH9h49LujIDQyFT18_tb/view?usp=drive_link)
  - Unzip the downloaded archive directly into the `maps` directory using:
    ```bash
    unzip London_Education_Deprivation_Maps.zip -d benchmarks/vae/maps/
    ```
  - After extraction, the directory structure should be:
    ```
    <root_directory>/benchmarks/vae/
    ├── maps/
    │   ├── London_LSOA_education_deprivation_parsed_thrs_0/ ...
    │   └── London_MSOA_education_deprivation_parsed_thrs_0/ ...
    ```
  - Once the maps are in place, run:
    ```bash
    uv run python benchmarks/vae/london_lsoa.py
    ```
  - Notes:
    - Runs both MSOA and LSOA examples; the LSOA GP run is shortened as described in the paper.
    - Results will be saved under:
      - `results/London_MSOA_education_deprivation_parsed_thrs_0`
      - `results/London_LSOA_education_deprivation_parsed_thrs_0`


- **Multi-location experiment (Transformer training + inference)**
  ```bash
  uv run python benchmarks/vae/multi_locations.py
  ```
  - Notes:
    - Transformer training can be long (2M steps); results in `results/multi_locations`.

- **Ablation study**
  ```bash
  uv run python benchmarks/vae/ablation_test.py
  ```
  - Notes:
    - Aggregated ablation tables are saved at `results/ablation_tables`.
---

## Example Usage

A small example usage script:
```bash
uv run python benchmarks/vae/deep_rv_example.py
```

---

## Repository Structure

```
dl4bi/                   # Core model and training modules (Belongs to the parent repo DL4BI)
benchmarks/vae/          # Scripts for DeepRV's paper experiments and utilities
benchmarks/vae/utils/    # Experiment utilities and plotting functions
benchmarks/vae/maps/     # Shapefiles for London LSOA/MSOA analysis (download link in the reproduce paper section)
results/                 # Where results per experiment are kept
outputs/                 # Where processed plots and tables are saved (will be generated automatically)
README.md                # DL4BI parent repo's README
benchmarks/vae/README.md # DeepRV paper reproduction README
pyproject.toml           # Dependency and installation configuration
```
