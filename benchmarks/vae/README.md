# DeepRV

DeepRV is a decoder-only surrogate neural model designed to emulate Gaussian process priors and accelerate Bayesian inference for spatial and spatiotemporal data.  

## Environment Setup

Follow the steps below to configure a clean Python 3.12 environment and install all required dependencies.

### 1. Create and Activate Environment

Choose one option (optional):

#### • Pyenv
```bash
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.12
pyenv virtualenv 3.12 DeepRV
pyenv local DeepRV
```

#### • Conda
```bash
conda create -n DeepRV python=3.12
conda activate DeepRV
```

### 2. Install Dependencies

Ensure you are in the repository root where `pyproject.toml` is located.

#### • GPU Installation
For NVIDIA GPUs (e.g., RTX 5000 Ada, CUDA 12.8):
```bash
pip install jax[cuda12]==0.6.2
pip install -e .
```

#### • CPU Installation
For CPU-only machines:
```bash
pip install jax==0.6.2 jaxlib==0.6.2
pip install -e .
```

### 3. Verify Installation

Run the following snippet to confirm JAX detects your hardware backend:

```bash
python - <<'PY'
import jax
print(f"JAX version: {jax.__version__}")
print(f"Detected devices: {jax.devices()}")
# Expected Output:
# GPU → [GpuDevice(id=0, process_index=0)]
# CPU → [CpuDevice(id=0)]
PY
```

---

## Reproduce Paper

All experiments are runnable from the repository root. Results and processed outputs are saved under `results/<experiment>` or `outputs/<experiment>` as noted.


- **Benchmarking — Matérn-1/2**
  ```bash
  python benchmarks/vae/benchmark_matern_1_2.py
  ```
  - Notes:
    - Ground truth data and INLA outputs are precomputed at `results/INLA_raw_batch_results` by the `benchmarks/vae/inla.r` script. The script can be used to re-generate them by running `benchmarks/vae/inla.r` (requires R packages listed in that file).
    - Long runtime (machine-dependent; expect days on some setups).
    - Results saved to `results/Benchmark_Matern_1_2_ls_<lengthscale>` for lengthscales 10, 30, 50.
    - Interruption-safe: intermediate results are reloaded automatically.
    - For a more computationally intensive INLA configuration, you can uncomment the lines related to mesh spacing (line 60) in `benchmarks/vae/inla.r`.  
   This increases mesh resolution and computation time but did not yield significant improvements in our results. It remains available for full reproducibility.

- **Benchmarking — Matérn-3/2**
  ```bash
  python benchmarks/vae/benchmark_matern_3_2.py
  ```
  - Notes:
    - Same structure and runtime considerations as Matérn-1/2, without INLA.
    - Results saved to `results/Benchmark_Matern_3_2_ls_<lengthscale>`.

- **Spatiotemporal (non-separable kernel)**
  ```bash
  python benchmarks/vae/spatiotemporal_kernel.py
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
    python benchmarks/vae/london_lsoa.py
    ```
  - Notes:
    - Runs both MSOA and LSOA examples; the LSOA GP run is shortened as described in the paper.
    - Results will be saved under:
      - `results/London_MSOA_education_deprivation_parsed_thrs_0`
      - `results/London_LSOA_education_deprivation_parsed_thrs_0`


- **Multi-location experiment (Transformer training + inference)**
  ```bash
  python benchmarks/vae/multi_locations.py
  ```
  - Notes:
    - Transformer training can be long (2M steps); results in `results/multi_locations`.

- **Ablation study**
  ```bash
  python benchmarks/vae/ablation_test.py
  ```
  - Notes:
    - Aggregated ablation tables are saved at `results/ablation_tables`.
---

## Example Usage

A small example usage script:
```bash
python deep_rv_usage_example.py
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