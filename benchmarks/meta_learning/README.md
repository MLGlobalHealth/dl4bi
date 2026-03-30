# Meta-Learning Benchmarks

This directory contains the benchmark scripts used for the BSA-TNP paper, along
with a few extra experiments and plotting utilities.

## Cite

```bibtex
@inproceedings{jenson2026scalable,
  title = {Scalable Spatiotemporal Inference with Biased Scan Attention Transformer Neural Processes},
  author = {Jenson, Daniel and Navott, Jhonathan and Grynfelder, Piotr and Zhang, Mengyan and Sharma, Makkunda and Semenova, Elizaveta and Flaxman, Seth},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics},
  series = {Proceedings of Machine Learning Research},
  volume = {300},
  address = {Tangier, Morocco},
  year = {2026},
  publisher = {PMLR}
}
```

## Setup

From the repo root, create the environment and pick the JAX backend you want:

```bash
uv sync --extra cpu
# or: uv sync --extra cuda12
# or: uv sync --extra cuda13
```

Then run the benchmark commands from this directory:

```bash
cd benchmarks/meta_learning
```

Most scripts use Hydra config overrides and log to Weights & Biases by default.
For local runs without W&B, add `wandb=False`. Results are written under
`results/...`, and downloaded or extracted datasets are cached under `cache/...`.

## Reproduce The Paper

To run the full BSA-TNP paper sweep:

```bash
uv run python reproduce_paper.py --seed 88 --num_runs 5
```

This driver runs the paper benchmarks for:

- translation-invariant Gaussian processes
- multiscale 2D Gaussian processes
- rotated SO(3) Gaussian processes
- SIR forecasting
- ERA5 weather forecasting
- Beijing air quality forecasting
- Gneiting spatiotemporal Gaussian processes

For a quick smoke test of the same pipeline:

```bash
uv run python reproduce_paper.py --dry_run
```

The dry run disables W&B logging and reduces the number of training, validation,
and test steps.

## Run Individual Benchmarks

Synthetic benchmarks that work out of the box:

```bash
uv run python gp.py wandb=False data=2d kernel=rbf model=2d/bsa_tnp seed=42
uv run python multiscale_2d_gp.py wandb=False model=bsa_tnp seed=42
uv run python sir.py wandb=False model=bsa_tnp data=spatial_64x64 seed=42
uv run python gneiting_gp.py wandb=False model=bsa_tnp data=16x16 seed=42
uv run python generic_spatial.py wandb=False model=bsa_tnp seed=42
```

Real-data benchmarks:

```bash
uv run python era5.py wandb=False model=bsa_tnp data=reduced seed=42
uv run python beijing_air_quality.py wandb=False model=bsa_tnp seed=42
uv run python household_electric.py wandb=False model=bsa_tnp seed=42
uv run python dengue.py wandb=False model=bsa_tnp seed=42
uv run python heaton.py wandb=False model=bsa_tnp test=sim seed=42
uv run python mnist.py wandb=False seed=42
uv run python cifar_10.py wandb=False seed=42
uv run python celeba.py wandb=False seed=42
```

Useful override patterns (for some scripts):

- swap the model with `model=...`
- change the synthetic dataset with `data=...`
- disable training and evaluate a saved checkpoint with `evaluate_only=True`
- change the RNG seed with `seed=...`

## Data Notes

- `era5.py` expects ERA5 NetCDF files under
  `cache/era5/{central_europe,northern_europe,western_europe}/2019_*.nc`.
  The script includes a `download_if_not_cached()` helper, but it requires a
  configured CDS API key.
- `beijing_air_quality.py` expects the UCI Beijing Multi-Site Air Quality CSV
  files under `cache/beijing_air_quality/`. If they are missing, the script
  prints the download URL and the expected extraction layout.
- `household_electric.py` can populate its cache automatically from
  `ucimlrepo`.
- `heaton.py` expects `cache/heaton/sim.csv` or `cache/heaton/sat.csv`. The
  original Heaton data ships as `.RData`, so you need to convert it to CSV
  before running the benchmark.
- `celeba.py` and `cifar_10.py` prepare their own cached datasets under
  `cache/`.

## Utilities

The plotting and analysis helpers in this directory are intended to run after
training:

- `plot_gp_comparisons.py`
- `plot_sir_comparisons.py`
- `plot_era5_comparisons.py`
- `paired_t_test.py`
- `wandb_csv_to_latex.py`
- `apply_pfn.py`
- `generic_spatial_pfn.py`
