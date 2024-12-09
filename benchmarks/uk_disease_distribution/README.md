
# UK LTAs Disease Distribution

## Development Setup
1. Follow the installation setup in the main README file.
2. In the new env created perform `pip install seaborn numpyro geopandas`
3. Download all maps from https://drive.google.com/drive/folders/1n5cUL2FQrB9bY1RgvG4c1vmD62gi5IOT?usp=sharing
    and unzip them in benchmarks/uk_disease_distribution/maps. This should be performed automatically when running,
    however, sometimes downloading them directly invokes a google drive bug check prompt which breaks the download.


## Train VAE models
`python benchmarks/uk_disease_distribution/vae.py model=deep_cholesky seed=7 [wandb=False] [+name="Experiment name"]`

## Run Inference Models
### Run inference with Baseline GP or Graph models model
(TODO): Naming change
`python benchmarks/uk_disease_distribution/infer_gp.py infer.run_gp_baseline=True seed=7 [wandb=False] [+name="Experiment name"]`

### Run inference with surrogate model
`python benchmarks/uk_disease_distribution/infer_gp.py model=deep_cholesky seed=7 [wandb=False] [+name="Experiment name"]`

### partial information
The user can add infer.num_context_points=<some int> for any inference run to let the inference model observe only partial info.

## Run meta regression on map (currently unsupported)
`python benchmarks/uk_disease_distribution/meta_regression_gp.py model=tnp_kr seed=7 [wandb=False] [+name="Experiment name"]`