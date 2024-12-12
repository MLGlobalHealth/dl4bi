# UK LTAs Disease Distribution

## Development Setup
1. Follow the installation setup in the main README.
2. In the new environment, install required libraries:
   ```bash
   pip install seaborn numpyro geopandas
   ```
3. Download all maps from [Google Drive](https://drive.google.com/drive/folders/1n5cUL2FQrB9bY1RgvG4c1vmD62gi5IOT?usp=sharing) and unzip them into:
   ```
   benchmarks/uk_disease_distribution/maps
   ```
   > **Note:** The script attempts to download maps automatically. However, a Google Drive bug may interrupt the download.
4. To add a new map:
   - Save a GeoPandas instance under:
     ```
     benchmarks/uk_disease_distribution/maps/<map_name>/raw
     ```
     Ensure the GeoPandas object has a `geometry` attribute containing polygon data for each location.
   - Save any preprocessed data under:
     ```
     benchmarks/uk_disease_distribution/maps/<map_name>/processed
     ```
     For example preprocessing details, refer to the **Sampling Policies** in the [Run GP Meta Regression on Map](#run-gp-meta-regression-on-map) section.

## Train VAE Models
Run the following commands to train VAE models:
### Run VAE on GP
```bash
python benchmarks/uk_disease_distribution/vae.py model=deep_cholesky seed=7 [wandb=False] [+name="Experiment name"]
```
### Run VAE on graph models
```bash
python benchmarks/uk_disease_distribution/vae.py model=deep_cholesky seed=7 is_gp=False [wandb=False] [+name="Experiment name"]
```
## Run Inference Models
### Baseline GP
> **Note:** Graph models are not supported for inference currently.
```bash
python benchmarks/uk_disease_distribution/infer.py infer.run_gp_baseline=True seed=7 [wandb=False] [+name="Experiment name"]
```

### Surrogate Model
```bash
python benchmarks/uk_disease_distribution/infer.py model=deep_cholesky seed=7 [wandb=False] [+name="Experiment name"]
```

### Partial Information
Add the following to any inference command to restrict the model to partial observations:
```bash
infer.num_context_points=<some int>
```

## Run GP Meta Regression on Map
Run the following command:
```bash
python benchmarks/uk_disease_distribution/meta_regression_gp.py model=tnp_kr seed=7 [wandb=False] [+name="Experiment name"]
```

### Sampling Policies
- **Default:** Randomly sampled centroids for all counties in the map.
- **Grid:** Random points between the min and max x/y values (points may fall outside the map).
- **In-Map:** Random points within the map. To use this option:
  1. Save a file under:
     ```
     benchmarks/uk_disease_distribution/maps/<map_name>/processed/data.npz
     ```
     Include a field `all_grid_points` which has the all the grid points from which you want the `in_map` option to sample from. You may use the `create_uniform_grid` function in `map_utils.py` to generate this.
  2. Usage - add flag:
     ```bash
     data.sampling_policy=in_map
     ```
     
