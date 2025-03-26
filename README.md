# DeepRV

DeepRV is a VAE-decoder-only model designed to generate pre-trained priors for geospatial data and perform inference on geodata. This code base is part of the not yet published Deep Learning for Bayesian Inference Repository (dl4bi).

## Development Setup
- Install Python 3.12 with `pyenv`:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.12: `pyenv install 3.12`
- Create a virtualenv called `dl4bi-dev` using Python 3.12: `pyenv virtualenv 3.12 dl4bi-dev`
- Clone the repository and `cd` into it: `git clone git@github.com:MLGlobalHealth/dl4bi.git && cd dl4bi`
- Inside the `dl4bi` repository, tell `pyenv` to use the `dl4bi-dev` virtualenv: `pyenv local dl4bi-dev`
    - `pyenv local dl4bi-dev` creates a `.python-version` file that tells `pyenv`
        to automatically activate the `dl4bi-dev` virtualenv whenever you are
        working in the `dl4bi` repository, so all `python` and `pip` commands will
        execute within the `dl4bi-dev` virtualenv
- Inside the `dl4bi` directory, install the package to the `dl4bi-dev` virtualenv: `pip install -e .`
    - Installing this package locally means it is installed "live", i.e. it
        immediately reflects any changes you make (this only needs to be done
        once)
- Install additional dependencies for the pyenv generated in the previous step:
    ```bash
    pip install numpyro seaborn geopandas
    ```
- In case of GPU usage, install the JAX GPU library (e.g., for CUDA 12): 
    ```bash
    pip install -U "jax[cuda12]"
    ```

## Main Logic
- DeepRV enables training the model to replace spatial priors for data generation or inference in MCMC inference. The workflow involves:
    1. Training the decoder-only network to simulate spatial priors for specific geospatial inputs.
    2. Using the trained network to generate data or as a surrogate model during inference.

- Input requirements:
    - For inference on real data: Provide a GeoPandas-readable map path with a `data` column.
    - For data simulation: Provide a GeoPandas-readable map without a `data` column, or use a 1d, 2d grid.

## Training DeepRV Priors
- The user has the following choices:
    - **Spatial prior class**:  Available classes are the GP kernels: `rbf`, `matern_1_2`, `matern_3_2`, `matern_5_2`, `periodic`. And the graph-based CAR model `car`.
    - **Priors for hyperparameters**: Specify priors for heperparameters of the chosen spatial prior, which must be distributions supported by NumPyro. Refer to the [NumPyro documentation](https://num.pyro.ai/en/stable/distributions.html). For a documented example, see `benchmarks/vae/configs/inference_model/poisson.yaml`.
    - **Map choice**: The user needs to provide a GeoPandas-readable map to the path `data.map_path`.
    - **Grid**: In case the user doesn't provide a map, they can set `data=1d` or `data=2d` to work on a simple grid (CAR model is not supported for this case).
    - **Experiment name**: The user needs to provide a name to save the experiment under `exp_name=experiment_name`.

- Running a training process with map data:
    ```bash
    python benchmarks/vae/train.py exp_name=experiment_name data.map_path=user_path/map_path \
        [inference_model.spatial_prior.func=matern_3_2 (defaults to rbf)] \
        [seed=7 (defaults to 0)] \
        [wandb=False (defaults to True)]
    ```
    - The default training prior is the RBF GP kernel. To use a custom prior, either replace `inference_model.spatial_prior.func` as in the example above, or add a new inference model file to `benchmarks/vae/configs/inference_model/<custom_prior>.yaml` and set your spatial prior and training hyperparameter priors there.
    - To log results in Weights & Biases, set `wandb=True` and optionally specify a project name using `project=<wandb_project>` (default: `DeepRV`).
    - Results and models are saved in:
      ```
      results/deep_RV/<experiment_name>/<spatial_prior>/<seed>/
      ```
    - Provide meaningful experiment names and update them for new maps to avoid overwriting results.
    - `data.map_path` can be a path to a directory or a single file, depending on the GeoPandas file type.

- Running a training process with 1d grid:
    ```bash
    python benchmarks/vae/train.py exp_name=experiment_name data=1d \
        [inference_model.spatial_prior.func=matern_3_2 (defaults to rbf)] \
        [seed=7 (defaults to 42)] \
        [wandb=False (defaults to True)]
    ```

## Run Inference with DeepRV
- For inference:
    - Use a GeoPandas dataframe with a `data` column to indicate real data, or simulate data with training priors (similar to the training process).
    - Customize NumPyro inference models or use provided ones like `poisson` and `binomial`. These models use spatial priors(`rbf`, `matern_1_2`, `matern_3_2`, `matern_5_2`, `periodic`, `car`), with pre-specified hyperprior distributions.
    - To add custom inference models, add a Numpyro model `inference_models/inference_models.py` and create a corresponding configuration file in `benchmarks/vae/configs/inference_model/`, with `model.func` equal to the new model's name. For documented example please see `benchmarks/vae/configs/inference_model/poisson.yaml`.

    **Note:** The Delta distribution in NumPyro does not propagate gradients properly and is unsuitable to run during inference.

- Running inference:
    ```bash
    python benchmarks/vae/infer.py exp_name=experiment_name data.map_path=user_path/map_path \
        inference_model=poisson (defaults to poisson) \
        [inference_model.spatial_prior.func=matern_3_2 (defaults to rbf)] \
        [seed=7 (defaults to 42)] \
        [wandb=False (defaults to True)]
    ```
    - Ensure `exp_name`, `map_path`, `spatial_prior.func`, and `seed` match the trained DeepRV model for proper loading.
    - The model is sensitive to the order of conditionals; use the same order as during training.
    - A `None` value in the `data` column indicates missing observations which are masked during inference (Behavior of feature is not fully tested).

- Inference results are stored in:
    ```
    results/deep_RV/<experiment_name>/<training_prior>/<seed>/
    ```
    - `<model_name>_hmc_pp.pkl`: Contains `s` (centroids), `f` (data samples), `obs_idxs` (observed location indices), and sampled variables.
    - `<model_name>_hmc_samples.pkl`: Contains all MCMC samples for all chains.
    - `<model_name>_hmc_summary.txt`: Summary statistics (mean, median, percentiles).
    - `<model_name>_mcmc.pkl`: MCMC object which can be read with arviz.

## Full Train and Inference Examples
### RBF Kernel with Poisson Inference
```bash
python benchmarks/vae/train.py exp_name=experiment_name data.map_path=user_path/map_path inference_model.spatial_prior.func=rbf seed=19 wandb=False && \
python benchmarks/vae/infer.py exp_name=experiment_name data.map_path=user_path/map_path inference_model=poisson inference_model.spatial_prior.func=rbf seed=19 wandb=False
```

### CAR model with Poisson Inference
```bash
python benchmarks/vae/train.py exp_name=experiment_name data.map_path=user_path/map_path inference_model.spatial_prior.func=car seed=19 wandb=False && \
python benchmarks/vae/infer.py exp_name=experiment_name data.map_path=user_path/map_path inference_model=poisson_car_gp seed=19 wandb=False
```

### Periodic Kernel with binomial Inference
```bash
python benchmarks/vae/train.py exp_name=experiment_name data.map_path=user_path/map_path inference_model=binomial inference_model.spatial_prior.func=periodic seed=19 wandb=False && \
python benchmarks/vae/infer.py exp_name=experiment_name data.map_path=user_path/map_path inference_model=binomial inference_model.spatial_prior.func=periodic seed=19 wandb=False
```

## Validating pre-trained prior with empirical bayes
```bash
python benchmarks/vae/train.py exp_name=experiment_name data.map_path=user_path/map_path inference_model=spatial_only inference_model.spatial_prior.func=matern_5_2 seed=19 wandb=False && \
python benchmarks/vae/empirical_bayes.py exp_name=experiment_name data.map_path=user_path/map_path inference_model=spatial_only inference_model.spatial_prior.func=matern_5_2 seed=19 wandb=False
```

## Reproducing Paper Results

To reproduce the results, first download the UK LTLA, as well as the England & Wales LAD maps using the following link:  

[Download Maps](https://drive.google.com/file/d/15DAfsL5dtO-lM3ef0saJ-73hMcQx6XVK/view?usp=sharing)  

Once downloaded and unzipped in the DeepRV directory, ensure the directory structure is as follows: 
```bash
benchmarks/vae/
├── maps/
│   ├── UK/
│   │   └── shapefile_data
│   ├── female_under_50_cancer_mortality_LAD_2023/
│   │   └── shapefile_data
│   ├── male_under_50_cancer_mortality_LAD_2023/
│   │   └── shapefile_data
│   ├── zwe2016phia_fixed.geojson
```

After downloading the maps, run:  

```bash
python benchmarks/vae/reproduce_paper/deep_rv_experiments.py --real_data --simulated_data && \
python benchmarks/vae/reproduce_paper/deep_rv_plots.py
```
### Workflow

#### Simulated Data  
- Run 5 seeds across 5 GP kernels.  
- Train DeepRV & PriorCVAE on the UK LTLA map.  
- Perform Empirical Bayes inference using DeepRV, PriorCVAE, and the actual process.  
- Run MCMC inference for a single seed per kernel.  

#### Real Data (Cancer Mortality Under 50)  
- Train DeepRV on the England & Wales LAD map.  
- Perform MCMC inference for Poisson & Binomial models.  
- Use Matern 3/2 and Matern 1/2 kernels, separately for males and females.  
- Train DeepRV on the Zimbabwe 2016 HIV prevalence map. 
- Perform MCMC inference with the Matern 1/2 for the Zimbabwe 2016 HIV prevalence. 

### Output  

Results are stored in: 
results/Experiment_name/spatial_prior/seed/model_name/inference_model_name/
They are also uploaded to wandb.

The final plots are stored in /results/final_plots/ directory
