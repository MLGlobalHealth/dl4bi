# DeepRV

DeepRV is a VAE-decoder-only model designed to generate pre-trained priors for geospatial data and perform inference on geodata.

## Environment Setup
- Install Python 3.12 with `pyenv`:
    - Install `pyenv`: 
      ```bash
      curl https://pyenv.run | bash
      ```
    - Add the required lines to your `~/.bashrc` and reload:
      ```bash
      source ~/.bashrc
      ```
    - Install Python 3.12: 
      ```bash
      pyenv install 3.12
      ```
- Create a virtual environment called `DeepRV_env` using Python 3.12: 
    ```bash
    pyenv virtualenv 3.12 DeepRV_env
    ```
- Clone the repository and navigate to it: 
    ```bash
    git clone git@github.com:jhonathan-navott/DeepRV.git; cd DeepRV
    ```
- Set the `DeepRV_env` virtual environment for the repository:
    ```bash
    pyenv local DeepRV_env
    ```
- Install dependencies:
    ```bash
    pip install -r requirements.txt
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
    - For data simulation: Provide a GeoPandas-readable map without a `data` column.

## Training DeepRV Priors
- The user has the following choices:
    - **Spatial prior class**:  Available classes the GP kernels: `rbf`, `matern_1_2`, `matern_3_2`, `matern_5_2`, `periodic`. And the graph-based model `car`.
    - **Priors for hyperparameters**: Specify priors for heperparameters of the chosen kernel, which must be distributions supported by NumPyro. Refer to the [NumPyro documentation](https://num.pyro.ai/en/stable/distributions.html). For a documented example, see `configs/inference_model/poisson.yaml`.
    - **Map choice**: The user needs to provide a GeoPandas-readable map to the path `map_path`.
    - **Experiment name**: The user needs to provide a name to save the experiment under.

- Running a training process:
    ```bash
    python vae.py exp_name=experiment_name map_path=user_path/map_path \
        [inference_model.spatial_prior.func=matern_3_2 (defaults to rbf)] \
        [seed=7 (defaults to 42)] \
        [wandb=False (defaults to True)]
    ```
    - The default training prior is the RBF GP kernel. To use a custom prior, either replace `inference_model.spatial_prior.func` as in the example above, or add a new inference model file to `configs/inference_model/<custom_prior>.yaml` and set your spatial prior and training hyperparameter priors there.
    - To log results in Weights & Biases, set `wandb=True` and optionally specify a project name using `project=<wandb_project>` (default: `DeepRV`).
    - Results and models are saved in:
      ```
      results/deep_RV/<experiment_name>/<spatial_prior>/<seed>/
      ```
    - Provide meaningful experiment names and update them for new maps to avoid overwriting results.
    - `map_path` can be a apth to a directory or a single file, depending on the GeoPandas file type.

## Run Inference with DeepRV
- For inference:
    - Use a GeoPandas dataframe with a `data` column to indicate real data, or simulate data with training priors (similar to the training process).
    - Customize NumPyro inference models or use provided ones like `poisson`, `poisson_car` and `binomial`. These models use spatial priors(`rbf`, `matern_1_2`, `matern_3_2`, `matern_5_2`, `periodic`, `car`) and distributions such as Poisson or Binomial.
    - To add custom inference models, add a Numpyro model `inference_models/inference_models.py` and create a corresponding configuration file in `configs/inference_model/`, with `model.func` equal to the new model's name. For documented example please see `configs/inference_model/poisson_distance_based_gp.yaml`.

    **Note:** The Delta distribution in NumPyro does not propagate gradients properly and is unsuitable to run during inference.

- Running inference:
    ```bash
    python vae.py exp_name=experiment_name map_path=user_path/map_path \
        inference_model=poisson (defaults to poisson) \
        [inference_model.spatial_prior.func=matern_3_2 (defaults to rbf)] \
        [seed=7 (defaults to 42)] \
        [wandb=False (defaults to True)]
    ```
    - Ensure `exp_name`, `map_path`, `spatial_prior.func`, and `seed` match the trained DeepRV model for proper loading.
    - The model is sensitive to the order of conditionals; use the same order as during training.
    - A `None` value in the `data` column indicates missing observations (Behavior of feature is not fully tested).

- Inference results are stored in:
    ```
    results/deep_RV/<experiment_name>/<training_prior>/<seed>/
    ```
    - `<model_name>_hmc_pp.pkl`: Contains `s` (centroids), `f` (data samples), `obs_idxs` (observed location indices), and sampled variables.
    - `<model_name>_hmc_samples.pkl`: Contains all MCMC samples for all chains.
    - `<model_name>_hmc_summary.txt`: Summary statistics (mean, median, percentiles).

## Full Train and Inference Examples
### RBF Kernel with Poisson Inference
```bash
python vae.py exp_name=experiment_name map_path=user_path/map_path inference_model.spatial_prior.func=rbf seed=19 wandb=False && \
python infer.py exp_name=experiment_name map_path=user_path/map_path inference_model=poisson inference_model.spatial_prior.func=rbf seed=19 wandb=False
```

### CAR Kernel with Poisson Inference
```bash
python vae.py exp_name=experiment_name map_path=user_path/map_path inference_model.spatial_prior.func=car seed=19 wandb=False && \
python infer.py exp_name=experiment_name map_path=user_path/map_path inference_model=poisson_car_gp seed=19 wandb=False
```

## Reproducing Paper Results

To reproduce the results, first download the England, Wales, and Scotland LTLA, as well as the England and Wales LAD maps using the following link:  

[Download Maps](https://drive.google.com/file/d/1ktT6BDszL3X9B_e1gaZpQr7F6sJengow/view?usp=drive_link)  

Once downloaded and unzipped in the DeepRV directory, ensure the directory structure is as follows: 
```bash
DeepRV/
├── maps/
│   ├── England/
│   │   └── shapefile_data
│   ├── female_under_50_cancer_mortality_LAD_2023/
│   │   └── shapefile_data
│   ├── male_under_50_cancer_mortality_LAD_2023/
│   │   └── shapefile_data
```

After downloading the maps, run:  

```bash
python -m reproduce_paper/reproduce_experiments --real_data --simulated_data && \
python -m reproduce_paper/reproduce_plots
```
### Workflow

#### Simulated Data  
- Run 5 seeds across 5 GP kernels.  
- Train DeepRV & PriorCVAE on the UK LTLA map.  
- Perform Empirical Bayes inference using DeepRV, PriorCVAE, and an actual GP.  
- Run MCMC inference for a single seed per kernel.  

#### Real Data (Cancer Mortality Under 50)  
- Train DeepRV on the England & Wales LAD map.  
- Perform MCMC inference for Poisson & Binomial models.  
- Use Matern 3/2 and RBF kernels, separately for males and females.  

### Output  

Results are stored in: 
results/Experiment type/GP kernel used/seed/model name/inference model used/
They are also uploaded to wandb.

The final plots are stored in /results/final_plots/ directory
