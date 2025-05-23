# Disease mapping

An example use of Neural Processes for conditioning an MCMC posterior for malaria mapping within the MGB framework.

## Requirements

Extra Python modules: `geopandas arviz rasterio`

## Use

1. Specify the prior in `model.py`, make sure there are no clashes in the sample site names.
2. `train.py` to train a neural process.
3. `mcmc.py` to run MCMC on the data to get samples of the spatial effect.
4. `predict.py model="path to the NP checkpoint"` or `path=gp` to run mapping.
    See `configs/inference.yaml` for more options.

The results are stored in `results`,
with intermediate MCMC results stored in `results/{hash_config(cfg.mcmc)}/mcmc.pickle`.
