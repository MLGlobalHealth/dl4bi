# Disease mapping

An example use of Neural Processes for conditioning an MCMC posterior for malaria mapping within the MGB framework.

## Requirements

Extra modules: `rpy2 pyDataverse geopandas arviz`

R + dependencies for data loading: `malariaAtlas`.

## Use

1. Specify the prior in `model.py`, make sure there are no clashes in the sample site names.
2. `train.py` to train a neural process.
3. `infer.py model="path to the NP checkpoint"` to run mapping. See `configs/inference.yaml` for more options.
