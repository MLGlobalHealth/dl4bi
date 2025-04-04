# Disease mapping

An example use of Neural Processes for conditioning an MCMC posterior for malaria mapping within the MGB framework.

## Requirements

Extra Python modules: `rpy2 pyDataverse geopandas arviz`

*R* is used for some data loading, and requires the package `malariaAtlas` to be installed.
Should `malariaAtlas` installation fail, first install `Rcpp`, 
and make sure R is configured to compile with C++17.

## Use

1. Specify the prior in `model.py`, make sure there are no clashes in the sample site names.
2. `train.py` to train a neural process.
3. `infer.py model="path to the NP checkpoint"` or `path=gp` to run mapping. 
    See `configs/inference.yaml` for more options.
