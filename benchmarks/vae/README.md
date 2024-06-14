# Testing PriorCVAE and DeepChol

`python 1D_GP.py +kernel=rbf +model=prior_cvae +seed=0 [+wandb=True]`

To use the model for inference, e.g. in NumPyro, this call will be useful:
`decode = jit(lambda z, var, ls: model.apply({"params": params}, z, var, ls, method="decode"))`


# Testing PiVAE

`python pi_vae_gp.py`
