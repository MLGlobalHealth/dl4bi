# Deep Stochastic Processes (dsp)

## Install
1. Install [jax](https://jax.readthedocs.io/en/latest/installation.html)
2. Install the `dsp` package from git:
```bash
pip install -U --force-reinstall git+ssh://git@github.com/MLGlobalHealth/dsp.git
```

## View Documentation (Locally)
```bash
pip install pdoc
git clone git@github.com:MLGlobalHealth/dsp.git
cd dsp
pdoc --docformat google --math dsp
```
Example scripts can be found [here](https://github.com/MLGlobalHealth/dsp/tree/main/benchmarks).

## Warnings & Caveats
- When using high precision models, i.e. transformer-based models, we recommend
using the `min_std` argument because the optimizer can learn to "hack" rewards
by arbitrarily decreasing standard deviation at observed context points (where
the standard deviation is theoretically 0) creating arbitrarily large negative
log likelihood scores, which destabilizes training. The only exception to this
is when using a model with `MultiheadFastAttention` as the softmax approximation
provides a measure of regularization which often prevents such degeneracy.

## Development Setup
- Install Python 3.12 with `pyenv`:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.12: `pyenv install 3.12`
- Create a virtualenv called `dsp-dev` using Python 3.12: `pyenv virtualenv 3.12 dsp-dev`
- Clone the repository and `cd` into it: `git clone git@github.com:MLGlobalHealth/dsp.git && cd dsp`
- Inside the `dsp` repository, tell `pyenv` to use the `dsp-dev` virtualenv: `pyenv local dsp-dev`
    - `pyenv local dsp-dev` creates a `.python-version` file that tells `pyenv`
        to automatically activate the `dsp-dev` virtualenv whenever you are
        working in the `dsp` repository, so all `python` and `pip` commands will
        execute within the `dsp-dev` virtualenv
- Inside the `dsp` directory, install the package to the `dsp-dev` virtualenv: `pip install -e .`
    - Installing this package locally means it is installed "live", i.e. it
        immediately reflects any changes you make (this only needs to be done
        once)
