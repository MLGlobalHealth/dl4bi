# Deep Learning for Bayesian Inference (dl4bi)

This repository contains code for two papers accepted to **AISTATS 2026**:

- **DeepRV** — [arxiv.org/abs/2503.21473](https://arxiv.org/abs/2503.21473)
  Reproduction code: [benchmarks/vae/](benchmarks/vae/)

- **BSA-TNP** — [arxiv.org/abs/2506.09163](https://arxiv.org/abs/2506.09163)
  Reproduction code: [benchmarks/meta_learning/](benchmarks/meta_learning/)

## Install
1. Install [jax](https://jax.readthedocs.io/en/latest/installation.html)
2. Install [numpyro](https://num.pyro.ai/en/stable/getting_started.html#installation)
3. Install the `dl4bi` package from git:
```bash
pip install -U --force-reinstall git+ssh://git@github.com/MLGlobalHealth/dl4bi.git
```

## View Documentation (Locally)
```bash
pip install pdoc
git clone git@github.com:MLGlobalHealth/dl4bi.git
cd dl4bi
pdoc --docformat google --math dl4bi
```
Example scripts can be found [here](https://github.com/MLGlobalHealth/dl4bi/tree/main/benchmarks).

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