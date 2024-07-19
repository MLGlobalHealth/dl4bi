# Deep Stochastic Processes (dsp)

## Install
1. Install [jax](https://jax.readthedocs.io/en/latest/installation.html)
2. Install [numpyro](https://num.pyro.ai/en/stable/getting_started.html)
3. Install the `dsp` package from git:
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
using the `min_std` or `bound_std` arguments because the optimizer can learn to
"hack" rewards by arbitrarily decreasing standard deviation at observed context
points (where the standard deviation is theoretically 0) creating arbitrarily
large negative log likelihood scores, which destabilizes training. The only
exception to this is when using a model with `MultiheadFastAttention` as the
softmax approximation provides a measure of regularization which often prevents
such degeneracy.

## Development Setup
- Install Python 3.12:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.12: `pyenv install 3.12`
    - Make Python 3.12 your default: `pyenv global 3.12`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd dsp && poetry install`
- Run tests: `poetry run pytest`
