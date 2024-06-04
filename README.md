# Deep Generative Emulation (dge)

## TODO
- [ ] Meta Regression
    - [ ] Add diagnostic plot comparison
        - [ ] Diagnose TNP-ND uncertainty?
        - [ ] Diagnose NP, ANP large losses
    - [ ] Test self-attn decoder to test locs
    - [ ] Add HMC baseline
    - [ ] Add BNP/BANP
    - [ ] Add ConvNP
    - [ ] Test larger jumps in residual connections
    - [ ] Add s directly to GFF PE embedding
- [ ] Clean up VAE benchmarks


## Install
1. Install [jax](https://jax.readthedocs.io/en/latest/installation.html)
2. Install [numpyro](https://num.pyro.ai/en/stable/getting_started.html)
3. Install the `dge` package from git:
```bash
pip install git+ssh://git@github.com/MLGlobalHealth/dge.git
```

## View Documentation (Locally)
```bash
pip install pdoc
git clone git@github.com:MLGlobalHealth/dge.git
cd dge
pdoc --docformat google --math dge
```
Example scripts can be found [here](https://github.com/MLGlobalHealth/dge/tree/main/examples).

## Development Setup
- Install Python 3.12:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.12: `pyenv install 3.12`
    - Make Python 3.12 your default: `pyenv global 3.12`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd dge && poetry install`
- Run tests: `poetry run pytest`
