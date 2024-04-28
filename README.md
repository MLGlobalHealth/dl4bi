# Deep Generative Emulation (dge)

## TODO
- [ ] Implement Diffusion
- [X] Implement ANP
    - [X] Fix identity embedding
    - [X] Compare embeddings: identity, sinusidal, nerf, fourier
    - [ ] MLP after cross-attn
    - [ ] Use joint MLP network for mu and var (z and f)
    - [ ] Test optimized var in fourier embedding
    - [ ] Add optimized vars to other embeddings
    - [ ] Use same self-attn network for local and global paths
    - [ ] Incorporate valid lens
    - [ ] Try different pooling mechanism for global zs, e.g. max pooling
    - [ ] Test KL loss term on global zs
    - [ ] Test generalization across multiple sinusoidal tasks
- [X] Implement piVAE
- [X] Implement PriorVAE
- [X] Implement DeepChol

## Install
1. Install [jax](https://jax.readthedocs.io/en/latest/installation.html)
2. Install the `dge` package from git:
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
