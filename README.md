# Deep Generative Emulation (dge)

## TODO
- [ ] Implement ANP
    - [ ] Build training script for matern 3/2 and periodic dges
        - [X] Incorporate valid lens
        - [X] Average results over z samples per instance in loss
    - [ ] Test embeddings
        - [ ] Test optimized var in fourier embedding
        - [ ] Add parameterization for other embeddings, e.g. period in sinusoidal
    - [ ] Test attention
        - [ ] Use same self-attn network for local and global paths
        - [ ] Mixed-multihead attention, e.g. matern, rbf, periodic, learned
        - [ ] Implement performer linear attention
    - [ ] Test decoders
        - [ ] Use joint MLP network for decoders (z and f)
        - [ ] Build transformer decoder -> pe embedding + z samples as input
    - [ ] Test pooling mechanisms for zs
        - [ ] Try max-pooling, read set transformer paper
    - [ ] Build PriorCVAE version of model
        - [ ] Test KL loss term on global zs
    - [ ] Losses
        - [ ] Test importance sampling version of NPML loss
    - [ ] Test hierarchical version like wavenet
- [ ] Implement Diffusion
- [X] Implement piVAE
- [X] Implement PriorVAE
- [X] Implement DeepChol

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
- Setup env: `cd dge && poetry install [--with examples]`
- Run tests: `poetry run pytest`
