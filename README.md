# Deep Generative Emulation (dge)

## TODO
- [ ] Implement Diffusion
- [X] Implement SPVAE
    - [ ] Keep randomness constant, change locations - smooth curve?
    - [ ] Use additive positional encodings
    - [ ] Can we do this only with Attention and relative embeddings?
        - https://arxiv.org/abs/1803.02155
        - https://arxiv.org/abs/2006.03654
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
