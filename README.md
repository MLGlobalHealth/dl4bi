# Deep Generative Emulation (dge)

## TODO
- [ ] Implement SPVAE
    - [ ] Keep randomness constant, change locations - smooth curve?
- [ ] Implement piVAE
- [ ] Implement Diffusion
- [X] Implement PriorVAE
- [X] Implement DeepChol
- [ ] Add optional installs for examples
- [ ] How do you properly use clu for metrics?

## Install
```bash
pip install git+ssh://git@github.com/danjenson/dge.git
```

## View Documentation (Locally)
```bash
pip install pdoc
git clone git@github.com:danjenson/dge.git
cd dge
pdoc --docformat google --math dge
```
Example scripts can be found [here](https://github.com/danjenson/dge/tree/main/examples).

## Development Setup
- Install Python 3.12:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.12: `pyenv install 3.12`
    - Make Python 3.12 your default: `pyenv global 3.12`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd dge && poetry install`
- Run tests: `poetry run pytest`
