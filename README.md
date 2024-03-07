# Deep Generative Emulation (dge)

## TODO
[ ] Implement PriorVAE
[ ] Implement piVAE
[X] Implement DeepChol
[ ] Implement Diffusion
[ ] Add optional installs for examples
[ ] How do you properly use clu for metrics?

## Install
```bash
pip install git+ssh://git@github.com/MLGlobalHealth/sps.git
```

## View Documentation
```bash
pip install pdoc
pdoc --docformat google --math sps
```
Example scripts can be found [here](https://github.com/danjenson/dge/tree/main/examples).

## Setup
- Install Python 3.12:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.12: `pyenv install 3.12`
    - Make Python 3.12 your default: `pyenv global 3.12`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd sps && poetry install`
- Run tests: `poetry run pytest`
