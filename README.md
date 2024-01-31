# Deep Generative Emulation (dge)

## Setup
- Install Python 3.11:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.11: `pyenv install 3.11`
    - Make Python 3.11 your default: `pyenv global 3.11`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd sps && poetry install`
- Run tests: `poetry run pytest`
