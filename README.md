# Deep Learning for Bayesian Inference (dl4bi)

## Install
Install with the appropriate command. If JAX isn't installed already, we recommend using one of the `dl4bi[<jax-version>]` installs.
```bash
pip install dl4bi # dl4bi
pip install dl4bi[cpu] # dl4bi + jax for CPU
pip install dl4bi[cuda12] # dl4bi + jax for CUDA-12
pip install dl4bi[cuda13] # dl4bi + jax for CUDA-13
```

## View Documentation (Locally)
```bash
git clone git@github.com:MLGlobalHealth/dl4bi.git
cd dl4bi
uv run --with pdoc pdoc --docformat google --math dl4bi
```

## Cite
If you're using this package or some of its code, please cite the relevant paper(s):
```bibtex
@inproceedings{jenson2026scalable,
  title = {Scalable Spatiotemporal Inference with Biased Scan Attention Transformer Neural Processes},
  author = {Jenson, Daniel and Navott, Jhonathan and Grynfelder, Piotr and Zhang, Mengyan and Sharma, Makkunda and Semenova, Elizaveta and Flaxman, Seth},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics},
  series = {Proceedings of Machine Learning Research},
  volume = {300},
  address = {Tangier, Morocco},
  year = {2026},
  publisher = {PMLR}
}

@inproceedings{navott2026deeprv,
  title = {{DeepRV}: Accelerating Spatiotemporal Inference with Pre-trained Neural Priors},
  author = {Navott, Jhonathan and Jenson, Daniel and Flaxman, Seth and Semenova, Elizaveta},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics},
  series = {Proceedings of Machine Learning Research},
  volume = {300},
  address = {Tangier, Morocco},
  year = {2026},
  publisher = {PMLR}
}
```
Benchmarks are available for BSA-TNP [here](https://github.com/MLGlobalHealth/dl4bi/tree/main/benchmarks/meta_learning) and for DeepRV [here](https://github.com/MLGlobalHealth/dl4bi/tree/main/benchmarks/vae).

## Development Setup
- Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Clone the repository and `cd` into it: `git clone git@github.com:MLGlobalHealth/dl4bi.git && cd dl4bi`
- Install the latest Python with `uv`: `uv python install`
- Sync the project environment:
    - CPU JAX: `uv sync --extra cpu`
    - CUDA 12 JAX: `uv sync --extra cuda12`
    - CUDA 13 JAX: `uv sync --extra cuda13`
- `uv sync` creates `.venv`, installs the project in editable mode, includes the default `dev` dependency group, and picks a Python interpreter compatible with the project's `requires-python`
- Before making changes, install the shared development hooks: `uv run pre-commit install --install-hooks`
- Verify the hook setup once per clone with: `uv run pre-commit run --all-files`
- Keep the hooks installed for local development; commits on `main` run `pytest -q tests` through the shared `pre-commit` setup
- Run project commands through `uv`, e.g. `uv run pytest` or `uv run python gp.py`
- If you want to activate the virtualenv directly, use `source .venv/bin/activate`

## Build and Publish to PyPI
Create a local `.env` file with the publish tokens:
```env
TEST_PYPI_TOKEN=pypi-...
PYPI_TOKEN=pypi-...
```

Run the release helper from a clean `main` checkout:
```bash
uv run python scripts/release.py .env "AISTATS 2026"
```

The helper bumps the patch version, commits and tags `v<version> <message>`,
rebuilds `dist/`, publishes to TestPyPI and PyPI, pushes `main` and the tag,
and smoke-tests the published install targets.
