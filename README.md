# Deep Learning for Bayesian Inference (dl4bi)

## Install
1. Install [jax](https://jax.readthedocs.io/en/latest/installation.html)
2. Install the `dl4bi` package from git with `uv`:
```bash
uv pip install -U --force-reinstall git+ssh://git@github.com/MLGlobalHealth/dl4bi.git
```

## View Documentation (Locally)
```bash
git clone git@github.com:MLGlobalHealth/dl4bi.git
cd dl4bi
uv run --with pdoc pdoc --docformat google --math dl4bi
```
Example scripts can be found [here](https://github.com/MLGlobalHealth/dl4bi/tree/main/benchmarks).

## Development Setup
- Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Clone the repository and `cd` into it: `git clone git@github.com:MLGlobalHealth/dl4bi.git && cd dl4bi`
- Install the latest Python with `uv`: `uv python install`
- Sync the project environment:
    - CPU JAX: `uv sync --extra cpu`
    - CUDA 12 JAX: `uv sync --extra cuda12`
    - CUDA 13 JAX: `uv sync --extra cuda13`
- `uv sync` creates `.venv`, installs the project in editable mode, includes the default `dev` dependency group, and picks a Python interpreter compatible with the project's `requires-python`
- Run project commands through `uv`, e.g. `uv run pytest`
- If you want to activate the virtualenv directly, use `source .venv/bin/activate`

## Build and Publish to PyPI
- PyPI receives a single distribution, `dl4bi`; the CUDA variants are published as extras on that distribution, not as separate wheels
- With the current metadata, the published extra install targets are `dl4bi[cpu]`, `dl4bi[cuda12]`, and `dl4bi[cuda13]`
- Extras are always opt-in; Python packaging does not have a "default extra", so `dl4bi` is only the CPU/default install if CPU JAX is included in the base `[project.dependencies]`
- With the current metadata, plain `dl4bi` does not install JAX; users must either install JAX separately or select one of the JAX extras

1. Bump the package version:
```bash
uv version --bump patch --frozen
```

2. Build the source distribution and wheel:
```bash
uv build --no-sources
```

3. Publish to TestPyPI first:
```bash
UV_PUBLISH_TOKEN=$TEST_PYPI_TOKEN uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --check-url https://test.pypi.org/simple/
```

4. After validating the release, publish the same artifacts to PyPI:
```bash
UV_PUBLISH_TOKEN=$PYPI_TOKEN uv publish
```

5. Smoke-test the published install targets in fresh environments:
```bash
uv run --isolated --with "dl4bi==<version>" --no-project -- python -c "import dl4bi"
uv run --isolated --with "dl4bi[cpu]==<version>" --no-project -- python -c "import dl4bi"
uv run --isolated --with "dl4bi[cuda12]==<version>" --no-project -- python -c "import dl4bi"
uv run --isolated --with "dl4bi[cuda13]==<version>" --no-project -- python -c "import dl4bi"
```
