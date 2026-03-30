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
Example benchmarks can be found [here](https://github.com/MLGlobalHealth/dl4bi/tree/main/benchmarks).

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
1. Bump the package version:
```bash
uv version --bump patch --frozen
git tag -a <version> -m "<message>"
git commit [--no-verify] -am "<version> <message>"
git push origin main --follow-tags
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
