# pipetransport

Scientific Python package for timeseries analysis of water quality in branched drinking water distribution networks.

## Commands

```bash
# Setup (fresh environment, separate from user's .venv)
# Windows: replace `env VAR=val cmd` with `set "VAR=val" && cmd`
rm -rf .venv-claude
env UV_PROJECT_ENVIRONMENT=.venv-claude uv sync --all-extras -q
git config core.hooksPath .githooks               # Enable pre-commit hook

# Testing (run before committing)
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q pytest tests/src -n auto                   # Unit tests
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q pytest tests/examples -n auto              # Example notebooks
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q pytest tests/docs -n auto                  # Documentation code snippets

# Linting (run before committing)
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q ruff format .                              # Format code
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q ruff check --fix .                         # Lint and auto-fix
npx prettier --check "**/*.{yaml,yml,md}"         # Format markdown/yaml

# Type checking (run before committing)
uv tool update -q ty & uv tool run -q ty check .

# Documentation
uv tool run -q --from sphinx --with-editable ".[docs]" sphinx-build -j auto -b linkcheck docs/source docs/build/linkcheck
rm -rf docs/build && uv tool run -q --from sphinx --with-editable ".[docs]" sphinx-build -j 1 -b html docs/source docs/build/html
```

## CI/CD

All checks must pass before merging. Pipeline tests on Python 3.12 (minimum deps) and 3.14 (latest deps). See `.github/workflows/` for details.

## Project Layout

- `src/pipetransport/` -- Package source code
- `tests/src/` -- Unit tests (one test file per module)
- `tests/examples/` -- Jupyter notebook execution tests
- `tests/docs/` -- Documentation code snippet tests
- `examples/` -- Example Jupyter notebooks
- `docs/source/` -- Sphinx documentation source

## Philosophy

You are a quality gatekeeper, not just an implementer. Before writing code:

- **Understand the physics.** This is a scientific package -- correctness of physical equations, units, and boundary conditions matters more than code elegance. If unsure about the physics, ask.
- **Check for dead code.** After every change, verify no unused imports, functions, or variables remain. Remove them.
- **Keep API and docs consistent.** When changing a public function signature, update its docstring, any cross-references, and affected example notebooks.
- **Re-read the request.** Before finishing, re-read the original question to verify you actually answered it.

## Code Style

- **Docstrings**: NumPy style.
- **Line length**: 120 characters.
- **Type hints**: Required for all public functions. Use `npt.ArrayLike` for array inputs, `npt.NDArray[np.floating]` for array outputs, `pd.DatetimeIndex` for time edges. Use built-in Python generics (`list`, `tuple`, `dict`, `X | None`) -- NEVER import from `typing`.
- **Vectorization**: ALWAYS prefer vectorized NumPy/SciPy/pandas operations over Python for-loops. The one accepted loop is over the segments of a path (path depth is small and the maps compose sequentially).
- **Leanness -- one path**: Avoid alternative code paths for special cases that the main computation already handles. The decay-weighted cell integral reduces exactly to the plain cell width at zero decay; do not branch on it.
- **Leanness -- no cruft**: Don't introduce small helper functions used only once or twice -- inline them. This package has no backwards-compatibility requirement, so there are no shims, aliases, or deprecated paths to preserve.
- **Formatting**: Enforced by linting with ruff and prettier. Do not fight the formatter.
- **Parameter names**: `flow` for endmember demand, `cin`/`cout` for concentrations, `tedges`/`cout_tedges` for time edges, `network` for the `PipeNetwork`.

## Domain Conventions

**IMPORTANT**: These conventions are load-bearing for correctness.

- **Bin-edge pattern**: Time is `tedges` (`pd.DatetimeIndex`, n+1 edges) with n values constant over each interval `[tedges[i], tedges[i+1])`.
- **Single source, tree topology**: One production point at the root; flow splits but never merges. A split leaves concentration unchanged, which is what makes the single-`cin` model exact.
- **Endmember demand drives everything**: `flow` is the demand at every endmember (leaf). Every internal segment flow follows from mass conservation as the sum of the demands downstream of it.
- **Label coordinate**: transport is built on the cumulative throughflow volume at the reporting node, not on time. Output bin averages are uniform in that coordinate, which is exactly flow weighting.
- **Paired operations**: Functions come in forward (`source_to_endmember`) and reverse (`endmember_to_source`) variants.
- **Units**: Must be consistent within a calculation. The package does not enforce units -- the user is responsible. Internally: days, m, m³, m³/day, 1/day.

## Testing

- Use fixtures from `tests/src/conftest.py` for common test data.
- Tests MUST be exact to machine precision. Use `np.testing.assert_allclose(actual, expected)`.
- Validate physical correctness: mass conservation, analytic limits, the constant-flow-fraction reduction to `sum(V_i / f_i)`.
- Tests MUST be meaningful -- not trivial identity checks. `tests/src/_oracle.py` holds an independent brute-force Lagrangian implementation; cross-check against it.
- Run specific tests with: `uv run pytest tests/src/test_transport.py -v`

## Git

- Do NOT include Claude-related signatures in commit messages or PR descriptions.
- Base your PR message on the template at `.github/PULL_REQUEST_TEMPLATE.md`.
- Run formatting, linting, and type checking before committing.
