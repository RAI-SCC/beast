# AGENTS.md

Guidance for AI coding agents working in the BEAST repository.

## What this repository is

BEAST is a PyTorch library for probabilistic weather models that run across many
GPUs. It contains the model, layers, loss, ERA5 datasets and evaluation metrics,
but no training loop. Two properties shape most of the code:

- **Models can be Bayesian.** Weights become distributions through `torch-blue`,
  and the loss in `beast/losses/up_loss.py` is a variational objective.
- **Models are sharded over a 4-D process mesh.** Inside a module, tensors are
  usually local shards. Grid and channel sharding come from `jigsaw`.

## Where to read first

| Topic | File |
|-------|------|
| Package layout, model structure, sharding | `docs/architecture.md` |
| Configuration keys, data format, environment variables | `docs/configuration.md` |
| The process mesh and a full training step | `docs/quickstart.md` |
| Writing and running tests | `docs/testing.md` |
| Code and docstring conventions | `CONTRIBUTING.md` |
| Common errors | `docs/troubleshooting.md` |

## Setup and commands

```console
git submodule update --init --recursive
python -m venv .venv && source .venv/bin/activate
pip install -e libs/jigsaw/
pip install -e ".[dev]"
pre-commit install
```

| Task | Command |
|------|---------|
| Tests, single process | `pytest` |
| Tests, 4 ranks | `mpirun -n 4 pytest --device cpu` |
| Lint and format | `ruff check --fix . && ruff format .` |
| Type check (whole package) | `mypy` |
| Docstrings | `numpydoc lint $(git ls-files 'beast/*.py')` |
| Documentation | `sphinx-build -W -b html docs docs/_build/html` |
| All hooks | `pre-commit run --all-files` |

## Rules

- Docstrings follow numpydoc: constructor arguments go in the class docstring,
  every function (including private ones) has a full docstring, and docstrings
  describe what the code does, not its history.
- Refer to process groups by their mesh names: `DP`, `SP`, `JSpatial`,
  `JChannel`, `DTP`, `DDP`.
- If a change touches distributed code, run the tests with
  `mpirun -n 4 pytest --device cpu`. Without `--device cpu`, a machine with fewer
  than four GPUs fails with NCCL errors.
- The pre-commit hooks run the full test suite. Do not disable them.
- Pre-commit type-checks only the committed files; run `mypy` on the whole
  package before pushing.
- In Hypothesis tests whose values reach a collective, wrap the strategy in
  `dist_st`, otherwise the ranks disagree and the test hangs.
- When you add a configuration key or environment variable, document it in
  `docs/configuration.md` in the same change.
- `tests/test_grad_reduction.py` checks that gradients do not depend on the mesh
  layout. Run it under MPI after changing anything in `beast/comm.py` or the
  gradient hooks.
