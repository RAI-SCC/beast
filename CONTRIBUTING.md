# Contributing to BEAST

Thanks for your interest in BEAST. This guide covers how to get a working
development environment and what we expect from a pull request.

If you are an AI coding agent, read [AGENTS.md](AGENTS.md) instead — it has the
same information plus the codebase-specific detail you need.

## Development setup

BEAST depends on the `jigsaw` submodule. It is a hard import dependency: without
it, `import beast.model` fails.

```console
git clone https://github.com/RAI-SCC/beast.git
cd beast
git submodule update --init --recursive

python -m venv .venv
source .venv/bin/activate

pip install -e libs/jigsaw/
pip install -e ".[dev]"
pre-commit install
```

Python 3.12 or newer is required.

## Running the tests

```console
TMPDIR=/var/tmp pytest
```

**Set `TMPDIR` to disk-backed storage.** The suite writes roughly 8 GB of zarr
fixtures per run, and pytest retains the last three runs. On a distribution that
mounts `/tmp` as a tmpfs, a plain `pytest` exhausts it partway through and
reports around ten failures that all read `OSError: [Errno 28] No space left on
device`. Those are not real failures. See [TESTING.md](TESTING.md).

Other useful invocations:

```console
TMPDIR=/var/tmp pytest --device cuda        # include GPU tests
TMPDIR=/var/tmp mpirun -n 4 pytest          # include 4-rank distributed tests
```

On CPU with a single rank, expect 155 passed and 73 skipped; the skips are the
GPU and multi-rank tests.

## Code quality

`pre-commit install` wires up the checks that CI also runs:

```console
ruff check --fix .     # lint and sort imports
ruff format .          # format
mypy                   # type check the beast package
pre-commit run --all-files
```

The pre-commit configuration runs the **full pytest suite** as a local hook, so
commits take a couple of minutes. That is deliberate — please do not disable it.

## Conventions

- **Type annotations are required** on functions in `beast/`
  (`disallow_untyped_defs`). `tests/` and `beast/model/test_models/` are exempt.
- **NumPy-style docstrings** on public functions, enforced by ruff's `D` rules.
  Include `Parameters` and `Returns` sections.
- Python 3.12 idioms: `X | None` rather than `Optional[X]`, built-in generics.
- Line length is not enforced, but match the surrounding code.

## Tests

- Test files mirror the source layout: `beast/layers/mlp.py` →
  `tests/layer/test_mlp.py`.
- `conftest.py` initializes a real `torch.distributed` process group for the whole
  session, so even a single-rank run has `dist.is_initialized() == True`.
- Mark GPU-only tests `@pytest.mark.gpu` and multi-rank tests `@pytest.mark.dist`
  (or `@pytest.mark.dist(4)` for an exact rank count).
- Use the `shared_tmp_path` fixture instead of `tmp_path` in distributed tests, or
  each rank will get a different directory.
- Wrap Hypothesis strategies in `dist_st` when the drawn value feeds a collective,
  so all ranks draw the same value. Without it, ranks disagree on shapes and the
  test hangs rather than failing.
- Use the `BEASTLING` configuration in tests. The production configurations will
  not fit on a single test device.

**If your change touches distributed code, run the suite under MPI at least once.**
A single-rank pass proves very little about sharded behaviour.

## Pull requests

1. Branch off `main`. Do not commit to `main` directly.
2. Keep the change focused; unrelated refactors make review harder.
3. Make sure `pre-commit run --all-files` passes.
4. Update the docs in the same PR:
   - public API changes → [docs/architecture.md](docs/architecture.md)
   - new `config_dict` keys → [docs/configuration.md](docs/configuration.md)
   - new environment variables → both of the above and [AGENTS.md](AGENTS.md)
5. Describe what you changed and how you verified it. If you could not run the
   distributed or GPU tests, say so explicitly.

## Reporting bugs

Open an issue with the BEAST configuration ID, the process-mesh dimensions, the
number of ranks, and whether the run used real or dummy data. For a distributed
hang, the output of `visualize_process_mesh()` is usually the most informative
thing you can attach.

Before filing a training bug, check the rank-0 log for
`not found. Running dummy dataset` — `get_dataloader` falls back to random data
when the configured `data_path` does not exist, and a job that trains on noise
looks a lot like a broken model.

## Code of conduct

Be respectful and constructive. Report unacceptable behaviour to the maintainers.
