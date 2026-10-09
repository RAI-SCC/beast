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
pytest                                # 155 passed, 81 skipped, ~25 s
pytest --device cuda                  # include GPU tests
mpirun -n 4 pytest --device cpu       # include 4-rank distributed tests
```

The skips in a single-rank run are the GPU and multi-rank tests.

Pass `--device cpu` to `mpirun` unless the machine has at least as many GPUs as
ranks — otherwise every test errors with a NCCL `DistBackendError`.

A run writes about 490 MB of zarr fixtures to `TMPDIR` (2.1 GB for 4 ranks), and
pytest retains the last three runs. If you hit `OSError: [Errno 28] No space left
on device`, that is disk exhaustion rather than a real failure; clear it with
`rm -rf "${TMPDIR:-/tmp}"/pytest-of-*` or set `TMPDIR=/var/tmp`. See
[TESTING.md](TESTING.md).

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

**Run bare `mypy` before you push, separately from pre-commit.** The pre-commit
mypy hook sets `pass_filenames: true`, so it only type-checks the files in your
commit. CI runs `mypy` over the whole `beast` package. A change can therefore pass
pre-commit and still fail CI.

## Conventions

- **Type annotations are required** on functions in `beast/`
  (`disallow_untyped_defs`). `tests/` is exempt.
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
- Use `small_beast_config()` from `tests/small_models.py` for model tests; the
  production configurations will not fit on a single test device.
- Use the `cached_mock_zarr` fixture rather than calling `generate_mock_zarr_data`
  directly, so zarr stores are shared between tests instead of duplicated.
- An autouse `_reset_process_mesh` fixture tears down any mesh a test creates.
  `beast.comm` keeps the mesh in module-level state, so leaving one up makes later
  tests depend on collection order.

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

## Before you debug something surprising

Read [docs/pitfalls.md](docs/pitfalls.md). BEAST is distributed and probabilistic,
so a mistake usually produces plausible-looking numbers or a silent hang rather
than an exception, and several common "failures" are environment problems rather
than bugs. The list covers setup, testing, configuration, distributed execution
and the model API.

The three that catch people most often:

- A missing `data_path` makes `get_dataloader` train on random noise, warning on
  rank 0 only.
- A Hypothesis strategy feeding a collective makes the suite *hang*, with no
  traceback, unless it is wrapped in `dist_st`.
- `mpirun -n 4 pytest` without `--device cpu` fails with NCCL errors on any
  machine with fewer than four GPUs.

If you find a new way to fail quietly, add it to that file in the same PR.

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
