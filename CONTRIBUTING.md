# Contributing to BEAST

Thank you for your interest in contributing to BEAST! This guide walks you through
a contribution step by step. If you are unsure whether a change fits, open an
issue first and describe what you have in mind.

## How to contribute

1. **Fork the repository** on GitHub. Members of the BEAST team can create a
   branch in the repository directly.

2. **Clone it** together with the jigsaw library:

   ```console
   git clone --recurse-submodules https://github.com/<your-account>/beast.git
   cd beast
   ```

3. **Install BEAST for development** in a new virtual environment, and set up the
   pre-commit hooks. They run ruff, mypy, numpydoc validation and the test suite
   on every commit.

   ```console
   python -m venv .venv
   source .venv/bin/activate
   pip install -e libs/jigsaw/
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Create a branch** with a prefix that says what kind of change it is:
   `feat/`, `fix/`, `chore/` or `docs/`, for example
   `git checkout -b feat/hourly-sampling`.

5. **Make your changes**, following the conventions below.

6. **Test your changes** as described in the [testing guide](docs/testing.md). If
   your change touches distributed code, also run the tests with `mpirun`.

7. **Update the documentation** in `docs/` if your change affects how BEAST is
   used, for example new configuration keys or environment variables. Check that
   it builds without warnings:

   ```console
   pip install -r docs/requirements.txt
   sphinx-build -W -b html docs docs/_build/html
   ```

8. **Commit** with a short message that says what the change does, using the same
   prefixes as for branches, for example `fix: skip SST weights without SST`.

9. **Open a pull request** against `main` and fill in the checklist of the
   template.

## Conventions

- **Type hints** are required for every function in `beast/`. Use the Python
  3.12 forms, such as `int | None` and `list[str]`. Pre-commit only checks the
  files of a commit, so run `mypy` on its own before you push.
- **Docstrings** follow the
  [numpydoc standard](https://numpydoc.readthedocs.io/en/latest/format.html) and
  are checked by numpydoc's validator. Every function has one, including private
  functions and methods. A class documents its constructor arguments in the class
  docstring; `__init__` has no docstring of its own.
- **Write docstrings for the reader.** Start with a one-line summary, then
  describe what the code does. Leave out its history and design discussions.
- **Process groups** are referred to by their names in the process mesh: `DP`,
  `SP`, `JSpatial`, `JChannel`, `DTP` and `DDP`.

A class and a function documented this way:

```python
class Downsample(nn.Module):
    """Downsample by averaging over blocks of ``factor`` along each dimension.

    Parameters
    ----------
    factor : int
        Downsampling factor.
    dims : tuple[int, ...], default: (-1,)
        Dimensions to downsample over, all positive or all negative.

    Attributes
    ----------
    negative_dims : bool
        Whether ``dims`` holds negative indices.
    """

    def __init__(self, factor: int, dims: tuple[int, ...] = (-1,)) -> None: ...


def mse(prediction: Tensor, target: Tensor, batch_dimension: int = 0) -> Tensor:
    """Compute the mean squared error over the batch dimension.

    Parameters
    ----------
    prediction : Tensor
        Model prediction.
    target : Tensor
        Reference values, of the shape of ``prediction``.
    batch_dimension : int, default: 0
        Dimension to average over.

    Returns
    -------
    Tensor
        Squared error averaged over ``batch_dimension``.
    """
```

## Code of conduct

Everyone taking part in BEAST is expected to follow our
[code of conduct](https://github.com/RAI-SCC/beast/blob/main/CODE_OF_CONDUCT.md).

## Questions and bug reports

Please open an [issue](https://github.com/RAI-SCC/beast/issues). For bugs, the
issue template asks for the information we need to reproduce the problem.
