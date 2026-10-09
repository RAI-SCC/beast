# Description

<!-- Summarize the change and why it is needed. -->

Fixes # (issue)

## Type of change

<!-- Delete the options that do not apply. -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (existing code using BEAST has to change)
- [ ] Documentation only

## Checklist

<!-- If you cannot check a box, explain why below. -->

- [ ] My code follows the [contributing guidelines](../CONTRIBUTING.md), including numpydoc docstrings
- [ ] I have reviewed my own changes
- [ ] I have added tests that cover my changes
- [ ] `pre-commit run --all-files` and `mypy` pass
- [ ] The tests pass with `mpirun -n 4 pytest --device cpu`, if the change touches distributed code
- [ ] I have updated the documentation, and it builds with `sphinx-build -W`
