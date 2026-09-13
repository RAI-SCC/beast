# Publishing checklist

Tracking what remains before making this repository public (target: November 2026).

## Done

- [x] `LICENSE` — MIT
- [x] `README.md` — description, usage, acknowledgments and citation sections filled in
- [x] `CONTRIBUTING.md` — setup, conventions, PR expectations
- [x] `CITATION.cff` — machine-readable citation metadata
- [x] `AGENTS.md` / `CLAUDE.md` — orientation for AI coding agents
- [x] `docs/architecture.md` — module map and process-mesh semantics
- [x] `docs/configuration.md` — complete `config_dict` and environment-variable reference
- [x] `TESTING.md` — expanded with disk requirements and distributed fixtures
- [x] `.github/workflows/ci.yml` — lint, type check, tests, build
- [x] Issue and pull-request templates
- [x] **Lint and format clean.** `ruff check .` and `ruff format --check .` both
      pass as of `5ccf6da`; the three errors and six unformatted files recorded
      here previously were fixed on `main`.
- [x] **`mypy` clean.** All 22 pre-existing errors fixed; the CI `typecheck` job
      is blocking. One documented `# type: ignore[override]` remains, on
      `EWBEra5.__getitem__` — see the follow-up below.
- [x] **Test-suite disk usage reduced to a CI-viable level.** 8.1 GB → 490 MB for
      a single-rank run (2.1 GB for 4 ranks), and 68 s → 25 s. Roughly 5x came
      from the smaller model fixtures on `main` (`tests/small_models.py`); the rest
      from the session-scoped `cached_mock_zarr` fixture, which stopped twenty-two
      byte-identical zarr stores being rebuilt per parametrisation, and from
      dropping the feature count of the one full-resolution generator test.

## Needs a human decision

- [ ] **Confirm the `LICENSE` copyright line.** It currently reads
      `Karlsruhe Institute of Technology and the BEAST contributors`. Institutional
      software releases usually need sign-off on the exact legal entity, and
      contributors from CSC and NVIDIA may need their employers' agreement.
- [ ] **Review `CITATION.cff` authorship.** The author list is ordered by commit
      count, which is rarely the intended academic ordering. Several contributors
      have no ORCID or affiliation recorded. Two contributors appear under multiple
      git identities and were merged by hand — verify nobody was dropped.
- [ ] **Add funding and compute-allocation acknowledgments.** Marked with a `TODO`
      in `README.md`. EuroHPC access agreements for JUPITER and LUMI typically
      require specific wording and project IDs.
- [ ] **Add the paper citation** once published (`TODO` in `README.md`, and
      `preferred-citation` in `CITATION.cff`).
- [ ] **Decide whether `beast/model/test_models/` ships.** It holds ~4000 lines of
      experimental architectures, excluded from mypy and ruff, with a name that
      actively misleads readers into thinking it contains model tests. Options:
      keep it (and rename to `experimental/`), move it to a separate repository,
      or drop it. Shipping it as-is invites confusion on day one.
- [ ] **Confirm `torch-blue` and `jigsaw` are public** and will stay at the pinned
      refs. `pyproject.toml` pins `torch-blue` to a git branch
      (`@stable-beast`) and `.gitmodules` tracks `jigsaw` at `branch = main`. Both
      are moving targets: an external user installing in six months may get
      something different from what was tested. Consider pinning to tags or commit
      SHAs before release.

## Technical, before release

- [ ] **Model `EWBEra5` with composition rather than inheritance.** It subclasses
      `ERA5Distributed` to reuse the sharding and normalization setup, but its
      `__getitem__` returns `(event, lead_time, inputs)` where the parent returns
      an `(input, target)` training pair. The two are not interchangeable, so the
      override carries a documented `# type: ignore[override]`. Extracting the
      shared loading logic into a helper that both hold would remove the
      suppression and the trap. Low urgency, but it is the one place where the
      type checker is being told to look away.

- [ ] **Validate the CI workflow on GitHub.** The commands it runs are all verified
      locally (lint, format, mypy, single-rank pytest, 4-rank pytest), but the
      workflow itself has never executed on a runner. Expect the usual first-run
      friction: action versions, the OpenMPI install, and submodule checkout.
- [ ] **Consider validating `config_dict`.** There is no schema and no validation;
      a typo in a key is silently ignored, and `set_dtype` silently falls back to
      float32 on an unrecognised string. A dataclass or TypedDict would turn a
      class of silent misconfiguration into errors.
- [ ] **Reconsider the silent dummy-data fallback.** `get_dataloader` substitutes
      random data when `data_path` is missing, warning on rank 0 only. For a public
      release this should probably be opt-in (an explicit `use_dummy=True`) rather
      than the default behaviour on a typo'd path.
- [ ] **Add a minimal training example.** The repository is a library with no
      training loop, so there is currently no end-to-end path a new user can run.
      A short script against `BEASTLING` and `ERA5Dummy` would make the package
      immediately evaluable.
- [ ] Decide whether to publish to PyPI. If so, the `torch-blue` direct git
      reference in `pyproject.toml` must be replaced — PyPI rejects direct URL
      dependencies.
- [ ] Enable branch protection on `main` and require the CI checks.
- [ ] Add `SECURITY.md` if the institution requires a disclosure policy.
- [ ] Consider a `CODE_OF_CONDUCT.md` (Contributor Covenant is the usual choice).
- [ ] Archive a release on Zenodo to get a DOI, and add it to `CITATION.cff`.

## Nice to have

- [ ] Published API documentation (Sphinx or MkDocs). Docstring coverage is
      already good and NumPy-style, so `sphinx.ext.napoleon` would need little work.
- [ ] Pretrained checkpoints, or a pointer to where they can be obtained.
- [ ] A short quickstart notebook.
