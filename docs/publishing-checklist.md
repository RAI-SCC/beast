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

- [ ] **Fix the pre-existing lint and format failures on `main`.** The new CI
      workflow's `lint` job fails today, before any of the other jobs matter. As of
      commit `da65c3a`, `ruff check .` reports three errors:

      | File | Rule | Issue |
      |------|------|-------|
      | `beast/model/utils.py:3` | `F401` | `ast.Mod` imported but unused |
      | `beast/model/utils.py:3` | `I001` | Import block un-sorted |
      | `beast/data/dataloader.py:65` | `D417` | `get_dataloader` docstring missing `**kwargs` |

      and `ruff format --check .` wants to reformat six files:
      `beast/data/dataloader.py`, `beast/data/era5/__init__.py`,
      `beast/layers/transformer.py`, `beast/losses/up_loss.py`,
      `beast/model/gb/beast_model.py`, `beast/model/utils.py`.

      `ruff check --fix . && ruff format .` resolves everything except the `D417`,
      which needs a `**kwargs` entry in the docstring. This was deliberately *not*
      done as part of the documentation work, because reformatting six source files
      would conflict with the branches currently in flight. Land it on `main` once
      those have merged, ideally as a single formatting-only commit that can be
      added to `.git-blame-ignore-revs`.

- [ ] **Validate the CI workflow.** It has never run — it was written against the
      repository's documented commands but not executed. The distributed job is
      marked `continue-on-error: true` for that reason; remove that once it is
      green. Expect disk space to be the first thing that breaks (see below).
- [ ] **Reduce test-suite disk usage.** A single run writes ~8 GB of zarr fixtures
      to `TMPDIR`, and pytest retains three runs. This is the single biggest
      obstacle to CI on a standard GitHub-hosted runner (~14 GB free), and it makes
      a first-time contributor's `pytest` fail confusingly on any machine with a
      tmpfs `/tmp`. Smaller fixtures or session-scoped reuse would fix both.
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
