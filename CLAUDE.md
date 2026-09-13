# CLAUDE.md

See **[AGENTS.md](AGENTS.md)** — it is the single source of truth for agent
guidance in this repository, and it applies to Claude Code as written.

Keep this file a pointer. Do not duplicate guidance here; the two copies will
drift and an agent reading only one of them will act on stale instructions.

A 30-second summary, so you know what you are looking at before you open it:

- BEAST is a **library**, not an application. No training loop lives here.
- The model is **Bayesian** (`torch-blue` variational inference) and **sharded
  across a 4-D process mesh** (`jigsaw`). Tensors inside modules are local shards.
- `libs/jigsaw` is a git submodule and a hard import dependency. Run
  `git submodule update --init --recursive` before anything else.
- `pytest` needs ~8 GB of scratch; run it as `TMPDIR=/var/tmp pytest` or you will
  get spurious `No space left on device` failures.
- `beast/model/test_models/` holds experimental architectures, not model tests.
