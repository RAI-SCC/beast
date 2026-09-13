## What this changes

<!-- One or two sentences. What behaviour is different after this PR? -->

## Why

<!-- Link the issue, or explain the motivation. -->

## How it was verified

<!-- Tick what you actually ran. Leave the rest unticked rather than guessing. -->

- [ ] `pre-commit run --all-files`
- [ ] `TMPDIR=/var/tmp pytest` (single rank, CPU)
- [ ] `TMPDIR=/var/tmp pytest --device cuda` (GPU)
- [ ] `TMPDIR=/var/tmp mpirun -n 4 pytest` (distributed)

If this touches distributed code — the process mesh, sharded layers, gradient
reduction, or the data shards — the multi-rank run is not optional. A single-rank
pass does not exercise the collectives.

## Docs

- [ ] Public API changed → updated `docs/architecture.md`
- [ ] New `config_dict` key → updated `docs/configuration.md`
- [ ] New environment variable → updated `docs/configuration.md` and `AGENTS.md`
- [ ] No docs change needed

## Anything reviewers should look at closely

<!-- Known gaps, decisions you were unsure about, things you could not test. -->
