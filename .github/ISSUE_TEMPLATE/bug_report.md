---
name: Bug report
about: Report incorrect behaviour, a crash, or a distributed hang
title: ''
labels: bug
assignees: ''
---

## What happened

## What you expected

## Environment

- BEAST commit / version:
- `jigsaw` submodule commit:
- Python version:
- PyTorch version:
- Machine (JUPITER / LUMI / local / other):

## Run configuration

- Model configuration ID (e.g. `BEASTLING`, `BEAST-LUMI-1.2B-1.8K`):
- Process-mesh dimensions (`DP`, `SP`, `JSpatial`, `JChannel`):
- Number of ranks:
- Real ERA5 data, or the dummy dataset?
- Relevant environment variables (`BEAST_DL_DSYNC_INIT`, `BEAST_DS_HOURLY_SUBSAMPLING`, `BEAST_USE_TE_DPA`):

## Reproduction

<!-- Smallest script or command that shows the problem. -->

## Logs

<!--
Paste the traceback or the relevant log excerpt.

For a distributed hang, the output of beast.comm.visualize_process_mesh() is
usually the most useful thing you can attach.
-->

## Checked already

Please skim [docs/pitfalls.md](../../docs/pitfalls.md) first — it catalogues the
ways BEAST fails quietly, and several common "bugs" turn out to be environment
problems.

- [ ] The rank-0 log does not contain `not found. Running dummy dataset`
      (BEAST silently trains on random data when `data_path` is missing).
- [ ] `df -h` shows free space on `TMPDIR` — exhausting it produces
      `OSError: [Errno 28] No space left on device` failures that are not real bugs.
- [ ] If this is a NCCL error under `mpirun`, the machine has at least as many
      GPUs as ranks, or the run used `--device cpu`.
- [ ] If the suite hangs rather than failing, no Hypothesis strategy feeding a
      collective is missing its `dist_st` wrapper.
