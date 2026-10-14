---
name: Bug report
about: Report incorrect behaviour, a crash, or a distributed hang
title: ''
labels: bug
assignees: ''
---

<!-- Common errors and their fixes are listed in docs/troubleshooting.md. -->

## What happened

## What you expected

## Environment

- BEAST commit / version:
- jigsaw version / commit:
- Python version:
- PyTorch version:
- Machine (JUPITER / LUMI / local / other):

## Run configuration

- Model configuration ID (e.g. `BEASTLING`, `BEAST-LUMI-1.1B-64K`):
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
