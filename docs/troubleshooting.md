# Troubleshooting

## The loss does not improve

Check the log of rank 0 for `not found. Running dummy dataset`. If `data_path`
does not exist, {func}`~beast.data.dataloader.get_dataloader` uses random data
instead, and only rank 0 warns about it.

## The job hangs at startup

{func}`~beast.comm.create_process_mesh_dict`,
{func}`~beast.comm.visualize_process_mesh` and
{func}`~beast.comm.execute_in_waves` are collective calls: every process has to
call them. A job hangs if some processes skip them, for example inside an
`if rank == 0:` block.

## `ValueError: Product of mesh dimensions must equal world size.`

The four sizes passed to {func}`~beast.comm.create_process_mesh_dict` multiply to
a different number than the number of processes. Adjust the sizes, or the number
of processes you launch.

## `ValueError: persistent_workers option needs num_workers > 0`

The data loader keeps its workers alive between epochs, which needs at least one
worker. Set `num_workers` in the `data` section of the configuration to 1 or
more.

## `AssertionError: Width of ... layer is not divisible by ... window size.`

The grid does not fit the attention windows of the model. See the grid
requirements under [Model presets](configuration.md#model-presets).

## The model trains, but on the wrong part of the data

`get_dataloader` takes the ranks of two different groups: `dtp_rank` and
`dtp_size` choose the part of each sample a process loads, and `dp_rank` and
`dp_size` choose which samples it sees. Passing the `DP` ranks as `DTP` ranks,
or the other way round, raises no error.
