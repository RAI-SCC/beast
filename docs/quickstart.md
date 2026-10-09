# Quickstart

This page walks through one training step of a Bayesian BEAST model on several
GPUs, in the order the pieces have to be set up. Launch the script with one
process per GPU, for example with `torchrun`.

## 1. Build the process mesh

Every process belongs to a four-dimensional mesh. Its dimensions, in order, are:

| Name | Splits | Meaning |
|------|--------|---------|
| `DP` | data items | Data parallelism: each group of ranks sees different samples. |
| `SP` | ensemble members | Sample parallelism: each rank draws its own ensemble members. |
| `JSpatial` | the grid | Each rank holds a tile of the latitude-longitude grid. |
| `JChannel` | the channels | Each rank holds a slice of the variables. |

The product of the four sizes must equal the number of processes. The mesh also
provides two combined groups: `DTP` (`JSpatial` x `JChannel`, the ranks that share
one data item) and `DDP` (`DP` x `SP` x `JSpatial`, the ranks gradients are
averaged over).

```python
import os

import torch
import torch.distributed as dist

from beast import get_model
from beast.comm import create_process_mesh_dict, wrap_ddp
from beast.data.dataloader import get_dataloader
from beast.losses.up_loss import UPAllGaussLoss
from beast.model import convert_jigsaw_vi
from beast.utils import get_spatial_weights

dist.init_process_group(backend="nccl")
device_id = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(device_id)
device = torch.device("cuda", device_id)

# (DP, SP, JSpatial, JChannel): 16 processes
groups = create_process_mesh_dict([2, 2, 2, 2])
```

## 2. Load the data

{func}`~beast.data.dataloader.get_dataloader` reads its settings from a
configuration dictionary, described in [Configuration](configuration.md). The
`DTP` rank decides which shard of each sample a process loads, and the `DP` rank
decides which samples it sees.

```python
loader = get_dataloader(
    config_dict,
    dtp_rank=groups["DTP"].rank(),
    dtp_size=groups["DTP"].size(),
    dp_rank=groups["DP"].rank(),
    dp_size=groups["DP"].size(),
    mode="train",
)
dataset = loader.dataset
if isinstance(dataset, torch.utils.data.Subset):
    dataset = dataset.dataset
```

## 3. Build the model

{func}`~beast.get_model` builds a named configuration and accepts any of its
arguments as overrides. Pass the process groups to shard the model, and take the
grid size from the data. The JUPITER and LUMI configurations are Bayesian, so the
model is converted with {func}`~beast.model.convert_jigsaw_vi` and then wrapped
for data parallelism.

```python
model = get_model(
    "BEAST-JUPITER-0.7B-1.8K",
    height=dataset.xlat,
    width=dataset.xlon,
    device=device,
    channel_group=groups["JChannel"],
    spatial_group=groups["JSpatial"],
    dtp_group=groups["DTP"],
)
convert_jigsaw_vi(model, groups["JChannel"], groups["JSpatial"])
model.return_log_probs = False
model = wrap_ddp(model, device_id=device_id, ddp_group=groups["DDP"])
```

## 4. Set up the loss

{class}`~beast.losses.up_loss.UPAllGaussLoss` combines the ensemble members of
all `SP` ranks. The feature weights exclude sea surface temperature over land,
and the latitude weights account for the smaller area of grid cells near the
poles.

```python
loss_fn = UPAllGaussLoss(
    model,
    dataset_size=len(dataset),
    up_group=groups["SP"],
    feature_weights=dataset.get_feature_weights(groups["JChannel"].rank()).to(device),
)
spatial_weights = torch.sqrt(get_spatial_weights(dataset.xlat, device))
```

## 5. Run a training step

Each batch is followed by two timestamps of the batch fetch. `samples` sets the
number of ensemble members drawn on each rank.

```python
x, y, _, _ = next(iter(loader))
pred = model(x.to(device), samples=4)
loss = loss_fn(pred * spatial_weights, y.to(device) * spatial_weights)
loss.backward()
```

## Running on one device

For a single GPU, use the small `BEASTLING` configuration and a mesh of
`[1, 1, 1, 1]`. On the CPU, also initialize the process group with the `gloo`
backend, call `create_process_mesh_dict([1, 1, 1, 1], device_type="cpu")` and pass
`device_id=None` to `wrap_ddp`.
