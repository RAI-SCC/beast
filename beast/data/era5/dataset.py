"""ERA5 datasets: zarr-backed, dummy and event-based for evaluation."""

import dataclasses as dc
import os
from datetime import datetime, timedelta
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import Tensor
from torch.utils.data import Dataset

from beast.comm import execute_in_waves

SUBSAMPLING_FACTOR = int(os.getenv("BEAST_DS_HOURLY_SUBSAMPLING", 1))


class ERA5Distributed(Dataset):
    """ERA5 dataset of ``(input, target)`` pairs read from a zarr directory.

    Inputs and targets stack one or more timesteps along the channel dimension,
    and inputs are followed by the constant masks. Values are normalized with
    the stored per-variable mean and standard deviation.

    If the ``DTP`` group has more than one rank, each rank loads only its
    shard: one of two channel halves and a slice of the longitudes. With an odd
    number of channels, constant masks are moved between the halves and the
    second half is zero-padded, so that both halves have the same size.

    Parameters
    ----------
    pg_rank : int
        Rank in the ``DTP`` process group.
    pg_size : int
        Size of the ``DTP`` process group.
    data_path : str
        Path to the zarr directory.
    constant_masks_path : str
        Directory containing the constant mask files.
    constant_masks : List[str]
        File names of the constant masks to use.
    normalization_dir : str
        Directory containing ``norm_mean.npy`` and ``norm_std.npy``.
    dt : int, default: 6
        Time between timesteps, in hours.
    n_in_timesteps : int, default: 1
        Number of input timesteps.
    n_out_timesteps : int, default: 1
        Number of output timesteps.
    dtype : torch.dtype, default: torch.float32
        Data type of inputs and targets.
    dsync_access : bool, default: False
        Whether to open the data in waves, to spread out file system access.
        Recommended beyond 512 nodes.
    time_range : Tuple[str, str], optional
        Start (inclusive) and end (exclusive) of the samples to use, formatted
        ``%Y-%m-%d``.
    transform : Callable, optional
        Transformation applied to the input.
    target_transform : Callable, optional
        Transformation applied to the target. Defaults to ``transform``.
    rollout : int, default: 1
        Number of rollout steps. The target holds ``n_out_timesteps * rollout``
        timesteps.

    Attributes
    ----------
    zarr_data : xarray.Dataset
        The opened data, restricted to ``time_range``.
    norm_values : dict[str, np.ndarray]
        Normalization ``"mean"`` and ``"std"`` per variable.
    n_data_items : int
        Number of timesteps in ``zarr_data``.
    xlat, xlon : int
        Global grid size, rounded down to an even number.
    n_variables : int
        Global number of variables.
    constant_masks : np.ndarray
        Constant masks, of shape ``(n_masks, xlat, xlon)``.
    n_masks : int
        Number of constant masks.
    in_var : Tuple[int, int]
        Range of variables loaded by this rank.
    in_masks : Tuple[int, int]
        Range of constant masks used by this rank.
    in_out_xlon : Tuple[int, int]
        Range of longitudes loaded by this rank.
    input_pad_oddrank, output_pad_oddrank : int
        Channels of zero padding appended to inputs and targets on odd ranks.
    """

    def __init__(
        self,
        pg_rank: int,
        pg_size: int,
        data_path: str,
        constant_masks_path: str,
        constant_masks: List[str],
        normalization_dir: str,
        dt: int = 6,
        n_in_timesteps: int = 1,
        n_out_timesteps: int = 1,
        dtype: torch.dtype = torch.float32,
        dsync_access: bool = False,
        time_range: Tuple[str, str] | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        rollout: int = 1,
    ):
        self.dt = dt
        self.in_timesteps = n_in_timesteps
        self.out_timesteps = n_out_timesteps

        self.dtype = dtype
        self.world_size = pg_size
        self.local_rank = pg_rank

        self.datapath = data_path
        self.years = time_range
        self.mask_path = constant_masks_path
        self.rollout = rollout
        mask_list = constant_masks

        self.norm_values: dict[str, np.ndarray] = {}

        def load_data() -> None:
            """Open the zarr data, restrict it to ``time_range`` and load masks and normalization."""
            try:
                self.zarr_data = xr.open_dataset(self.datapath, engine="zarr")
            except Exception:
                raise OSError(
                    "Given path does not point to a zarr directory. Data must be in zarr format."
                )

            try:
                self.xlat = self.zarr_data.sizes["latitude"]
                self.xlon = self.zarr_data.sizes["longitude"]
            except Exception:
                raise Exception(
                    "Zarr directory has no dimensions `latitude` and/or `longitude`. Cannot access spatial dimension"
                )

            if self.xlat % 2 == 1:
                self.xlat -= 1
            if self.xlon % 2 == 1:
                self.xlon -= 1

            try:
                times = pd.to_datetime(self.zarr_data.time)
            except Exception:
                raise Exception(
                    "Time values of data cannot be accessed. Zarr data must be chunked in the time domain. "
                )

            if self.years is not None:
                train_years = times[
                    (times >= self.years[0])
                    & (times < self.years[1])
                    & (times.hour % SUBSAMPLING_FACTOR == 0)
                ]
                self.zarr_data = self.zarr_data.sel(time=train_years)

            try:
                self.n_data_items = self.zarr_data.sizes["time"]
            except Exception:
                raise Exception(
                    "Zarr directory has no dimension `time`. Cannot access temporal dimension"
                )

            try:
                self.n_variables = self.zarr_data.sizes["feature"]
            except Exception:
                raise Exception(
                    "Zarr directory has no dimension `feature`. Cannot access feature dimension"
                )

            self.constant_masks = self._load_constant_mask_2d(self.mask_path, mask_list)
            self.n_masks = self.constant_masks.shape[0]

            self.norm_values["mean"] = np.load(f"{normalization_dir}/norm_mean.npy")
            self.norm_values["std"] = np.load(f"{normalization_dir}/norm_std.npy")

        if dsync_access:
            execute_in_waves(load_data)
        else:
            load_data()

        if self.world_size > 1:
            row, col = self.local_rank % 2, self.local_rank // 2
            # rank 0: [0,0], rank 1 = [1,0], rank 2 = [0,1], rank 3 = [1,1]
            if row == 0:
                self.in_var = (
                    0,
                    (
                        self.n_variables // 2
                        if self.n_variables % 2 == 0
                        else self.n_variables // 2 + 1
                    ),
                )
            else:
                self.in_var = (
                    (
                        self.n_variables // 2
                        if self.n_variables % 2 == 0
                        else self.n_variables // 2 + 1
                    ),
                    self.n_variables,
                )

            # We split the channels into two ranks; however, if we have an odd number of channels this will
            # result in rank 0 having one more variable.
            # since we do this for each timestep we accumulate an inbalance in the rank=1 channel group
            input_inbalance = (self.n_variables % 2) * self.in_timesteps
            output_inbalance = (self.n_variables % 2) * self.out_timesteps
            # We can use the constant masks to counteract this
            # First, we compute how many masks we can add to rank=1 channel group to counteract the imbalance
            self.input_pad_oddrank = 0
            self.output_pad_oddrank = output_inbalance

            # Two cases:
            # If we have more masks then the inbalance, we need to distribute the remaining masks evenly accross channel ranks
            if (self.n_masks - input_inbalance) >= 0:
                remaining_masks = self.n_masks - input_inbalance
                if row == 0:
                    self.in_masks = (0, remaining_masks - remaining_masks // 2)
                else:
                    self.in_masks = (
                        remaining_masks - remaining_masks // 2,
                        self.n_masks,
                    )

                if remaining_masks % 2 == 1:
                    self.input_pad_oddrank = 1

            # If we have fewer masks then the imbalance, we need to pad on rank 1
            else:
                if row == 0:
                    self.in_masks = (0, 0)
                else:
                    self.in_masks = (0, self.n_masks)

                self.input_pad_oddrank = input_inbalance - self.n_masks

            # split into 2 "longitude" pieces for 4-way parallel and 4 for 8-way
            self.in_out_xlon = (
                col * self.xlon // (self.world_size // 2),
                (col + 1) * self.xlon // (self.world_size // 2),
            )

        else:
            # Single-rank (no domain parallelism): all variables.
            self.in_var = (0, self.n_variables)
            self.in_masks = (0, self.n_masks)
            self.in_out_xlon = (0, self.xlon)
            self.input_pad_oddrank = 0
            self.output_pad_oddrank = 0

        self.transform = transform
        if target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = target_transform

        try:
            _ = self.__getitem__(0)
        except Exception as err:
            raise RuntimeError(
                "Dataset was instantiated, but data items cannot be loaded. Data "
                "seems to be empty, or does not contain `fields`."
            ) from err

    def __len__(self) -> int:
        """Return the number of samples that have a complete input and target.

        Returns
        -------
        int
            The number of samples.
        """
        return int(
            self.n_data_items
            - (self.in_timesteps + self.out_timesteps - 1)
            * (self.dt // SUBSAMPLING_FACTOR)
            * self.rollout
        )

    def _get_sample_list(self, idx: int, steps: int) -> list[torch.Tensor]:
        """Load and normalize consecutive timesteps of this rank's shard.

        Parameters
        ----------
        idx : int
            Index of the first timestep.
        steps : int
            Number of timesteps to load, ``dt`` hours apart.

        Returns
        -------
        list[torch.Tensor]
            One tensor of shape ``(channel shard, xlat, longitude shard)`` per
            timestep.
        """
        sample_list: list[torch.Tensor] = []
        for i in range(steps):
            sample = self.zarr_data.isel(
                time=(idx + i * (self.dt // SUBSAMPLING_FACTOR)),
                feature=range(self.in_var[0], self.in_var[1]),
                latitude=range(0, self.xlat),
                longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
            )["fields"].values

            sample = (
                sample
                - self.norm_values["mean"][self.in_var[0] : self.in_var[1]].reshape(
                    -1, 1, 1
                )
            ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]].reshape(
                -1, 1, 1
            )

            sample_list.append(torch.tensor(sample, dtype=self.dtype))

        return sample_list

    def _padsample(self, sample: torch.Tensor, inbalance: int) -> torch.Tensor:
        """Zero-pad the channels on ranks that hold the second channel half.

        Parameters
        ----------
        sample : torch.Tensor
            Sample with channels in the third-to-last dimension, of shape
            ``(..., channels, xlat, longitude shard)``.
        inbalance : int
            Number of zero channels to append.

        Returns
        -------
        torch.Tensor
            The padded sample on odd ranks of a distributed dataset, otherwise
            ``sample`` unchanged.
        """
        if self.world_size > 1:
            # Odd ranks hold the second half of the channels
            if (inbalance != 0) and (self.local_rank % 2 == 1):
                # Pads the channel dimension (third from last) at the end
                sample = torch.nn.functional.pad(sample, (0, 0, 0, 0, 0, inbalance))
        return sample

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one normalized ``(input, target)`` pair.

        Parameters
        ----------
        idx : int
            Global index of the first input timestep.

        Returns
        -------
        input : torch.Tensor
            Shape ``(channel shard * n_in_timesteps + masks, xlat, longitude shard)``.
        target : torch.Tensor
            Shape ``(channel shard * n_out_timesteps * rollout, xlat, longitude shard)``.
        """
        input_list = self._get_sample_list(idx, steps=self.in_timesteps)
        input_list.append(self.get_constant_masks().to(self.dtype))
        input = torch.cat(input_list, dim=0)
        input = self._padsample(input, self.input_pad_oddrank)

        # Output data
        target_idx = idx + self.in_timesteps * (self.dt // SUBSAMPLING_FACTOR)

        target_list = self._get_sample_list(
            target_idx, steps=self.out_timesteps * self.rollout
        )
        target = torch.cat(target_list, dim=0)
        target = self._padsample(target, self.output_pad_oddrank)

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def get_constant_masks(self) -> torch.Tensor:
        """Return the constant masks of this rank.

        Returns
        -------
        torch.Tensor
            Shape ``(masks of this rank, xlat, longitude shard)``.
        """
        return torch.tensor(
            self.constant_masks[
                self.in_masks[0] : self.in_masks[1],
                : self.xlat,
                self.in_out_xlon[0] : self.in_out_xlon[1],
            ]
        )

    def get_feature_weights(self, channel_rank: int, sst_idx: int = 4) -> torch.Tensor:
        """Return per-feature loss weights that mask SST over land.

        Parameters
        ----------
        channel_rank : int
            Rank in the ``JChannel`` process group. SST is masked only on
            rank 0, which holds the first channel half.
        sst_idx : int, default: 4
            Index of SST among the variables.

        Returns
        -------
        torch.Tensor
            Weights of shape ``(channel shard * n_out_timesteps, xlat, longitude
            shard)``, 1 everywhere except 0 for SST over land.
        """
        feature_weights = torch.ones(
            (self.in_var[1] - self.in_var[0]) * self.out_timesteps,
            self.xlat,
            self.in_out_xlon[1] - self.in_out_xlon[0],
        )
        if channel_rank == 0:
            for i in range(self.out_timesteps):
                feature_weights[i * (self.in_var[1] - self.in_var[0]) + sst_idx] = (
                    1 - self.get_land_sea_mask(folder_path=self.mask_path)
                )

        return feature_weights

    def get_land_sea_mask(
        self, folder_path: str, sst_mask_name: str = "sst_mask.npy"
    ) -> torch.Tensor:
        """Return the land-sea mask of this rank, 1 over land.

        Parameters
        ----------
        folder_path : str
            Directory containing the mask file.
        sst_mask_name : str, default: "sst_mask.npy"
            Name of the mask file.

        Returns
        -------
        torch.Tensor
            Float32 mask of shape ``(xlat, longitude shard)``.
        """
        return torch.tensor(
            np.load(os.path.join(folder_path, sst_mask_name))[
                : self.xlat, self.in_out_xlon[0] : self.in_out_xlon[1]
            ]
        ).to(torch.float32)

    def _load_constant_mask_2d(
        self, folder_path: str, mask_list: List[str]
    ) -> np.ndarray:
        """Load the constant masks that are appended to the input.

        Parameters
        ----------
        folder_path : str
            Directory containing the mask files.
        mask_list : List[str]
            File names of the masks.

        Returns
        -------
        np.ndarray
            Masks stacked to shape ``(n_masks, n_lat, n_lon)``.
        """
        # Load data from numpy files

        data = []
        for mask in mask_list:
            data.append(np.load(os.path.join(folder_path, mask)))

        masks = np.stack(data, axis=0)
        return masks

    def input_from_list(self, sample_list: list[torch.Tensor]) -> torch.Tensor:
        """Build a model input from a list of timesteps.

        The last ``in_timesteps`` entries are concatenated along the channel
        dimension, followed by the constant masks, and padded as in
        :meth:`__getitem__`.

        Parameters
        ----------
        sample_list : list[torch.Tensor]
            Batched timesteps, each ``(batch, channel shard, xlat, longitude shard)``.

        Returns
        -------
        torch.Tensor
            The model input.
        """
        input_samples = sample_list[-self.in_timesteps :]
        batch_dim = input_samples[0].shape[0]
        cmasks = self.get_constant_masks().unsqueeze(0).repeat(batch_dim, 1, 1, 1)
        input_samples.append(cmasks.to(self.dtype))
        input = torch.cat(input_samples, dim=-3)
        return self._padsample(input, self.input_pad_oddrank)


class ERA5Dummy(Dataset):
    """Dataset of random or constant ``(input, target)`` pairs of a given shape.

    Inputs and targets stack one or more timesteps along the channel dimension.
    If the ``DTP`` group has more than one rank, each rank generates only its
    shard: one of two channel halves and a slice of the longitudes.

    Parameters
    ----------
    pg_rank : int
        Rank in the ``DTP`` process group.
    pg_size : int
        Size of the ``DTP`` process group.
    global_shape : Tuple[int, int, int]
        Global ``(channel, latitude, longitude)`` shape of one timestep.
    fill_value : float, optional
        Value to fill inputs and targets with. If None or 0, they are random.
    time_range : Tuple[str, str], optional
        Start and end of the time range, formatted ``%Y-%m-%d``. Defaults to
        the year 2000.
    dt : int, default: 6
        Time between timesteps, in hours.
    n_in_timesteps : int, default: 1
        Number of input timesteps.
    n_out_timesteps : int, default: 1
        Number of output timesteps.
    dtype : torch.dtype, default: torch.float32
        Data type of inputs and targets.
    transform : Callable, optional
        Transformation applied to the input.
    target_transform : Callable, optional
        Transformation applied to the target. Defaults to ``transform``.

    Attributes
    ----------
    n_data_items : int
        Number of hourly timesteps in the time range.
    num_vars, xlat, xlon : int
        Global channel and grid sizes, with the grid rounded down to an even
        number.
    in_var : List[int]
        Range of channels generated by this rank.
    in_out_xlat, in_out_xlon : List[int]
        Ranges of latitudes and longitudes generated by this rank.
    """

    def __init__(
        self,
        pg_rank: int,
        pg_size: int,
        global_shape: Tuple[int, int, int],
        fill_value: float | None = None,
        time_range: Tuple[str, str] | None = None,
        dt: int = 6,
        n_in_timesteps: int = 1,
        n_out_timesteps: int = 1,
        dtype: torch.dtype = torch.float32,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.dt = dt
        self.in_timesteps = n_in_timesteps
        self.out_timesteps = n_out_timesteps

        self.dtype = dtype
        self.world_size = pg_size
        self.local_rank = pg_rank
        self.years = time_range

        self.fill_value = fill_value
        # Filter desired time points based on parameters and return file statistics.
        if self.years is not None:
            max_date = datetime.strptime(self.years[1], "%Y-%m-%d")
            min_date = datetime.strptime(self.years[0], "%Y-%m-%d")
        else:
            max_date = datetime.strptime("2000-12-31", "%Y-%m-%d")
            min_date = datetime.strptime("2000-01-01", "%Y-%m-%d")
        print(f"Dummy Dataset max_date {max_date}, min_date {min_date} ")

        self.num_vars, self.xlat, self.xlon = global_shape
        self.n_data_items = int((max_date - min_date).days * 24)
        print(f"Dummy Dataset n_data_items {self.n_data_items} ")

        if self.xlat % 2 == 1:
            self.xlat -= 1
        if self.xlon % 2 == 1:
            self.xlon -= 1

        self.in_out_xlat = [0, self.xlat]
        if self.world_size > 1:
            row, col = self.local_rank % 2, self.local_rank // 2
            # rank 0: [0,0], rank 1 = [1,0], rank 2 = [0,1], rank 3 = [1,1]
            self.in_var = [row * self.num_vars // 2, (row + 1) * self.num_vars // 2]
            self.in_out_xlon = [
                col * self.xlon // (self.world_size // 2),
                (col + 1) * self.xlon // (self.world_size // 2),
            ]

        else:
            self.in_var = [0, self.num_vars]
            self.in_out_xlon = [0, self.xlon]

        self.transform = transform
        if target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the number of samples that have a complete input and target.

        Returns
        -------
        int
            The number of samples.
        """
        return int(
            self.n_data_items - (self.in_timesteps + self.out_timesteps - 1) * self.dt
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return one generated ``(input, target)`` pair.

        Parameters
        ----------
        idx : int
            Index of the item. Items are generated, so every index returns
            equivalent data.

        Returns
        -------
        input : Tensor
            Shape ``(channel shard * n_in_timesteps, xlat, longitude shard)``.
        target : Tensor
            Shape ``(channel shard * n_out_timesteps, xlat, longitude shard)``.
        """
        shape_in = (
            (self.in_var[1] - self.in_var[0]) * self.in_timesteps,
            self.in_out_xlat[1] - self.in_out_xlat[0],
            self.in_out_xlon[1] - self.in_out_xlon[0],
        )
        shape_out = (
            (self.in_var[1] - self.in_var[0]) * self.out_timesteps,
            self.in_out_xlat[1] - self.in_out_xlat[0],
            self.in_out_xlon[1] - self.in_out_xlon[0],
        )

        if self.fill_value is not None:
            input = torch.full(shape_in, self.fill_value, dtype=self.dtype)
            target = torch.full(shape_out, self.fill_value, dtype=self.dtype)
        else:
            input = torch.rand(shape_in, dtype=self.dtype)
            target = torch.rand(shape_out, dtype=self.dtype)

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target


@dc.dataclass
class Event:
    """An extreme weather event from EWB, with its time window and bounding box.

    Attributes
    ----------
    id : str
        Event identifier.
    event_type : str
        Type of the event.
    title : str
        Event title.
    start_date, end_date : datetime
        Start and end of the event.
    lat_min, lat_max, lon_min, lon_max : float
        Latitude and longitude bounds of the event region.
    """

    id: str
    event_type: str
    title: str
    start_date: datetime
    end_date: datetime
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class EWBEra5(ERA5Distributed):
    """ERA5 initial conditions for forecasting ExtremeWeatherBench (EWB) events.

    Each item is one event at one lead time, together with the input timesteps
    to start the forecast from. Data is read, normalized and sharded as in
    :class:`ERA5Distributed`.

    Parameters
    ----------
    event_list : Sequence[Event]
        Events to forecast.
    lead_times : Sequence[int]
        Lead times to forecast each event at, in hours.
    pg_rank : int
        Rank in the ``DTP`` process group.
    pg_size : int
        Size of the ``DTP`` process group.
    data_path : str
        Path to the zarr directory.
    constant_masks_path : str
        Directory containing the constant mask files.
    constant_masks : List[str]
        File names of the constant masks to use.
    normalization_dir : str
        Directory containing ``norm_mean.npy`` and ``norm_std.npy``.
    dt : int, default: 6
        Time between timesteps, in hours.
    n_in_timesteps : int, default: 1
        Number of input timesteps.
    n_out_timesteps : int, default: 1
        Number of output timesteps per model step.
    dtype : torch.dtype, default: torch.float32
        Data type of the inputs.
    dsync_access : bool, default: False
        Whether to open the data in waves, to spread out file system access.
        Recommended beyond 512 nodes.
    time_range : Tuple[str, str], optional
        Start (inclusive) and end (exclusive) of the samples to use, formatted
        ``%Y-%m-%d``.
    transform : Callable, optional
        Transformation applied to the input.
    target_transform : Callable, optional
        Transformation applied to the target. Defaults to ``transform``.

    Attributes
    ----------
    event_lead_list : list[tuple[Event, int]]
        Every combination of event and lead time, in item order.
    """

    def __init__(
        self,
        event_list: Sequence["Event"],
        lead_times: Sequence[int],
        pg_rank: int,
        pg_size: int,
        data_path: str,
        constant_masks_path: str,
        constant_masks: List[str],
        normalization_dir: str,
        dt: int = 6,
        n_in_timesteps: int = 1,
        n_out_timesteps: int = 1,
        dtype: torch.dtype = torch.float32,
        dsync_access: bool = False,
        time_range: Tuple[str, str] | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        # Setup event list
        self.event_lead_list: list[tuple[Event, int]] = [
            (event, lead_time) for event in event_list for lead_time in lead_times
        ]

        super().__init__(
            pg_rank,
            pg_size,
            data_path,
            constant_masks_path,
            constant_masks,
            normalization_dir,
            dt,
            n_in_timesteps,
            n_out_timesteps,
            dtype,
            dsync_access,
            time_range,
            transform,
            target_transform,
        )

    def __len__(self) -> int:
        """Return the number of events times the number of lead times.

        Returns
        -------
        int
            The number of items.
        """
        return len(self.event_lead_list)

    def input_dates(self, start_date: datetime, lead_time: int) -> list[datetime]:
        """Return the input timestamps of one forecast, oldest first.

        The forecast is initialized at ``start_date - lead_time``. That is the
        last input, preceded by ``in_timesteps - 1`` earlier inputs spaced ``dt``
        hours apart.

        Parameters
        ----------
        start_date : datetime
            Start of the event being forecast.
        lead_time : int
            Lead time in hours. May be negative, which initializes the forecast
            after the event has started.

        Returns
        -------
        list[datetime]
            The ``in_timesteps`` input timestamps, in ascending order.
        """
        first_date = start_date - timedelta(
            hours=(lead_time + self.dt * (self.in_timesteps - 1))
        )
        return [
            first_date + timedelta(hours=i * self.dt) for i in range(self.in_timesteps)
        ]

    def __getitem__(self, idx: int) -> tuple[Event, int, list[torch.Tensor]]:  # type:ignore[override]
        """Return one event, its lead time and the input timesteps to forecast from.

        Parameters
        ----------
        idx : int
            Index of the item.

        Returns
        -------
        event : Event
            The event to forecast.
        lead_time : int
            Lead time of the forecast, in hours.
        inputs : list[torch.Tensor]
            The normalized input timesteps of this rank's shard, oldest first,
            each of shape ``(1, channel shard, xlat, longitude shard)``.
        """
        event, lead_time = self.event_lead_list[idx]

        samples = []
        for date in self.input_dates(event.start_date, lead_time):
            sample = (
                self.zarr_data.sel(time=date)
                .isel(
                    feature=slice(self.in_var[0], self.in_var[1]),
                    latitude=slice(0, self.xlat),
                    longitude=slice(self.in_out_xlon[0], self.in_out_xlon[1]),
                )["fields"]
                .values
            )

            sample = (
                sample
                - self.norm_values["mean"][self.in_var[0] : self.in_var[1]].reshape(
                    -1, 1, 1
                )
            ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]].reshape(
                -1, 1, 1
            )

            samples.append(torch.tensor(sample, dtype=self.dtype).unsqueeze(0))

        return event, lead_time, samples

    def list_from_output(
        self, output: torch.Tensor, feature_weights: torch.Tensor, fill: float
    ) -> list[torch.Tensor]:
        """Split the model output into timesteps and mask fixed values.

        Parameters
        ----------
        output : torch.Tensor
            Model output of shape ``(batch, channel shard * out_timesteps, xlat, longitude shard)``.
        feature_weights : torch.Tensor
            Loss weights, see :meth:`ERA5Distributed.get_feature_weights`.
            Output where they are 0 is replaced by ``fill``.
        fill : float
            Value for the masked output.

        Returns
        -------
        list[torch.Tensor]
            One tensor per output timestep, moved to the CPU so that long
            rollouts do not fill up GPU memory.
        """
        # Output shape: LSN, LBS, C, lat, lon

        target_list = []
        n_vars = self.in_var[1] - self.in_var[0]
        output = torch.where(feature_weights == 0, fill, output)
        for i in range(self.out_timesteps):
            target_list.append(
                output[:, n_vars * i : n_vars * (i + 1), :, :].contiguous().cpu()
            )

        return target_list
