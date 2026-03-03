#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# All variables in desired order.
ORDERED_VARIABLES = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "sea_surface_temperature",
    "total_precipitation",
    "geopotential_1000",
    "geopotential_925",
    "geopotential_850",
    "geopotential_700",
    "geopotential_600",
    "geopotential_500",
    "geopotential_400",
    "geopotential_300",
    "geopotential_250",
    "geopotential_200",
    "geopotential_150",
    "geopotential_100",
    "geopotential_50",
    "specific_humidity_1000",
    "specific_humidity_925",
    "specific_humidity_850",
    "specific_humidity_700",
    "specific_humidity_600",
    "specific_humidity_500",
    "specific_humidity_400",
    "specific_humidity_300",
    "specific_humidity_250",
    "specific_humidity_200",
    "specific_humidity_150",
    "specific_humidity_100",
    "specific_humidity_50",
    "temperature_1000",
    "temperature_925",
    "temperature_850",
    "temperature_700",
    "temperature_600",
    "temperature_500",
    "temperature_400",
    "temperature_300",
    "temperature_250",
    "temperature_200",
    "temperature_150",
    "temperature_100",
    "temperature_50",
    "u_component_of_wind_1000",
    "u_component_of_wind_925",
    "u_component_of_wind_850",
    "u_component_of_wind_700",
    "u_component_of_wind_600",
    "u_component_of_wind_500",
    "u_component_of_wind_400",
    "u_component_of_wind_300",
    "u_component_of_wind_250",
    "u_component_of_wind_200",
    "u_component_of_wind_150",
    "u_component_of_wind_100",
    "u_component_of_wind_50",
    "v_component_of_wind_1000",
    "v_component_of_wind_925",
    "v_component_of_wind_850",
    "v_component_of_wind_700",
    "v_component_of_wind_600",
    "v_component_of_wind_500",
    "v_component_of_wind_400",
    "v_component_of_wind_300",
    "v_component_of_wind_250",
    "v_component_of_wind_200",
    "v_component_of_wind_150",
    "v_component_of_wind_100",
    "v_component_of_wind_50",
    "vertical_velocity_1000",
    "vertical_velocity_925",
    "vertical_velocity_850",
    "vertical_velocity_700",
    "vertical_velocity_600",
    "vertical_velocity_500",
    "vertical_velocity_400",
    "vertical_velocity_300",
    "vertical_velocity_250",
    "vertical_velocity_200",
    "vertical_velocity_150",
    "vertical_velocity_100",
    "vertical_velocity_50",
]


def load_checkpoint(
    checkpoint_path: Path, num_variables: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load the checkpoint to resume calculation.

    Args:
        checkpoint_path (Path): Path to checkpoint file.
        num_variables (int): Number of variables to load.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: Tuple of current sum, squared sum, count, min, max, and index.
    """
    actual_path = (
        checkpoint_path.with_suffix(".npz")
        if checkpoint_path.suffix != ".npz"
        else checkpoint_path
    )

    if actual_path.exists():
        print(f"Loading checkpoint from {actual_path}. Resuming...")
        data = np.load(actual_path)
        return (
            data["sum"],
            data["squared_sum"],
            data["count"],
            data["min"],
            data["max"],
            int(data["idx"]),
        )
    else:
        print("No checkpoint found. Starting from scratch.")
        return (
            np.zeros(num_variables, dtype=np.float64),
            np.zeros(num_variables, dtype=np.float64),
            np.zeros(num_variables, dtype=np.float64),
            np.full(num_variables, np.inf),
            np.full(num_variables, -np.inf),
            0,
        )


def save_checkpoint(
    checkpoint_path: Path,
    running_sum: np.ndarray,
    running_squared_sum: np.ndarray,
    running_count: np.ndarray,
    running_min: np.ndarray,
    running_max: np.ndarray,
    idx: int,
) -> None:
    """Saves current processing progress.

    Args:
        checkpoint_path (Path): Path to checkpoint file.
        running_sum (np.ndarray): Current running sum.
        running_squared_sum (np.ndarray): Current running squared sum.
        running_count (np.ndarray): Current running count.
        running_min (np.ndarray): Current running min.
        running_max (np.ndarray): Current running max.
        idx (int): Current chunk index.
    """
    np.savez(
        checkpoint_path,
        sum=running_sum,
        squared_sum=running_squared_sum,
        count=running_count,
        min=running_min,
        max=running_max,
        idx=idx,
    )


def calculate_and_save_normalization_values(
    output_path: str,
    data_path: str,
    checkpoint_path: str | None = None,
    save_interval: int = 50,
    sampling_step: int = 1,
) -> None:
    """Calculate mean and standard deviation for each variable and save to Xarray dataset.

    Args
        output_path (str): Path to save the normalization values.
        data_path (str): Path to the ERA5 zarr file.
        checkpoint_path (str, optional): Path to checkpoint file, if not provided then no checkpoint is used.
        save_interval (int, optional): Interval to save normalization values to checkpoint.
        sampling_step (int, optional): Temporal sampling step, sample only each "sampling_step" hour to speed up.
    """
    print(f"Starting normalization calculation. Values will be saved to {output_path}")

    # Convert to path.
    output_path = Path(output_path)
    data_path = Path(data_path)

    # Handle checkpoint.
    if checkpoint_path is None:
        resume = False
    else:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.suffix != ".npz":
            checkpoint_path = checkpoint_path.with_suffix(".npz")
        resume = True

    # Check and create directory for saving values.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy load dataset.
    ds = xr.open_zarr(data_path)

    # Save variable names.
    variables = ds.coords["feature"].values
    num_variables = len(variables)
    num_time = ds.sizes["time"]

    # Initialize empty arrays.
    (
        running_sum,
        running_squared_sum,
        running_count,
        running_min,
        running_max,
        start_idx,
    ) = (
        np.zeros(num_variables, dtype=np.float64),
        np.zeros(num_variables, dtype=np.float64),
        np.zeros(num_variables, dtype=np.float64),
        np.full(num_variables, np.inf),
        np.full(num_variables, -np.inf),
        0,
    )

    # Calculate sea surface temperature index for special handling.
    sea_surface_index = np.where(variables == "sea_surface_temperature")[0][0]

    # Load data from checkpoint if resuming from a crash.
    if resume:
        (
            running_sum,
            running_squared_sum,
            running_count,
            running_min,
            running_max,
            start_idx,
        ) = load_checkpoint(checkpoint_path, num_variables)

    # Calculate steps for pretty tqdm display.
    total_steps = (num_time + sampling_step - 1) // sampling_step
    initial_steps = start_idx // sampling_step

    # Start calculating.
    print(
        f"Calculating normalization values with step size of {sampling_step}. Therefore we have {total_steps} samples in total!"
    )
    print(
        "Starting normalization calculation. This will take a long time - go grab a coffee and chill!\n"
    )

    for i in tqdm(
        range(start_idx, num_time, sampling_step),
        initial=initial_steps,
        total=total_steps,
    ):
        # Access data.
        data_slice = ds["fields"].isel(time=i).compute().values
        data_slice = data_slice.astype(np.float64)

        # Special handling for sea temperature since zeros indicate missing data and should not be considered.
        # Turn zeros back to nans, so that nansum ignores them.
        data_slice[sea_surface_index][data_slice[sea_surface_index] == 0] = np.nan

        # Statistics for each time chunk - collapsing latitude and longitude
        chunk_sum = np.nansum(data_slice, axis=(1, 2))
        chunk_squared_sum = np.nansum(data_slice**2, axis=(1, 2))
        chunk_count = np.sum(~np.isnan(data_slice), axis=(1, 2))
        chunk_min = np.nanmin(data_slice, axis=(1, 2))
        chunk_max = np.nanmax(data_slice, axis=(1, 2))

        running_sum += chunk_sum
        running_squared_sum += chunk_squared_sum
        running_count += chunk_count
        running_min = np.fmin(running_min, chunk_min)
        running_max = np.fmax(running_max, chunk_max)

        del data_slice

        # Save progress.
        if (
            resume
            and (i - start_idx) > 0
            and (i - start_idx) % (save_interval * sampling_step) == 0
        ):
            save_checkpoint(
                checkpoint_path,
                running_sum,
                running_squared_sum,
                running_count,
                running_min,
                running_max,
                i + sampling_step,
            )

    # Calculate man and standard deviation.
    mean = running_sum / running_count
    var = (running_squared_sum / running_count) - (mean**2)
    std = np.sqrt(np.maximum(0, var))

    print("Computation finished. Packing into NetCDF...")

    # Create Xarray Dataset for saving.
    ds_statistics = xr.Dataset(
        data_vars={
            "mean": (["feature"], mean.astype(np.float32)),
            "std": (["feature"], std.astype(np.float32)),
            "min": (["feature"], running_min.astype(np.float32)),
            "max": (["feature"], running_max.astype(np.float32)),
        },
        coords={"feature": variables},
    )

    # Sanity check to make sure the order is correct - this shouldn't cause any errors!
    try:
        ds_ordered = ds_statistics.sel(feature=ORDERED_VARIABLES)
        print("Yay - the correct order was achieved!.")
    except KeyError as e:
        print(
            "We have a problem - the names in the requested list do not match the names in the Zarr file!"
        )
        print(f"The first missing variable is: {e}")
        print("We will save with the original order in order to not lose the data :)")
        ds_ordered = ds_statistics

    # Save to NetCDF.
    ds_ordered.to_netcdf(output_path)

    # Clean up checkpoint.
    if resume and checkpoint_path.exists():
        print(f"Removing checkpoint file: {checkpoint_path}")
        checkpoint_path.unlink()

    print(f"Done. Normalization statistics saved to {output_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-d",
        required=True,
        type=str,
        help="Path to ERA5 Zarr file containing the data.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        required=True,
        type=str,
        help="Path to save the normalization information as NETCDF file, e.g. your/path/here/file.nc",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-c",
        required=True,
        type=str,
        help="Path to save checkpoint file, e.g. your/path/here/checkpoint",
    )
    parser.add_argument(
        "--save_interval",
        "-i",
        default=48,
        type=int,
        help="Interval for frequency of saving to checkpoint file.",
    )
    parser.add_argument(
        "--sampling_step",
        "-s",
        default=1,
        type=int,
        help="Temporal sampling step - i.e. sample only ever s-th hour. Speeds up computation.",
    )

    args = parser.parse_args()
    calculate_and_save_normalization_values(
        output_path=args.output_path,
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        save_interval=args.save_interval,
        sampling_step=args.sampling_step,
    )
