import argparse
import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
import xarray as xr

# Surface and pressure-level variables to stack
SURF_VARS: List[str] = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "sea_surface_temperature",
    "total_precipitation",
]
PLEV_VARS: List[str] = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]


def stack_to_feature_dim(subset: xr.Dataset) -> xr.Dataset:
    """
    Stack selected variables into a single DataArray named 'fields' with shape:
      fields: (time, feature, latitude, longitude)
    Pressure-level variables are expanded per level as independent features.
    Returns a Dataset with coords: time, feature, latitude, longitude.
    """
    s_arrays = []
    p_arrays = []

    # Surface vars: (time, lat, lon) -> add feature=var
    for v in SURF_VARS:
        a = subset[v].values
        s_arrays.append(a)  # (time, feature=1, latitude, longitude)

    # Pressure-level vars: (time, level, lat, lon) -> split by level, add feature=f"{var}|level={lev}"
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    for v in PLEV_VARS:
        # for lev in levels:
        a = subset[v].sel(level=levels).values  # (time, latitude, longitude)
        p_arrays.append(a)

    fields = np.stack(s_arrays, axis=1)
    final_list = [fields]
    final_list.extend(p_arrays)  # list:
    fields = np.concatenate(final_list, axis=1)

    fields = {"fields": (["time", "feature", "latitude", "longitude"], fields)}
    feature_list = SURF_VARS + [var + "_" + str(p) for var in PLEV_VARS for p in levels]
    # Build minimal dataset
    out = xr.Dataset(
        data_vars=fields,
        coords=dict(
            time=subset.time.values,
            feature=feature_list,
            latitude=subset.latitude,
            longitude=subset.longitude,
        ),
        attrs=dict(description="ERA5 data stacked into feature dimension."),
    )
    return out


def write_out(start_date, end_date, path_out):
    path_to_data = "/hkfs/work/workspace/scratch/xk5289-the_grand_bellbeast/data/era5_hourly_37levels.zarr"
    ds = xr.open_dataset(path_to_data, consolidated=True, engine="zarr")

    start_date, end_date = (
        datetime.strptime(start_date, "%Y-%m-%d"),
        datetime.strptime(end_date, "%Y-%m-%d"),
    )
    # Download 1 days of data at a time
    dt = 1
    current_date = start_date + timedelta(days=(dt - 1))
    print(f"Downloading data from {start_date} to {current_date}")

    # Prepare encoding once using the first slice to get feature count and grid sizes
    enc = {"fields": {"chunks": (1, 84, 721, 1440)}}

    # Change path out

    while current_date < end_date:  # download 10 days of data
        print(
            f"Downloading {(start_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))}"
        )

        subset = ds.sel(
            time=slice(
                start_date.strftime("%Y-%m-%d"), current_date.strftime("%Y-%m-%d")
            )
        )
        dataset = stack_to_feature_dim(subset)

        if not os.path.isdir(path_out):
            dataset.to_zarr(
                store=path_out, mode="w", zarr_format=2, encoding=enc, consolidated=True
            )
        else:
            dataset.to_zarr(
                store=path_out,
                mode="a",
                zarr_format=2,
                append_dim="time",
                consolidated=True,
            )

        start_date += timedelta(days=dt)
        current_date += timedelta(days=dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", default="1979-01-01")
    parser.add_argument("--end_date", default="2023-01-01")
    parser.add_argument(
        "--path_out",
        default="/hkfs/work/workspace/scratch/ke4365-jigsaw/GrandBellBeast/data/era5_stackedvars_test.zarr",
    )

    # parser.add_argument("--n_threads", default=)
    args = parser.parse_args()
    write_out(
        start_date=args.start_date, end_date=args.end_date, path_out=args.path_out
    )
