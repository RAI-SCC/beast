import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def check_variable(variable: str, file_out: Path, start_date: str, end_date: str):
    invalid_times = []
    path_to_data = "/hkfs/work/workspace/scratch/xk5289-the_grand_bellbeast/data/era5_hourly_37levels.zarr"

    ds = xr.open_dataset(path_to_data, engine="zarr")
    start_date, end_date = (
        datetime.datetime.strptime(start_date, "%Y-%m-%d"),
        datetime.datetime.strptime(end_date, "%Y-%m-%d"),
    )

    ds = ds[variable]  # filter by variable
    times = pd.to_datetime(ds["time"].values)
    filtered_times = times[(times < end_date) & (times >= start_date)]
    ds = ds.sel(time=filtered_times)

    # go through times
    for index, t in enumerate(filtered_times):
        if index % 100 == 0:
            print(
                f"Currently at index {index}, time {t}\nInvalid times: {invalid_times}"
            )
        ds_i = ds.sel(time=t)
        tmp = ds_i.values
        if np.isnan(np.dot(tmp.flatten(), tmp.flatten())):
            invalid_times.append(t)

    with open(f"{file_out}", "a") as f:
        f.write(f"From {start_date} to {end_date}: invalid times: \n{invalid_times}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable")
    parser.add_argument("--path_out")
    parser.add_argument("--start_date", default="1979-01-01")
    parser.add_argument("--end_date", default="2023-01-01")
    # parser.add_argument("--n_threads", default=)
    args = parser.parse_args()
    check_variable(
        variable=args.variable,
        file_out=args.path_out,
        start_date=args.start_date,
        end_date=args.end_date,
    )
