import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def check_variable(data_path: Path, file_out: Path, start_date: str, end_date: str):
    invalid_times = []
    invalid_indices = []
    dataset_path = os.getenv("DATA_DIR") + data_path
    ds = xr.open_dataset(dataset_path, engine="zarr")
    start_date, end_date = (
        datetime.datetime.strptime(start_date, "%Y-%m-%d"),
        datetime.datetime.strptime(end_date, "%Y-%m-%d"),
    )

    baseline_date = datetime.datetime.strptime("1979-01-01-00", "%Y-%m-%d-%H")

    ds = ds["fields"]  # filter by variable
    times = pd.to_datetime(ds["time"].values)
    filtered_times = times[(times < end_date) & (times >= start_date)]
    ds = ds.sel(time=filtered_times)

    # go through times
    for index, t in enumerate(filtered_times):
        if index % 10 == 0:
            print(
                f"Currently at index {index}, time {t}\nInvalid times: {invalid_times}\nInvalid indices {invalid_indices}"
            )
        ds_i = ds.sel(time=t)
        tmp = ds_i.values
        if np.isnan(np.dot(tmp.flatten(), tmp.flatten())):
            invalid_indices.append(((t - baseline_date).total_seconds()) // 3600)
            invalid_times.append(t)

    with open(f"{file_out}", "a") as f:
        f.write(f"From {start_date} to {end_date}: invalid times: \n{invalid_times}")
        f.write(
            f"From {start_date} to {end_date}: invalid indices: \n{invalid_indices}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="/scratch/project_462001248/data/era5_preprocessed.zarr"
    )
    parser.add_argument("--path_out")
    parser.add_argument("--start_date", default="1979-01-01")
    parser.add_argument("--end_date", default="2023-01-01")
    args = parser.parse_args()
    check_variable(
        data_path=args.data_path,
        file_out=args.path_out,
        start_date=args.start_date,
        end_date=args.end_date,
    )
