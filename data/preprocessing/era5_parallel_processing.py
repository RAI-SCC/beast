import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Set

import numpy as np
import pandas as pd
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

# Thread-safe lock for zarr writing
write_lock = threading.Lock()


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
        a = subset[v].sel(level=levels).values  # (time, latitude, longitude)
        p_arrays.append(a)

    fields = np.stack(s_arrays, axis=1)
    final_list = [fields]
    final_list.extend(p_arrays)
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


def get_existing_dates(path_out) -> Set[str]:
    """
    Get the set of dates already present in the output zarr store.

    Args:
        path_out: path to zarr store

    Returns:
        Set of date strings in 'YYYY-MM-DD' format
    """
    if not os.path.isdir(path_out):
        return set()

    try:
        ds = xr.open_dataset(path_out, engine="zarr", consolidated=True)
        # Convert times to date strings
        existing_dates = set()
        for t in ds.time.values:
            date_str = pd.Timestamp(t).strftime("%Y-%m-%d")
            existing_dates.add(date_str)
        ds.close()
        return existing_dates
    except Exception as e:
        print(f"Warning: Could not read existing dates: {e}")
        return set()


def process_chunk(ds, start_date, end_date, path_out, enc, chunk_id, is_first_chunk):
    """
    Process a single time chunk: read, transform, and write to zarr.

    Args:
        ds: xarray dataset (opened once, shared across threads)
        start_date: datetime for chunk start
        end_date: datetime for chunk end
        path_out: output zarr path
        enc: encoding dict for zarr
        chunk_id: identifier for logging
        is_first_chunk: whether this is the first chunk to write
    """
    try:
        print(
            f"[Chunk {chunk_id}] Processing {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        # Read data for this chunk
        subset = ds.sel(
            time=slice(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        )

        # Check if we actually got data
        if len(subset.time) == 0:
            print(f"[Chunk {chunk_id}] WARNING: No data found for this time range")
            return False

        print(f"[Chunk {chunk_id}] Retrieved {len(subset.time)} time steps")

        # Transform data
        dataset = stack_to_feature_dim(subset)

        # Write to zarr (thread-safe)
        with write_lock:
            if is_first_chunk and not os.path.isdir(path_out):
                # First chunk creates the store
                dataset.to_zarr(
                    store=path_out,
                    mode="w",
                    zarr_format=2,
                    encoding=enc,
                    consolidated=True,
                )
                print(
                    f"[Chunk {chunk_id}] Created zarr store with {len(dataset.time)} time steps"
                )
            else:
                # Subsequent chunks append
                dataset.to_zarr(
                    store=path_out,
                    mode="a",
                    zarr_format=2,
                    append_dim="time",
                    consolidated=True,
                )
                print(
                    f"[Chunk {chunk_id}] Appended {len(dataset.time)} time steps to zarr store"
                )

        print(f"[Chunk {chunk_id}] Completed successfully")
        return True

    except Exception as e:
        print(f"[Chunk {chunk_id}] ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def write_out_parallel(
    start_date, end_date, path_out, n_threads=4, chunk_days=1, resume=False
):
    """
    Parallel version of write_out using ThreadPoolExecutor.

    Args:
        start_date: start date string (YYYY-MM-DD)
        end_date: end date string (YYYY-MM-DD)
        path_out: output zarr path
        n_threads: number of parallel threads
        chunk_days: number of days per chunk (default 1)
        resume: if True, skip dates that already exist in output
    """
    path_to_data = "/hkfs/work/workspace/scratch/xk5289-the_grand_bellbeast/data/era5_hourly_37levels.zarr"

    # Open dataset once (will be shared across threads - xarray is thread-safe for reads)
    print(f"Opening dataset from {path_to_data}")
    ds = xr.open_dataset(path_to_data, consolidated=True, engine="zarr")

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Check for existing dates if resuming
    existing_dates = set()
    if resume:
        print("Resume mode: checking for existing dates...")
        existing_dates = get_existing_dates(path_out)
        if existing_dates:
            print(f"Found {len(existing_dates)} existing dates in output")
            # Show date range of existing data
            sorted_dates = sorted(existing_dates)
            print(f"Existing data range: {sorted_dates[0]} to {sorted_dates[-1]}")
        else:
            print("No existing data found")

    # Prepare encoding
    enc = {"fields": {"chunks": (1, 84, 721, 1440)}}

    # Generate list of chunks to process
    chunks = []
    current_start = start_date
    chunk_id = 0
    skipped_chunks = 0

    while current_start <= end_date:
        current_end = current_start + timedelta(days=chunk_days)
        # FIX: Use min to avoid going past end_date
        if current_end > end_date:
            current_end = end_date
        else:
            # Subtract 1 day since we want chunk_days of data
            current_end = current_end - timedelta(days=1)

        # Check if we should skip this chunk (resume mode)
        if resume and existing_dates:
            # Check if all dates in this chunk already exist
            chunk_dates = []
            check_date = current_start
            while check_date <= current_end:
                chunk_dates.append(check_date.strftime("%Y-%m-%d"))
                check_date += timedelta(days=1)

            # If all dates exist, skip this chunk
            if all(d in existing_dates for d in chunk_dates):
                print(
                    f"[Chunk {chunk_id}] Skipping {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')} (already exists)"
                )
                skipped_chunks += 1
                chunk_id += 1
                current_start = current_end + timedelta(days=1)
                continue

        is_first_chunk = len(chunks) == 0
        chunks.append(
            (ds, current_start, current_end, path_out, enc, chunk_id, is_first_chunk)
        )
        chunk_id += 1
        current_start = current_end + timedelta(days=1)

    if skipped_chunks > 0:
        print(f"Skipped {skipped_chunks} chunks that already exist")

    if len(chunks) == 0:
        print("No chunks to process - all data already exists!")
        return

    print(f"\nTotal chunks to process: {len(chunks)}")
    print(f"Using {n_threads} threads")

    # Process chunks in ORDER to maintain time sequence
    # Using a sequential approach for the first chunk, then parallel for the rest
    successful = 0
    failed = 0

    if len(chunks) > 0:
        # Process first chunk sequentially to create the store
        first_chunk = chunks[0]
        print("\nProcessing first chunk sequentially to initialize store...")
        result = process_chunk(*first_chunk)
        if result:
            successful += 1
        else:
            failed += 1
            print("ERROR: First chunk failed. Aborting.")
            return

        # Process remaining chunks in parallel
        if len(chunks) > 1:
            print("\nProcessing remaining chunks in parallel...")
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Submit remaining tasks
                future_to_chunk = {
                    executor.submit(process_chunk, *chunk_args): chunk_args[5]
                    for chunk_args in chunks[1:]
                }

                # Process completed tasks
                for future in as_completed(future_to_chunk):
                    chunk_id = future_to_chunk[future]
                    try:
                        result = future.result()
                        if result:
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"[Chunk {chunk_id}] Unexpected error: {str(e)}")
                        failed += 1

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"Successful chunks: {successful}/{len(chunks)}")
    print(f"Failed chunks: {failed}/{len(chunks)}")

    # Verify the output
    if os.path.isdir(path_out):
        print("\nVerifying output...")
        try:
            ds_out = xr.open_dataset(path_out, engine="zarr", consolidated=True)
            print(f"Output dimensions: {dict(ds_out.dims)}")
            print(f"Time range: {ds_out.time.values[0]} to {ds_out.time.values[-1]}")
            print(f"Total time steps: {len(ds_out.time)}")
            ds_out.close()
        except Exception as e:
            print(f"Warning: Could not verify output: {e}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel ERA5 data processing with multithreading (FIXED VERSION)"
    )
    parser.add_argument(
        "--start_date", default="1979-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", default="2023-01-01", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--path_out",
        default="/hkfs/work/workspace/scratch/ke4365-jigsaw/GrandBellBeast/data/era5_stackedvars_test.zarr",
        help="Output zarr path",
    )
    parser.add_argument(
        "--n_threads", type=int, default=4, help="Number of parallel threads"
    )
    parser.add_argument(
        "--chunk_days", type=int, default=1, help="Number of days per chunk"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip already processed dates)",
    )

    args = parser.parse_args()

    write_out_parallel(
        start_date=args.start_date,
        end_date=args.end_date,
        path_out=args.path_out,
        n_threads=args.n_threads,
        chunk_days=args.chunk_days,
        resume=args.resume,
    )
