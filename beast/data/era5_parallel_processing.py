import argparse
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
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

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
FEATURE_LIST = SURF_VARS + [var + "_" + str(p) for var in PLEV_VARS for p in LEVELS]


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
        if v == "sea_surface_temperature":
            a[np.isnan(a)] = 0
        s_arrays.append(a)

    # Pressure-level vars: (time, level, lat, lon) -> split by level
    for v in PLEV_VARS:
        a = subset[v].sel(level=LEVELS).values
        p_arrays.append(a)

    fields = np.stack(s_arrays, axis=1)
    final_list = [fields]
    final_list.extend(p_arrays)
    fields = np.concatenate(final_list, axis=1)

    # Explicitly copy coordinate values to avoid holding references to source dataset
    time_vals = subset.time.values.copy()
    lat_vals = subset.latitude.values.copy()
    lon_vals = subset.longitude.values.copy()

    # Clear intermediate arrays
    del s_arrays, p_arrays, final_list

    # Build minimal dataset
    out = xr.Dataset(
        data_vars={"fields": (["time", "feature", "latitude", "longitude"], fields)},
        coords=dict(
            time=time_vals,
            feature=FEATURE_LIST,
            latitude=lat_vals,
            longitude=lon_vals,
        ),
        attrs=dict(description="ERA5 data stacked into feature dimension."),
    )
    return out


def get_existing_dates(path_out) -> Set[str]:
    """
    Get the set of dates already present in the output zarr store.
    Memory-efficient version that doesn't iterate over all timestamps.
    """
    if not os.path.isdir(path_out):
        return set()

    try:
        # Open with chunks=None to avoid loading data, only metadata
        ds = xr.open_dataset(path_out, engine="zarr", consolidated=True, chunks=None)

        # Get time values more efficiently - convert to pandas and use vectorized operations
        times = pd.to_datetime(ds.time.values)
        # Use pandas' vectorized strftime which is much more memory efficient
        existing_dates = set(times.strftime("%Y-%m-%d").unique())

        ds.close()
        del ds, times
        gc.collect()

        return existing_dates
    except Exception as e:
        print(f"Warning: Could not read existing dates: {e}")
        return set()


def process_chunk_compute(
    path_to_data: str, start_date: datetime, end_date: datetime, chunk_id: int
):
    """
    Process a single time chunk: read and transform data.
    Opens its own dataset handle to avoid shared state issues.

    Args:
        path_to_data: path to source zarr store
        start_date: datetime for chunk start
        end_date: datetime for chunk end
        chunk_id: identifier for logging

    Returns
    -------
        Transformed xarray Dataset or None if processing failed
    """
    ds = None
    subset = None

    try:
        print(
            f"[Chunk {chunk_id}] Processing {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        # Each worker opens its own dataset handle - avoids shared cache issues
        ds = xr.open_dataset(path_to_data, consolidated=True, engine="zarr")

        # Read data for this chunk
        subset = ds.sel(
            time=slice(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        )

        # Check if we actually got data
        if len(subset.time) == 0:
            print(f"[Chunk {chunk_id}] WARNING: No data found for this time range")
            return None

        print(f"[Chunk {chunk_id}] Retrieved {len(subset.time)} time steps")

        # Transform data - this loads everything into memory
        dataset = stack_to_feature_dim(subset)
        print(f"[Chunk {chunk_id}] Transformation complete")

        return dataset

    except Exception as e:
        print(f"[Chunk {chunk_id}] ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        # Always clean up, even on exception
        if subset is not None:
            try:
                subset.close()
            except Exception:
                pass
            del subset
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass
            del ds
        gc.collect()


def write_out_parallel(
    start_date, end_date, path_out, n_threads=4, chunk_days=1, resume=False
):
    """
    Parallel version of write_out using ThreadPoolExecutor.
    Computes chunks in parallel but writes sequentially to maintain time order.

    FIXED: Memory leaks from unbounded futures list and shared dataset caching.
    """
    path_to_data = "/hkfs/work/workspace/scratch/xk5289-the_grand_bellbeast/data/era5_hourly_37levels.zarr"

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Check for existing dates if resuming
    if resume:
        print("Resume mode: checking for existing dates...")
        existing_dates = get_existing_dates(path_out)
        if existing_dates:
            print(f"Found {len(existing_dates)} existing dates in output")
            sorted_dates = sorted(existing_dates)
            print(f"Existing data range: {sorted_dates[0]} to {sorted_dates[-1]}")

            last_existing = datetime.strptime(sorted_dates[-1], "%Y-%m-%d")
            resume_start = last_existing + timedelta(days=1)

            if resume_start > start_date:
                print(
                    f"Adjusting start_date from {start_date.strftime('%Y-%m-%d')} to {resume_start.strftime('%Y-%m-%d')}"
                )
                start_date = resume_start

            # Free memory from date checking
            del existing_dates, sorted_dates
            gc.collect()
        else:
            print("No existing data found")

    # Prepare encoding
    enc = {"fields": {"chunks": (1, 84, 721, 1440)}}
    start_time = time.time()

    # Generate list of chunk parameters (lightweight - just dates and IDs)
    chunks = []
    current_start = start_date
    chunk_id = 0

    while current_start <= end_date:
        current_end = current_start + timedelta(days=chunk_days)
        if current_end > end_date:
            current_end = end_date
        else:
            current_end = current_end - timedelta(days=1)

        # Store only the essential parameters, not dataset references
        chunks.append((current_start, current_end, chunk_id))
        chunk_id += 1
        current_start = current_end + timedelta(days=1)

    if len(chunks) == 0:
        print("No chunks to process - all data already exists!")
        return

    print(f"\nTotal chunks to process: {len(chunks)}")
    print(f"Using {n_threads} threads for computation")

    store_exists = os.path.isdir(path_out)
    should_create_store = not (store_exists and resume)

    successful = 0
    failed = 0

    # Process first chunk separately
    if len(chunks) > 0:
        first_chunk = chunks[0]

        if should_create_store:
            print("\nProcessing first chunk to initialize store...")
        else:
            print("\nProcessing first chunk (appending to existing store)...")

        dataset = process_chunk_compute(
            path_to_data, first_chunk[0], first_chunk[1], first_chunk[2]
        )

        if dataset is not None:
            if should_create_store:
                dataset.to_zarr(
                    store=path_out,
                    mode="w",
                    zarr_format=2,
                    encoding=enc,
                    consolidated=True,
                )
                print(
                    f"[Chunk {first_chunk[2]}] Created zarr store with {len(dataset.time)} time steps"
                )
            else:
                dataset.to_zarr(
                    store=path_out,
                    mode="a",
                    zarr_format=2,
                    append_dim="time",
                    consolidated=True,
                )
                print(
                    f"[Chunk {first_chunk[2]}] Appended {len(dataset.time)} time steps to existing zarr store"
                )
            successful += 1

            dataset.close()
            del dataset
            gc.collect()
        else:
            failed += 1
            print("ERROR: First chunk failed. Aborting.")
            return

        # Process remaining chunks with bounded prefetch queue
        if len(chunks) > 1:
            print(
                "\nProcessing remaining chunks in parallel (with sequential writing)..."
            )

            remaining_chunks = chunks[1:]

            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # FIX: Use a bounded sliding window instead of unbounded list
                # Only keep n_threads futures in flight at a time

                pending_futures = {}  # chunk_id -> future
                next_write_idx = (
                    0  # Index into remaining_chunks for next chunk to write
                )
                next_submit_idx = (
                    0  # Index into remaining_chunks for next chunk to submit
                )

                # Submit initial batch
                initial_batch = min(n_threads, len(remaining_chunks))
                for i in range(initial_batch):
                    chunk = remaining_chunks[i]
                    future = executor.submit(
                        process_chunk_compute,
                        path_to_data,
                        chunk[0],
                        chunk[1],
                        chunk[2],
                    )
                    pending_futures[chunk[2]] = future
                    next_submit_idx = i + 1

                # Process in order
                while next_write_idx < len(remaining_chunks):
                    chunk = remaining_chunks[next_write_idx]
                    chunk_id = chunk[2]

                    # Wait for this specific chunk
                    if chunk_id not in pending_futures:
                        # Submit it if not already submitted
                        future = executor.submit(
                            process_chunk_compute,
                            path_to_data,
                            chunk[0],
                            chunk[1],
                            chunk[2],
                        )
                        pending_futures[chunk_id] = future

                    future = pending_futures[chunk_id]

                    try:
                        dataset = future.result()

                        # CRITICAL: Remove future from dict immediately after getting result
                        del pending_futures[chunk_id]

                        if dataset is not None:
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
                            successful += 1

                            dataset.close()
                            del dataset
                        else:
                            failed += 1

                    except Exception as e:
                        print(f"[Chunk {chunk_id}] Unexpected error: {str(e)}")
                        import traceback

                        traceback.print_exc()
                        failed += 1

                        # Remove failed future
                        if chunk_id in pending_futures:
                            del pending_futures[chunk_id]

                    next_write_idx += 1

                    # Submit next chunk to keep pipeline full
                    if next_submit_idx < len(remaining_chunks):
                        next_chunk = remaining_chunks[next_submit_idx]
                        if next_chunk[2] not in pending_futures:
                            future = executor.submit(
                                process_chunk_compute,
                                path_to_data,
                                next_chunk[0],
                                next_chunk[1],
                                next_chunk[2],
                            )
                            pending_futures[next_chunk[2]] = future
                        next_submit_idx += 1

                    # Periodic aggressive garbage collection
                    if next_write_idx % 10 == 0:
                        gc.collect()
                        # Log memory status periodically
                        if next_write_idx % 100 == 0:
                            try:
                                import psutil

                                process = psutil.Process()
                                mem_gb = process.memory_info().rss / (1024**3)
                                print(
                                    f"[Memory] After {next_write_idx} chunks: {mem_gb:.2f} GB RSS"
                                )
                            except ImportError:
                                pass

    end_time = time.time()
    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"Successful chunks: {successful}/{len(chunks)}")
    print(f"Failed chunks: {failed}/{len(chunks)}")
    print(f"Processed {successful} in {(end_time - start_time) / 60} [min]")

    # Verify the output
    if os.path.isdir(path_out):
        print("\nVerifying output...")
        try:
            ds_out = xr.open_dataset(path_out, engine="zarr", consolidated=True)
            print(f"Output dimensions: {dict(ds_out.dims)}")
            print(f"Time range: {ds_out.time.values[0]} to {ds_out.time.values[-1]}")
            print(f"Total time steps: {len(ds_out.time)}")

            times = pd.DatetimeIndex(ds_out.time.values)
            time_diffs = times[1:] - times[:-1]
            unique_diffs = time_diffs.unique()
            print(f"Time step intervals: {unique_diffs}")

            is_sorted = (times[1:] >= times[:-1]).all()
            if is_sorted:
                print("✓ Time dimension is properly sorted")
            else:
                print("✗ WARNING: Time dimension is NOT sorted!")

            ds_out.close()
        except Exception as e:
            print(f"Warning: Could not verify output: {e}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel ERA5 data processing with time-ordered writing"
    )
    parser.add_argument(
        "--start_date", default="1979-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", default="2023-01-01", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--path_out",
        default="/lsdf/kit/scc/projects/bellbeast/era5_stackedvars_test.zarr",
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
