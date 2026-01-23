#!/usr/bin/env python3
"""
Generate SLURM job scripts for parallel ERA5 processing.
Creates one job per year (or other time period) for embarrassingly parallel execution.
"""

import argparse
import os

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=era5_{year}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
#SBATCH --output=logs/era5_{year}_%j.out
#SBATCH --error=logs/era5_{year}_%j.err


# Activate conda environment
source activate /hkfs/work/workspace/scratch/ke4365-jigsaw/Jigsaw/.venv/bin/activate

# Run processing for this year
python era5_parallel_processing.py \\
    --start_date {start_date} \\
    --end_date {end_date} \\
    --path_out {output_path} \\
    --n_threads {n_threads} \\
    --chunk_days {chunk_days}

echo "Job completed for year {year}"
"""


def generate_job_scripts(
    start_year,
    end_year,
    output_base_dir,
    scripts_dir="slurm_scripts",
    partition="cpuonly,",
    cpus=16,
    memory="128G",
    time_limit="24:00:00",
    n_threads=16,
    chunk_days=1,
):
    """
    Generate SLURM job scripts for parallel processing by year.

    Args:
        start_year: First year to process
        end_year: Last year to process
        output_base_dir: Base directory for output zarr stores
        scripts_dir: Directory to save job scripts
        partition: SLURM partition name
        cpus: CPUs per task
        memory: Memory allocation
        time_limit: Job time limit
        n_threads: Number of threads per job
        chunk_days: Days per processing chunk
    """
    # Create directories
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    job_scripts = []

    for year in range(start_year, end_year + 1):
        # Define date range for this year
        start_date = f"{year}-01-01"
        end_date = f"{year}-01-31"

        # Output path for this year
        output_path = os.path.join(output_base_dir, f"era5_preprocessed_{year}.zarr")

        # Create job script from template
        job_content = SLURM_TEMPLATE.format(
            year=year,
            partition=partition,
            cpus=cpus,
            memory=memory,
            time_limit=time_limit,
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            n_threads=n_threads,
            chunk_days=chunk_days,
        )

        # Write job script
        script_path = os.path.join(scripts_dir, f"process_{year}.sh")
        with open(script_path, "w") as f:
            f.write(job_content)

        os.chmod(script_path, 0o755)  # Make executable
        job_scripts.append(script_path)

        print(f"Created job script: {script_path}")

    # Create master submission script
    master_script = os.path.join(scripts_dir, "submit_all.sh")
    with open(master_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all ERA5 processing jobs\n\n")
        for script in job_scripts:
            f.write(f"sbatch {script}\n")

    os.chmod(master_script, 0o755)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(job_scripts)} job scripts")
    print(f"Master submission script: {master_script}")
    print("\nTo submit all jobs, run:")
    print(f"  bash {master_script}")
    print("\nOr submit individually:")
    print(f"  sbatch {scripts_dir}/process_<year>.sh")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SLURM job scripts for parallel ERA5 processing"
    )
    parser.add_argument(
        "--start_year", type=int, default=1979, help="First year to process"
    )
    parser.add_argument(
        "--end_year", type=int, default=2023, help="Last year to process"
    )
    parser.add_argument(
        "--output_base_dir",
        default="/hkfs/work/workspace/scratch/ke4365-jigsaw/GrandBellBeast/data/era5_yearly",
        help="Base directory for output zarr stores",
    )
    parser.add_argument(
        "--scripts_dir", default="slurm_scripts", help="Directory for job scripts"
    )
    parser.add_argument("--partition", default="gpu_4", help="SLURM partition")
    parser.add_argument("--cpus", type=int, default=16, help="CPUs per task")
    parser.add_argument("--memory", default="128G", help="Memory allocation")
    parser.add_argument("--time_limit", default="24:00:00", help="Job time limit")
    parser.add_argument("--n_threads", type=int, default=8, help="Threads per job")
    parser.add_argument("--chunk_days", type=int, default=1, help="Days per chunk")

    args = parser.parse_args()

    generate_job_scripts(
        start_year=args.start_year,
        end_year=args.end_year,
        output_base_dir=args.output_base_dir,
        scripts_dir=args.scripts_dir,
        partition=args.partition,
        cpus=args.cpus,
        memory=args.memory,
        time_limit=args.time_limit,
        n_threads=args.n_threads,
        chunk_days=args.chunk_days,
    )
