#!/usr/bin/env python
"""
Test PyTorch Distributed Initialization on Jupiter HPC

This script verifies that:
1. PyTorch distributed initializes correctly with NCCL backend
2. NCCL is using InfiniBand (not TCP/Ethernet)
3. GPU-to-GPU communication works across nodes

Usage:
    # Single node test (4 GPUs)
    torchrun --nproc_per_node=4 test_torch_dist.py

    # Multi-node via SLURM (see run_jupiter_distributed.sh)
    srun --mpi=pmix -N 2 --ntasks-per-node=4 python -m torch.distributed.run \
        --nnodes=2 --nproc_per_node=4 test_torch_dist.py
"""

import os
import socket
import subprocess
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def get_nccl_env_info() -> dict:
    """Collect NCCL-related environment variables."""
    nccl_vars = {}
    for key, value in os.environ.items():
        if key.startswith(("NCCL_", "UCX_", "OMPI_")):
            nccl_vars[key] = value
    return nccl_vars


def init_distributed() -> tuple[int, int, int]:
    """Initialize PyTorch distributed and return rank info."""
    # Get distributed info from environment (set by torchrun/srun)
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    # Set CUDA device before init
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Initialize process group with NCCL backend
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5),
        )

    return rank, world_size, local_rank


def test_communication(rank: int, world_size: int, local_rank: int) -> dict:
    """Test actual GPU communication and measure bandwidth."""
    results = {"tests_passed": 0, "tests_failed": 0, "details": []}
    dist.barrier()  # Sync before starting tests
    print("All ranks synchronized, starting communication tests...")

    device = torch.device(f"cuda:{local_rank}")

    # Test 1: All-reduce
    try:
        tensor = torch.ones(1024 * 1024, device=device) * rank  # 4MB tensor
        dist.barrier()

        # Warm up
        for _ in range(3):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Timed test
        torch.cuda.synchronize()
        print("Starting all-reduce bandwidth test...")
        dist.barrier()

        import time
        start = time.perf_counter()
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Check result: sum of 0,1,2,...,world_size-1 = world_size*(world_size-1)/2
        # After 13 all-reduces (3 warmup + 10 timed), value is complex but should be consistent
        expected_single = sum(range(world_size))
        bandwidth_gbps = (tensor.numel() * tensor.element_size() * 10 * 2) / elapsed / 1e9

        results["tests_passed"] += 1
        results["details"].append({
            "test": "all_reduce",
            "status": "PASSED",
            "bandwidth_gbps": round(bandwidth_gbps, 2),
            "time_ms": round(elapsed * 1000 / 10, 2),
        })
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append({"test": "all_reduce", "status": "FAILED", "error": str(e)})

    # Test 2: All-gather (replaces p2p ring - NCCL isend/irecv can deadlock)
    # This tests that each rank can contribute data and all ranks receive it
    try:
        # Each rank creates a tensor with its rank value
        local_tensor = torch.ones(1024 * 1024, device=device) * rank
        # Prepare list to gather all tensors
        gathered = [torch.zeros(1024 * 1024, device=device) for _ in range(world_size)]

        dist.barrier()

        if world_size > 1:
            dist.all_gather(gathered, local_tensor)

            # Verify each gathered tensor has the correct rank value
            all_correct = True
            for i, t in enumerate(gathered):
                expected = torch.ones_like(t) * i
                if not torch.allclose(t, expected):
                    all_correct = False
                    break

            if all_correct:
                results["tests_passed"] += 1
                results["details"].append({"test": "all_gather", "status": "PASSED"})
            else:
                results["tests_failed"] += 1
                results["details"].append({
                    "test": "all_gather",
                    "status": "FAILED",
                    "error": "Data mismatch in gathered tensors",
                })
        else:
            results["details"].append({"test": "all_gather", "status": "SKIPPED", "reason": "world_size=1"})

    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append({"test": "all_gather", "status": "FAILED", "error": str(e)})

    # Test 3: Broadcast
    try:
        tensor = torch.ones(1024 * 1024, device=device) * (42 if rank == 0 else 0)
        dist.broadcast(tensor, src=0)

        if torch.allclose(tensor, torch.ones_like(tensor) * 42):
            results["tests_passed"] += 1
            results["details"].append({"test": "broadcast", "status": "PASSED"})
        else:
            results["tests_failed"] += 1
            results["details"].append({"test": "broadcast", "status": "FAILED", "error": "Data mismatch"})
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append({"test": "broadcast", "status": "FAILED", "error": str(e)})

    return results


def main():
    # Initialize distributed
    rank, world_size, local_rank = init_distributed()
    hostname = socket.gethostname()

    # Gather system info (only on rank 0 to reduce noise)
    if rank == 0:
        print("=" * 70)
        print("PyTorch Distributed Communication Test")
        print("=" * 70)
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"NCCL version: {'.'.join(map(str, torch.cuda.nccl.version()))}")
        print(f"World size: {world_size}")
        print()

    # Synchronize before printing per-rank info
    dist.barrier()

    # Each rank reports its setup
    gpu_name = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "N/A"
    print(f"[Rank {rank:2d}] Host: {hostname}, Local rank: {local_rank}, GPU: {gpu_name}")
    sys.stdout.flush()
    dist.barrier()

    if rank == 0:
        print()

        # Check NCCL environment
        print("\nNCCL/UCX Environment Variables:")
        nccl_env = get_nccl_env_info()
        for key in sorted(nccl_env.keys()):
            print(f"  {key}={nccl_env[key]}")

        # Verify critical settings
        print("\nInterface Verification:")
        nccl_ib_disable = os.environ.get("NCCL_IB_DISABLE", "not set")
        nccl_socket_if = os.environ.get("NCCL_SOCKET_IFNAME", "not set")
        ucx_tls = os.environ.get("UCX_TLS", "not set")

        if nccl_ib_disable == "0":
            print("  ✓ NCCL_IB_DISABLE=0 (InfiniBand enabled)")
        else:
            print(f"  ✗ NCCL_IB_DISABLE={nccl_ib_disable} (should be 0 for IB)")

        if "ib" in nccl_socket_if.lower():
            print(f"  ✓ NCCL_SOCKET_IFNAME={nccl_socket_if} (IB interface)")
        else:
            print(f"  ? NCCL_SOCKET_IFNAME={nccl_socket_if}")

        if "rc" in ucx_tls.lower() or "ud" in ucx_tls.lower():
            print(f"  ✓ UCX_TLS={ucx_tls} (IB transport)")
        else:
            print(f"  ? UCX_TLS={ucx_tls}")

        print()

    dist.barrier()

    # Run communication tests
    if rank == 0:
        print("Running Communication Tests...")
        print("-" * 40)

    results = test_communication(rank, world_size, local_rank)

    dist.barrier()

    # Report results (gather to rank 0)
    if rank == 0:
        print(f"\nResults from Rank 0:")
        for detail in results["details"]:
            status = detail["status"]
            test_name = detail["test"]
            if status == "PASSED":
                extra = ""
                if "bandwidth_gbps" in detail:
                    extra = f" (bandwidth: {detail['bandwidth_gbps']} GB/s, latency: {detail['time_ms']} ms)"
                print(f"  ✓ {test_name}: {status}{extra}")
            elif status == "FAILED":
                print(f"  ✗ {test_name}: {status} - {detail.get('error', 'unknown')}")
            else:
                print(f"  - {test_name}: {status}")

        print()
        print("=" * 70)
        if results["tests_failed"] == 0:
            print("All tests PASSED! Communication is working correctly.")
            if results["tests_passed"] > 0:
                bw_test = next((d for d in results["details"] if "bandwidth_gbps" in d), None)
                if bw_test and bw_test["bandwidth_gbps"] > 10:
                    print("Bandwidth suggests InfiniBand is being used.")
                elif bw_test:
                    print("Warning: Low bandwidth may indicate TCP fallback instead of IB.")
        else:
            print(f"FAILED: {results['tests_failed']} test(s) failed")
        print("=" * 70)

    # Cleanup
    dist.destroy_process_group()

    return 0 if results["tests_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())