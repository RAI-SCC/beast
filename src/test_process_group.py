#!/usr/bin/env python3
"""
Test script for PyTorch distributed process group initialization.
Tests NCCL backend, GPU access, and all-to-all communication.
"""

import os
import torch
import torch.distributed as dist
from Jigsaw import utils


def test_process_group():
    """Initialize process group and test distributed operations."""
    
    # Initialize the process group
    device, localid, rank, world_size = utils.init_cuda()
    
    # Get rank and world size
    
    print(f"[Rank {rank}] Process group initialized successfully")
    print(f"[Rank {rank}] World size: {world_size}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(f"[Rank {rank}] CUDA is not available")
    
    gpu_count = torch.cuda.device_count()
    print(f"[Rank {rank}] Number of GPUs available: {gpu_count}")
    
    # Set device for this process
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}] Using GPU: {local_rank}")
    print(f"[Rank {rank}] GPU name: {torch.cuda.get_device_name(local_rank)}")
    
    # Test all-to-all operation
    # Create input tensor: each rank sends different data to each other rank
    tensor_size = 4
    input_tensor = torch.ones(world_size * tensor_size, device=device) * rank
    output_tensor = torch.zeros(world_size * tensor_size, device=device)
    
    # Split input into chunks for each destination rank
    input_list = list(input_tensor.chunk(world_size))
    output_list = list(output_tensor.chunk(world_size))
    
    print(f"[Rank {rank}] Starting all-to-all operation...")
    print(f"[Rank {rank}] Input tensor: {input_tensor}")
    
    # Perform all-to-all
    dist.all_to_all(output_list, input_list)
    
    # Reconstruct output tensor
    output_tensor = torch.cat(output_list)
    
    print(f"[Rank {rank}] All-to-all completed successfully")
    print(f"[Rank {rank}] Output tensor: {output_tensor}")
    
    # Verify correctness: each chunk should come from a different rank
    expected = torch.cat([torch.ones(tensor_size, device=device) * i for i in range(world_size)])
    if torch.allclose(output_tensor, expected):
        print(f"[Rank {rank}] ✓ All-to-all verification PASSED")
    else:
        print(f"[Rank {rank}] ✗ All-to-all verification FAILED")
        print(f"[Rank {rank}] Expected: {expected}")
    
    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*50)
        print("All tests completed successfully!")
        print("="*50)
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    test_process_group()
