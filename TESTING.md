# How to test the beast

## Requirenments

- pytest
- hypothesis
- coverage
- mpi4py

## Structure

Everything goes in the test folder, and the structure should mirror the source code directory, with filenames called `test_<module>.py`

## QoL

### GPU test

GPU tests can be marked by decorating the method with the `gpu` mark. A fixture with the correct device is also available. See example bellow:

```python
@pytest.mark.gpu
def test_gpu_marker(self, device: torch.device) -> None:
    """Test that the gpu marker correctly selects a CUDA device."""
    assert device.type == "cuda"
```

To activate GPU tests, pass the `--device cuda` option on the command line:

```console
λ pytest --device cuda
```

If no GPUs are available, the test will be skipped. The device fixture can still be used, in case the tests also need to run on CPU.


### Distributed tests

If the tests need to be run on more than 1 rank, use the `dist` marker.

```python
@pytest.mark.dist()
def test_dist_marker_no_ranks(self) -> None:
    assert (
        dist.get_world_size() > 1
    )  # This test should only run if there are more than 1 ranks
```

If a specific number of ranks is needed, add the number of ranks as an argument to the marker.

```python
@pytest.mark.dist(4)
def test_dist_marker_with_ranks(self) -> None:
    """Test that the dist marker with rank specification works."""
    assert dist.is_initialized()
    assert dist.get_world_size() == 4
```

To parallelize the tests, start pytest with mpi.

```console
mpirun -n 4 pytest
```
