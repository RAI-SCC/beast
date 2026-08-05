"""Tests for the WeatherBench2 initialization-time rollout dataset."""

from datetime import datetime, timedelta

import pytest
import torch
from pytest import TempPathFactory

from beast.data.era5 import Event, EWBEra5, WB2Era5
from tests.data.test_dataset import (
    generate_mock_masks,
    generate_mock_normalization,
    generate_mock_zarr_data,
)

N_TIME, N_FEATURES, N_LAT, N_LON, N_MASKS = 24, 8, 60, 120, 3
FIRST_TIME = datetime(2024, 1, 1, 0)


@pytest.fixture(scope="module")
def mock_paths(tmp_path_factory: TempPathFactory):
    """Build a small ERA5-shaped zarr plus masks and normalization files."""
    data_path = tmp_path_factory.mktemp("wb2")
    zarr_path = data_path / "test_data.zarr"
    generate_mock_zarr_data(zarr_path, N_TIME, N_FEATURES, N_LAT, N_LON)
    mask_list = generate_mock_masks(data_path, N_LAT, N_LON, N_MASKS)
    generate_mock_normalization(data_path, N_FEATURES)
    return zarr_path, data_path, mask_list


def make_dataset(
    mock_paths, init_times, *, dt=1, in_timesteps=1, out_timesteps=1, max_lead_hours=6
):
    """Construct a single-rank dataset over ``init_times``."""
    zarr_path, data_path, mask_list = mock_paths
    return WB2Era5(
        init_times=init_times,
        max_lead_hours=max_lead_hours,
        pg_rank=0,
        pg_size=1,
        data_path=zarr_path,
        constant_masks_path=data_path,
        constant_masks=mask_list,
        normalization_dir=data_path,
        dt=dt,
        n_in_timesteps=in_timesteps,
        n_out_timesteps=out_timesteps,
        dtype=torch.float32,
        time_range=None,
    )


def test_len_is_the_number_of_init_times(mock_paths):
    """One item per initialization time, unlike EWBEra5's event x lead product."""
    init_times = [FIRST_TIME + timedelta(hours=h) for h in (4, 5, 6)]
    assert len(make_dataset(mock_paths, init_times)) == 3


@pytest.mark.parametrize("in_timesteps", [1, 2])
def test_warmup_window_ends_at_the_init_time(mock_paths, in_timesteps):
    """``samples[-1]`` must be the analysis the forecast starts from."""
    dt = 2
    init_time = FIRST_TIME + timedelta(hours=10)
    dataset = make_dataset(mock_paths, [init_time], dt=dt, in_timesteps=in_timesteps)

    returned_init, max_lead, samples = dataset[0]

    assert returned_init == init_time
    assert max_lead == 6
    assert len(samples) == in_timesteps

    for offset, sample in enumerate(reversed(samples)):
        expected_time = init_time - timedelta(hours=dt * offset)
        expected = (
            dataset.zarr_data.sel(time=expected_time)["fields"]
            .isel(latitude=slice(0, dataset.xlat))
            .values
        )
        # Mock normalization is mean 0 / std 1, so values pass through unchanged.
        torch.testing.assert_close(
            sample[0], torch.tensor(expected, dtype=torch.float32)
        )


def test_getitem_returns_correct(mock_paths):
    """The rollout loop unpacks ``(init_time, max_lead_hours, samples)``."""
    init_time = FIRST_TIME + timedelta(hours=4)
    item = make_dataset(mock_paths, [init_time], max_lead_hours=12)[0]

    assert len(item) == 3
    init, max_lead, samples = item
    assert isinstance(init, datetime)
    assert max_lead == 12
    assert isinstance(samples, list)
    # (batch, channel, latitude, longitude) -- the leading axis is the ensemble
    # slot the model fills in during rollout.
    assert samples[0].ndim == 4
    assert samples[0].shape[0] == 1


def test_input_from_list_appends_masks(mock_paths):
    """The stacked input is ``in_timesteps`` fields plus the constant masks."""
    dataset = make_dataset(mock_paths, [FIRST_TIME + timedelta(hours=4)])
    _, _, samples = dataset[0]

    stacked = dataset.input_from_list(list(samples))

    n_channels = dataset.in_var[1] - dataset.in_var[0]
    expected = n_channels * dataset.in_timesteps + dataset.get_constant_masks().shape[0]
    assert stacked.shape[-3] == expected


@pytest.mark.parametrize("in_timesteps", [1, 2])
def test_input_from_list_uses_the_last_timesteps(mock_paths, in_timesteps):
    """The rollout appends to ``samples``, so only its tail may be read."""
    dataset = make_dataset(
        mock_paths, [FIRST_TIME + timedelta(hours=6)], dt=2, in_timesteps=in_timesteps
    )
    _, _, samples = dataset[0]
    # Stand-ins for earlier rollout states, distinguishable from the real ones.
    stale = [torch.full_like(samples[0], -999.0) for _ in range(3)]

    stacked = dataset.input_from_list(stale + [s.clone() for s in samples])

    assert not (stacked == -999.0).any()
    torch.testing.assert_close(
        stacked, dataset.input_from_list([s.clone() for s in samples])
    )


def make_ewb_dataset(mock_paths, *, dt=1, in_timesteps=1):
    """Construct an ``EWBEra5`` over the same mock archive, for parity checks."""
    zarr_path, data_path, mask_list = mock_paths
    event = Event(
        id="mock_heat_wave_1",
        event_type="heat_wave",
        title="Mock",
        start_date=FIRST_TIME + timedelta(hours=6),
        end_date=FIRST_TIME + timedelta(hours=8),
        lat_min=0.0,
        lat_max=10.0,
        lon_min=0.0,
        lon_max=10.0,
    )
    return EWBEra5(
        event_list=[event],
        lead_times=[2],
        pg_rank=0,
        pg_size=1,
        data_path=zarr_path,
        constant_masks_path=data_path,
        constant_masks=mask_list,
        normalization_dir=data_path,
        dt=dt,
        n_in_timesteps=in_timesteps,
        n_out_timesteps=1,
        dtype=torch.float32,
        time_range=None,
    )


def test_single_rank_padding_defaults_to_zero(mock_paths):
    """WB2Era5 must survive single-rank construction, where the pad is unset."""
    dataset = make_dataset(mock_paths, [FIRST_TIME + timedelta(hours=4)])

    assert dataset._input_pad == 0
    # No padding channels are appended beyond fields plus masks.
    _, _, samples = dataset[0]
    n_channels = dataset.in_var[1] - dataset.in_var[0]
    assert dataset.input_from_list(list(samples)).shape[-3] == (
        n_channels * dataset.in_timesteps + dataset.get_constant_masks().shape[0]
    )


def test_list_from_output_matches_ewb_era5(mock_paths):
    """Same drift guard for the output half of the rollout step."""
    wb2 = make_dataset(mock_paths, [FIRST_TIME + timedelta(hours=4)])
    ewb = make_ewb_dataset(mock_paths)
    n_vars = wb2.in_var[1] - wb2.in_var[0]
    output = torch.randn(2, n_vars, wb2.xlat, wb2.xlon)
    weights = torch.ones(n_vars, 1, 1)

    wb2_out = wb2.list_from_output(output.clone(), weights, -24.0)
    ewb_out = ewb.list_from_output(output.clone(), weights, -24.0)

    assert len(wb2_out) == len(ewb_out)
    for from_wb2, from_ewb in zip(wb2_out, ewb_out):
        torch.testing.assert_close(from_wb2, from_ewb)


@pytest.mark.parametrize("out_timesteps", [1, 2])
def test_list_from_output_splits_one_timestep_per_entry(
    mock_paths, monkeypatch, out_timesteps
):
    """Each entry is one timestep's channels, ready to extend the sample list."""
    dataset = make_dataset(mock_paths, [FIRST_TIME + timedelta(hours=4)])
    monkeypatch.setattr(dataset, "out_timesteps", out_timesteps)
    n_vars = dataset.in_var[1] - dataset.in_var[0]
    output = torch.randn(2, n_vars * out_timesteps, dataset.xlat, dataset.xlon)
    # `torch.where` broadcasts the weights against the whole output, so they
    # cover every timestep's channels, not just one timestep's.
    weights = torch.ones(n_vars * out_timesteps, 1, 1)

    outputs = dataset.list_from_output(output, weights, -24.0)

    assert len(outputs) == out_timesteps
    for i, timestep in enumerate(outputs):
        assert timestep.shape == (2, n_vars, dataset.xlat, dataset.xlon)
        torch.testing.assert_close(
            timestep, output[:, n_vars * i : n_vars * (i + 1)].cpu()
        )


def test_list_from_output_masks_zero_weighted_features(mock_paths):
    """Zero-weighted channels are overwritten with ``fill``, not left as model noise."""
    dataset = make_dataset(mock_paths, [FIRST_TIME + timedelta(hours=4)])
    n_vars = dataset.in_var[1] - dataset.in_var[0]
    output = torch.randn(1, n_vars, dataset.xlat, dataset.xlon)
    weights = torch.ones(n_vars, 1, 1)
    weights[0] = 0.0

    (timestep,) = dataset.list_from_output(output.clone(), weights, -24.0)

    assert torch.all(timestep[:, 0] == -24.0)
    torch.testing.assert_close(timestep[:, 1:], output[:, 1:].cpu())


def test_missing_init_time_names_the_timestamp(mock_paths):
    """A gap in the archive must fail at construction, not hours into a job."""
    missing = FIRST_TIME + timedelta(days=400)
    with pytest.raises(ValueError, match=str(missing.date())):
        make_dataset(mock_paths, [missing])


def test_missing_warmup_frame_is_caught(mock_paths):
    """With in_timesteps=2 the frame before the init time must exist too."""
    # The archive starts at FIRST_TIME, so its own warm-up frame is missing.
    with pytest.raises(ValueError, match="which is not in"):
        make_dataset(mock_paths, [FIRST_TIME], dt=1, in_timesteps=2)


@pytest.mark.parametrize(
    ("init_times", "max_lead", "match"),
    [
        pytest.param([], 6, "at least one", id="no-init-times"),
        pytest.param([FIRST_TIME + timedelta(hours=4)], 0, "positive", id="zero-lead"),
        pytest.param(
            [FIRST_TIME + timedelta(hours=4)], -6, "positive", id="negative-lead"
        ),
    ],
)
def test_invalid_arguments_are_rejected(mock_paths, init_times, max_lead, match):
    """Bad configuration should fail immediately with a clear message."""
    with pytest.raises(ValueError, match=match):
        make_dataset(mock_paths, init_times, max_lead_hours=max_lead)
