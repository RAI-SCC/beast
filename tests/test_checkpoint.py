import pathlib

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from beast.checkpoint import (
	load_checkpoint,
	load_state_dict,
	save_checkpoint,
)
from beast.comm import create_process_mesh_dict, get_process_group
from beast.model import convert_jigsaw_vi, get_model
from beast.model.gb.beast_config import BEAST_CONFIGURATIONS
from beast.utils import set_all_seeds


class TestCheckpoint:
	def test_save_checkpoint(
		self,
		shared_tmp_path: pathlib.Path,
		device: torch.device,
	) -> None:
		"""Test to save and load a dummy model. If running distributed, assumes only ddp group, so a single checkpoint should be written."""
		model = torch.nn.Linear(10, 10)
		optimizer = torch.optim.AdamW(model.parameters())
		loss_value = torch.tensor([1.2], device=device)

		full_path = save_checkpoint(
			str(shared_tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
		)
		dist.barrier()
		checkpoint_list = list(shared_tmp_path.iterdir())
		assert len(checkpoint_list) == 1

		checkpoint = load_checkpoint(full_path)

		assert isinstance(checkpoint["model_state_dict"], dict)
		assert isinstance(checkpoint["optimizer_state_dict"], dict)
		assert isinstance(checkpoint["epoch"], int)
		assert isinstance(checkpoint["step"], int)
		assert checkpoint["loss"] == loss_value

	@pytest.mark.dist(4)
	def test_dist_save_checkpoint(
		self,
		shared_tmp_path: pathlib.Path,
		device: torch.device,
	) -> None:
		"""Test to save and load a dummy model. DTP parallelism, should produce two checkpoint files."""
		create_process_mesh_dict(mesh_dims=[1, 1, 1, 2, 2])
		model = torch.nn.Linear(10, 10)
		optimizer = torch.optim.AdamW(model.parameters())
		loss_value = torch.tensor([1.2], device=device)

		full_path = save_checkpoint(
			str(shared_tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
		)
		dist.barrier()
		checkpoint_list = list(shared_tmp_path.iterdir())
		assert len(checkpoint_list) == 2
		checkpoint = load_checkpoint(full_path)

		assert isinstance(checkpoint["model_state_dict"], dict)
		assert isinstance(checkpoint["optimizer_state_dict"], dict)
		assert isinstance(checkpoint["epoch"], int)
		assert isinstance(checkpoint["step"], int)
		assert checkpoint["loss"] == loss_value

	@pytest.mark.gpu
	@pytest.mark.dist(4)
	@pytest.mark.parametrize("bayesian", [False])
	@pytest.mark.parametrize("mesh_dims", [[1,4,1,1,1], [1,1,1,2,2]])
	def test_save_beast_checkpoint(
		self,
		device: torch.device,
		shared_tmp_path: pathlib.Path,
		bayesian: bool,
		mesh_dims: list[int],
	) -> None:
		"""Test that initializing, saving, and loading beast works."""
		create_process_mesh_dict(mesh_dims=mesh_dims)
		channel_group = get_process_group("JChannel")
		spatial_group = get_process_group("JSpatial")
		ddp_group = get_process_group("DDP")

		config = BEAST_CONFIGURATIONS["BEASTLING"]["config"].copy()
		model = get_model("BEASTLING", bayesian=bayesian, device=device).to(device)
		if bayesian:
			convert_jigsaw_vi(model, channel_group, spatial_group)
			model.return_log_probs = False

		dist.barrier()
		model = DDP(
			model,
			device_ids=[device],
			output_device=device,
			process_group=ddp_group,
		)	

		height = config["height"]
		width = config["width"]
		in_channels = config["in_channels"]
		input_beastling = torch.ones((1, in_channels, height, width), device=device)
		set_all_seeds(42)
		if bayesian:
			output_native = model(input_beastling, samples=1)
		else:
			output_native = model(input_beastling)

		optimizer = torch.optim.AdamW(model.parameters())
		loss_value = torch.tensor([1.2], device=device)

		full_path = save_checkpoint(
			str(shared_tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
		)
		dist.barrier()
		checkpoint_list = list(shared_tmp_path.iterdir())
		assert len(checkpoint_list) == channel_group.size()

		checkpoint = load_checkpoint(full_path)

		assert isinstance(checkpoint["model_state_dict"], dict)
		assert isinstance(checkpoint["optimizer_state_dict"], dict)
		assert isinstance(checkpoint["epoch"], int)
		assert isinstance(checkpoint["step"], int)
		assert checkpoint["loss"] == loss_value

		model_loaded, optimizer_loaded, start_epoch = load_state_dict(
			model,
			optimizer,
			checkpoint["model_state_dict"],
			checkpoint["optimizer_state_dict"],
			checkpoint["epoch"],
		)

		set_all_seeds(42)
		if bayesian:
			output_loaded = model_loaded(input_beastling, samples=1)
		else:
			output_loaded = model_loaded(input_beastling)
		assert torch.allclose(output_loaded, output_native)
