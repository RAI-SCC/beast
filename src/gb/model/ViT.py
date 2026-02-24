import torch
from Jigsaw import attention, linear, patch_embedding


class Transformer_Sequential(torch.nn.Module):
    def __init__(
        self,
        config_dict,
        rank,
        device,
        parallelism,
        model_group,
        row_subgroup,
        col_subgroup,
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.column_subgroup = col_subgroup
        self.softmax_subgroup = row_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        transformer_dimension = config_dict["model"]["transformer_dim"]
        heads = config_dict["model"]["heads"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        padding = (0, 0)
        self.patch_size = config_dict["model"]["patch_size"]
        self.stride = config_dict["model"]["stride"]
        self.n_patches = (
            self.xlat * self.xlon - 1 * (self.patch_size - 1) - 1
        ) // self.stride + 1

        self.dropout = config_dict["model"]["dropout"]
        self.embedding = patch_embedding.PatchEmbedding1D(
            config_dict,
            self.patch_size,
            hidden_dimension,
            device=device,
            in_channels=72,
            padding=padding,
            stride=self.stride,
            xlat=self.xlat,
        )
        self.n_blocks = config_dict["model"]["n_blocks"]
        self.blocks = torch.nn.ModuleList(
            [
                attention.DistributedTransformer(
                    hidden_dimension,
                    transformer_dimension,
                    hidden_dimension,
                    heads,
                    rank,
                    self.device,
                    self.parallelism,
                    self.model_group,
                    self.softmax_subgroup,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
        self.num_variables = 69

        self.add = config_dict["model"]["add"]
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]
        if self.linear_residual:
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)
            self.layer_norm3 = torch.nn.LayerNorm(69 * 2)
            self.linearlayer = torch.nn.linear(
                self.num_variables * 2, self.num_variables
            )
        if self.concat:
            self.recovery = patch_embedding.PatchRecovery1D(
                self.patch_size,
                hidden_dimension,
                padding=padding,
                stride=self.stride,
                out_channels=69,
                concat=True,
                xlat=self.xlat,
                xlon=self.xlon,
            )
        else:
            self.recovery = patch_embedding.PatchRecovery1D(
                self.patch_size,
                hidden_dimension,
                padding=padding,
                stride=self.stride,
                out_channels=69,
                concat=False,
                xlat=self.xlat,
                xlon=self.xlon,
            )
        self.activation = torch.nn.GELU()

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )

    def forward(self, x, rollout_=1):
        x_shape = x.shape
        x_res = x.clone()
        x_res = x_res.permute(0, 2, 3, 1)

        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)  # x: latent_t

        x_res2 = x.clone()

        # x shape: [1, 1152, 2000]
        for rollout in torch.arange(rollout_):
            for i in range(self.n_blocks):
                x = self.blocks[i](x)  # x: latent_tplus1

        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)

        x = self.recovery(x)

        if self.add:  # Recovery
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x)  # layer norm must go before concat
            x = torch.concat((x[:, :, :, : self.num_variables], x_res[..., :69]), dim=3)
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat * self.xlon, self.num_variables * 2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
        return x


class ViT(torch.nn.Module):
    def __init__(
        self,
        config_dict,
        rank,
        device,
        parallelism,
        model_group,
        row_subgroup,
        col_subgroup,
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.column_subgroup = col_subgroup
        self.softmax_subgroup = row_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        transformer_dimension = config_dict["model"]["transformer_dim"]
        heads = config_dict["model"]["heads"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        self.patch_size = config_dict["model"]["patch_size"]
        self.stride = config_dict["model"]["stride"]
        self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size

        self.embedding = patch_embedding.Distributed1DPatchEmbedding(
            self.xlat,
            self.xlon,
            self.patch_size,
            72,
            hidden_dimension,
            rank,
            device,
            parallelism,
            model_group,
            bias=False,
        )

        self.n_blocks = config_dict["model"]["n_blocks"]
        self.blocks = torch.nn.ModuleList(
            [
                attention.TransformerLayer(
                    hidden_dimension,
                    transformer_dimension,
                    hidden_dimension,
                    heads,
                    rank,
                    self.device,
                    self.parallelism,
                    self.model_group,
                    self.softmax_subgroup,
                    self.column_subgroup,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension // 2)
        self.num_variables = 72 // (parallelism // 2)  # TODO: Not applicable for 8-way

        self.add = config_dict["model"]["add"]
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]

        if self.linear_residual:
            self.layer_norm2 = torch.nn.LayerNorm(
                self.num_variables if parallelism == 4 else self.num_variables // 2
            )  # 36 for two, 72 for four
            self.layer_norm3 = torch.nn.LayerNorm(
                self.num_variables * 2 if parallelism == 4 else self.num_variables
            )  # 72 for two, 72 for four
            self.linearlayer = linear.DistributedXWT(
                self.num_variables * 2 if parallelism == 4 else self.num_variables,
                self.num_variables,
                rank,
                device,
                parallelism=parallelism,
                model_group=model_group,
                bias=False,
            )

        self.recovery = patch_embedding.Distributed1DPatchRecovery(
            self.patch_size,
            hidden_dimension,
            self.num_variables,
            self.rank,
            self.device,
            parallelism,
            model_group,
            concat=self.concat,
            stride=self.stride,
            bias=False,
        )

        self.activation = torch.nn.GELU()

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )

    def forward(self, x, rollout_=1):
        x_res = x
        x_res = x_res.flatten(start_dim=1, end_dim=2)
        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)  # x: latent_t

        x_res2 = x.clone()

        # x shape: [1, 1152, 2000]
        for i in range(self.n_blocks):
            x = self.blocks[i](x)  # x: latent_tplus1

        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=-2)
        else:
            x = self.layer_norm(x + x_res2)

        x = self.recovery(x, self.n_lat, self.n_lon // (self.parallelism // 2))
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.add:
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x)

            x = torch.concat((x, x_res), dim=-1)

            # After recovery: [n_batch, n_lat, n_lon, n_var]
            x = self.layer_norm3(x)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = self.linearlayer(x)
        return x


## Megatron Implementations
class Megatron_Transformer(torch.nn.Module):
    def __init__(
        self,
        config_dict,
        rank,
        device,
        parallelism,
        model_group,
        row_subgroup,
        col_subgroup,
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.column_subgroup = col_subgroup
        self.softmax_subgroup = row_subgroup

        hidden_dimension = config_dict["model"]["hidden_dim"]
        transformer_dimension = config_dict["model"]["transformer_dim"]
        heads = config_dict["model"]["heads"]
        self.xlat, self.xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
        self.patch_size = config_dict["model"]["patch_size"]
        self.stride = config_dict["model"]["stride"]
        self.n_lat, self.n_lon = self.xlat, self.xlon // self.patch_size
        padding = (0, 0)
        self.embedding = patch_embedding.PatchEmbedding1D(
            config_dict,
            self.patch_size,
            hidden_dimension,
            device=device,
            in_channels=72,
            padding=padding,
            stride=self.stride,
            xlat=self.xlat,
        )
        self.n_blocks = config_dict["model"]["n_blocks"]
        self.blocks = torch.nn.ModuleList(
            [
                attention.MegatronTransformerLayer(
                    hidden_dimension,
                    transformer_dimension,
                    hidden_dimension,
                    heads,
                    rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    self.softmax_subgroup,
                    self.column_subgroup,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
        self.num_variables = 69

        self.add = config_dict["model"]["add"]
        self.concat = config_dict["model"]["concat"]
        self.linear_residual = config_dict["model"]["linear_residual"]
        if self.linear_residual:
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)
            self.layer_norm3 = torch.nn.LayerNorm(self.num_variables * 2)
            self.linearlayer = torch.nn.linear(
                self.num_variables * 2,
                self.num_variables,
            )

        if self.concat:
            self.recovery = patch_embedding.PatchRecovery1D(
                self.patch_size,
                hidden_dimension,
                padding=padding,
                stride=self.stride,
                out_channels=69,
                concat=True,
                xlat=self.xlat,
                xlon=self.xlon,
            )
        else:
            self.recovery = patch_embedding.PatchRecovery1D(
                self.patch_size,
                hidden_dimension,
                padding=padding,
                stride=self.stride,
                out_channels=69,
                concat=False,
                xlat=self.xlat,
                xlon=self.xlon,
            )
        self.activation = torch.nn.GELU()

        if self.linear_residual:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList(
                        [
                            self.blocks,
                            self.layer_norm,
                            self.layer_norm2,
                            self.layer_norm3,
                            self.linearlayer,
                        ]
                    ),
                }
            )
        else:
            self.params = torch.nn.ModuleDict(
                {
                    "embedding": torch.nn.ModuleList([self.embedding]),
                    "recovery": torch.nn.ModuleList([self.recovery]),
                    "common": torch.nn.ModuleList([self.blocks, self.layer_norm]),
                }
            )

    def forward(self, x, rollout_=1):
        x_shape = x.shape
        x_res = x.clone()
        x_res = x_res.permute(0, 2, 3, 1)

        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)  # x: latent_t

        x_res2 = x.clone()

        # x shape: [1, 1152, 2000]
        for rollout in torch.arange(rollout_):
            for i in range(self.n_blocks):
                x = self.blocks[i](x)  # x: latent_tplus1

        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)

        x = self.recovery(x)

        if self.add:  # Recovery
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x)  # layer norm must go before concat
            x = torch.concat((x[:, :, :, : self.num_variables], x_res[..., :69]), dim=3)
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat * self.xlon, self.num_variables * 2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)

        return x
