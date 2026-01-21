import torch
import torch.nn as nn
from torch import permute, reshape
from torch.utils.checkpoint import checkpoint
from torch_blue.vi import VIModule, convert_to_vimodule

from .architecture import (
    MLP,
    EarthAttention2DNoBias,
    GroupedPatchEmbedding2D,
    GroupedPatchRecovery2D,
)


class NiceNorm(nn.Module):
    def __init__(self, normalized_shape, noise_dim, eps=1e-5, device=None, dtype=None):
        super().__init__()

        self.norm = nn.LayerNorm(
            normalized_shape,
            eps=eps,
            device=device,
            dtype=dtype,
            elementwise_affine=False,
        )

        self.noise_dim = noise_dim
        self.shape = normalized_shape

        self.activation = nn.GELU()
        self.linear_scale1 = nn.Linear(noise_dim, normalized_shape)
        self.linear_scale2 = nn.Linear(normalized_shape, normalized_shape)
        self.linear_bias1 = nn.Linear(noise_dim, normalized_shape)
        self.linear_bias2 = nn.Linear(normalized_shape, normalized_shape)

    def forward(self, x, noise):
        x = self.norm(x)

        scale = self.linear_scale1(noise)
        scale = self.activation(scale)
        scale = self.linear_scale2(scale)
        scale = scale.reshape(x.shape[0], *[1 for _ in x.shape[1:-1]], self.shape)

        bias = self.linear_bias1(noise)
        bias = self.activation(bias)
        bias = self.linear_bias2(bias)
        bias = bias.reshape(x.shape[0], *[1 for _ in x.shape[1:-1]], self.shape)

        x = scale * x + bias

        return x


############################ Upsampling and Downsampling


class NiceDownSample2D(nn.Module):
    """Downsample 2D."""

    def __init__(self, dim, noise_dim):
        """
        Initialize 2D Down-sampling operation.

        dim: int
            hidden dimension of input
        downsampling: Tuple(int, int)
            Number of patches to downsample in (lat, lon) dimensions.
        """
        super().__init__()
        # A linear function and a layer normalization
        self.linear = nn.Linear(
            4 * dim, 2 * dim, bias=False
        )  # TODO: Modify 4, 2 to correspond to different downsampling dimensions
        self.norm = NiceNorm(4 * dim, noise_dim)
        self.downsampling = (2, 2)

    def forward(self, x, n_patch_lat, n_patch_lon, noise):
        """
        Forward pass of Downsample 2D.

        x: Tensor
            input tensor of shape (n_batch, n_patch_lon*n_patch_lat, hidden_dimension)
        n_patch_lat: int
            number of patches in the latitude dimension.
        n_patch_lon: int
            number of patches in the longitude dimension.

        Returns
        -------
        x: Tensor
            of shape (n_batch, n_patch_lat/downsampling[0], n_patch_lon/downsampling[1], hidden_dim*2)
        """
        # Reshape x to three dimensions for downsampling
        x = reshape(x, shape=(x.shape[0], n_patch_lon, n_patch_lat, x.shape[-1]))

        # Padding the input to facilitate downsampling
        y1_pad = (
            (self.downsampling[0] - (x.shape[1] % self.downsampling[0]))
            % self.downsampling[0]
            // 2
        )
        y2_pad = (
            self.downsampling[0] - (x.shape[1] % self.downsampling[0])
        ) % self.downsampling[0] - y1_pad
        z1_pad = (
            (self.downsampling[1] - (x.shape[2] % self.downsampling[1]))
            % self.downsampling[1]
            // 2
        )
        z2_pad = (
            self.downsampling[1] - (x.shape[2] % self.downsampling[1])
        ) % self.downsampling[1] - z1_pad

        x = torch.nn.functional.pad(
            x, pad=(0, 0, z1_pad, z2_pad, y1_pad, y2_pad), mode="constant", value=0
        )

        # Reorganize x to reduce the resolution: simply change the order and downsample from (8, 360, 182) to (8, 180, 91)
        n_patch_lat, n_patch_lon = x.shape[1:3]
        # Reshape x to facilitate downsampling
        x = reshape(
            x, shape=(x.shape[0], n_patch_lat // 2, 2, n_patch_lon // 2, 2, x.shape[-1])
        )
        # Change the order of x
        x = permute(x, (0, 1, 3, 2, 4, 5))
        # Reshape to get a tensor of resolution (8, 180, 91)
        x = reshape(
            x,
            shape=(
                x.shape[0],
                (n_patch_lat // 2) * (n_patch_lon // 2),
                4 * x.shape[-1],
            ),
        )

        # Call the layer normalization
        x = self.norm(x, noise)

        # Decrease the channels of the data to reduce computation cost
        x = self.linear(x)
        return x


class NiceUpSample2D(nn.Module):
    """Upsample 2D."""

    def __init__(
        self,
        input_dim,
        output_dim,
        n_lat,
        n_lon,
        noise_dim,
        lat_crop=(0, 0),
        lon_crop=(0, 0),
    ):
        """
        Up-sampling operation.

        input_dim: int
            dimensions of the input data
        output_dim: int
            dimensions of output
        n_lat: int
            number of patches in the latitude dimension
        n_lon: int
            number of patches in the longitude dimension
        lat_crop: Tuple(int, int)
            cropping applied to the left and right dimensions to obtain the original size of the image
            if downsampled image is (8, 181, 360), lat_crop = (0, 1)
            if downsampled image is (8, 91, 180), lat_crop = (0, 1)
        lon_crop: Tuple(int, int)
            cropping applied to the top and bottom dimensions to obtain the original size of the image
            if downsampled image is (8, 181, 360), lon_crop= (0, 0)
            if downsampled image is (8, 91, 180), lon_crop= (0, 0)
        """
        # TODO: should we fix the relationship between the input and output dim? esp. because the output dimension is
        # changed (x4) here

        super().__init__()
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.lat_crop = lat_crop
        self.lon_crop = lon_crop

        # Linear layers without bias to increase channels of the data
        self.linear1 = nn.Linear(input_dim, output_dim * 4, bias=False)

        # Linear layers without bias to mix the data up
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)

        # Normalization
        self.norm = NiceNorm(output_dim, noise_dim)

    def forward(self, x, noise):
        """
        Forward pass of 2D Upsample.

        x: Tensor
            of shape (n_batch, n_patch_lat*n_patch_lon, input_dim)
            In lite model, n_patch_lat = 46, n_patch_lon = 90

        Returns
        -------
        x: Tensor
            of shape (n_batch, n_patch_lat*n_patch_lon*4, output_dim);
            middle dimension is cropped to shape before downsampling.
            i.e., in lite model, n_patch_lat = 91, n_patch_lon = 180
        """
        # Call the linear functions to increase channels of the data
        x = self.linear1(x)

        # Reorganize x to increase the resolution: simply change the order and upsample from (8, 180, 91) to (8, 360, 182)
        # Reshape x to facilitate upsampling.
        x = reshape(
            x, shape=(x.shape[0], self.n_lat, self.n_lon, 2, 2, x.shape[-1] // 4)
        )
        # Change the order of x
        x = permute(x, (0, 1, 3, 2, 4, 5))
        # After reshape: (n_batch, n_patch_lat, n_patch_lon, output_dim)
        x = reshape(x, shape=(x.shape[0], self.n_lat * 2, self.n_lon * 2, x.shape[-1]))

        # Crop the output to the input shape of the network
        x = x[
            :,
            self.lat_crop[0] : -self.lat_crop[1] or None,
            self.lon_crop[0] : -self.lon_crop[1] or None,
            :,
        ]

        # Reshape x back
        x = reshape(x, shape=(x.shape[0], x.shape[1] * x.shape[2], x.shape[-1]))

        # Call the layer normalization
        x = self.norm(x, noise)

        # Mixup normalized tensors
        x = self.linear2(x)
        return x


####################################### Attention mechanism


class NiceEarthSpecificBlock2DNoBias(nn.Module):
    """Class for one block of the 2D attention mechanism with no bias term."""

    def __init__(
        self,
        dim,
        # drop_path_ratio,
        heads,
        input_shape,
        device,
        input_resolution,
        window_size,
        noise_dim,
    ):
        """
        Initialize class for 2D attention mechanism with no bias.

        dim: int
            Size of hidden dimension
        drop_path_ratio: float
            Probability that a sample will be dropped during training
        heads: int
            Number of attention heads
        input_shape: torch.Shape
            of shape 2 describing the number of patches in the (lat, lon) dimensions
        device: String
            device that the code is being run on.
        input_resolution: List[int]
            List of length 2 that specifies the dimensions after the patch embedding step incl. padding.
            For Pangu-Weather, input resolution is (186, 360)
            For Pangu-Weather-Lite, input resolution is (93, 180)
        window_size: Tensor
            Tensor of length(2) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
        """
        # super().__init__(dim, drop_path_ratio, heads, input_shape, device, input_resolution, window_size)

        super().__init__()
        # Define the window size of the neural network
        self.window_size = window_size

        # Initialize several operations
        # self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = NiceNorm(dim, noise_dim)
        self.norm2 = NiceNorm(dim, noise_dim)
        self.linear = MLP(dim, 0)
        self.input_resolution = input_resolution

        # Define the window size of the neural network

        self.attention = EarthAttention2DNoBias(
            dim,
            heads,
            # 0,
            self.window_size,
        )

        # Only generate masks one time @ initialization
        # self.input_resolution = input_resolution

    def _window_partition(self, x, window_size):
        """
        Partitions input tensor into windows over which attention will be calculated.

        x: Tensor
            of shape (n_batch, n_patch_vert, n_patch_lat, n_patch_lon, C)
        window_size: Tensor
            of length 2, describing window size in (vert, lat, lon) dimensions
            = (window_size_lat, window_size_lon)

        Returns
        -------
        windows: Tensor
            of shape(num_windows*n_batch, window_size_lat, window_size_lon, C)
        """
        n_batch, n_patch_lat, n_patch_lon, hidden_dim = x.shape
        x = x.view(
            n_batch,
            n_patch_lat // window_size[0],
            window_size[0],
            n_patch_lon // window_size[1],
            window_size[1],
            hidden_dim,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0], window_size[1], hidden_dim)
        )
        return windows

    def _window_reverse(self, windows, window_size, n_patch_lat, n_patch_lon):
        """
        Reverse window operation to retrieve x Tensor in original shape.

        windows: Tensor
            of shape (num_windows*B, window_size_lat, window_size_lon, C) == nB,W0,W1,C
        window_size: Tensor
          of length 3, describing window size in (vert, lat, lon) dimensions
        n_patch_lat: int
            Number of patches in the latitude dimension
        n_patch_lon: int
            Number of patches in the longitude dimension

        Returns
        -------
        x: Tensor
            of shape (n_batch, n_patch_vert, n_patch_lat, n_patch_lon, C)
        """
        n_batch = int(
            windows.shape[0]
            / (n_patch_lat * n_patch_lon)
            * (window_size[0] * window_size[1])
        )
        x = windows.view(
            n_batch,
            n_patch_lat // window_size[0],
            n_patch_lon // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
        x = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(n_batch, n_patch_lat, n_patch_lon, -1)
        )

        return x

    def forward(self, x, n_patch_lat, n_patch_lon, roll, noise):
        """
        Forward pass of a single 2D attention block.

        n_patch_lat: int
            Number of patches in the latitude dimension
        n_patch_lon: int
            Number of patches in the longitude dimension
        roll: bool
            Specifying if the layer is rolled

        Returns
        -------
        x: Tensor
            of shape (B, n_patch_lat*n_patch_lon, C)
        """
        # Save the shortcut for skip-connection
        shortcut = x.clone()

        # Reshape input to three dimensions to calculate window attention
        x = reshape(x, shape=(x.shape[0], n_patch_lat, n_patch_lon, x.shape[2]))

        # Zero-pad input accordign to window sizes
        x1_pad = (
            (self.window_size[0] - (x.shape[1] % self.window_size[0]))
            % self.window_size[0]
            // 2
        )
        x2_pad = (
            self.window_size[0] - (x.shape[1] % self.window_size[0])
        ) % self.window_size[0] - x1_pad
        y1_pad = (
            (self.window_size[1] - (x.shape[2] % self.window_size[1]))
            % self.window_size[1]
            // 2
        )
        y2_pad = (
            self.window_size[1] - (x.shape[2] % self.window_size[1])
        ) % self.window_size[1] - y1_pad

        x = nn.functional.pad(
            x, pad=(0, 0, y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
        )

        # Store the shape of the input for restoration
        ori_shape = x.shape

        if roll:
            # Roll x for half of the window for 3 dimensions
            x = torch.roll(
                x,
                shifts=(self.window_size[0] // 2, self.window_size[1] // 2),
                dims=(1, 2),
            )

        # Generate mask of attention masks
        # If two pixels are not adjacent, then mask the attention between them
        # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
        # Reorganize data to calculate window attention
        x_window = self._window_partition(
            x, self.window_size
        )  # n_windows * B, window_size[0], window_size[1], window_size[2], C

        # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube
        x_window = x_window.view(
            -1, self.window_size[0] * self.window_size[1], x_window.shape[-1]
        )  # n_windows * B, window_size[0], window_size[1], window_size[2], C

        mask = self._gen_mask_(n_patch_lat=ori_shape[1], n_patch_lon=ori_shape[2]).to(
            x.dtype
        )

        # Apply 3D window attention with Earth-Specific bias
        if roll:
            attention_windows = self.attention(x_window, mask)
        else:
            zero_mask = torch.zeros(mask.shape).to(x.dtype)
            attention_windows = self.attention(x_window, zero_mask)

        attention_windows = attention_windows.view(
            -1, self.window_size[0], self.window_size[1], x.shape[-1]
        )

        x = self._window_reverse(
            attention_windows,
            self.window_size,
            n_patch_lat=x.shape[1],
            n_patch_lon=x.shape[2],
        )

        # Reshape the tensor back to its original shape
        x = reshape(x, shape=ori_shape)  # B, n_patch_lat, n_patch_lon, C

        if roll:
            # Roll x back for half of the window
            x = torch.roll(
                x,
                shifts=(-self.window_size[0] // 2, -self.window_size[1] // 2),
                dims=(1, 2),
            )

        # Crop the zero padding
        x = x[:, x2_pad : x2_pad + n_patch_lat, y2_pad : y2_pad + n_patch_lon, :]

        # Reshape the tensor back to the input shape
        x = reshape(x, shape=(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))

        # Main calculation stages
        # x = shortcut + self.drop_path(self.norm1(x))
        x = shortcut + self.norm1(x, noise)
        # x = x + self.drop_path(self.norm2(self.linear(x)))
        x = x + self.norm2(self.linear(x), noise)

        return x

    def _gen_mask_(self, n_patch_lat, n_patch_lon):
        """
        Generate mask for attention mechanism when fields are rolled.

        n_patch_vert: int
            Number of patches in the vertical (first) dimension
        n_patch_lat: int
            Number of patches in the latitude (second) dimension
        n_patch_lon: int
            Number of patches in the longitude (third) dimension

        Returns
        -------
        attention_mask: Tensor
            of shape (n_windows, n_patch_in_window, n_patch_in_window)
            attention mask with value of 0 when attention should not be masked and -10000 when masked.
        """
        img_mask = torch.zeros(
            (1, n_patch_lat, n_patch_lon, 1)
        )  # 1 n_patch_lat n_patch_lon 1
        h_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.window_size[0] // 2),
            slice(-self.window_size[0] // 2, None),
        )

        cnt = 0
        for h in h_slices:
            img_mask[:, h, :, :] = cnt
            cnt += 1

        # TODO: check implementation with SWIN transformer architecture
        mask_windows = self._window_partition(
            img_mask, self.window_size
        )  # n_windows, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0, float(-10000.0)
        ).masked_fill(attention_mask == 0, float(0.0))
        return attention_mask


class NiceEarthSpecificLayer2DNoBias(nn.Module):
    """Earth-Specific Layer for 2D attention mechanism without bias term."""

    def __init__(
        self,
        depth,
        dim,
        # drop_path_ratio_list,
        heads,
        device,
        noise_dim,
        window_size=torch.tensor([6, 12]),
        checkpointing=False,
    ):
        """
        Initialize layer of network that calls EarthSpecificBlock2DNoBias.

        depth: int
            Number of layers
        dim: int
            Size of hidden dimension
        drop_path_ratio_list: List[float]
            List of length = layers specifying drop_path_ratio for each layer.
        device: String
            device that the code is being run on.
        input_resolution: List[int]
            List of length 3 that specifies the dimensions after the patch embedding step incl. padding.
            For Pangu-Weather, input resolution is (8, 186, 360)
            For Pangu-Weather-Lite, input resolution is (8, 93, 180)
        window_size: Tensor
            Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
        """
        super().__init__()
        self.depth = depth
        self.dim = dim

        # Construct basic blocks
        self.blocks = nn.ModuleList(
            [
                NiceEarthSpecificBlock2DNoBias(
                    dim,
                    # drop_path_ratio_list[i],
                    heads,
                    input_shape=[0, 0],
                    device=device,
                    input_resolution=(0, 0),
                    window_size=window_size,
                    noise_dim=noise_dim,
                ).to(device)
                for i in torch.arange(depth)
            ]
        )
        self.checkpointing = checkpointing

    def forward(self, x, n_patch_lat, n_patch_lon, noise):
        """
        Forward pass of Earth-Specific layers for 2D attention mechanism with absolute positional bias term.

        x: Tensor
            Input to model of shape (B, n_patch_lat, n_patch_lon, n_variables*n_patch_vert) #TODO correct?
        n_patch_lat: int
            Number of patches in the latitude dimension.
        n_patch_lon: int
            Number of patches in the longitude dimension.

        Returns
        -------
        x: Tensor
        """
        if self.checkpointing:
            for i in range(self.depth):
                # Roll the input every two blocks
                if i % 2 == 0:
                    x = checkpoint(
                        self.blocks[i],
                        x,
                        n_patch_lat,
                        n_patch_lon,
                        False,
                        use_reentrant=False,
                    )
                    torch.cuda.empty_cache()
                else:
                    x = checkpoint(
                        self.blocks[i],
                        x,
                        n_patch_lat,
                        n_patch_lon,
                        True,
                        use_reentrant=False,
                    )
                    torch.cuda.empty_cache()
        else:
            for i in range(self.depth):
                # Roll the input every two blocks
                if i % 2 == 0:
                    x = self.blocks[i](x, n_patch_lat, n_patch_lon, False, noise)
                else:
                    x = self.blocks[i](x, n_patch_lat, n_patch_lon, True, noise)
        return x


########################################### Full model


class NiceNoEmbeddingModel(nn.Module):
    """Class definition of 2D model."""

    def __init__(self, config: dict, device: torch.device = "cpu") -> None:
        """Initialize 2D Lite model.

        dim: int
        patch_size = Tuple(int, int, int)
                patch size in the height, lat, long directions
        device: String
                device that the code is offloaded onto.
        """
        nn.Module.__init__(self)
        # Drop path rate is linearly increased as the depth increases
        # drop_list = linspace(0, 0.2, 8)  # used to be drop_path_list
        self.config = config

        self.dim = config["model"]["hidden_dim"]
        self.patch_size = (config["model"]["patch_size"], config["model"]["patch_size"])

        if "embedding_kernel_size" in config["model"].keys():
            self.embedding_kernel_size = config["model"]["embedding_kernel_size"]
        else:
            self.embedding_kernel_size = config["model"]["patch_size"]

        if self.embedding_kernel_size < self.patch_size[0]:
            self.embedding_kernel_size = self.patch_size[0]

        self.resolution = (config["data"]["x_lat"], config["data"]["x_lon"])
        self.padded_resolution = (
            self.resolution[0]
            + (self.patch_size[0] - self.resolution[0] % self.patch_size[0])
            % self.patch_size[0]
            + self.embedding_kernel_size
            - self.patch_size[0],
            self.resolution[1]
            + (self.patch_size[1] - self.resolution[1] % self.patch_size[1])
            % self.patch_size[1]
            + self.embedding_kernel_size
            - self.patch_size[1],
        )
        self.padding = (
            (
                (self.padded_resolution[0] - self.resolution[0]) % 2,
                (self.padded_resolution[0] - self.resolution[0])
                - (self.padded_resolution[0] - self.resolution[0]) % 2,
            ),
            (
                (self.padded_resolution[1] - self.resolution[1]) % 2,
                (self.padded_resolution[1] - self.resolution[1])
                - (self.padded_resolution[1] - self.resolution[1]) % 2,
            ),
        )
        self.patch_resolution = (
            (
                self.resolution[0]
                + (self.patch_size[0] - self.resolution[0] % self.patch_size[0])
                % self.patch_size[0]
            )
            // self.patch_size[0],
            (
                self.resolution[1]
                + (self.patch_size[1] - self.resolution[1] % self.patch_size[1])
                % self.patch_size[1]
            )
            // self.patch_size[1],
        )
        self.downsampled_resolution = (
            self.patch_resolution[0] // 2 + self.patch_resolution[0] % 2,
            self.patch_resolution[1] // 2 + self.patch_resolution[1] % 2,
        )

        self.n_in_vars = (
            len(config["data"]["in_surface_variables"])
            + len(config["data"]["in_pressure_variables"])
            * len(config["data"]["in_pressure_levels"])
            + len(config["data"]["constant_masks"])
        )
        self.n_out_vars = len(config["data"]["out_surface_variables"]) + len(
            config["data"]["out_pressure_variables"]
        ) * len(config["data"]["out_pressure_levels"])

        # self.n_in_vars = config['model']['n_in_vars']
        # self.n_out_vars = config['model']['n_out_vars']

        # Patch embedding
        self._input_layer = GroupedPatchEmbedding2D(
            self.patch_size,
            self.embedding_kernel_size,
            dim=self.dim,
            device=device,
            in_channels=self.n_in_vars,
            num_surface=len(config["data"]["in_surface_variables"]),
            num_pressure=len(config["data"]["in_pressure_variables"]),
            num_levels=len(config["data"]["in_pressure_levels"]),
        )

        # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_resolution[0], self.dim))
        # trunc_normal_(self.absolute_pos_embed, std=0.02)

        window_size = torch.tensor([6, 12])

        # Four transformer layers
        self.layer1 = NiceEarthSpecificLayer2DNoBias(
            depth=6,
            dim=self.dim,
            # drop_path_ratio_list=drop_list[:6],
            heads=12,
            device=device,
            window_size=window_size,
            checkpointing=config["training"]["checkpointing"],
            noise_dim=config["model"]["noise_dim"],
        )

        self.layer2 = NiceEarthSpecificLayer2DNoBias(
            depth=2,
            dim=2 * self.dim,
            # drop_path_ratio_list=drop_list[6:],
            heads=24,
            device=device,
            window_size=window_size,
            checkpointing=config["training"]["checkpointing"],
            noise_dim=config["model"]["noise_dim"],
        )

        self.layer3 = NiceEarthSpecificLayer2DNoBias(
            depth=2,
            dim=2 * self.dim,
            # drop_path_ratio_list=drop_list[6:],
            heads=24,
            device=device,
            window_size=window_size,
            checkpointing=config["training"]["checkpointing"],
            noise_dim=config["model"]["noise_dim"],
        )

        self.layer4 = NiceEarthSpecificLayer2DNoBias(
            depth=6,
            dim=self.dim,
            # drop_path_ratio_list=drop_list[:6],
            heads=12,
            device=device,
            window_size=window_size,
            checkpointing=config["training"]["checkpointing"],
            noise_dim=config["model"]["noise_dim"],
        )

        # Upsample and downsample blocks
        self.upsample = NiceUpSample2D(
            input_dim=self.dim * 2,
            output_dim=self.dim,
            n_lat=self.downsampled_resolution[0],
            n_lon=self.downsampled_resolution[1],
            lat_crop=(0, self.patch_resolution[0] % 2),
            lon_crop=(0, self.patch_resolution[1] % 2),
            noise_dim=config["model"]["noise_dim"],
        )

        self.downsample = NiceDownSample2D(
            dim=self.dim, noise_dim=config["model"]["noise_dim"]
        )

        # Patch Recovery
        self._output_layer = GroupedPatchRecovery2D(
            patch_size=self.patch_size,
            kernel_size=self.embedding_kernel_size,
            dim=2 * self.dim,
            out_channels=self.n_out_vars,
            num_surface=len(config["data"]["out_surface_variables"]),
            num_pressure=len(config["data"]["out_pressure_variables"]),
            num_levels=len(config["data"]["out_pressure_levels"]),
        )  # added patch size

    def forward(
        self, input_data: torch.Tensor, noise, rollout: int = 1
    ) -> torch.Tensor:
        """Forward pass of 2D Pangu model with positional embedding.

        input: Tensor
          of shape (n_batch,  n_fields*n_vert, n_lat, n_lon)
        input_surface: Tensor
          of shape (n_batch, n_fields, n_lat, n_lon)

        Returns
        -------
        output: Tensor
          of shape (n_batch,  n_fields*n_vert, n_lat, n_lon)
        output_surface: Tensor
          of shape (n_batch, n_fields, n_lat, n_lon)
        """
        # Embed the input fields into patches

        input_data = input_data.reshape(
            input_data.shape[0],
            -1,
            self.padded_resolution[0],
            self.padded_resolution[1],
        )
        x = self._input_layer(input_data)
        x = x.reshape(
            x.shape[0], self.patch_resolution[0], self.patch_resolution[1], self.dim
        )
        x = x.reshape(x.shape[0], -1, self.dim)

        # Store the tensor for skip-connection
        skip = x.clone()

        for i in range(rollout):
            # Encoder, composed of two layers
            # Layer 1, shape e.g. (B, 181 * 360, C), C = 192 as in the original paper
            x = self.layer1(
                x, self.patch_resolution[0], self.patch_resolution[1], noise
            )

            in_loop_skip = x.clone()

            # Downsample from (B, 181 * 360, C) to (B, 91 * 180, 2C)
            x = self.downsample(
                x, self.patch_resolution[0], self.patch_resolution[1], noise
            )

            # Layer 2, shape (B, 91 * 180, 2C), C = 192 as in the original paper
            x = self.layer2(
                x, self.downsampled_resolution[0], self.downsampled_resolution[1], noise
            )

            # Decoder, composed of two layers
            # Layer 3, shape (B, 91 * 180, 2C), C = 192 as in the original paper
            x = self.layer3(
                x, self.downsampled_resolution[0], self.downsampled_resolution[1], noise
            )

            # Upsample from (B, 91 * 180, 2C) to (B, 181 * 360, C)
            x = self.upsample(x, noise)

            x = x + in_loop_skip

            # Layer 4, shape (B, 181 * 360, C), C = 192 as in the original paper
            x = self.layer4(
                x, self.patch_resolution[0], self.patch_resolution[1], noise
            )

        # Skip connect, in last dimension(C from 192 to 384)
        x = torch.cat((skip, x), dim=2)

        # Recover the output fields from patches
        output = self._output_layer(
            x, self.patch_resolution[0], self.patch_resolution[1]
        )
        return output


class NiceBNNNoEmbeddingModel(VIModule):
    def __init__(self, config: dict, device: torch.device = "cpu"):
        super().__init__()

        if "noise_dim" not in config["model"].keys():
            config["model"]["noise_dim"] = 32

        self.noise_dim = config["model"]["noise_dim"]

        model = NiceNoEmbeddingModel(config, device)
        convert_to_vimodule(model)

        self.model = model

        self.padding = model.padding
        self.padded_resolution = model.padded_resolution

    def forward(self, x, rollout=1):
        batch_size = x.shape[0]
        noise = torch.randn(batch_size, self.noise_dim).to(x.device, x.dtype)

        x = self.model(x, noise, rollout=rollout)

        return x
