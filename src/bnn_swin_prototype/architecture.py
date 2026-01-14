import torch
import torch.nn as nn
from torch import permute, reshape
from torch.utils.checkpoint import checkpoint
from torch_blue.vi import VIModule, convert_to_vimodule

############################ Upsampling and Downsampling


class DownSample2D(nn.Module):
    """Downsample 2D."""

    def __init__(self, dim):
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
        self.norm = nn.LayerNorm(4 * dim)
        self.downsampling = (2, 2)

    def forward(self, x, n_patch_lat, n_patch_lon):
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
        x = self.norm(x)

        # Decrease the channels of the data to reduce computation cost
        x = self.linear(x)
        return x


class UpSample2D(nn.Module):
    """Upsample 2D."""

    def __init__(
        self, input_dim, output_dim, n_lat, n_lon, lat_crop=(0, 0), lon_crop=(0, 0)
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
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
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
        x = self.norm(x)

        # Mixup normalized tensors
        x = self.linear2(x)
        return x


################################ Patch embedding and recovery


class GroupedPatchEmbedding2D(nn.Module):
    """2D Patch Embedding operation."""

    def __init__(
        self,
        patch_size,
        kernel_size,
        dim,
        device,
        in_channels=72,
        num_surface=4,
        num_pressure=5,
        num_levels=13,
    ):
        """
        Initialize patch embedding operation.

        patch_size: Tuple(int, int)
            Number of pixels in (lat, lon) dimensions per patch
        dim: int
            Hidden dimension
        device: String
            Device that the operation is running on
        in_channels: int
            Total number of channels
            equal to n_pressure_levels * n_pressure_fields + n_surface_fields + masks
            = 13 pressure levels * 5 pressure fields + 4 surface fields + 3 masks
        """
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim

        self.num_surface = num_surface
        self.num_pressure = num_pressure
        self.num_levels = num_levels

        self.dim_factor = dim // in_channels
        if self.dim_factor < 1:
            raise ValueError(
                "Number of hidden features must be greater than the total number of input variables and masks."
            )
        # print(f"Embedding: dimensionality factor = {self.dim_factor}")

        self.mask_input_indices = in_channels - (
            num_surface + num_pressure * num_levels
        )
        self.mask_hidden_indices = dim - self.dim_factor * (
            num_surface + num_pressure * num_levels
        )

        self.conv_surface = nn.Conv2d(
            in_channels=num_surface,
            out_channels=(self.dim_factor * num_surface),
            kernel_size=kernel_size,
            stride=patch_size,
            groups=num_surface,
        )

        self.conv_masks = nn.Conv2d(
            in_channels=self.mask_input_indices,
            out_channels=self.mask_hidden_indices,
            kernel_size=kernel_size,
            stride=patch_size,
        )

        self.conv_pressure = nn.Conv2d(
            in_channels=(num_pressure * num_levels),
            out_channels=(self.dim_factor * num_pressure * num_levels),
            kernel_size=kernel_size,
            stride=patch_size,
            groups=num_pressure,
        )

    def forward(self, input):
        # Input should be padded already, according to the patch size
        input_surface = input[:, : self.num_surface, :, :]
        input_pressure = input[
            :,
            self.num_surface : (self.num_surface + self.num_pressure * self.num_levels),
            :,
            :,
        ]
        input_masks = input[
            :, (self.num_surface + self.num_pressure * self.num_levels) :, :, :
        ]

        x_surf = self.conv_surface(input_surface)
        x_pressure = self.conv_pressure(input_pressure)
        x_masks = self.conv_masks(input_masks)

        x = torch.cat((x_surf, x_pressure, x_masks), dim=1)

        # Reshape x for calculation of linear projections

        # Dimensions before: (batch_size, fields, latitude, longitude)
        # Dimensions after: (batch_size, latitude, longitude, fields)
        x = permute(x, (0, 2, 3, 1))

        # Dimensions after: (batch_size,  latitude * longitude, fields)
        x = reshape(x, shape=(x.shape[0], x.shape[1] * x.shape[2], x.shape[-1]))

        return x


class GroupedPatchRecovery2D(nn.Module):
    """2D Patch recovery option."""

    def __init__(
        self,
        patch_size,
        kernel_size,
        dim,
        out_channels=69,
        num_surface=4,
        num_pressure=5,
        num_levels=13,
        dim_factor=None,
    ):
        """
        2D Patch recovery.

        A transpose convolution operation is performed over the pressure and surface outputs to recover the forecasted fields.

        patch_size: Tuple(int, int, int)
            Number of pixels in (lat, lon) dimensions per patch
        dim: int
          Hidden dimension
        out_channels: int
          Total number of channels
          equal to n_pressure_levels * n_pressure_fields + n_surface_fields
          = 13 pressure levels * 5 pressure fields + 4 surface fields = 69
        """
        super().__init__()

        self.patch_size = patch_size
        self.dim = dim

        self.num_surface = num_surface
        self.num_pressure = num_pressure
        self.num_levels = num_levels

        if dim_factor is None:
            self.dim_factor = dim // (2 * out_channels)
        else:
            self.dim_factor = dim_factor

        if self.dim_factor < 1:
            raise ValueError(
                "Number of hidden features must be greater than the total number of input variables and masks."
            )

        # print(f"Recovery: dimensionality factor = {self.dim_factor}")

        self.mask_hidden_indices = dim // 2 - self.dim_factor * (
            num_surface + num_pressure * num_levels
        )

        self.conv_surface = nn.ConvTranspose2d(
            out_channels=num_surface,
            in_channels=(2 * self.dim_factor * num_surface),
            kernel_size=kernel_size,
            stride=patch_size,
            groups=num_surface,
        )

        self.conv_other = nn.ConvTranspose2d(
            out_channels=out_channels,
            in_channels=(2 * self.mask_hidden_indices),
            kernel_size=kernel_size,
            stride=patch_size,
        )

        self.conv_pressure = nn.ConvTranspose2d(
            out_channels=(num_pressure * num_levels),
            in_channels=(2 * self.dim_factor * num_pressure * num_levels),
            kernel_size=kernel_size,
            stride=patch_size,
            groups=num_pressure,
        )

    def forward(self, x, n_patch_lat, n_patch_lon):
        """
        2D inverse operation of the patch embedding operation.

        x: Tensor
          of shape (n_batch, n_patch_lat*n_patch_lon, 2*hidden_dim)
        n_patch_lat: int
          number of patches in the lat dimension
        n_patch_lon: int
          number of patches in the lon dimension

        Returns
        -------
        output: Tensor
          of shape (n_batch, n_levels * n_fields, n_lat, n_lon)
        output_surface: Tensor
          of shape (n_batch, n_fields, n_lat, n_lon)
        """
        # Reshape x back to three dimensions
        # Dimensions: (nData, pressure level * latitude * longitude, fields)
        x = permute(x, (0, 2, 1))
        # Dimensions: (nData, fields, pressure level, latitude, longitude)
        x_shape = x.shape
        x = reshape(x, shape=(x_shape[0], x_shape[1], n_patch_lat, n_patch_lon))

        x_mix = x.reshape(x_shape[0], 2, x_shape[1] // 2, n_patch_lat, n_patch_lon)
        x_mix = x_mix.transpose(1, 2)
        x_mix = x_mix.reshape(x_shape[0], x_shape[1], n_patch_lat, n_patch_lon)

        x_surf = x_mix[:, : (2 * self.dim_factor * self.num_surface), :, :]
        x_pressure = x_mix[
            :,
            (2 * self.dim_factor * self.num_surface) : (
                2 * self.dim_factor * self.num_surface
                + 2 * self.dim_factor * self.num_pressure * self.num_levels
            ),
            :,
            :,
        ]
        x_other = x_mix[
            :,
            (
                2 * self.dim_factor * self.num_surface
                + 2 * self.dim_factor * self.num_pressure * self.num_levels
            ) :,
            :,
            :,
        ]

        out_surface = self.conv_surface(x_surf)
        out_pressure = self.conv_pressure(x_pressure)
        out_mod = self.conv_other(x_other)

        output = 0.75 * torch.cat((out_surface, out_pressure), dim=1) + 0.25 * out_mod

        # Note: Output is still padded. Cropping only occurs outside of training loop.
        return output


####################################### Attention mechanism


class MLP(nn.Module):
    """MLP."""

    def __init__(
        self,
        dim,
        hidden_dim=None,
        # dropout_rate
    ):
        """
        Initialize MLP.

        dim: int
        dropout_rate: float
        """
        super().__init__()
        """MLP layers, same as most vision transformer architectures."""

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()
        # self.drop = Dropout(p=dropout_rate)

    def forward(self, x):
        """Forward function of MLP."""
        x = self.linear1(x)
        x = self.activation(x)
        # x = self.drop(x)
        x = self.linear2(x)
        # x = self.drop(x)
        return x


class EarthAttention2DNoBias(nn.Module):
    """
    2D Attention mechanism with absolute position bias term.

    Attention(Q, K, V) = (SoftMax(Q@K^T/sqrt(D)) @ V
    """

    def __init__(
        self,
        dim,
        heads,
        # dropout_rate,
        window_size,
    ):
        """
        Initialize base class for attention.

        dim: int
            Size of hidden dimension
        heads: int
            Number of attention heads
        dropout_rate: float
            probability that value is dropped during training
        window_size: Tensor
            Tensor of length(3) describing the window size in (vert, lat, long) dimensions to be used in the attention mechanism.
        """
        super().__init__()

        # Initialize several operations
        # Should make sense to use dim*3 to generate the vectors for a qkv matrix
        self.linear1 = nn.Linear(dim, dim * 3, bias=True)
        self.linear2 = nn.Linear(dim, dim)
        self.Softmax = nn.Softmax(dim=-1)
        # self.dropout = Dropout(dropout_rate)

        # Store several attributes
        self.head_number = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.window_size = window_size
        self.total_window_size = torch.prod(window_size)

    def calculate_attention(self, x):
        """
        Calculate attention.

        x: Tensor
            of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)

        Returns
        -------
        attention: Tensor
            of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
        value: Tensor
            of shape (n_batch * n_windows, n_heads, n_patch_in_window, hidden_dim // n_heads)
        original_shape: torch.Size
            of x at input
        """
        # Linear layer to create query, key and value
        # Record the original shape of the input BEFORE linear layer (correct?)
        original_shape = x.shape  # x shape: (B*n_windows, W0*W1*W2, dim)

        x = self.linear1(x)  # x shape: (B*n_windows, W0*W1*W2, 3*dim)
        # reshape the data to calculate multi-head attention
        # q shape: (B*n_windows, W0*W1*W2, 3, nHead, dim/nHead)
        qkv = reshape(
            x,
            shape=(
                x.shape[0],
                x.shape[1],
                3,
                self.head_number,
                self.dim // self.head_number,
            ),
        )
        # query after permute: (nB, nHead, W0*W1*W2, C/nHead)
        query, key, value = permute(qkv, (2, 0, 3, 1, 4))
        # Scale the attention
        query = query * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        # Attention shape: B*n_windows, nHeads, N, N (n_patch_in_window = W0*W1*W2)
        attention = query @ key.transpose(-2, -1)  # @ denotes matrix multiplication
        return attention, value, original_shape

    def mask_attention(self, x, mask, attention, original_shape):
        """
        Masks attention between non-adjacent pixels. -10000 is added to masked elements.

        x: Tensor
        mask: Tensor
            of shape (n_windows, n_patch_in_window, n_patch_in_window)
        attention: Tensor
            of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
        value: Tensor
            of shape (n_batch * n_windows, n_heads, n_patch_in_window, hidden_dim // n_heads)
        original_shape: torch.Size
            of x at input = (n_batch * n_windows, n_patch_in_window, hidden_dim)

        Returns
        -------
        attention: Tensor
            masked attention of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
        """
        if attention.get_device() < 0:
            device = "cpu"
        else:
            device = attention.get_device()
        mask = mask.to(device)

        # attention shape: n_windows * B, nHeads, W[0]*W[1]*W[2], W[0]*W[1]*W[2]
        # mask shape: n_windows, W[0]*W[1]*W[2], W[0]*W[1]*W[2]
        n_windows = mask.shape[0]  # n_windows
        n_patch_in_window = original_shape[1]  # W0*W1*W2
        n_windows_total = original_shape[0]  # B*n_windows

        # original attention shape: (n_windows * B, nHead, W0*W1*W2, W0*W1*W2)
        attention = attention.view(
            n_windows_total // n_windows,
            n_windows,
            self.head_number,
            n_patch_in_window,
            n_patch_in_window,
        )  # (B, n_windows, nHead, W0*W1*W2, W0*W1*W2)
        attention = (
            attention + mask.unsqueeze(1).unsqueeze(0)
        )  # (B, n_windows, nHead, W0*W1*W2, W0*W1*W2) + (1,  n_windows, 1, W0*W1*W2, W0*W1*W2)
        attention = attention.view(
            n_windows_total, self.head_number, n_patch_in_window, n_patch_in_window
        )  # (B*n_windows, nHead, W0*W1*W2, W0*W1*W2)
        return attention

    def activate(self, attention):
        """
        Apply Softmax activation to attention.

        attention: Tensor
            of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)

        Returns
        -------
        attention: Tensor
            of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
        """
        attention = self.Softmax(attention)
        # attention = self.dropout(attention)
        return attention

    def mixing_linear_layer(
        self, attention, value, original_shape
    ):  # attention.view(batches*n_windows, nHeads, N, N)
        """
        Compute (Q@K^T)@V and linear layer to allow mixing.

        attention: Tensor
            = (Q@K^T)/(sqrt(D)) + bias + masked_attention + Softmax activation.
            of shape (n_batch * n_windows, n_heads, n_patch_in_window, n_patch_in_window)
        value: Tensor
            of shape (n_batch * n_windows, n_heads, n_patch_in_window, hidden_dim // n_heads)
        original_shape: torch.Size
            of x at input = (n_batch * n_windows, n_patch_in_window, hidden_dim)

        Returns
        -------
        x: Tensor
            of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
        """
        # x shape after mat mul: (nB, nHeads, W0*W1*W2, C/nHeads)
        x = attention @ value
        # x shape after transpose: (B*n_windows, W0*W1*W2, nHeads, C/nHeads)
        x = torch.transpose(x, 1, 2)

        # original shape: (B*n_windows, W0*W1*W2, dim)
        # Reshape tensor to the original shape
        x = reshape(x, shape=original_shape)

        # Linear layer to post-process operated tensor
        x = self.linear2(x)
        # x = self.dropout(x)
        return x

    def forward(self, x, mask):
        """
        Forward pass of 2D attention mechanism with absolute positional bias term.

        x: Tensor
            of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
        mask: Tensor
            of shape (n_windows, n_patch_in_window, n_patch_in_window)

        Returns
        -------
        x: Tensor
            of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
        """
        # Linear layer to create query, key and value
        # Record the original shape of the input BEFORE linear layer (correct?)
        if x.get_device() < 0:
            device = "cpu"
        else:
            device = x.get_device()
        mask = mask.to(device=device, dtype=x.dtype)

        # x shape: (n_batch*n_windows, W0, W1, W2, C)
        original_shape = x.shape
        n_windows_total = original_shape[0]

        # x shape: (B*n_windows, W0*W1*W2, C)
        x = self.linear1(x)

        # reshape the data to calculate multi-head attention
        qkv = reshape(
            x,
            shape=(
                x.shape[0],
                x.shape[1],
                3,
                self.head_number,
                self.dim // self.head_number,
            ),
        )
        query, key, value = permute(qkv, (2, 0, 3, 1, 4))

        # Scale the attention
        query = query * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        # Attention shape: nB * n_windows, nHeads, N, N (n_patch_in_window = W0*W1*W2)
        attention = query @ key.transpose(-2, -1)  # @ denotes matrix multiplication

        # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
        # from SWIN paper
        n_windows = mask.shape[0]
        n_patch_in_window = original_shape[1]

        attention = attention.view(
            n_windows_total // n_windows,
            n_windows,
            self.head_number,
            n_patch_in_window,
            n_patch_in_window,
        )
        attention = attention + mask.unsqueeze(1).unsqueeze(0)
        attention = attention.view(
            -1, self.head_number, n_patch_in_window, n_patch_in_window
        )

        attention = self.activate(attention)

        # Calculated the tensor after spatial mixing.
        x = self.mixing_linear_layer(attention, value, original_shape)
        return x


class EarthSpecificBlock2DNoBias(nn.Module):
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
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
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

    def forward(self, x, n_patch_lat, n_patch_lon, roll):
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
        x = shortcut + self.norm1(x)
        # x = x + self.drop_path(self.norm2(self.linear(x)))
        x = x + self.norm2(self.linear(x))

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


class EarthSpecificLayer2DNoBias(nn.Module):
    """Earth-Specific Layer for 2D attention mechanism without bias term."""

    def __init__(
        self,
        depth,
        dim,
        # drop_path_ratio_list,
        heads,
        device,
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
                EarthSpecificBlock2DNoBias(
                    dim,
                    # drop_path_ratio_list[i],
                    heads,
                    input_shape=[0, 0],
                    device=device,
                    input_resolution=(0, 0),
                    window_size=window_size,
                ).to(device)
                for i in torch.arange(depth)
            ]
        )
        self.checkpointing = checkpointing

    def forward(self, x, n_patch_lat, n_patch_lon):
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
                    x = self.blocks[i](x, n_patch_lat, n_patch_lon, False)
                else:
                    x = self.blocks[i](x, n_patch_lat, n_patch_lon, True)
        return x


########################################### Full model


class NoEmbeddingModel(nn.Module):
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
        self.layer1 = EarthSpecificLayer2DNoBias(
            depth=6,
            dim=self.dim,
            # drop_path_ratio_list=drop_list[:6],
            heads=12,
            device=device,
            window_size=window_size,
            checkpointing=config["training"]["checkpointing"],
        )

        self.layer2 = EarthSpecificLayer2DNoBias(
            depth=2,
            dim=2 * self.dim,
            # drop_path_ratio_list=drop_list[6:],
            heads=24,
            device=device,
            window_size=window_size,
            checkpointing=config["training"]["checkpointing"],
        )

        self.layer3 = EarthSpecificLayer2DNoBias(
            depth=2,
            dim=2 * self.dim,
            # drop_path_ratio_list=drop_list[6:],
            heads=24,
            device=device,
            window_size=window_size,
            checkpointing=config["training"]["checkpointing"],
        )

        self.layer4 = EarthSpecificLayer2DNoBias(
            depth=6,
            dim=self.dim,
            # drop_path_ratio_list=drop_list[:6],
            heads=12,
            device=device,
            window_size=window_size,
            checkpointing=config["training"]["checkpointing"],
        )

        # Upsample and downsample blocks
        self.upsample = UpSample2D(
            input_dim=self.dim * 2,
            output_dim=self.dim,
            n_lat=self.downsampled_resolution[0],
            n_lon=self.downsampled_resolution[1],
            lat_crop=(0, self.patch_resolution[0] % 2),
            lon_crop=(0, self.patch_resolution[1] % 2),
        )

        self.downsample = DownSample2D(dim=self.dim, downsampling=(2, 2))

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

    def forward(self, input_data: torch.Tensor, rollout: int = 1) -> torch.Tensor:
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
            x = self.layer1(x, self.patch_resolution[0], self.patch_resolution[1])

            in_loop_skip = x.clone()

            # Downsample from (B, 181 * 360, C) to (B, 91 * 180, 2C)
            x = self.downsample(x, self.patch_resolution[0], self.patch_resolution[1])

            # Layer 2, shape (B, 91 * 180, 2C), C = 192 as in the original paper
            x = self.layer2(
                x, self.downsampled_resolution[0], self.downsampled_resolution[1]
            )

            # Decoder, composed of two layers
            # Layer 3, shape (B, 91 * 180, 2C), C = 192 as in the original paper
            x = self.layer3(
                x, self.downsampled_resolution[0], self.downsampled_resolution[1]
            )

            # Upsample from (B, 91 * 180, 2C) to (B, 181 * 360, C)
            x = self.upsample(x)

            x = x + in_loop_skip

            # Layer 4, shape (B, 181 * 360, C), C = 192 as in the original paper
            x = self.layer4(x, self.patch_resolution[0], self.patch_resolution[1])

        # Skip connect, in last dimension(C from 192 to 384)
        x = torch.cat((skip, x), dim=2)

        # Recover the output fields from patches
        output = self._output_layer(
            x, self.patch_resolution[0], self.patch_resolution[1]
        )
        return output


def BNNNoEmbeddingModel(config: dict, device: torch.device = "cpu") -> VIModule:
    model = NoEmbeddingModel(config, device)
    convert_to_vimodule(model)
    return model
