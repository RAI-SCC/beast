# For an AB multiplication,
# define based on global in/hid/out dimensions
def ab_flop_global(in_dim, hidden_dim, out_dim):
    """Calc global FLOPs in distributed matrix multiplication."""
    return in_dim * out_dim * (2 * hidden_dim - 1) / 1e12


def transformer_block(ntokens, hidden_dim, transformer_dim, out_dim, heads):
    """Calc FLOPs and comm in a Transformer block."""
    # input shape = output of patch embedding shape
    flop = 0

    # qkv calculation
    # X: (n_tokens, hidden_dim) @ (hidden_dim, 3 * transformer_dimension)
    flop = flop + ab_flop_global(ntokens, hidden_dim, 3 * transformer_dim)

    # Q @ K.mT
    # Q: (n_tokens, transformer_dim) K.mT : (transformer_dim, n_tokens)
    flop = flop + heads * ab_flop_global(ntokens, transformer_dim // heads, ntokens)

    # (Q@K.mT) @ V
    # (Q@K.mT): (n_tokens, n_tokens) @ (n_tokens, transformer_dim)
    flop = flop + heads * ab_flop_global(ntokens, ntokens, transformer_dim // heads)

    # concatenator
    flop = (
        flop
        + ab_flop_global(ntokens, transformer_dim, out_dim)
        + ab_flop_global(ntokens, out_dim, out_dim)
    )

    return flop  # FLOP: GigaFLOP, Comm: GBs


def convolution(lat, lon, c_in, c_out):
    flop = c_out * (c_out + c_in * (lat * lon))
    return flop / 1e12


def swin(n_windows, n_tokens, hidden_dim, transformer_dim, out_dim, heads):
    flop = n_windows * transformer_block(
        n_tokens, hidden_dim, transformer_dim, out_dim, heads
    )
    # flop = flop + n_windows * 2 * ab_flop_global(out_dim, out_dim, out_dim)
    return flop


def nice_norm(out_dim, noise_dim):
    flop = 2 * ab_flop_global(noise_dim, out_dim, out_dim) + ab_flop_global(
        out_dim, out_dim, out_dim
    )
    return flop


def calc_flops_gb_model(config):
    """Calc FLOPs based on config file."""
    nlat, nlon = config["data"]["xlat"], config["data"]["xlon"]
    nvars_in, nvars_out = (
        config["model"]["n_input_channels"],
        config["model"]["n_output_channels"],
    )
    heads = config["model"]["n_attn_heads"]
    patch_size = config["model"]["patch_size"]
    embed_dim = config["model"]["embedding_dimension"]
    window_size_outer = config["model"]["window_size_outer"]
    window_size_inner = config["model"]["window_size_inner"]
    noise_dim = config["model"]["noise_dim"]
    n_attn_blocks = config["model"]["n_attn_blocks"]
    lbs, lsn = config["training"]["batch_size"], config["training"]["n_samples"]

    flop_patch_embedding = convolution(nlat, nlon, nvars_in, embed_dim)
    flop_downsample = convolution(
        nlat // patch_size, nlon // patch_size, embed_dim, embed_dim * 2
    )
    flop_upsample = convolution(
        nlat // patch_size, nlon // patch_size, embed_dim, embed_dim * 2
    )
    flop_patch_recovery = convolution(nlat, nlon, nvars_out, embed_dim)

    n_windows_outer = (
        nlat * nlon / patch_size**2 / window_size_outer[0] / window_size_outer[1]
    )
    n_windows_inner = (
        nlat * nlon / patch_size**2 / 4 / window_size_inner[0] / window_size_inner[1]
    )
    flop_transformer = swin(
        n_windows_outer,
        window_size_outer[0] * window_size_outer[1],
        embed_dim,
        embed_dim,
        embed_dim,
        heads,
    ) + nice_norm(embed_dim, noise_dim)
    flop_transformer_down = swin(
        n_windows_inner,
        window_size_inner[0] * window_size_inner[1],
        embed_dim * 2,
        embed_dim * 2,
        embed_dim * 2,
        heads,
    ) + nice_norm(embed_dim * 2, noise_dim)
    print(
        f"convolution flops: {
            (
                flop_patch_embedding
                + flop_downsample
                + flop_upsample
                + flop_patch_recovery
            )
            * lbs
            * lsn
        } "
    )
    print(
        f"attention flops: {
            (
                flop_transformer * (2 * (n_attn_blocks[0] + n_attn_blocks[2]))
                + flop_transformer_down * (2 * n_attn_blocks[1])
            )
            * lbs
            * lsn
        } "
    )
    flop = (
        flop_patch_embedding
        + flop_downsample
        + flop_upsample
        + flop_patch_recovery
        + flop_transformer * (2 * (n_attn_blocks[0] + n_attn_blocks[2]))
        + flop_transformer_down * (2 * n_attn_blocks[1])
    )
    return flop * lbs * lsn
