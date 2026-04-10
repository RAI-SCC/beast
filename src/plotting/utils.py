import matplotlib.pyplot as plt

# ── KIT Color Palette ─────────────────────────────────────────────────────────
KIT_GREEN = (0 / 255, 150 / 255, 130 / 255)
KIT_BLUE = (70 / 255, 100 / 255, 170 / 255)
KIT_RED = (162 / 255, 34 / 255, 35 / 255)
KIT_ORANGE = (223 / 255, 155 / 255, 27 / 255)
KIT_LIGHTGREEN = (140 / 255, 182 / 255, 60 / 255)
KIT_PURPLE = (163 / 255, 16 / 255, 124 / 255)
KIT_CYAN = (35 / 255, 161 / 255, 224 / 255)
KIT_BROWN = (167 / 255, 130 / 255, 46 / 255)

KIT_GRAY10 = (0.9, 0.9, 0.9)
KIT_GRAY30 = (0.7, 0.7, 0.7)
KIT_GRAY60 = (0.4, 0.4, 0.4)

KIT_PALETTE = [
    KIT_BLUE,
    KIT_GREEN,
    KIT_ORANGE,
    KIT_RED,
    KIT_CYAN,
    KIT_PURPLE,
    KIT_LIGHTGREEN,
    KIT_BROWN,
]


def set_size(width="thesis", fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def init_matplotlib():
    plt.style.use("seaborn-v0_8-paper")

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # DPI
        "figure.dpi": 200,
    }

    plt.rcParams.update(tex_fonts)
