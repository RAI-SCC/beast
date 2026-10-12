"""Sphinx configuration for the BEAST documentation."""

project = "BEAST"

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

# API reference, generated from the source without importing it, so the
# documentation builds without torch, jigsaw or GPU libraries installed.
autoapi_dirs = ["../beast"]
autoapi_root = "api"
autoapi_python_class_content = "class"
autoapi_member_order = "groupwise"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

# Docstrings follow numpydoc
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Render Attributes sections as fields, since autoapi also lists the attributes
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

myst_heading_anchors = 3
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = "BEAST"
