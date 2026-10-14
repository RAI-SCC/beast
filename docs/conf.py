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
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

# Docstrings follow numpydoc. Each entry of an Attributes section becomes an
# attribute of the class, so autoapi does not list attributes separately.
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Single backticks mark names of parameters, attributes and other Python
# objects, as numpydoc recommends. They link to the object if it is documented
# and otherwise render as code.
default_role = "py:obj"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

myst_heading_anchors = 3
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = "BEAST"
