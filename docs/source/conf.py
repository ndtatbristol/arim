# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))
import arim  # noqa: E402

# -- Project information -----------------------------------------------------

# Keep identical to pyproject.toml
project = "arim"
author = "arim contributors"
copyright = f"2016–{datetime.now().year}, {author}"
release = arim.__version__


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {"navigation_depth": 4}
#     "navbar_start": ["navbar-logo"],
#     "navbar_center": [
#         {
#             "type": "dropdown",
#             "label": "Installation",
#             "items": [
#                 {"type": "link", "label": "User Installation", "url": "installation//index.html"},
#                 {"type": "link", "label": "Development", "url": "installation//development.html"},
#             ],
#         },
#     ],
#     "navbar_end": ["navbar-icon-links"],
#     "navbar_persistent": ["search-field"],
#     "navbar_align": "content",
#     "show_nav_level": 2,
#     "collapse_navigation": False,
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- Extension settings  -----------------------------------------------------

# Autodoc settings
# http://www.sphinx-doc.org/en/stable/ext/autodoc.html
autodoc_member_order = "groupwise"
autodoc_default_flags = ["members", "show-inheritance"]
# autodoc_default_flags = 'special-members'

# Autosummary settings
autosummary_generate = True
