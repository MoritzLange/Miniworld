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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
import os
from typing import Any, Dict

import miniworld

project = "Miniworld"
copyright = "2022 Farama Foundation"
author = "Farama Foundation"

# The full version, including alpha/beta/rc tags
release = miniworld.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "Miniworld Documentation"
html_baseurl = "https://miniworld.farama.org"
html_copy_source = False
html_favicon = "_static/img/miniworld-favicon.png"
html_theme_options = {
    "light_logo": "img/miniworld.svg",
    "dark_logo": "img/miniworld-white.svg",
    "gtag": "",
    "versioning": True,
}
html_context: Dict[str, Any] = {}
html_context["conf_py_path"] = "/docs/"
html_context["display_github"] = True
html_context["github_user"] = "Farama-Foundation"
html_context["github_repo"] = "Miniworld"
html_context["github_version"] = "master"
html_context["slug"] = "miniworld"

html_static_path = ["_static"]
html_css_files = []

# -- Generate Changelog -------------------------------------------------

sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
