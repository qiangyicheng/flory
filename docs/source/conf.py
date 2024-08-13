# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'flory'
copyright = '2024, Yicheng Qiang'
author = 'Yicheng Qiang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
import pathlib

proj_path=pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(proj_path))

print(proj_path)

import flory

extensions = [
    "sphinx.ext.napoleon", # load napoleon after sphinx_autodoc_typehints
    'sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.coverage",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    'sphinx_paramlinks',
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
autodoc_member_order = "bysource"
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
autodoc_typehints = "signature" # Sphinx-native method
add_module_names = False # Remove namespaces from class/method signatures
always_use_bars_union = True # display Union's using the | operator for sphinx_autodoc_typehints
typehints_defaults = "comma"

napoleon_custom_sections = [('Returns', 'params_style')]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Readthedocs theme
html_theme = "sphinx_rtd_theme"

# html_theme = 'alabaster'
html_static_path = ['_static']
