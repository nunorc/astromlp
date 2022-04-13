
import os, sys

sys.path.insert(0, os.path.abspath('../../'))

from astromlp import __version__


# Project information

project = 'astromlp'
copyright = '2022, Nuno Ramos Carvalho'
author = 'Nuno Ramos Carvalho'
version = __version__


# General configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon' ]
templates_path = ['_templates']
exclude_patterns = []


# Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_mock_imports = ['astromlp.api']
