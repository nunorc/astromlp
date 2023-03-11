
from setuptools import setup, find_packages
from astromlp import __version__

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(name = 'astromlp',
      version = __version__,
      url = 'https://github.com/nunorc/astromlp',
      author = 'Nuno Carvalho',
      author_email = 'narcarvalho@gmail.com',
      description = 'framework for building deep learning models and pipelines for astrophysics applications',
      long_description = long_description,
      long_description_content_type = 'text/x-rst',
      license = 'MIT',
      packages = find_packages(),
      install_requires = ['numpy', 'pandas', 'tensorflow', 'astropy', 'scikit-learn'])

