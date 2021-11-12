
""" astromlp init """

import logging

from .classes import Dataset, Model

logging.basicConfig(format = '%(asctime)s | %(levelname)s: %(message)s',
                    datefmt = "%Y-%m-%d %H:%M:%S",
                    level = logging.INFO)

__version__ = '0.0.1a1'
