
import os, logging, copy
import tensorflow as tf
from statistics import mean
import numpy as np
import concurrent.futures

from .sdss.helper import Helper
from .sdss.predictor import Predictor
from .sdss.shared import CLASSES
from .sdss.skyserver import SkyServer

from .pipelines import MapReducePipeline

logger = logging.getLogger(__name__)

class One2One(MapReducePipeline):
    """ Pipeline for processing SDSS galaxy object and a infer a set of properties using an ensemble of models.

        Attributes:
            model_store (str): location of the astromlp-models model store, detaults to `./astromlp-models.git/model_store`
        Returns:
            :code:`PipelineResult`
    """
    def __init__(self, model_store='./astromlp-models.git/model_store', helper=None):
        models = {
            'redshift': ['i2r', 'f2r', 's2r', 'ss2r', 'b2r', 'w2r'],
            'smass': ['i2sm', 'f2sm', 's2sm', 'ss2sm', 'b2sm', 'w2sm'],
            'subclass': ['i2s', 'f2s', 's2s', 'ss2s', 'b2s', 'w2s'],
            'gz2c': ['i2g', 'f2g', 's2g', 'ss2g', 'b2g', 'w2g']
        }
        MapReducePipeline.__init__(self, models, model_store=model_store, helper=helper)

class CherryPicked(MapReducePipeline):
    """ Pipeline for processing SDSS galaxy object and a infer a set of properties using an ensemble of models.

        Attributes:
            model_store (str): location of the astromlp-models model store, detaults to `./astromlp-models.git/model_store`
        Returns:
            :code:`PipelineResult`
    """
    def __init__(self, model_store='./astromlp-models.git/model_store', helper=None):
        models = {
            'redshift': ['s2r', 'ss2r', 'iFsSSbW2r'],
            'smass': ['f2sm'],
            'subclass': ['iFsSSbW2s'],
            'gz2c': ['i2g', 'f2g', 'iFsSSbW2g']
        }
        MapReducePipeline.__init__(self, models, model_store=model_store, helper=helper)
