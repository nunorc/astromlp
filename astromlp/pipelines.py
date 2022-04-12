
import os, logging, copy
import tensorflow as tf
from statistics import mean
import numpy as np
import concurrent.futures

from .sdss.helper import Helper
from .sdss.predictor import Predictor
from .sdss.shared import CLASSES
from .sdss.skyserver import SkyServer

logger = logging.getLogger(__name__)

class PipelineResult:
    """ Class for storing the result of processing an object using a pipeline for
        processing SDSS galaxy object and a infer a set of properties using an ensemble of models.

        Attributes:
            result (object): location of the astromlp-models model store, detaults to `./astromlp-models/model_store`
    """
    def __init__(self, result):
        self.objid = result['objid']
        self.models = result['models']
        self.obj = result['obj']
        self.map = result['map']
        self.output = result['output']

    def __str__(self):
        return self._to_string()

    def __repr__(self):
        return f"PipelineResult({ self._to_string() })"

    def _to_string(self):
        vals = []
        for k, v in self.output.items():
            if isinstance(v, str):
                vals.append(f"{ k }='{ v }'")
            else:
                vals.append(f"{ k }={ v }")

        return ", ".join(vals)

class MapReducePipeline:
    """ Base class for processing an object using a map-reduce approach.

        Attributes:
            models (object): dictionary of outputs, and ensemble of models per output
            model_store (str): location of the astromlp-models model store, detaults to `./astromlp-models/model_store`
    """
    def __init__(self, models, model_store='./astromlp-models/model_store', helper=None):
        self.models = models
        self.model_store = model_store
        self.skyserver = SkyServer()

        if helper:
            self.helper = helper
        else:
            self.helper = Helper()

        if not os.path.exists(self.model_store):
            logger.warn(f'Model store not found: { self.model_store }')

        self.predictors = {}
        for k in self.models.keys():
            predictors = []
            for m in self.models[k]:
                filename = os.path.join(self.model_store, m)
                if os.path.exists(filename):
                    predictors.append(Predictor(m, model_store=self.model_store, helper=self.helper))
                else:
                    logger.warn(f'Model not found { filename }')
            self.predictors[k] = predictors

    def _get_predict(self, p, k, objid):
        _output = p.predict(objid, extra=False)['output']
        idx = p.y.index(k)

        return _output[idx]

    def process(self, objid):
        result = { 'objid': objid, 'models': copy.deepcopy(self.models) }
        result['obj'] = self.skyserver.get_obj(objid, wise=False)

        # map
        _map = {}
        for k in self.models.keys():
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for p in self.predictors[k]:
                    futures.append(executor.submit(self._get_predict, p, k, objid))

                    results = [x.result() for x in futures]
                _map[k] = results
        result['map'] = _map

        # reduce
        _outputs = {}
        for k in result['map'].keys():
            if k in CLASSES.keys():
                _outputs[k] = CLASSES[k][np.argmax(np.add.reduce(result['map'][k]))]
            else:
                _outputs[k] = mean(result['map'][k])
        result['output'] = _outputs

        return PipelineResult(result)









