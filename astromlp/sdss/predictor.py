
import os, random, requests, time, pathlib, base64, tempfile, io, logging, subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
from ImageCutter.ImageCutter import FITSImageCutter
from astropy.io import fits
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from .helper import Helper
from .skyserver import SkyServer
from .shared import CLASSES

class Predictor:
    """ A predictor class for predicting data using `astromlp-models <https://github.com/nunorc/astromlp-models>`_.

        Attributes:
            model (str): the astromlp-model identifier (eg, `i2r`, `f2s`)
            model_store (str): location of the model store, defaults to `'./astromlp-models/model_store'`
    """
    def __init__(self, model, model_store='./astromlp-models/model_store', x=None, y=None, helper=None, tmp_dir='/tmp/mysdss'):
        if helper:
            self.helper = helper
        else:
            self.helper = Helper()

        self.skyserver = SkyServer()

        if model:
            if isinstance(model, str):
                if not os.path.exists(model):
                    filename = os.path.join(model_store, model)
                else:
                    filename = model
                if os.path.exists(filename):
                    self.model = tf.keras.models.load_model(filename)
                else:
                    logger.warn(f'Model not found { filename }')
            else:
                self.model = model

        if x:
            self.x = x
        else:
            if self.model:
                self.x = self.model.input_names
        if y:
            self.y = y
        else:
            if self.model:
                self.y = self.model.output_names

        self.tmp_dir = tmp_dir
        pathlib.Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)

    def _handle_img(self, obj, extra=True):
        _input, _extra = None, None

        if self.helper._has_img(obj['objid']):
            filename = self.helper._img_filename(obj['objid'])
        else:
            filename = os.path.join(self.tmp_dir, str(obj['objid'])+'.jpg')

        if self.helper.save_img(obj, filename=filename):
            _input = np.array([self.helper.load_img(filename)])

            if extra:
                with open(filename, 'rb') as fin:
                    _extra = base64.b64encode(fin.read()).decode('utf-8')

        return _input, _extra

    def _handle_fits(self, obj, extra=True):
        _input, _extra = None, None

        if self.helper._has_fits(obj['objid']):
            filename = self.helper._fits_filename(obj['objid'])
        else:
            filename = os.path.join(self.tmp_dir, f"{ obj['objid'] }.npy")

        print('filename', filename)
        print('base_dir', self.tmp_dir)
        data = self.helper.save_fits(obj, filename=filename, base_dir=self.tmp_dir)
        _input = np.array([data])

        if extra:
            _extra = []
            for i in range(5):
                _tmp_file = f"{ obj['objid'] }_band_{ i }.jpg"
                if not os.path.exists(_tmp_file):
                    plt.imsave(os.path.join(self.tmp_dir, _tmp_file), data[:, :, i])
                with open(os.path.join(self.tmp_dir, _tmp_file), 'rb') as fin:
                    _extra.append(base64.b64encode(fin.read()).decode('utf-8'))

        return _input, _extra

    def _handle_spectra(self, obj, extra=True):
        _input, _extra = None, None

        if self.helper._has_spectra(obj['objid']):
            filename = self.helper._spectra_filename(obj['objid'])
        else:
            filename = os.path.join(self.tmp_dir, f"{ obj['objid'] }_spectra.csv")

        self.helper.save_spectra(obj, filename=filename)
        spectra, waves = self.helper.load_spectra(filename)
        _input = np.array([spectra])

        if extra:
            _extra = waves.tolist()

        return _input, _extra

    def _handle_ssel(self, obj, extra=True):
        _input, _extra = None, None

        if self.helper._has_ssel(obj['objid']):
            filename = self.helper._ssel_filename(obj['objid'])
        else:
            filename = os.path.join(self.tmp_dir, f"{ obj['objid'] }_ssel.csv")
 
        spectra_filename = os.path.join(self.tmp_dir, f"{ obj['objid'] }_spectra.csv")
        self.helper.save_ssel(obj, filename=filename, spectra_filename=spectra_filename)
        ssel, waves = self.helper.load_ssel(filename)
        _input = np.array([ssel])

        if extra:
            _extra = waves.tolist()

        return _input, _extra

    def predict(self, objid, extra=True, return_input=True):
        """ Perform a prediction on a model for a SDSS object identifier.

            Args:
                objid (id): SDSS object identifier
            Returns:
                an object where the key `output` contains the resulting prediction

        """
        need_wise = 'wise' in self.x
        obj = self.helper.get_obj(objid, wise=need_wise)
        if obj is None:
            return None

        _input, _extra, _result = {}, {}, { 'obj': obj }
        if 'img' in self.x:
            _input['img'], _extra['img'] = self._handle_img(obj, extra=extra)

        if 'fits' in self.x:
            _input['fits'], _extra['fits'] = self._handle_fits(obj, extra=extra)

        if 'spectra' in self.x:
            _input['spectra'], _extra['spectra'] = self._handle_spectra(obj, extra=extra)

        if 'ssel' in self.x:
            _input['ssel'], _extra['ssel'] = self._handle_ssel(obj, extra=extra)

        if 'bands' in self.x:
            _input['bands'] = np.array([[obj['modelMag_u'], obj['modelMag_g'], obj['modelMag_r'], obj['modelMag_i'], obj['modelMag_z']]])

        if 'wise' in self.x:
            _input['wise'] = np.array([[obj['w1mag'], obj['w2mag'], obj['w3mag'], obj['w4mag']]])

        _output = self.model.predict(_input)

        _result['_classes'] = {}
        for i in self.y:
            if i in CLASSES:
                _result['_classes'][i] = CLASSES[i][np.argmax(_output[self.y.index(i)])]

        _result['x'] = self.x
        _result['y'] = self.y

        if return_input:
            _result['input'] = dict([(x, _input[x].tolist()) for x in _input.keys()])

        # predict output
        if len(self.y) > 1:
            _result['output'] = [x.tolist()[0] for x in _output]
            _result['output'] = [x[0] if len(x)==1 else x for x in _result['output'] ]
        else:
            l = _output.tolist()
            if len(l[0]) == 1:
                _result['output'] = l[0]
            else:
                _result['output'] = l

        if extra:
            _result['extra'] = _extra

        return _result


