
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
    def __init__(self, model, model_store='../astromlp-models.git/model_store', x=None, y=None, helper=None, tmp_dir='/tmp/mysdss'):
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

    def predict(self, objid, extra=True, return_input=True):
        obj = self.skyserver.get_obj(objid)
        if obj is None:
            return None

        _input, _result = {}, { 'obj': obj }
        if 'img' in self.x:
            filename = os.path.join(self.tmp_dir, str(objid)+'.jpg')
            if self.helper.save_img(obj, filename=filename):
                _input['img'] =  np.array([self.helper.load_img(filename)])

                if extra:
                    with open(filename, 'rb') as fin:
                        _result['_img_base64'] = base64.b64encode(fin.read()).decode('utf-8')

        if 'fits' in self.x:
            filename = os.path.join(self.tmp_dir, f"{ obj['objid'] }.npy")
            data = self.helper.save_fits(obj, filename=filename, base_dir=self.tmp_dir)
            _input['fits'] = np.array([data])

            if extra:
                _result['_fits_base64'] = []
                for i in range(5):
                    _tmp_file = f"{ obj['objid'] }_band_{ i }.jpg"
                    if not os.path.exists(_tmp_file):
                        plt.imsave(os.path.join(self.tmp_dir, _tmp_file), data[:, :, i])
                    with open(os.path.join(self.tmp_dir, _tmp_file), 'rb') as fin:
                        _result['_fits_base64'].append(base64.b64encode(fin.read()).decode('utf-8'))

        if 'spectra' in self.x:
            filename = os.path.join(self.tmp_dir, f"{ obj['objid'] }_spectra.csv")
            self.helper.save_spectra(obj, filename=filename)
            spectra, waves = self.helper.load_spectra(filename)
            _input['spectra'] = np.array([spectra])

            if extra:
                _result['_waves_spectra'] = waves.tolist()

        if 'ssel' in self.x:
            filename = os.path.join(self.tmp_dir, f"{ obj['objid'] }_ssel.csv")
            spectra_filename = os.path.join(self.tmp_dir, f"{ obj['objid'] }_spectra.csv")
            self.helper.save_ssel(obj, filename=filename, spectra_filename=spectra_filename)
            ssel, waves = self.helper.load_ssel(filename)
            _input['ssel'] = np.array([ssel])

            if extra:
                _result['_waves_ssel'] = waves.tolist()

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

        return _result


