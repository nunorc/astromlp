
import os, logging, random
import tensorflow as tf
import numpy as np

from .helper import Helper
from .shared import CLASSES, SM_FACTOR

logger = logging.getLogger(__name__)

# batch data generator
class DataGen(tf.keras.utils.Sequence):
    def __init__(self, ids, x=['img','spectra','bands'], y=['redshift', 'subclass'], classes=CLASSES, batch_size=64, helper=None):
        self.batch_size = batch_size
        self.ids = ids
        self.x = x
        self.y = y
        self.classes = classes
        
        if helper is None:
            self.helper = Helper()
        else:
            self.helper = helper

    def __getitem__(self, index):
        _from, _to = index*self.batch_size, (index+1)*self.batch_size
        _ids = self.ids[_from:_to]
        _y = self.y[_from:_to]

        X = {}
        if 'img' in self.x:
            X['img'] = self.helper.load_imgs(_ids)
        if 'fits' in self.x:
            X['fits'] = self.helper.load_fits(_ids)
        if 'spectra' in self.x:
            X['spectra'] = self.helper.load_spectras(_ids)
        if 'ssel' in self.x:
            X['ssel'] = self.helper.load_ssels(_ids)
        if 'bands' in self.x:
            X['bands'] = self.helper.load_bands(_ids)
        if 'wise' in self.x:
            X['wise'] = self.helper.load_wises(_ids)

        y = {}
        if 'redshift' in self.y:
            y['redshift'] = np.array(self.helper.y_list(_ids, 'redshift'))
        if 'subclass' in self.y:
            _y, _classes = self.helper.y_list_class(_ids, 'subclass', self.classes['subclass'])
            y['subclass'] = np.array(_y)
        if 'smass' in self.y:
            y['smass'] = np.array(self.helper.y_list(_ids, 'stellarmass')) / SM_FACTOR
        if 'gz2c' in self.y:
            _y, _classes = self.helper.y_list_class(_ids, 'gz2c', self.classes['gz2c'])
            y['gz2c'] = np.array(_y)

        return X, y

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.ids)







