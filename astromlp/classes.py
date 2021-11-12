
""" Base classes definition. """

import logging, os
import numpy as np
import tensorflow as tf

from .env import BASEDIR, BASEURL
from .utils import download_file

logger = logging.getLogger(__name__)

class Dataset:
    """ A wrapper class for datasets.

    Attributes
    ----------
        id : str
            dataset unique identifier

    """
    def __init__(self, id):
        self.id = id

        self.filename, self.url = self.__filename_and_url()
        self.__download()
        self.__setup()

    def __filename_and_url(self):
        filename = os.path.join(BASEDIR, 'datasets', self.id+'.npz')

        url = "/".join([BASEURL, 'datasets', self.id+'.npz'])

        return filename, url

    def __download(self):
        if not os.path.exists(self.filename):
            logger.info('data file not found, downloading..')
            download_file(self.url, self.filename)
            logger.info(f'data file saved as { self.filename }')

    def __setup(self):
        if os.path.exists(self.filename):
            npzfile = np.load(self.filename, allow_pickle=True)

            nparrays = []
            for file in npzfile.files:
                setattr(self, file, npzfile[file])
                nparrays.append(file)
            self.nparrays = nparrays
        else:
            logger.warning('data file not found')

    def __str__(self):
        return self.__to_string()

    def __repr__(self):
        return f"Dataset({ self.__to_string() })"

    def __to_string(self):
        return f"id='{ self.id }', filename='{ self.filename }', url={ self.url }, nparrays={ self.nparrays }"


class Model:
    """ A wrapper class for models.

    Attributes
    ----------
        id : str
            model unique identifier
        backend : str
            backend identifer, defaults to 'keras'

    Methods
    -------
        fit(X, y)
            fit model to data
        predict(X)
            predict values

    """
    def __init__(self, id, backend='keras'):
        self.id = id
        self.backend = backend
        self.history = None

        self.filename, self.url = self.__filename_and_url()
        self.__download()
        self.__setup()

    def __filename_and_url(self):
        ext = 'h5'   # TODO: update ext based on backend
        file = f'{ self.id }.{ ext }'

        filename = os.path.join(BASEDIR, 'models', file)

        url = '/'.join([BASEURL, 'models', file])

        return filename, url

    def __download(self):
        if not os.path.exists(self.filename):
            logger.info('data file not found, downloading..')
            download_file(self.url, self.filename)
            logger.info('data file saved as ' + self.filename)

    def __setup(self):
        if os.path.exists(self.filename):
            self.model = tf.keras.models.load_model(self.filename)
            self.desc = f'Inputs: { self.model.input_shape } -> Outputs: { self.model.output_shape }'
        else:
            logger.warning('data file not found')    

    def fit(self, X, y, batch_size=64, epochs=10):
        logger.info(f'batch_size={ batch_size} epochs={ epochs }')
        history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

        self.history = history

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return self.__to_string()

    def __repr__(self):
        return f"Model({ self.__to_string() })"

    def __to_string(self):
        return f"id='{ self.id }', filename='{ self.filename }', url='{ self.url }', backend='{ self.backend }'"

    def __call__(self, *args):
        return self.predict(*args)


