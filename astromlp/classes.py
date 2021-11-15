
""" Base classes definition. """

import logging, os, pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .env import BASEDIR, BASEURL
from .utils import filename_and_url, download_file

logger = logging.getLogger(__name__)

class Data:
    """ A wrapper class for a data source.

    Attributes
    ----------
        uid : str
            data source unique identifier
    """
    def __init__(self, uid):
        self.uid = uid
        self.filename, self.url = filename_and_url(uid, 'data', 'npz')

        # download data file if required
        if not os.path.exists(self.filename):
            logger.info('data file not found, downloading..')
            download_file(self.url, self.filename)
            logger.info(f'data file saved as { self.filename }')

        # setup
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
        return f"Data({ self.__to_string() })"

    def __to_string(self):
        return f"uid='{ self.uid }', nparrays={ self.nparrays }"

class Dataset:
    """ A wrapper class for datasets.

    Attributes
    ----------
        features : nparray
            dataset features tensor
        target : nparray
            dataset target tensor
    """
    def __init__(self, features=None, target=None):
        self.features = features
        self.target = target

    def __str__(self):
        return self.__to_string()

    def __repr__(self):
        return f"Dataset({ self.__to_string() })"

    def __to_string(self):
        if type(self.features) is list:
            features = [x.shape for x in self.features]
            target = [x.shape for x in self.target]
        else:
            features = self.features.shape
            target = self.target.shape

        return f"features={ features }, target={ target }"


class Model:
    """ A wrapper class for models.

    Attributes
    ----------
        uid : str
            model unique identifier
        val : bool
            use validation set while fitting if possible, defaults to `False`
        backend : str
            backend identifer, defaults to 'keras'

    Methods
    -------
        fit(X, y)
            fit model to data
        predict(X)
            predict values

    """
    def __init__(self, uid, backend='keras', batch_size=64, epochs=10, validation_data=False, do_fit=True):
        self.uid = uid
        self.backend = backend
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_data = validation_data
        self.do_fit = do_fit

        self.history = None
        self.filename, self.url = filename_and_url(uid, 'models', 'h5')   # TODO: update ext based on backend

        # download model file if required
        if not os.path.exists(self.filename):
            logger.info('model file not found, downloading..')
            download_file(self.url, self.filename)
            logger.info(f'model file saved as { self.filename }')

        # setup model
        if os.path.exists(self.filename):
            self.model = tf.keras.models.load_model(self.filename)
            self.desc = f'Inputs: { self.model.input_shape } -> Outputs: { self.model.output_shape }'
        else:
            logger.warning('data file not found')    

    def fit(self, dataset, **kwargs):
        if type(dataset.features) is list:
            if self.validation_data:
                history = self.model.fit(dataset.features[0], dataset.target[0],
                                         validation_data=(dataset.features[1], dataset.target[1]),
                                         batch_size=self.batch_size, epochs=self.epochs)
            else:
                history = self.model.fit(dataset.features[0], dataset.target[0],
                                         batch_size=self.batch_size, epochs=self.epochs)
        else:
            history = self.model.fit(dataset.features, dataset.target,
                                     batch_size=self.batch_size, epochs=self.epochs)

        self.history = history

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return self.__to_string()

    def __repr__(self):
        return f"Model({ self.__to_string() })"

    def __to_string(self):
        return f"uid='{ self.uid }', backend='{ self.backend }', validation_data='{ self.validation_data }', do_fit={ self.do_fit }"

    def __call__(self, *args):
        if self.do_fit:
            return self.fit(*args)
        else:
            return self.predict(*args)


class Splitter:
    def __init__(self, test_size=0.2, random_state=None, shuffle=True):
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

    def transform(self, dataset):
        X = dataset.features
        y = dataset.target

        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, shuffle=self.shuffle)
            dataset.features = [X_train, X_test]
            dataset.target = [y_train, y_test]
        else:
            X_train, X_test = train_test_split(X, test_size=self.test_size, random_state=self.random_state, shuffle=self.shuffle)
            dataset.features = [X_train, X_test]

        return dataset

    def __str__(self):
        return self.__to_string()

    def __repr__(self):
        return f"Splitter({ self.__to_string() })"

    def __to_string(self):
        return f"test_size={ self.test_size }, random_state={ self.random_state }"

    def __call__(self, dataset):
        return self.transform(dataset)


class Scaler:
    def __init__(self):
        self.backend = StandardScaler()

    def transform(self, dataset):
        dataset.features = self.backend.fit_transform(dataset.features)

        return dataset

    def save(self, filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def __str__(self):
        return self.__to_string()

    def __repr__(self):
        return f"Scaler({ self.__to_string() })"

    def __to_string(self):
        return f"backend={ self.backend }"

    def __call__(self, dataset):
        return self.transform(dataset)


class Pipeline:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.callbacks = []

    def apply(self, dataset):
        res = dataset
        for s in self.steps:
            res = s(res)

            for cb in self.callbacks:
                if isinstance(s, cb[0]):
                    cb[1](s)

        return res

    def predict(self, X):
        if isinstance(self.steps[-1], Model):
            return self.steps[-1].predict(X)
        else:
            logger.warning('last step of the pipeline is not a model')
            return None

    def add_callback(self, cl, func):
        self.callbacks.append((cl, func))

    def __call__(self, dataset):
        return self.apply(dataset)




