
astromlp
=====================================

**Experimental and under development!**

A framework for building machine learning pipelines for astrophysics
models and applications.

All the datasets are pre-built and made available as illustrated
in the next section. Models architectures are also available. Once
you use a dataset or model, a local copy is downloaded to your 
machine (some of the files are *largish*), once downloaded the
the local file is then used in future instantiations of the
dataset/model. The downloaded files are stored in `$HOME/.astromlp`.

`Repository <https://github.com/nunorc/astromlp>`_ | `Documentation <https://nunorc.github.io/astromlp>`_

Installation
-------------------------------------

Install package from the git repository:

.. code-block:: bash

    $ pip install git+https://github.com/nunorc/astromlp@master

Quick Start
-------------------------------------

Do some imports:

.. code-block:: python

    from astromlp import DataSource, Dataset, Model

Setup a data source readily available, which provides a set of `nparrays`, in this example
data from `specphoto` table in the SDSS database:

.. code-block:: python

    >>> sdss = DataSource('sdss_specphoto')
    >>> sdss.nparrays
    ['id', 'ugriz', 'redshift', 'class_', 'class1hot']
    >>> sdss.ugriz
    array([[19.78247, 18.51326, 17.70357, 17.26893, 16.97296],
           [20.85782, 18.6041 , 17.62333, 17.17327, 16.83212],
           [18.25609, 17.21735, 16.89063, 16.75081, 16.66703],
           ...,
           [22.22028, 21.68921, 21.66217, 21.37846, 20.71212],
           [23.31769, 23.67536, 21.14215, 20.11948, 19.61694],
           [23.35438, 21.86783, 20.52837, 19.7718 , 19.45774]])
    >>> sdss.ugriz.shape
    (4613773, 5)
    >>> sdss.class_
    array(['GALAXY', 'GALAXY', 'STAR', ..., 'QSO', 'GALAXY', 'GALAXY'],
          dtype=object)
    >>> sdss.class_.shape
    (4613773,)


Setup a dataset with data from the previous source, in this example the features are
the band values, and the target in the redshift:

.. code-block:: python

    >>> ds = Dataset(features=sdss.ugriz, target=sdss.redshift)
    >>> ds
    Dataset(features=(4613773, 5), target=(4613773,))

Setup a previously created untrained model given an unique identifier and fit the data:

.. code-block:: python

    >>> ml = Model('keras_ugriz_redshift')
    >>> ml.desc
    'Inputs: (None, 5) -> Outputs: (None, 1)'
    >>> ml.fit(ds)
    2021-11-10 12:20:16 | INFO: batch_size=64 epochs=10
    ...
    Epoch 1/10
    5311/72091 [=>............................] - ETA: 1:11 - loss: 214.0605 - mean_squared_error: 214.0605

Once the model is trained it can be applied to new data, i.e. predict the redshift
given band values:

.. code-block:: python

    >>> data = np.array([[18.27583, 19.274648, 19.19848, 16.38564, 17.17455]])
    >>> data.shape
    (1, 5)
    >>> ml.predict(data)
    array([[-0.27960014]], dtype=float32)

Models are instantiated using a backend, Keras by default, and the
underlying model is always available in the `model` attribute:

.. code-block:: python

    >>> ml.model
    <keras.engine.functional.Functional object at 0x159ab75e0>
    >>> ml.model.summary()
    Model: "keras_ugriz_redshift"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_8 (InputLayer)         [(None, 5)]               0         
    _________________________________________________________________
    dense_45 (Dense)             (None, 32)                192       
    _________________________________________________________________
    dense_46 (Dense)             (None, 16)                528       
    _________________________________________________________________
    dense_47 (Dense)             (None, 8)                 136       
    _________________________________________________________________
    dense_48 (Dense)             (None, 1)                 9         
    =================================================================
    Total params: 865
    Trainable params: 865
    Non-trainable params: 0
    _________________________________________________________________


More complex workflows can be achieved, let's do some more imports:

.. code-block:: python

    from astromlp import Scaler, Splitter, Pipeline

A common operation before fitting the model to the actual data is to scale the
values, we can create a new `Scaler` object that used the `StandardScaler`
from scikit-learn by default:

.. code-block:: python

    >>> sc = Scaler()
    >>> sc
    Scaler(backend=StandardScaler())

We can also split the data into a train and validation set, to do this
we can use a `Splitter` object:

.. code-block:: python

    >>> sp = Splitter()
    >>> sp
    Splitter(test_size=0.2, random_state=None)

We need to recreate our model instance to tell it we want to use a validation
set:

.. code-block:: python

    >>> ml = Model('keras_ugriz_redshift', validation_data=True)

And instead of manually applying all these operations we can create a pipeline
specifying the individual steps, and apply this sequence of operations to the
dataset:

.. code-block:: python

    >>> pl = Pipeline(sc, sp, ml)
    >>> pl(ds)
    (...)
    Epoch 1/10
     3737/57673 [>.............................] - ETA: 57s - loss: 0.5945 - mean_squared_error: 0.5945

Callbacks can be added after each step to perform any extra operations, for example
save to file the `Scaler` instance to use in future data:

.. code-block:: python

    >>> pl.add_callback(Scaler, lambda s: s.save('my-scaler.pkl'))


Available Data Sources and Models
-------------------------------------

Data sources:

* `sdss_specphoto`

Models:

* `keras_ugriz_redshift`
* `keras_ugriz_class`
