
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

    from astromlp import Dataset, Model
    import numpy as np

Setup a dataset given an unique identifier, which contains data as a set of `numpy` arrays:

.. code-block:: python

    >>> ds = Dataset('sdss_specphoto_ugriz_redshift')
    >>> ds.nparrays
    ['ids', 'X', 'y', 'X_cols', 'y_cols']
    >>> ds.X.shape
    (4613773, 5)
    >>> ds.X
    array([[19.78247, 18.51326, 17.70357, 17.26893, 16.97296],
           [20.85782, 18.6041 , 17.62333, 17.17327, 16.83212],
           [18.25609, 17.21735, 16.89063, 16.75081, 16.66703],
           ...
    >>> ds.X_cols
    array(['u', 'g', 'r', 'i', 'z'], dtype=object)

Setup a previously created untrained model given an unique identifier and fit the data:

.. code-block:: python

    >>> ml = Model('keras_ugriz_redshift')
    >>> ml.desc
    'Inputs: (None, 5) -> Outputs: (None, 1)'
    >>> ml.fit(ds.X, ds.y)
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
    >>> ml(data)
    array([[-0.27960014]], dtype=float32)

Models are instantiated using a backend, `Keras` by default, and the
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


Available Datasets and Models
-------------------------------------

Datasets:

* `sdss_specphoto_ugriz_redshift`
* `sdss_specphoto_ugriz_class`

Models:

* `keras_ugriz_redshift`
* `keras_ugriz_class`
