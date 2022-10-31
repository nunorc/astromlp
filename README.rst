
astromlp
=====================================

**Experimental and under development!**

A framework for building deep learning models and pipelines for astrophysics applications.
You can explore the available pipelines and related models online from the `astromlp-app <https://nunorc.github.io/astromlp-app/>`_.

`Repository <https://github.com/nunorc/astromlp>`_ | `Documentation <https://nunorc.github.io/astromlp>`_

Installation
-------------------------------------

Install package from the git repository:

.. code-block:: bash

    $ pip install git+https://github.com/nunorc/astromlp@master

The collection of models available from the
`astromlp-models <https://github.com/nunorc/astromlp-models>`_ repository is required,
quick clone:

.. code-block:: bash

    $ git clone https://github.com/nunorc/astromlp-models.git

And set the :code:`model_store` accordingly when necessary.

For development and exploring just clone **astromlp** recursively:

.. code-block:: bash

    $ git clone --recurse-submodules https://github.com/nunorc/astromlp.git

and run code from the repository root, **astromlp-models** is set as a `submodule` of the
**astromlp** repository, and this is the default location for :code:`model_store` when used.
Just make sure all the requirements are available in your environment, check the
`Installation <https://nunorc.github.io/astromlp/html/install.html>`_ section for details.

Quick Start
-------------------------------------

Import pipelines for a specific topic, for example to import 
the :code:`One2One`, :code:`CherryPicked` and :code:`Universal` pipelines for galaxies characterization:

.. code-block:: python

    >>> from astromlp.galaxies import One2One, CherryPicked, Universal

Next, create an instance of the `One2One` pipeline, you may need to provide the location
of the `astromlp-models/model_store` directory where the actual models live,
for example:

.. code-block:: python

    >>> pipeline = One2One(model_store='./astromlp-models/model_store')

The galaxies pipelines are based on SDSS data, so the input to the pipeline
if an SDSS object identifier (`objid`), for example to process the object
`1237648720693755918 <https://skyserver.sdss.org/dr17/VisualTools/explore/summary?id=1237648720693755918>`_
using the selected pipeline run:

.. code-block:: python

    >>> result = pipeline.process(1237648720693755918)

The `result` object is an instance of :code:`PipelineResult`, the outputs of the pipeline
processing:

.. code-block:: python

    >>> result
    PipelineResult(redshift=0.0869317390024662, smass=23.44926865895589,
    subclass='STARFORMING', gz2c='ScR')

The :code:`PipelineResult` object implements other methods that provide extra data, namely:

- :code:`objid`: returns the SDSS object identifier;
- :code:`obj`: returns some information about the object from SDSS data;
- :code:`models`: returns the ensemble of models used;
- :code:`map`: returns the list of results of applying each individual model for each output.

You can easily create new ensembles of models using the :code:`MapReducPipeline` and passing the
list of outputs and corresponding models. For example, to create a pipeline that computes
the `redshift` using the `i2r` and `f2r` models:

.. code-block:: python

    >>> from astromlp.galaxies import MapReducePipeline
    >>> pipeline = MapReducePipeline({ 'redshift': ['i2r', 'f2r'] })

Acknowledgments
===============

Thank you to Dr. Andrew Humphrey for helping spawning this project and his contributions that helped improve this work.

