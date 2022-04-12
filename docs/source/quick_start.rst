
Quick Start
======================

Import pipelines for a specific topic, for example to import 
the :code:`One2One` and :code:`CherryPicked` pipelines for galaxies characterization:

.. code-block:: python

    >>> from astromlp.galaxies import One2One, CherryPicked

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