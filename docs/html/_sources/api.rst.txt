
API
==================

A simple REST API implemented using `Flask <https://flask.palletsprojects.com/en/2.1.x/>`_
is provided by `astromlp.api`. To run the API locally clone the repository and run:

.. code-block:: bash

    $ python astromlp/api.py

By default the API listens on :code:`http://localhost:8010` and the following requests are available:

- :code:`/infer/<model>/<objid>`: request for prediction for SDSS object identifier :code:`objid` using model identifier :code:`model`
- :code:`/proc/<pipeline>/<objid>`: request for process an SDSS object identifier :code:`objid` using pipeline identifier :code:`pipeline`

