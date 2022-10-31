
API
==================

A simple REST API implemented using `Flask <https://flask.palletsprojects.com/en/2.1.x/>`_
is provided by `astromlp.api`. To run the API locally clone the repository and run:

.. code-block:: bash

    $ python astromlp/api.py

By default the API listens on :code:`http://localhost:8010` and the following requests are available:

- :code:`/infer/<model>/<objid>`: request for prediction for SDSS object identifier :code:`objid` using model identifier :code:`model`
- :code:`/proc/<pipeline>/<objid>`: request for process an SDSS object identifier :code:`objid` using pipeline identifier :code:`pipeline`

Running the API using Docker
----------------------------

A `Docker <https://www.docker.com>`_ file is also available to run the API, to build the Docker image run:

.. code-block:: bash

    $ docker build -t astromlp-api:latest .

And to run a container:

.. code-block:: bash

    $ docker run -d --rm -p 8500:8500 astromlp-api

The API is available in :code:`http://localhost:8500`, and the same methods illustrated before can be used to send queries.
For example:

.. code-block:: bash

    $ curl http://127.0.0.1:8500/infer/i2r/1237648720693755918
    (...) "output":[0.09091393649578094],"x":["img"],"y":["redshift"]}
