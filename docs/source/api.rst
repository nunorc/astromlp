
API
==================

A simple REST API implemented using `FastAPI <https://fastapi.tiangolo.com/>`_
is provided by `astromlp.api`. To run the API locally, for example using `uvicorn <https://www.uvicorn.org/>`_,
clone the repository and run:

.. code-block:: bash

    $ uvicorn astromlp.api:app

By default the API listens on :code:`http://127.0.0.1:8000` and the following requests are available:

- :code:`/infer/<model>/<objid>`: request for prediction for SDSS object identifier :code:`objid` using model identifier :code:`model`
- :code:`/proc/<pipeline>/<objid>`: request for process an SDSS object identifier :code:`objid` using pipeline identifier :code:`pipeline`

Running the API using Docker
----------------------------

A `Docker <https://www.docker.com>`_ file is also available to run the API in a container,
to build the Docker image run from the repository:

.. code-block:: bash

    $ docker build -t astromlp-api:latest .

And then to run a container:

.. code-block:: bash

    $ docker run -d --rm -p 8500:8500 astromlp-api

The API is available from :code:`http://127.0.0.1:8500`, and the same methods illustrated before can be used to send queries,
for example:

.. code-block:: bash

    $ curl http://127.0.0.1:8500/infer/i2r/1237648720693755918
    (...) "output":[0.09091393649578094],"x":["img"],"y":["redshift"]}

An image is also available from `Docker Hub <https://hub.docker.com/repository/docker/nunorc/astromlp-api>`_,
to pull the image run:

.. code-block:: bash

    $ docker pull nunorc/astromlp-api