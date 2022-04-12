
Installation
============

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
Just make sure all the requirements are available in your environment.

Install the requirements using for example:

.. code-block:: bash

    $ pip install -r requirements.txt

The `ImageCutter <https://github.com/nunorc/ImageCutter>`_ package is required for saving
FITS data files. the version in GitHub
does not work with the more recent FITS data available from SDSS, so consider installing
this `fork <https://github.com/nunorc/ImageCutter>`_  from the repository:

.. code-block:: bash

    $ pip install pip install git+https://github.com/nunorc/ImageCutter.git

