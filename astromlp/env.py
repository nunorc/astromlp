
""" Environment definition.

    Attributes
    ----------
        BASEDIR : str
            base directory for local files, defaults to `$HOME/.astromlp`
        BASEURL : str
            base URL for remote files
"""

import os

BASEDIR = os.path.join(os.path.expanduser("~"), '.astromlp')
BASEURL = 'http://nrc.pt/astromlp'
