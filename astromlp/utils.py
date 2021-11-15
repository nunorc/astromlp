
""" Utility functions definition. """

import logging, os, requests
from tqdm import tqdm

from .env import BASEDIR, BASEURL

logger = logging.getLogger(__name__)

def filename_and_url(uid, sub, ext):
    filename = os.path.join(BASEDIR, sub, f'{ uid }.{ ext }')
    url = '/'.join([BASEURL, sub, f'{ uid }.{ ext }'])

    return filename, url

def download_file(url, filename):
    """ Function to download a file.

    Arguments
    ----------
        url : str
            file url
        filename : str
            local file name
    """
    dirname = os.path.dirname(filename)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    response = requests.get(url, stream=True)
    size = int(response.headers.get('content-length', 0))
    block = 1024 * 1024  # 1 MB
    progress_bar = tqdm(total=size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block):
            progress_bar.update(len(data))
            file.write(data)
        progress_bar.close()
