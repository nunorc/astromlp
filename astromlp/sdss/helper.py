
import os, random, logging, requests, subprocess, tempfile
import numpy as np
from pandas import read_csv, concat
from astropy.io import fits
import tensorflow.keras.preprocessing.image as keras
from ImageCutter.ImageCutter import FITSImageCutter
import concurrent.futures

from .skyserver import SkyServer

logger = logging.getLogger(__name__)

class Helper:
    """ A helper class providing set of helper functions to deal with the
        `SDSS Galaxy Subset <https://zenodo.org/record/6501642>`_ dataset,
        henceforth refereed to as `sdss-ds`.

        Attributes:
            ds (str): location of the `sdss-ds` dataset, detauls to `'../sdss-gs'`
    """
    def __init__(self, ds: str ='../sdss-gs'):
        """ Constructor method """
        self.ds = ds

        if not os.path.exists(self.ds):
            logger.warn(f'Dataset files directory not found: { self.ds }')

        self.df = None
        _filename = os.path.join(self.ds, 'data.csv')
        if os.path.exists(_filename):
            self.df = read_csv(_filename)
        else:
            logger.warn(f'Data file not found: { _filename }')

        self.ss = SkyServer()

    def ids_list(self, has_img=False, has_fits=False, has_spectra=False, has_ssel=False, has_bands=False, has_wise=False, has_gz2c=False):
        """ Build a list of SDSS objects identifiers from the `sdss-ds` dataset.

            Args:
                has_img (bool): include identifier only if RGB image file is available, defaults to `False`
                has_fits (bool): include identifier only if FITS data file is available, defaults to `False`
                has_spectra (bool): include identifier only if spectra data file is available, defaults to `False`
                has_ssel (bool): include identifier only if spectra selected bands data file is available, defaults to `False`
                has_bands (bool): include identifier only if bands data is available, defaults to `False`
                has_wise (bool): include identifier only if WISE data is available, defaults to `False`
            Returns:
                list of SDSS objects identifiers
        """
        _df = self.df.copy()

        if has_bands:
            _df = _df[~_df['modelMag_u'].isna() & ~_df['modelMag_g'].isna() & ~_df['modelMag_r'].isna() & ~_df['modelMag_i'].isna() & ~_df['modelMag_z'].isna()]
        if has_wise:
            _df = _df[~_df['w1mag'].isna() & ~_df['w2mag'].isna() & ~_df['w3mag'].isna() & ~_df['w4mag'].isna()]
        if has_gz2c:
            _df = _df[~_df['gz2c_f'].isna() & ~_df['gz2c_s'].isna()]
        _ids = _df['objid'].tolist()

        if has_img:
            _ids = [x for x in _ids if self._has_img(x)]
        if has_fits:
            _ids = [x for x in _ids if self._has_fits(x)]
        if has_spectra:
            _ids = [x for x in _ids if self._has_spectra(x)]
        if has_ssel:
            _ids = [x for x in _ids if self._has_ssel(x)]

        return _ids

    def _has_img(self, _id):
        return os.path.exists(self._img_filename(_id))

    def _has_fits(self, _id):
        return os.path.exists(self._fits_filename(_id))

    def _has_spectra(self, _id):
        filename = self._spectra_filename(_id)
        if os.path.exists(filename):
            _df = read_csv(filename)
            if len(_df)>0 and 'Wavelength' in _df.columns and 'BestFit' in _df.columns:
                _x = _df[(_df['Wavelength']>=4000) & (_df['Wavelength']<=9000.0)]['BestFit'].to_numpy()
                if len(_x) == 3522:
                    return True
        return False

    def _has_ssel(self, _id):
        filename = self._ssel_filename(_id)
        if os.path.exists(filename):
            _df = read_csv(filename)
            _x = _df['BestFit'].to_numpy()
            if _x.shape == (1423,):
                return True

        return False

    def _img_filename(self, objID, DIR='img'):
        return os.path.join(self.ds, DIR, str(objID)+'.jpg')

    def _fits_filename(self, objID, DIR='fits'):
        return os.path.join(self.ds, DIR, str(objID)+'.npy')

    def _spectra_filename(self, objID, DIR='spectra'):
        return os.path.join(self.ds, DIR, str(objID)+'.csv')

    def _ssel_filename(self, objID, DIR='ssel'):
        return os.path.join(self.ds, DIR, str(objID)+'.csv')

    def get_obj(self, id, wise=False):
        """ Retrieve information for a SDSS object.

            Args:
                id (int): a SDSS object identifier
            Returns:
                a `Dict` containing proprieties available for the object from the `sdss-ds`
        """
        if isinstance(id, str):
            id = int(id)

        if self.df is not None:
            sl = self.df[self.df['objid'] == id]

        if self.df is None or sl.empty:
            return self.ss.get_obj(id, wise=wise)
        else:
            return sl.replace({np.nan: None}).iloc[0].to_dict()

    def y_list(self, ids, target):
        """ Build a list of target data to use in the data generator for a continuous variable.

            Args:
                ids ([int]): list of SDSS object identifiers
                target (str): target variable
            Returns:
                a numpy array
        """
        if target in ['redshift', 'stellarmass']:
            res = []
            for i in ids:
                row = self.get_obj(i)
                res.append(row[target])

            return res

    def y_list_class(self, ids, target, classes):
        """ Build a list of target data to use in the data generator for a class variable.

            Args:
                ids ([int]): list of SDSS object identifiers
                target (str): target variable
                classes: classes object
            Returns:
                a numpy array, class set
        """
        n = len(classes)

        y = []
        for i in ids:
            _row = self.get_obj(i)
            if target == 'gz2c':     # exception for gz2class
                c = _row['gz2c_s']
            else:
                c = _row[target]
            tmp = np.zeros(n)
            tmp[classes.index(c)] = 1
            y.append(tmp)

        return y, classes

    def load_img(self, filename):
        """ Load RGB image into a numpy array from file.

            Args:
                filename (str): RGB image filename
            Returns:
                a numpy array
        """
        img = keras.load_img(filename)
        x = keras.img_to_array(img)/255

        return x

    def load_imgs(self, _ids):
        """ Load list of RGB images into a numpy array given list of SDSS object identifiers.

            Args:
                ids ([int]): list of SDSS object identifiers
            Returns:
                a numpy array
        """
        X_img = []

        for i in _ids:
            filename = self._img_filename(i)
            X_img.append(self.load_img(filename))

        return np.array(X_img)

    def load_fits(self, _ids):
        """ Load list of FITS data into a numpy array given list of SDSS object identifiers.

            Args:
                ids ([int]): list of SDSS object identifiers
            Returns:
                a numpy array
        """
        X_fits = []

        for i in _ids:
            filename = self._fits_filename(i)
            with open(filename, 'rb') as fin:
                X_fits.append(np.load(fin))

        return np.array(X_fits)

    def load_spectra(self, filename):
        """ Load spectra data into a numpy array from file.

            Args:
                filename (str): spectra data filename
            Returns:
                a numpy array
        """
        if os.path.exists(filename):
            df = read_csv(filename)
            if len(df)>0 and 'Wavelength' in df.columns and 'BestFit' in df.columns:
                _df = df[(df['Wavelength']>=4000) & (df['Wavelength']<=9000.0)]
                x = _df['BestFit'].to_numpy()
                w = _df['Wavelength'].to_numpy()
                if len(x) == 3522:
                    return x, w

        return None

    def load_spectras(self, ids):
        """ Load list of spectra data into a numpy array given list of SDSS object identifiers.

            Args:
                ids ([int]): list of SDSS object identifiers
            Returns:
                a numpy array
        """
        X_spectra = []

        for i in ids:
            x, _ = self.load_spectra(self._spectra_filename(i))
            if x is not None:
                X_spectra.append(x)

        return np.array(X_spectra)

    def load_ssel(self, filename):
        """ Load spectra selected bands data into a numpy array from file.

            Args:
                filename (str): spectra data filename
            Returns:
                a numpy array
        """
        if os.path.exists(filename):
            df = read_csv(filename)
            x = df['BestFit'].to_numpy()
            w = df['Wavelength'].to_numpy()
            return x, w

        return None

    def load_ssels(self, ids):
        """ Load list of spectra selected bands data into a numpy array given list of SDSS object identifiers.

            Args:
                ids ([int]): list of SDSS object identifiers
            Returns:
                a numpy array
        """
        X_ssel = []

        for i in ids:
            x, _ = self.load_ssel(self._ssel_filename(i))
            if x is not None:
                X_ssel.append(x)

        return np.array(X_ssel)

    def load_bands(self, _ids):
        """ Load list of bands data into a numpy array given list of SDSS object identifiers.

            Args:
                ids ([int]): list of SDSS object identifiers
            Returns:
                a numpy array
        """
        X_bands = []

        for i in _ids:
            row = self.get_obj(i)
            x = [row['modelMag_u'], row['modelMag_g'], row['modelMag_r'], row['modelMag_i'], row['modelMag_z']]
            X_bands.append(x)

        return np.array(X_bands)

    def load_wises(self, _ids):
        """ Load list of WISE data into a numpy array given list of SDSS object identifiers.

            Args:
                ids ([int]): list of SDSS object identifiers
            Returns:
                a numpy array
        """
        X_wise = []

        for i in _ids:
            row = self.get_obj(i)
            x = [row['w1mag'], row['w2mag'], row['w3mag'], row['w4mag']]
            X_wise.append(x)

        return np.array(X_wise)

    def random_id(self):
        """ Return a random SDSS object identifier from the `sdss-ds` dataset.

            Returns:
                a random SDSS object identifier
        """
        _ids = self.df['objid'].tolist()

        return random.choice(_ids)

    def _frame_url(self, obj, band):
        return f"https://dr17.sdss.org/sas/dr17/eboss/photoObj/frames/{ obj['rerun'] }/{ obj['run'] }/{ obj['camcol'] }/frame-{ band }-{ str(obj['run']).zfill(6) }-{ obj['camcol'] }-{ str(obj['field']).zfill(4) }.fits.bz2"

    def _frame_filename(self, obj, band, base_dir, DIR='frames', bz=False):
        d = os.path.join(base_dir, DIR)
        os.makedirs(d, exist_ok=True)
        filename = os.path.join(d, str(obj['objid']) + '_' + band + '.fits')
        
        if bz:
            filename += '.bz2'

        return filename

    def _frames_urls_filenames(self, obj, base_dir='./'):
        r = []

        if obj:
            for b in ['u', 'g', 'r', 'i', 'z']:
                u = self._frame_url(obj, b)
                f = self._frame_filename(obj, b, base_dir, bz=True)

                r.append((u, f))

        return r

    def save_img(self, obj, filename=None):
        """ Retrieve and save RGB image for a given SDSS object.

            Args:
                obj: SDSS object
                filename (str): optional filename
            Returns:
                a numpy array
        """
        if filename is None:
            filename = self._img_filename(obj['objid'])

        if os.path.exists(filename):
            return filename

        return self.ss.save_jpeg(obj['objid'], filename, ra=obj['ra'], dec=obj['dec'], scale=0.2, width=150, height=150)

    def _save_frame(self, url, filename):
        if os.path.exists(filename) or os.path.exists(filename.replace('.bz2', '')):
            return

        r = requests.get(url)
        if r.status_code == 200:
            with open(filename, 'wb') as fout:
                fout.write(r.content)

    def save_fits(self, obj, filename=None, base_dir='./'):
        """ Retrieve and save FITS data file for a given SDSS object.

            Args:
                obj: SDSS object
                filename (str): optional filename
            Returns:
                a numpy array
        """
        if filename is None:
            filename = self._fits_filename(obj['objid'])

        if os.path.exists(filename):
            with open(filename, 'rb') as fin:
                return np.load(fin)

        # download frames files
        urls_files = self._frames_urls_filenames(obj, base_dir=base_dir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for u, f in urls_files:
                executor.submit(self._save_frame, u, f)

        # unzip files
        for _, f in urls_files:
            if os.path.exists(f):
                subprocess.run(['bunzip2', f])  # FIXME make more portable

        # build fits data
        _exists = []
        for u, f in urls_files:
            _exists.append(os.path.exists(f.replace('.bz2', '')))

        result = None
        if all(_exists):
            arr = []
            for _, f in urls_files:
                tmp = tempfile.NamedTemporaryFile()

                x = FITSImageCutter()
                x.prepare(f.replace('.bz2', ''))
                x.fits_cut(obj['ra'], obj['dec'], tmp.name, xs=0.4, ys=0.4)

                hdul = fits.open(tmp.name)
                data = hdul[0].data
                if data.shape == (61, 61):
                    arr.append(data)
                else:
                    logger.warn(f"Err shape { obj['objid'] } - { f.replace('.bz2', '') }")

                tmp.close()

            if len(arr) == 5:
                with open(filename, 'wb') as fout:
                    data = np.stack(arr, axis=-1)
                    np.save(fout, data)
                    result = data
            else:
                logger.warn(f"Err shape { obj['objid'] }")

        # delete fits files
        for _, f in urls_files:
            f = f.replace('.bz2', '')
            if os.path.exists(f):
                os.remove(f)

        return result

    def _spectra_url(self, obj):
        return f"https://dr16.sdss.org/optical/spectrum/view/data/format=csv/spec=lite?plateid={ obj['plate'] }&mjd={ obj['mjd'] }&fiberid={ obj['fiberid'] }"

    def save_spectra(self, obj, filename=None):
        """ Retrieve and save spectra data file for a given SDSS object.

            Args:
                obj: SDSS object
                filename (str): optional filename
            Returns:
                a numpy array
        """
        if filename is None:
            filename = self._spectra_filename(obj['objid'])

        if os.path.exists(filename):
            return filename

        url = self._spectra_url(obj)
        r = requests.get(url)

        if r.status_code == 200:
            with open(filename, 'wb') as fout:
                fout.write(r.content)
                return filename

        return None

    def save_ssel(self, obj, filename=None, spectra_filename=None):
        """ Retrieve and save spectra selected bands data file for a given SDSS object.

            Args:
                obj: SDSS object
                filename (str): optional filename
            Returns:
                a numpy array
        """
        if filename is None:
            filename = self._ssel_filename(obj['objid'])

        if os.path.exists(filename):
            return filename

        if spectra_filename is None:
            spectra_filename = self._spectra_filename(obj['objid'])
        if not os.path.exists(spectra_filename):
            self.save_spectra(obj, filename=spectra_filename)

        df = read_csv(spectra_filename)
        intervals = [(4000,4200),(4452,4474),(4514,4559),(4634,4720),(4800,5134),(5154,5196),(5245,5285),
           (5312,5352),(5387,5415),(5696,5720),(5776,5796),(5876,5909),(5936,5994),(6189,6272),
           (6500,6800),(7000,7300),(7500,7700)]
        dfs = []
        for i in intervals:
            dfs.append(df[(df['Wavelength']>=i[0]) & (df['Wavelength']<=i[1])])

        if len(dfs) > 0:
            final = concat(dfs)
            final.to_csv(filename, index=False)
            return filename

        return None


