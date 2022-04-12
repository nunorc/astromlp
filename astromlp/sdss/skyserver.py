
import os, requests, logging

logger = logging.getLogger(__name__)

class SkyServer:
    def __init__(self, base_url='http://skyserver.sdss.org/dr16/SkyServerWS'):
        self.base_url = base_url

    def _url(self, action):
        return f"{ self.base_url}{ action }"

    def get_obj(self, objid, wise=True):
        obj = None

        # SpecPhoto data
        sql = f"SELECT objID as objid, mjd, plate, tile, fiberID as fiberid, run, rerun, camcol, field, ra, dec, class, subClass as subclass, modelMag_u, modelMag_g, modelMag_r, modelMag_i, modelMag_z, z as redshift FROM SpecPhoto WHERE objID={ str(objid) } AND class='GALAXY' AND subClass is not null AND zwarning=0"
        payload = { 'cmd': sql, 'format': 'json' }
        r = requests.get(self._url('/SearchTools/SqlSearch'), params=payload)
        if r.status_code == 200:
            data = r.json()
            if len(data) == 2 and 'Rows' in data[0] and len(data[0]['Rows']) == 1:
                obj = data[0]['Rows'][0]

        if not wise:
            return obj

        # WISE_allsky data
        sql = f"SELECT s.objID, w.w1mag, w.w2mag, w.w3mag, w.w4mag FROM SpecPhoto s JOIN WISE_xmatch x ON x.sdss_objid = s.objID JOIN WISE_allsky w ON x.wise_cntr = w.cntr WHERE s.objID={ str(objid) }"
        payload = { 'cmd': sql, 'format': 'json' }
        r = requests.get(self._url('/SearchTools/SqlSearch'), params=payload)
        if r.status_code == 200:
            data = r.json()
            if len(data) == 2 and 'Rows' in data[0] and len(data[0]['Rows']) == 1 and len(data[0]['Rows'][0]) == 5:
                obj['w1mag'] = data[0]['Rows'][0]['w1mag']
                obj['w2mag'] = data[0]['Rows'][0]['w2mag']
                obj['w3mag'] = data[0]['Rows'][0]['w3mag']
                obj['w4mag'] = data[0]['Rows'][0]['w4mag']

        return obj

    def save_jpeg(self, objid, filename, ra=None, dec=None, scale=0.2, width=150, height=150):
        if os.path.exists(filename):
            return filename

        url = 'http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg'

        if ra is None or dec is None:
            obj = self.get_obj(objid)
            if obj is None:
                return None
            else:
                ra, dec = obj['ra'], obj['dec']

        payload = {
            'ra': ra,
            'dec': dec,
            'scale': scale,
            'width': width,
            'height': height,
            'opt': ''
        }
        r = requests.get(self._url('/ImgCutout/getjpeg'), params=payload)

        if r.status_code == 200:
            with open(filename, 'wb') as fout:
                fout.write(r.content)

        return filename