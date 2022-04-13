
import sys
sys.path.insert(0, '')

from flask import Flask, jsonify
from flask_cors import CORS

from astromlp.sdss.helper import Helper
from astromlp.sdss.predictor import Predictor
from astromlp.galaxies import One2One, CherryPicked

app = Flask(__name__)
CORS(app)

# initial setup
helper = Helper()
models = {
    'i2r': Predictor(model='i2r', helper=helper),
    'f2r': Predictor(model='f2r', helper=helper),
    's2r': Predictor(model='s2r', helper=helper),
    'ss2r': Predictor(model='ss2r', helper=helper),
    'b2r': Predictor(model='b2r', helper=helper),
    'w2r': Predictor(model='w2r', helper=helper),
    'i2sm': Predictor(model='i2sm', helper=helper),
    'f2sm': Predictor(model='f2sm', helper=helper),
    's2sm': Predictor(model='s2sm', helper=helper),
    'ss2sm': Predictor(model='ss2sm', helper=helper),
    'b2sm': Predictor(model='b2sm', helper=helper),
    'w2sm': Predictor(model='w2sm', helper=helper),
    'i2s': Predictor(model='i2s', helper=helper),
    'f2s': Predictor(model='f2s', helper=helper),
    's2s': Predictor(model='s2s', helper=helper),
    'ss2s': Predictor(model='ss2s', helper=helper),
    'b2s': Predictor(model='b2s', helper=helper),
    'w2s': Predictor(model='w2s', helper=helper),
    'i2g': Predictor(model='i2g', helper=helper),
    'f2g': Predictor(model='f2g', helper=helper),
    's2g': Predictor(model='s2g', helper=helper),
    'ss2g': Predictor(model='ss2g', helper=helper),
    'b2g': Predictor(model='b2g', helper=helper),
    'w2g': Predictor(model='w2g', helper=helper),
    'fSbW2rSM': Predictor(model='fSbW2rSM', helper=helper),
    'fSbW2sG': Predictor(model='fSbW2sG', helper=helper),
    'iFsSSbW2r': Predictor(model='iFsSSbW2r', helper=helper),
    'iFsSSbW2sm': Predictor(model='iFsSSbW2sm', helper=helper),
    'iFsSSbW2s': Predictor(model='iFsSSbW2s', helper=helper),
    'iFsSSbW2g': Predictor(model='iFsSSbW2g', helper=helper),
    'iFsSSbW2rSMsG': Predictor(model='iFsSSbW2rSMsG', helper=helper)
}

pipelines = {
    'one2one': One2One(helper=helper),
    'cherryPicked': CherryPicked(helper=helper)
}

app.logger.info('Setup done!')

@app.route('/')
def _root():
    return 'astromlp_api v.0.0.1'

@app.route('/infer/<model>/<objid>')
def _infer(model, objid):
    if model in models.keys():
        app.logger.info(f'Infer: { objid }')
        data = models[model].predict(objid)

        # FIXME
        data['obj']['objid'] = str(data['obj']['objid'])

        return jsonify(data)
    else:
        return 'model not found'

@app.route('/proc/<pl>/<objid>')
def _proc(pl, objid):
    print(pl, objid)
    if pl in pipelines.keys():
        app.logger.info(f'Proc: { objid }')
        result = pipelines[pl].process(objid)

        return result.to_json()
    else:
        return 'pipeline not found'

@app.route('/random/id')
def _random_id():
    data = helper.random_id()

    return str(data)

# main    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8010, debug=False)
