
import sys
sys.path.insert(0, '')

from flask import Flask, Response, jsonify
from flask_cors import CORS

from astromlp.sdss.helper import Helper
from astromlp.sdss.predictor import Predictor
from astromlp.galaxies import One2One, CherryPicked, Universal

app = Flask(__name__)
CORS(app)

# initial setup
helper = Helper()
models = {
    'i2r': Predictor('i2r', helper=helper),
    'f2r': Predictor('f2r', helper=helper),
    's2r': Predictor('s2r', helper=helper),
    'ss2r': Predictor('ss2r', helper=helper),
    'b2r': Predictor('b2r', helper=helper),
    'w2r': Predictor('w2r', helper=helper),
    'i2sm': Predictor('i2sm', helper=helper),
    'f2sm': Predictor('f2sm', helper=helper),
    's2sm': Predictor('s2sm', helper=helper),
    'ss2sm': Predictor('ss2sm', helper=helper),
    'b2sm': Predictor('b2sm', helper=helper),
    'w2sm': Predictor('w2sm', helper=helper),
    'i2s': Predictor('i2s', helper=helper),
    'f2s': Predictor('f2s', helper=helper),
    's2s': Predictor('s2s', helper=helper),
    'ss2s': Predictor('ss2s', helper=helper),
    'b2s': Predictor('b2s', helper=helper),
    'w2s': Predictor('w2s', helper=helper),
    'i2g': Predictor('i2g', helper=helper),
    'f2g': Predictor('f2g', helper=helper),
    's2g': Predictor('s2g', helper=helper),
    'ss2g': Predictor('ss2g', helper=helper),
    'b2g': Predictor('b2g', helper=helper),
    'w2g': Predictor('w2g', helper=helper),
    'fSbW2rSM': Predictor('fSbW2rSM', helper=helper),
    'fSbW2sG': Predictor('fSbW2sG', helper=helper),
    'iFsSSbW2r': Predictor('iFsSSbW2r', helper=helper),
    'iFsSSbW2sm': Predictor('iFsSSbW2sm', helper=helper),
    'iFsSSbW2s': Predictor('iFsSSbW2s', helper=helper),
    'iFsSSbW2g': Predictor('iFsSSbW2g', helper=helper),
    'iFsSSbW2rSMsG': Predictor('iFsSSbW2rSMsG', helper=helper)
}

pipelines = {
    'one2one': One2One(helper=helper),
    'cherryPicked': CherryPicked(helper=helper),
    'universal': Universal(helper=helper)
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
    if pl in pipelines.keys():
        app.logger.info(f'Proc: { objid }')
        result = pipelines[pl].process(objid)

        return Response(result.to_json(), mimetype='application/json')
    else:
        return 'pipeline not found'

@app.route('/random/id')
def _random_id():
    data = helper.random_id()

    return str(data)

# main    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8010, debug=False)
