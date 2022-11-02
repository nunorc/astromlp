
import sys, logging
sys.path.insert(0, '')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from astromlp.sdss.helper import Helper
from astromlp.sdss.predictor import Predictor
from astromlp.galaxies import One2One, CherryPicked, Universal

app = FastAPI(title = 'astromlp API',  version = 'v0.1')
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# initial setup
@app.on_event('startup')
def _init():
    global helper, models, pipelines
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

@app.get('/')
def _root():
    return { 'title': app.title, 'version': app.version }

@app.get('/infer/{model}/{objid}')
def _infer(model, objid):
    if model in models.keys():
        data = models[model].predict(objid)

        # FIXME
        data['obj']['objid'] = str(data['obj']['objid'])

        return data
    else:
        raise HTTPException(status_code=404, detail='Model not found')

@app.get('/proc/{pl}/{objid}')
def _proc(pl, objid):
    if pl in pipelines.keys():
        result = pipelines[pl].process(objid)

        return result.to_json()
    else:
        raise HTTPException(status_code=404, detail='Pipeline not found')

@app.get('/random/id')
def _random_id():
    data = helper.random_id()

    return str(data)

