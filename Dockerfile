
# astromlp-api

FROM python:3.9-slim

RUN pip install tensorflow tensorflow-io

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gunicorn flask flask-cors

RUN apt update
RUN apt install -y git git-lfs build-essential libbz2-dev
RUN pip install git+https://github.com/nunorc/ImageCutter.git

RUN mkdir /app
WORKDIR /app

RUN git clone --recurse-submodules https://github.com/nunorc/astromlp
WORKDIR /app/astromlp

EXPOSE 8500

ENTRYPOINT gunicorn -b :8500 --workers=1 --access-logfile - --error-logfile - astromlp.api:app

