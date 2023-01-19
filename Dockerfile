#FROM python:3.8-slim
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY . /app
COPY ./src/static /app/static
COPY ./src/templates /app/templates
COPY ./src/utils.py /app/utils.py
COPY ./src/inference.py /app/inference.py

#COPY ./src/weights/ /app/weights/
#RUN ls -la ./src/weights/*

COPY ./src/arcane_filter/ /app/arcane_filter/
RUN ls -la ./src/arcane_filter/*

COPY ./src/anime_filter/ /app/anime_filter/
RUN ls -la ./src/anime_filter/*

COPY ./src/enhance_filter/ /app/enhance_filter/
RUN ls -la ./src/enhance_filter/*



WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    gcc \
    make

RUN apt install -y \
    libgl1-mesa-glx \
    libglib2.0-dev

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install zip

# Create a virtual environment in /opt
RUN python3 -m venv /opt/venv

# Install requirments to new virtual environment
COPY ./src/requirements.txt /app/requirements.txt
RUN /opt/venv/bin/pip install -r /app/requirements.txt

RUN /opt/venv/bin/gdown --id 1WPBjn3vhQOJxafBa_89lbYwJfolKVMu6
RUN unzip weights.zip -d /app/

#COPY ./src/weights/ /app/weights/
#RUN ls -la ./src/weights/*

# purge unused
RUN apt-get remove -y --purge make gcc build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*


#RUN pip install gunicorn[gevent]
#EXPOSE 4040
#CMD gunicorn --worker-class gevent --workers 8 --bind 0.0.0.0:4040 src.app:app

# make entrypoint.sh executable
RUN chmod +x entrypoint.sh
CMD [ "./entrypoint.sh" ]

#EXPOSE 9999
#CMD ["/opt/venv/bin/uvicorn", "src.app:app", "--host", "localhost", "--port", "9999"]