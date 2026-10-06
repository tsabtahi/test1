# CPHD ground-projection viewer
#   docker build -t cphd-viewer .
#   docker run -d --name cphd-viewer --user $(id -u):$(id -g) -p 127.0.0.1:8095:8095 \
#       -v /path/to/cphd/folder:/data:ro -v /path/to/cache:/cache cphd-viewer
#   ssh -L 8095:localhost:8095 <server>      then open http://localhost:8095
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN pip install \
        numpy==2.5.3 scipy==1.18.1 pandas==3.0.5 lxml==6.1.3 \
        sarkit==1.12.0 pyproj==3.8.0 rasterio==1.5.2 \
        flask==3.1.3 pillow==12.3.0

WORKDIR /app
COPY cphd_viewer.py /app/cphd_viewer.py
RUN mkdir -p /data /cache && chmod 777 /cache

EXPOSE 8095
ENTRYPOINT ["python", "/app/cphd_viewer.py"]
CMD ["--data", "/data", "--cache", "/cache", "--port", "8095"]
