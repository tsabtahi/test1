# Umbra open-data phase (SICD/CPHD) fetcher
#   build:  docker build -t umbra-phase:latest .
#   run:    ./umbra-fetch.sh --limit 30 --dry-run
#
# Deliberately not a GDAL image: this container only talks to S3 and reads
# SICD/CPHD via sarpy. Keep ortho/GDAL work in the geohub3 GDAL image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    AWS_DEFAULT_REGION=us-west-2 \
    AWS_EC2_METADATA_DISABLED=true

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY umbra_phase_fetch.py sicd_info.py /app/

# /data is the mount point for the test set
VOLUME ["/data"]

ENTRYPOINT ["python", "/app/umbra_phase_fetch.py"]
CMD ["--help"]
