#!/usr/bin/env bash
# Thin wrapper around the umbra-phase container.
#
#   ./umbra-fetch.sh --limit 30 --dry-run
#   ./umbra-fetch.sh --limit 30
#   OUT=/home/tabtahi/SATLOCK/dataset/umbra_phase ./umbra-fetch.sh --limit 30 --products sicd,cphd
#
# Env:
#   IMAGE   container tag           (default umbra-phase:latest)
#   OUT     host output directory   (default ./umbra_phase)
set -euo pipefail

IMAGE=${IMAGE:-umbra-phase:latest}
OUT=${OUT:-$PWD/umbra_phase}

mkdir -p "$OUT"

# --user keeps the downloaded files owned by you, not root.
# Proxy vars are passed through in case geohub3 egress is proxied; unset is fine.
exec docker run --rm -it \
    --user "$(id -u):$(id -g)" \
    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
    -e http_proxy -e https_proxy -e no_proxy \
    -v "$OUT:/data" \
    "$IMAGE" --out /data "$@"
