#!/bin/bash
# Ablation on the TIE-POINTED scenes only (the ones we can score with CE90).
#
# Knobs (env vars):
#   BLOCKS="1024"            block sizes to run            (default "1024 2048")
#   MATCHERS="roma xoftr"    matchers (both -> agreement gate can be scored)
#   CLOUD=1                  OmniCloudMask cloud filter ON (default 0 = off)
#   NODATA=1                 SAR nodata filter ON          (default 1)
#   GPUS=auto                every idle GPU (<1 GB used), LANES_PER_GPU each
#   GPUS="1 1 2 2"           or explicit: one lane per listed id
#   LANES_PER_GPU=4  CPUS=4  MAX_CORES=48  MAX_SCENES=0 (all)  REUSE_SCENES=1
#
# Output folders encode the filters, so arms never collide or falsely skip:
#   out_height/<matcher>_b<block>[_cloud][_nond]/<scene>/
#   e.g. CLOUD=1 -> roma_b1024_cloud/, xoftr_b1024_cloud/
#
# Scheduling is a shared queue: each lane claims the next unclaimed job
# (atomic mkdir), so fast XoFTR jobs and slow RoMa jobs never strand a lane.
#
# VRAM per lane at block 1024: RoMa ~7-8 GB, XoFTR ~2.5 GB, OCM small.
# XoFTR input is capped at 1024 px (--xoftr-max-side); its coarse matching
# memory grows ~side^4, so native 2048 needs ~47 GB and does not fit.
#
#   CLOUD=1 BLOCKS="1024" GPUS=auto ./run_ablation_block.sh
set -u
cd /home/tabtahi/SATLOCK/gec_block_register
mkdir -p logs out_height
SCRATCH=/home/tabtahi/SATLOCK/scratch; mkdir -p "$SCRATCH"

RESID=${RESID:-/home/tabtahi/SATLOCK/ce90_out/residuals.csv}
MAX_SCENES=${MAX_SCENES:-0}
SEED=${SEED:-0}
WV_SUBSET=/home/kashley/satlock/mosaic_datasets/eo_sar_umbra_subset_HH_unique_v2/wv_dailytake_output
WV_FULL=/data3/sandbox/kashley/satlock/data_collection/outputs/wv_dailytake_output
CPUS=${CPUS:-4}
BLOCKS=${BLOCKS:-"1024 2048"}
MATCHERS=${MATCHERS:-"roma xoftr"}
CLOUD=${CLOUD:-0}
NODATA=${NODATA:-1}
LANES_PER_GPU=${LANES_PER_GPU:-4}

SUFFIX=""
FILTER_ARGS=""
if [ "$CLOUD" = "1" ]; then SUFFIX="${SUFFIX}_cloud"
else FILTER_ARGS="$FILTER_ARGS --no-cloud-filter"; fi
if [ "$NODATA" != "1" ]; then SUFFIX="${SUFFIX}_nond"; FILTER_ARGS="$FILTER_ARGS --no-nodata-filter"; fi

# --- GPUs -------------------------------------------------------------------------
if [ "${GPUS:-auto}" = "auto" ]; then
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2 < 1000 {print $1}')
  GPU_LIST=()
  for g in $FREE; do for i in $(seq 1 $LANES_PER_GPU); do GPU_LIST+=("$g"); done; done
  echo "auto: idle GPUs [$(echo $FREE)] x ${LANES_PER_GPU} lanes"
else
  GPU_LIST=($GPUS)
fi
NLANE=${#GPU_LIST[@]}
[ "$NLANE" -lt 1 ] && { echo "no idle GPUs found — pass GPUS=\"...\" explicitly"; exit 1; }
# shared box: keep lanes x CPUS under MAX_CORES (default 48 of 96)
MAX_CORES=${MAX_CORES:-48}
if [ $((NLANE * CPUS)) -gt "$MAX_CORES" ]; then
  CPUS=$(( MAX_CORES / NLANE )); [ "$CPUS" -lt 2 ] && CPUS=2
  echo "CPU cap: ${NLANE} lanes -> ${CPUS} CPUs/lane (<= ${MAX_CORES} cores total)"
fi

# --- preflight: OCM must be in the image before 12 lanes discover it isn't --------
if [ "$CLOUD" = "1" ]; then
  if ! docker run --rm --entrypoint python3 height-register -c "import omnicloudmask" >/dev/null 2>&1; then
    echo "FATAL: omnicloudmask not in the height-register image."
    echo "       Rebuild: docker build -t height-register -f Dockerfile.register ."
    exit 1
  fi
  echo "preflight: omnicloudmask present"
fi

# --- scenes that have tie points (pinned; reused so arms stay paired) -------------
if [ -s scenes_tiepoint.txt ] && [ "${REUSE_SCENES:-1}" = "1" ]; then
  echo "reusing scenes_tiepoint.txt ($(wc -l < scenes_tiepoint.txt) scenes)"
else
python3 - "$RESID" "$MAX_SCENES" "$SEED" > scenes_tiepoint.txt <<'PY'
import csv, sys, random, collections
resid, nmax, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
counts = collections.Counter()
with open(resid) as f:
    for r in csv.DictReader(f):
        s = r.get("scene")
        if s:
            counts[s] += 1
scenes = sorted(counts)
if nmax and nmax < len(scenes):
    random.Random(seed).shuffle(scenes)
    scenes = sorted(scenes[:nmax])
print("\n".join(scenes))
PY
fi

# --- job list + claim dir (fresh each launch; report.json marks done work) --------
> jobs_ablation_block.txt
while read s; do
  for b in $BLOCKS; do
    for m in $MATCHERS; do
      echo "$s $b $m" >> jobs_ablation_block.txt
    done
  done
done < scenes_tiepoint.txt
njobs=$(wc -l < jobs_ablation_block.txt)
CLAIMS=logs/claims_ablation; rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"
echo "$(date)  $(wc -l < scenes_tiepoint.txt) scenes | blocks [${BLOCKS}] | matchers [${MATCHERS}]" \
     "| cloud=${CLOUD} nodata=${NODATA} | ${njobs} jobs on ${NLANE} lanes (GPUs ${GPU_LIST[*]}) | ${CPUS} CPUs/lane"

worker() {
  lane=$1; gpu=$2; consec_fail=0; j=0
  while read s b m; do
    j=$((j+1))
    mkdir "$CLAIMS/$j" 2>/dev/null || continue          # another lane has it
    exp="${m}_b${b}${SUFFIX}"
    out="out_height/${exp}/${s}"
    [ -f "${out}/report.json" ] && continue
    sar="/home/tabtahi/SATLOCK/dataset/SAR_RPC_ellip/${s}/${s}_rpc_ellip.tif"
    wv=$(ls ${WV_SUBSET}/${s}/mosaic/*.tif 2>/dev/null | head -1)
    [ -z "$wv" ] && wv=$(ls ${WV_FULL}/${s}/mosaic/*.tif 2>/dev/null | head -1)
    if [ ! -f "$sar" ] || [ -z "$wv" ]; then
      echo "[lane${lane} gpu${gpu} skip] ${s} (sar or wv missing)"; continue
    fi
    for attempt in 1 2 3; do
      echo "[lane${lane} gpu${gpu} ${j}/${njobs}] $(date +%H:%M) ${exp} ${s} (try ${attempt})"
      docker run --rm --cpus ${CPUS} --gpus "\"device=${gpu}\"" \
        -v /home/tabtahi/SATLOCK/gec_block_register:/work \
        -v /home/tabtahi/SATLOCK/dataset:/home/tabtahi/SATLOCK/dataset \
        -v /home/kashley/satlock/mosaic_datasets:/home/kashley/satlock/mosaic_datasets:ro \
        -v /data3/sandbox/kashley/satlock:/data3/sandbox/kashley/satlock:ro \
        -v ${SCRATCH}:/tmp -e TMPDIR=/tmp -e CPL_TMPDIR=/tmp -e MPLCONFIGDIR=/tmp/mpl \
        -e OMP_NUM_THREADS=${CPUS} -e MKL_NUM_THREADS=${CPUS} \
        height-register \
        --sar "$sar" --wv "$wv" \
        --matcher "$m" --select all --device cuda \
        --block "$b" ${FILTER_ARGS} \
        --no-viz --no-tif \
        --out "/work/out_height/${exp}/${s}" \
        < /dev/null >> "logs/abl_${exp}_${s}.log" 2>&1
      rc=$?
      if [ -f "${out}/report.json" ]; then consec_fail=0; break; fi
      if [ "$rc" -eq 125 ]; then
        echo "[lane${lane} gpu${gpu} FATAL] docker daemon error (disk?) — lane stopping"; return 1; fi
      echo "[lane${lane} gpu${gpu} retry] ${exp} ${s} rc=${rc}; 60s"; sleep 60
    done
    if [ ! -f "${out}/report.json" ]; then
      consec_fail=$((consec_fail+1))
      [ "$consec_fail" -ge 5 ] && { echo "[lane${lane} gpu${gpu} FATAL] 5 consecutive failures — lane stopping"; return 1; }
    fi
  done < jobs_ablation_block.txt
  echo "[lane${lane} gpu${gpu}] $(date) lane done"
}

pids=""
for lane in $(seq 0 $((NLANE-1))); do
  g=${GPU_LIST[$lane]}
  worker "$lane" "$g" > "logs/abl_worker_lane${lane}_gpu${g}.log" 2>&1 &
  pids="$pids $!"
  sleep 3            # stagger starts so lanes don't hit the CPU prep phase in lockstep
done
echo "$(date)  ${NLANE} lanes launched"
wait $pids
echo "$(date)  matching complete"

# --- score --------------------------------------------------------------------------
for b in $BLOCKS; do
  echo ""
  echo "================ BLOCK ${b}${SUFFIX} ================"
  docker run --rm \
    -v /home/tabtahi/SATLOCK/gec_block_register:/work \
    -v /home/tabtahi/SATLOCK/ce90_out:/home/tabtahi/SATLOCK/ce90_out \
    -v ${SCRATCH}:/tmp -e TMPDIR=/tmp \
    --entrypoint python3 height-register \
    /work/compute_ce90_agree.py \
      --residuals "$RESID" \
      --out-root /work/out_height \
      --roma-exp "roma_b${b}${SUFFIX}" --xoftr-exp "xoftr_b${b}${SUFFIX}" \
      --medfilter-tol 10 --agree-tols 5 10 20 50 \
      --out "/home/tabtahi/SATLOCK/ce90_out/abl_block${b}${SUFFIX}" \
    2>&1 | tee "logs/abl_score_b${b}${SUFFIX}.log"
done
echo "$(date)  ablation complete"
