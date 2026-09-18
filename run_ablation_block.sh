#!/bin/bash
# Block-size ablation on the TIE-POINTED scenes only (the ones we can score).
#
#   A1: --block 1024  + SAR nodata filter   (cloud filter OFF)
#   A2: --block 2048  + SAR nodata filter   (cloud filter OFF)
#
# Both matchers each, so the two-matcher agreement gate can be scored per arm.
# Cloud filter is deliberately OFF so block size is the ONLY variable.
#
# Outputs:
#   out_height/roma_b1024/<scene>/   out_height/xoftr_b1024/<scene>/
#   out_height/roma_b2048/<scene>/   out_height/xoftr_b2048/<scene>/
#
# Lanes are set by repeating a device id in GPUS, e.g. 4 lanes on GPU 1:
#   GPUS="1 1 1 1" CPUS=5 ./run_ablation_block.sh
#
# VRAM budget per lane (54 GB card): RoMa ~7-8 GB (resizes internally, flat in
# block size); XoFTR runs NATIVELY so it scales with block^2 — ~2.5 GB at 1024,
# ~10 GB at 2048. Worst case is all lanes on XoFTR@2048, so keep
# lanes x 10 GB under the card: 4 lanes ~40 GB (safe), 5 ~50 GB (tight).
set -u
cd /home/tabtahi/SATLOCK/gec_block_register
mkdir -p logs out_height
SCRATCH=/home/tabtahi/SATLOCK/scratch; mkdir -p "$SCRATCH"

RESID=${RESID:-/home/tabtahi/SATLOCK/ce90_out/residuals.csv}
# MAX_SCENES caps the scene list (jobs = scenes x blocks x matchers).
# e.g. MAX_SCENES=25 -> 25 x 2 x 2 = 100 jobs. Sampling is seeded and pinned to
# scenes_tiepoint.txt, so BOTH arms score the identical scene set (paired) and
# re-runs reuse the same sample. Unset = all tie-pointed scenes.
MAX_SCENES=${MAX_SCENES:-0}
SEED=${SEED:-0}
WV_SUBSET=/home/kashley/satlock/mosaic_datasets/eo_sar_umbra_subset_HH_unique_v2/wv_dailytake_output
WV_FULL=/data3/sandbox/kashley/satlock/data_collection/outputs/wv_dailytake_output
CPUS=${CPUS:-5}
BLOCKS=${BLOCKS:-"1024 2048"}
MATCHERS=${MATCHERS:-"roma xoftr"}

if [ -n "${GPUS:-}" ]; then GPU_LIST=($GPUS)
else GPU_LIST=($(seq 0 $(( $(nvidia-smi -L 2>/dev/null | wc -l) - 1 )))); fi
NGPU=${#GPU_LIST[@]}
[ "$NGPU" -lt 1 ] && { echo "no GPUs"; exit 1; }

# --- scenes that have tie points -------------------------------------------------
if [ -s scenes_tiepoint.txt ] && [ "${REUSE_SCENES:-1}" = "1" ]; then
  echo "reusing existing scenes_tiepoint.txt ($(wc -l < scenes_tiepoint.txt) scenes)"
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
    random.Random(seed).shuffle(scenes)          # seeded -> reproducible
    scenes = sorted(scenes[:nmax])
print("\n".join(scenes))
PY
fi
total=$(wc -l < scenes_tiepoint.txt)
echo "$(date)  ${total} tie-pointed scenes | blocks [${BLOCKS}] x matchers [${MATCHERS}] | GPUs ${GPU_LIST[*]}"

# --- job list: scene x block x matcher -------------------------------------------
> jobs_ablation_block.txt
while read s; do
  for b in $BLOCKS; do
    for m in $MATCHERS; do
      echo "$s $b $m" >> jobs_ablation_block.txt
    done
  done
done < scenes_tiepoint.txt
njobs=$(wc -l < jobs_ablation_block.txt)
echo "$(date)  ${njobs} jobs across ${NGPU} lanes"

worker() {
  lane=$1; gpu=$2; j=0; consec_fail=0
  while read s b m; do
    j=$((j+1)); [ $(( (j-1) % NGPU )) -ne "$lane" ] && continue
    sar="/home/tabtahi/SATLOCK/dataset/SAR_RPC_ellip/${s}/${s}_rpc_ellip.tif"
    wv=$(ls ${WV_SUBSET}/${s}/mosaic/*.tif 2>/dev/null | head -1)
    [ -z "$wv" ] && wv=$(ls ${WV_FULL}/${s}/mosaic/*.tif 2>/dev/null | head -1)
    if [ ! -f "$sar" ] || [ -z "$wv" ]; then
      echo "[gpu${gpu} skip] ${s} (sar or wv missing)"; continue
    fi
    exp="${m}_b${b}"
    out="out_height/${exp}/${s}"
    [ -f "${out}/report.json" ] && { echo "[gpu${gpu} skip ${j}/${njobs}] ${exp} ${s}"; continue; }
    for attempt in 1 2 3; do
      echo "[gpu${gpu} ${j}/${njobs}] $(date +%H:%M) ${exp} ${s} (try ${attempt})"
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
        --block "$b" --no-cloud-filter \
        --no-viz --no-tif \
        --out "/work/out_height/${exp}/${s}" \
        < /dev/null >> "logs/abl_${exp}_${s}.log" 2>&1
      rc=$?
      if [ -f "${out}/report.json" ]; then consec_fail=0; break; fi
      if [ "$rc" -eq 125 ]; then
        echo "[gpu${gpu} FATAL] docker daemon error (disk?) — lane stopping"; return 1; fi
      echo "[gpu${gpu} retry] ${exp} ${s} rc=${rc}; 60s"; sleep 60
    done
    if [ ! -f "${out}/report.json" ]; then
      consec_fail=$((consec_fail+1))
      [ "$consec_fail" -ge 5 ] && { echo "[gpu${gpu} FATAL] 5 consecutive failures — lane stopping"; return 1; }
    fi
  done < jobs_ablation_block.txt
  echo "[gpu${gpu}] $(date) lane done"
}

pids=""
for lane in $(seq 0 $((NGPU-1))); do
  g=${GPU_LIST[$lane]}
  worker "$lane" "$g" > "logs/abl_worker_lane${lane}_gpu${g}.log" 2>&1 &
  pids="$pids $!"; echo "lane ${lane} -> gpu${g} pid $!"
done
wait $pids
echo "$(date)  ablation matching complete"

# --- score both arms -------------------------------------------------------------
for b in $BLOCKS; do
  echo ""
  echo "================ BLOCK ${b} ================"
  docker run --rm \
    -v /home/tabtahi/SATLOCK/gec_block_register:/work \
    -v /home/tabtahi/SATLOCK/ce90_out:/home/tabtahi/SATLOCK/ce90_out \
    -v ${SCRATCH}:/tmp -e TMPDIR=/tmp \
    --entrypoint python3 height-register \
    /work/compute_ce90_agree.py \
      --residuals "$RESID" \
      --out-root /work/out_height \
      --roma-exp "roma_b${b}" --xoftr-exp "xoftr_b${b}" \
      --medfilter-tol 10 --agree-tols 5 10 20 50 \
      --out /home/tabtahi/SATLOCK/ce90_out/abl_block${b} \
    2>&1 | tee logs/abl_score_b${b}.log
done
echo "$(date)  ablation complete"
