cd /home/tabtahi/SATLOCK && unzip -o gec_block_register.zip     # code only, results untouched
cd gec_block_register && chmod +x *.sh

GPUS="1 1 1 1" CPUS=5 nohup ./run_ablation_block.sh > logs/ablation_block.log 2>&1 &
sleep 20; head -4 logs/ablation_block.log


tail -f logs/abl_worker_lane*_gpu1.log
ls out_height/{roma,xoftr}_b{1024,2048}/*/report.json 2>/dev/null | wc -l   # → 264
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader              # watch GPU 1


cd /home/tabtahi/SATLOCK/gec_block_register

MAX_SCENES=50 GPUS="1 1 1 1" CPUS=5 nohup ./run_ablation_block.sh > logs/ablation_block.log 2>&1 &

sleep 20; head -5 logs/ablation_block.log

pkill -f run_ablation_block.sh
docker kill $(docker ps -q --filter ancestor=height-register) 2>/dev/null
docker ps        # verify empty


```
cd /home/tabtahi/SATLOCK/gec_block_register
echo "=== alive ==="; ps aux | grep run_ablation_block | grep -v grep | wc -l          # 5 = parent + 4 lanes
echo "=== progress ==="
t=$(( $(wc -l < scenes_tiepoint.txt) * 4 ))
for e in roma_b1024 xoftr_b1024 roma_b2048 xoftr_b2048; do
  echo "  $e: $(ls out_height/$e/*/report.json 2>/dev/null | wc -l) / $(wc -l < scenes_tiepoint.txt)"
done
d=$(ls out_height/{roma,xoftr}_b{1024,2048}/*/report.json 2>/dev/null | wc -l); echo "  total $d / $t jobs"
echo "=== lanes ==="; tail -qn1 logs/abl_worker_lane*_gpu1.log
echo "=== problems ==="; grep -c "FATAL\|retry" logs/abl_worker_lane*_gpu1.log; grep -l "out of memory" logs/abl_*.log 2>/dev/null | head -3
echo "=== gpu 1 ==="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed -n 2p
echo "=== scored? ==="; ls /home/tabtahi/SATLOCK/ce90_out/abl_block*/summary.txt 2>/dev/null
```

```
cd /home/tabtahi/SATLOCK/gec_block_register
ps aux | grep run_ablation_block | grep -v grep | wc -l     # >0 = still running
grep -E "matching complete|BLOCK|ablation complete" logs/ablation_block.log
ls -la logs/abl_score_b*.log /home/tabtahi/SATLOCK/ce90_out/abl_block*/summary.txt 2>&1
```

```
for b in 1024 2048; do
  echo "======== BLOCK $b ========"
  docker run --rm \
    -v /home/tabtahi/SATLOCK/gec_block_register:/work \
    -v /home/tabtahi/SATLOCK/ce90_out:/home/tabtahi/SATLOCK/ce90_out \
    --entrypoint python3 height-register \
    /work/compute_ce90_agree.py \
      --residuals /home/tabtahi/SATLOCK/ce90_out/residuals.csv \
      --out-root /work/out_height \
      --roma-exp roma_b$b --xoftr-exp xoftr_b$b \
      --medfilter-tol 10 --agree-tols 5 10 20 50 \
      --out /home/tabtahi/SATLOCK/ce90_out/abl_block${b}_partial
done
```
```
cd /home/tabtahi/SATLOCK && unzip -o gec_block_register.zip
cd gec_block_register
BLOCKS="2048" MATCHERS="xoftr" GPUS="1 1 1 1" CPUS=5 \
  nohup ./run_ablation_block.sh > logs/ablation_xoftr2048.log 2>&1 &
sleep 60; tail -qn1 logs/abl_worker_lane*_gpu1.log; grep -c FATAL logs/abl_worker_lane*_gpu1.log
```


```
cd /home/tabtahi/SATLOCK && unzip -o gec_block_register.zip
cd gec_block_register
ls -la .dockerignore          # must exist, or the build ships out_height/ again
docker build -t height-register -f Dockerfile.register .
```
```
gdalinfo $(ls /data3/sandbox/kashley/satlock/data_collection/outputs/wv_dailytake_output/*/mosaic/*.tif | head -1) | grep -E "^Band|ColorInterp"
```

```
cd /home/tabtahi/SATLOCK/gec_block_register
docker ps | grep -c height-register                    # containers actually running
tail -qn2 logs/abl_worker_lane*.log                    # what each lane last did
tail -8 $(ls -t logs/abl_xoftr_b2048_*.log | head -1)  # inside the newest job
for g in 1 2; do docker run --rm --gpus "\"device=$g\"" --entrypoint python3 height-register \
```


```
pkill -f run_ablation_block.sh
docker kill $(docker ps -q --filter ancestor=height-register) 2>/dev/null

cd /home/tabtahi/SATLOCK && unzip -o gec_block_register.zip && cd gec_block_register
grep -c "max_side" matchers/__init__.py          # must be non-zero now

docker run --rm -v $PWD:/work --entrypoint python3 height-register -c \
  "import sys; sys.path.insert(0,'/work'); from matchers import XoFTRMatcher; print('xoftr max_side =', XoFTRMatcher().max_side)"

BLOCKS="2048" MATCHERS="xoftr" GPUS="1 1 1 1 2 2 2 2" CPUS=5 \
  nohup ./run_ablation_block.sh > logs/ablation_xoftr2048.log 2>&1 &
```
  -c "import torch; print('gpu$g', torch.cuda.is_available(), torch.cuda.device_count())"; done
```
