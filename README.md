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
