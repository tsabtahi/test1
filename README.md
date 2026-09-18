cd /home/tabtahi/SATLOCK && unzip -o gec_block_register.zip     # code only, results untouched
cd gec_block_register && chmod +x *.sh

GPUS="1 1 1 1" CPUS=5 nohup ./run_ablation_block.sh > logs/ablation_block.log 2>&1 &
sleep 20; head -4 logs/ablation_block.log


tail -f logs/abl_worker_lane*_gpu1.log
ls out_height/{roma,xoftr}_b{1024,2048}/*/report.json 2>/dev/null | wc -l   # → 264
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader              # watch GPU 1
