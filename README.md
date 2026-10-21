


```
docker run -d --name train_adj --gpus '"device=1"' --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \
  -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  cphd-gpu /app/train_raw.py train \
  --crops /crops --ann-root /ann --out /runs/fourier_adj_r1 --only-adjusted --arch fourier --amp \
  --samples-per-epoch 2000 --collage-every 2


docker run -d --name tb_adj --user $(id -u):$(id -g) -e HOME=/tmp -p 127.0.0.1:6007:6006 \
  -v /home/tabtahi/SATLOCK/runs:/runs:ro --entrypoint tensorboard cphd-gpu \
  --logdir /runs/fourier_adj_r1 --host 0.0.0.0 --port 6006 --load_fast=false

docker run --rm --gpus '"device=1"' --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $PWD/train_raw.py:/app/train_raw.py:ro \
  -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  cphd-gpu /app/train_raw.py predict --ckpt /runs/fourier_adj_r1/best.pt --crops /crops --ann-root /ann \
  --split val --score-thrs 0.05 0.1 0.15 0.2 --batch 1 --workers 2 --collage-n 8 --collage-pixel-m 4 \
  --out /runs/fourier_adj_r1/pred_val_coll
```
