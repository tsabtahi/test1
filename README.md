


```
docker run -d --name train_fixed_aug --gpus '"device=1"' --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $PWD/train_raw.py:/app/train_raw.py:ro \
  -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  cphd-gpu /app/train_raw.py train \
  --crops /crops --ann-root /ann --out /runs/fourier_adj_fixed_aug \
  --only-adjusted --arch fourier --fixed-dft --amp --aug \
  --epochs 10 --samples-per-epoch 2000 --lr 1.5e-4 --shift-m 128 \
  --collage-every 2 --train-eval-n 400

docker run --rm --gpus '"device=1"' --user $(id -u):$(id -g) -e HOME=/tmp \
  -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  cphd-gpu /app/train_raw.py predict --ckpt /runs/fourier_adj_fixed_aug/best.pt --crops /crops --ann-root /ann \
  --split val --score-thrs 0.1 0.3 --batch 1 --collage-n 0 --out /runs/fourier_adj_fixed_aug/pred_val \
  --save-heat /runs/fourier_adj_fixed_aug/heat_val
```
