


```
docker run -d --name train_adj --gpus '"device=1"' --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \
  -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  cphd-gpu /app/train_raw.py train \
  --crops /crops --ann-root /ann --out /runs/fourier_adj_r1 --only-adjusted --arch fourier --amp```
```
