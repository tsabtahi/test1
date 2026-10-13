


```
docker run -d --rm --name f200_shift --gpus '"device=2"' --ipc=host --user $(id -u):$(id -g) \
  -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/crops_small:/small:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro \
  -v /home/tabtahi/SATLOCK/runs:/runs \
  cphd-gpu /app/train_raw.py train --crops /small --ann-root /ann --out /runs/f200_shift \
  --arch fourier --overfit --shift-m 64 --amp --batch 8 --collage-every 2

docker run -d --rm --name ftrain --gpus '"device=3"' --ipc=host --user $(id -u):$(id -g) \
  -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  --entrypoint torchrun cphd-gpu --nproc_per_node=4 /app/train_raw.py train \
  --crops /crops --ann-root /ann --out /runs/fourier_full --arch fourier --amp --batch 8

for c in $(docker ps -q); do echo "$(docker inspect -f '{{.Name}}  GPUs: {{.HostConfig.DeviceRequests}}' $c)"; done

docker run -d --rm --name ftrain2 --gpus '"device=2,3"' --ipc=host --user $(id -u):$(id -g) \
  -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  --entrypoint torchrun cphd-gpu --nproc_per_node=2 /app/train_raw.py train \
  --crops /crops --ann-root /ann --arch fourier --amp --batch 8 \
  --resume /runs/fourier_full/last.pt --out /runs/fourier_full_r1 \
  --epochs 100 --samples-per-epoch 4000 --collage-every 1
```
