


```
docker stop ftrain2
docker run -d --rm --name ftrain4 --gpus '"device=A,B,C,D"' --ipc=host --user $(id -u):$(id -g) \
  -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  --entrypoint torchrun cphd-gpu --nproc_per_node=4 /app/train_raw.py train \
  --crops /crops --ann-root /ann --arch fourier --amp --batch 8 \
  --resume /runs/fourier_full_r1/last.pt --out /runs/fourier_full_r2 \
  --epochs 250 --samples-per-epoch 4000 --collage-every 5
```
