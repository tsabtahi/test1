


```
docker run -d --name layers --gpus '"device=1"' --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \
  -p 127.0.0.1:8770:8770 \
  -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  cphd-gpu /app/layer_viewer.py --ckpt /runs/det_frcnn/best.pt --crops /crops --ann-root /ann \
  --split val --host 0.0.0.0
"
```
