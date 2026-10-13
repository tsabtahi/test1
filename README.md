


```
docker run -d --name train --gpus '"device=2,3,4,5"' --ipc=host --user $(id -u):$(id -g) \
    -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro \
    -v /home/tabtahi/SATLOCK/runs:/runs \
    --entrypoint torchrun cphd-gpu --nproc_per_node=4 /app/train_raw.py train \
    --crops /crops --ann-root /ann --out /runs/raw1 --amp --workers 6 --epochs 100
```
