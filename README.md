


```
docker run --rm -it --name train --gpus '"device=0"' --ipc=host \
    --user $(id -u):$(id -g) -e HOME=/tmp \
    -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro \
    -v /home/tabtahi/SATLOCK/runs:/runs \
    cphd-gpu /app/train_raw.py train \
    --crops /crops --ann-root /ann --out /runs/raw_1gpu --amp
```
