


```
mkdir -p ~/SATLOCK/dataset/crops_small
python3 - <<'EOF'
import json, os, random
root = os.path.expanduser('~/SATLOCK/dataset/cphd_crops')
rows = [json.loads(l) for l in open(root + '/manifest.jsonl')]
random.seed(0)
pos = [r for r in rows if r['label'] == 'positive']; neg = [r for r in rows if r['label'] != 'positive']
pick = random.sample(pos, min(100, len(pos))) + random.sample(neg, min(100, len(neg)))
with open(os.path.expanduser('~/SATLOCK/dataset/crops_small/manifest.jsonl'), 'w') as f:
    for r in pick:
        r['cphd'] = '/crops/' + r['cphd']; r['json'] = '/crops/' + r['json']
        f.write(json.dumps(r) + '\n')
print(len(pick), 'crops')
EOF
wc -l ~/SATLOCK/dataset/crops_small/manifest.jsonl

docker run -d --name s200 --gpus '"device=2"' --ipc=host --user $(id -u):$(id -g) \
    -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro \
    -v /home/tabtahi/SATLOCK/dataset/crops_small:/small:ro \
    -v /home/tabtahi/SATLOCK/runs:/runs \
    cphd-gpu /app/train_raw.py train --crops /small --ann-root /ann --out /runs/s200 \
    --overfit --shift-m 0 --epochs 10 --samples-per-epoch 2000 --batch 8 --lr 3e-4 \
    --warmup 100 --amp --workers 6 --log-every 50 --collage-every 2

docker run -d --name s200shift --gpus '"device=3"' --ipc=host --user $(id -u):$(id -g) \
    -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro \
    -v /home/tabtahi/SATLOCK/dataset/crops_small:/small:ro \
    -v /home/tabtahi/SATLOCK/runs:/runs \
    cphd-gpu /app/train_raw.py train --crops /small --ann-root /ann --out /runs/s200_shift \
    --overfit --shift-m 64 --epochs 10 --samples-per-epoch 2000 --batch 8 --lr 3e-4 \
    --warmup 100 --amp --workers 6 --log-every 50 --collage-every 2

docker run -d --name full_noshift --gpus '"device=4,5"' --ipc=host --user $(id -u):$(id -g) \
    -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro \
    -v /home/tabtahi/SATLOCK/runs:/runs \
    --entrypoint torchrun cphd-gpu --nproc_per_node=2 /app/train_raw.py train \
    --crops /crops --ann-root /ann --out /runs/raw_noshift --amp --workers 6 \
    --epochs 30 --shift-m 0 --lr 1e-4
```
