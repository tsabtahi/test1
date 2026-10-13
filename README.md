


```
mkdir -p ~/SATLOCK/dataset/crops_tiny
python3 - <<'EOF'
import json, os, random
root = os.path.expanduser('~/SATLOCK/dataset/cphd_crops')
rows = [json.loads(l) for l in open(root + '/manifest.jsonl')]
random.seed(0)
pick = random.sample([r for r in rows if r['label'] == 'positive'], 8)
with open(os.path.expanduser('~/SATLOCK/dataset/crops_tiny/manifest.jsonl'), 'w') as f:
    for r in pick:
        r['cphd'] = '/crops/' + r['cphd']; r['json'] = '/crops/' + r['json']
        f.write(json.dumps(r) + '\n')
print(len(pick), 'crops')
EOF

docker run --rm -it --name tiny --gpus '"device=2"' --ipc=host --user $(id -u):$(id -g) \
    -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro \
    -v /home/tabtahi/SATLOCK/dataset/crops_tiny:/small:ro \
    -v /home/tabtahi/SATLOCK/runs:/runs \
    cphd-gpu /app/train_raw.py train --crops /small --ann-root /ann --out /runs/tiny \
    --overfit --shift-m 0 --epochs 1 --max-iters 1500 --samples-per-epoch 12000 \
    --batch 8 --lr 3e-4 --warmup 50 --pos-frac 1.0 --amp --workers 4 --log-every 50 --collage-every 1
```
