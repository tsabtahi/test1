


```
docker run -d --name train_det --gpus '"device=1"' --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \
  -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
  cphd-gpu /app/train_raw.py train --crops /crops --ann-root /ann --out /runs/det_frcnn \
  --only-adjusted --arch det --detector fasterrcnn --det-backbone resnet50 --fixed-dft --aug --amp \
  --batch 4 --epochs 10 --samples-per-epoch 2000 --lr 1e-4 --shift-m 128 --collage-every 2

docker rm -f tb_adj 2>/dev/null
docker run -d --name tb_adj --user $(id -u):$(id -g) -e HOME=/tmp -p 127.0.0.1:6007:6006 \
  -v /home/tabtahi/SATLOCK/runs:/runs:ro --entrypoint tensorboard cphd-gpu \
  --logdir_spec det:/runs/det_frcnn,fixed_aug:/runs/fourier_adj_fixed_aug --host 0.0.0.0 --port 6006 --load_fast=false

python3 -c "
import json
for l in open('/home/tabtahi/SATLOCK/runs/det_frcnn/metrics.jsonl'):
    d=json.loads(l)
    for k in ('eval','eval_train'):
        if k in d: m=d[k]; print(f\"{k:10s} ep {m['epoch']:3d}  AP30 {m['ap30']:.3f}  rec25m {m.get('recall_within_25m',0):.3f}\")
"
```
