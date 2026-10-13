


```
docker run -d --name train --gpus '"device=2,3,4,5"' --ipc=host --user $(id -u):$(id -g) \
    -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro \
    -v /home/tabtahi/SATLOCK/runs:/runs \
    --entrypoint torchrun cphd-gpu --nproc_per_node=4 /app/train_raw.py train \
    --crops /crops --ann-root /ann --out /runs/raw1 --amp --workers 6 --epochs 100


docker run -d --name tb --rm -p 127.0.0.1:6006:6006 --user $(id -u):$(id -g) \
    -v /home/tabtahi/SATLOCK/runs:/runs:ro \
    --entrypoint tensorboard cphd-gpu --logdir /runs --host 0.0.0.0 --port 6006

cd ~/SATLOCK/dataset
ls /data3/sandbox/jhu/satlock/annotation_tool/test_aligned | sed 's/_GEC$//' > all_scenes.txt
ls -d cphd_crops/*/ | xargs -n1 basename | while read s; do
  [ -f cphd_crops/$s/manifest.jsonl ] && echo $s; done > done_scenes.txt
grep -v -F -f done_scenes.txt all_scenes.txt > ~/satlock_scenes_todo.txt
wc -l all_scenes.txt done_scenes.txt ~/satlock_scenes_todo.txt

docker run --rm --user $(id -u):$(id -g) \
    -v ~/satlock_scenes_todo.txt:/in/scenes.txt:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd:/out \
    umbra-cphd-fetch --scenes /in/scenes.txt --limit 20
```
