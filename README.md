


```
docker build -t cphd-gpu .
docker run -d --name boxadj --gpus '"device=0"' --user $(id -u):$(id -g) -p 127.0.0.1:8765:8765 \
  -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
  -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann_adj \
  -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops \
  cphd-gpu /app/box_adjust_server.py --crops /crops --ann-root /ann --ann-out /ann_adj --host 0.0.0.0 --device cuda:0
```
