


```
ls /data3/sandbox/jhu/satlock/annotation_tool/test_aligned > ~/satlock_scenes.txt

docker run --rm --user $(id -u):$(id -g) \
    -v ~/satlock_scenes.txt:/in/scenes.txt:ro \
    -v /home/tabtahi/SATLOCK/dataset/cphd:/out \
    umbra-cphd-fetch --scenes /in/scenes.txt --dry-run
```
