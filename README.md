


```
docker run --rm -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/out \
    --entrypoint chown cphd-crops -R $(id -u):$(id -g) /out
```
