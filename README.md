


```
for f in ~/SATLOCK/dataset/cphd_crops/_logs/shard*.log; do
  echo "$(basename $f): $(grep -o '^\[[0-9]*/[0-9]*\]' $f | tail -1)"
done
```
