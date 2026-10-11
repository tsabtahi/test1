#!/usr/bin/env python3
"""Download Umbra open-data CPHD files for a list of scene names.

Input : a JSON (or plain text) file holding names such as "2025-10-29-03-46-55_UMBRA-10_GEC".
Output: <out>/<YYYY-MM-DD-HH-MM-SS_UMBRA-NN>/<YYYY-MM-DD-HH-MM-SS_UMBRA-NN>_CPHD.cphd

    python umbra_cphd_fetch.py --scenes completed_scenes.json --out /data/cphd --dry-run     # what would be fetched
    python umbra_cphd_fetch.py --scenes completed_scenes.json --out /data/cphd --limit 5     # first 5 scenes
    python umbra_cphd_fetch.py --scenes completed_scenes.json --out /data/cphd               # everything

How it finds the files: the bucket keeps scenes in three trees (seen 2026-10-11),
    sar-data/task-data/<task id>/<scene>/        CPHD, CSI, GEC, SICD, SIDD
    sar-data/tasks/<task name>/.../<scene>/
    open-data/<id>/<scene>/                      GEC, SICD, SIDD only in the folders sampled
and S3 can only search by the start of a key, so the scene name alone is not enough to address a file. The script
lists those trees once (many folders side by side), keeps that list in <out>/_index/, and looks every scene up in it:
any key that contains the scene name and ends in .cphd counts. Later runs reuse the list; --refresh rebuilds it.

CPHD files are large (1.6 to 55 GB each in a sample of ten), so look at the --dry-run total first.

Safe to rerun: a file already present with the right size is skipped, and a file is only given its final name after
its size has been checked against the bucket.
"""
import argparse, csv, glob, gzip, os, re, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config

BUCKET, REGION, PREFIXES = "umbra-open-data-catalog", "us-west-2", ["sar-data/", "open-data/"]
SCENE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_UMBRA-\d+")
GB = 1e9


def say(*a):
    print(*a, flush=True)


def read_scenes(path):
    """{scene: name as written in the file}, in file order, without repeats. Suffixes such as _GEC are not part of
    the scene; file extensions are dropped from the written name."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    out = {}
    for m in re.finditer(r"(" + SCENE.pattern + r")((?:_[A-Za-z0-9]+)*)", text):
        out.setdefault(m.group(1), m.group(1) + m.group(2))
    return out


def folder_for(scene, written, mode):
    return {"scene": scene, "json": written, "date": scene.split("_")[0]}[mode]


# ------------------------------------------------------------------ bucket listing
def make_client(region, endpoint, pool):
    return boto3.client("s3", region_name=region, endpoint_url=endpoint,
                        config=Config(signature_version=UNSIGNED, max_pool_connections=pool,
                                      retries={"max_attempts": 10, "mode": "adaptive"}))


def list_level(s3, bucket, prefix):
    """One folder level: (sub-folders, [(key, size)] directly inside)."""
    subs, keys = [], []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        subs += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        keys += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return subs, keys


def list_under(s3, bucket, prefix):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return keys


def build_inventory(s3, bucket, prefixes, threads, depth):
    """Every key under `prefixes`. The top `depth` folder levels are walked first so that the folders below can be
    listed side by side; files sitting directly in a walked folder are collected on the way."""
    t0 = time.time()
    level, keys = list(prefixes), []
    with ThreadPoolExecutor(threads) as ex:
        for _ in range(depth):
            nxt = []
            for subs, k in ex.map(lambda p: list_level(s3, bucket, p), level):
                nxt += subs
                keys += k
            level = nxt
        say(f"listing s3://{bucket}/ under {' and '.join(prefixes)}: {len(level):,} folders, {threads} at a time")
        done, t_say = 0, time.time()
        for fut in as_completed([ex.submit(list_under, s3, bucket, p) for p in level]):
            keys += fut.result()
            done += 1
            if time.time() - t_say > 15:
                say(f"  {done:,} of {len(level):,} folders, {len(keys):,} files so far")
                t_say = time.time()
    say(f"listed {len(keys):,} files in {time.time() - t0:.0f} s")
    return keys


def save_inventory(path, keys):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path + ".tmp", "wt", encoding="utf-8") as f:
        for key, size in keys:
            f.write(f"{size}\t{key}\n")
    os.replace(path + ".tmp", path)


def load_inventory(path):
    keys = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            size, key = line.rstrip("\n").split("\t", 1)
            keys.append((key, int(size)))
    return keys


# ------------------------------------------------------------------ matching
def plan(scenes, keys, out, mode):
    """One row per scene: what the bucket holds for it and which file to fetch."""
    wanted = set(scenes)
    by_scene = {}
    for key, size in keys:
        for s in set(SCENE.findall(key)):
            if s in wanted:
                by_scene.setdefault(s, []).append((key, size))
    rows = []
    for s in scenes:
        objs = by_scene.get(s, [])
        products = sorted({m.group(1).upper() for k, _ in objs
                           for m in [re.search(re.escape(s) + r"_([A-Za-z0-9]+)", os.path.basename(k))] if m})
        row = dict(scene=s, status="", products=" ".join(products), files=[], note="")
        cphd = [(k, n) for k, n in objs if k.lower().endswith(".cphd")]
        if not objs:
            row["status"] = "not in bucket"
        elif not cphd:
            row["status"] = "no CPHD published"
        else:
            best = {}                                   # same file under several task folders: keep one copy
            for k, n in sorted(cphd):
                name = os.path.basename(k)
                if name not in best or n > best[name][1]:
                    best[name] = (k, n)
            if len(cphd) > len(best):
                row["note"] = f"{len(cphd)} copies in the bucket, 1 kept per file name"
            row["files"] = [dict(scene=s, key=k, size=n, dest=os.path.join(out, folder_for(s, scenes[s], mode), name), status="")
                            for name, (k, n) in sorted(best.items())]
            row["status"] = "to download"
        rows.append(row)
    return rows


# ------------------------------------------------------------------ download
class Progress:
    def __init__(self, total_bytes, total_files, every):
        self.total, self.n_files, self.every = total_bytes, total_files, every
        self.done, self.files_done, self.t0 = 0, 0, time.time()
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def add(self, n):
        with self.lock:
            self.done += n

    def file_done(self):
        with self.lock:
            self.files_done += 1

    def _loop(self):
        while not self.stop.wait(self.every):
            el = time.time() - self.t0
            rate = self.done / max(el, 1e-3)
            line = f"  {self.done / GB:.1f} of {self.total / GB:.1f} GB, {self.files_done} of {self.n_files} files"
            if self.done > 0:
                left = (self.total - self.done) / rate
                eta = f"{left / 3600:.1f} h" if left >= 5400 else f"{left / 60:.0f} min"
                line += f", {rate / 1e6:.0f} MB/s, about {eta} left"
            say(line)


def fetch(s3, bucket, item, tcfg, progress, tries=3):
    dest, size = item["dest"], item["size"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part, err = dest + ".part", None
    for attempt in range(1, tries + 1):
        for stale in glob.glob(glob.escape(part) + "*"):         # leftovers of an interrupted run
            os.remove(stale)
        got = [0]
        def count(n):
            got[0] += n
            progress.add(n)
        try:
            s3.download_file(bucket, item["key"], part, Config=tcfg, Callback=count)
            have = os.path.getsize(part)
            if have != size:
                raise IOError(f"size {have} differs from the bucket's {size}")
            os.replace(part, dest)
            return "downloaded"
        except Exception as e:
            err = e
            progress.add(-got[0])
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if str(code) in ("403", "404", "AccessDenied", "NoSuchKey"):     # will not get better on a retry
                break
            if attempt < tries:
                time.sleep(10 * attempt)
    for stale in glob.glob(glob.escape(part) + "*"):
        os.remove(stale)
    return f"failed: {type(err).__name__}: {err}"


def write_report(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "status", "size_gb", "products_in_bucket", "s3_uri", "local_path", "note"])
        for r in rows:
            if not r["files"]:
                w.writerow([r["scene"], r["status"], "", r["products"], "", "", r["note"]])
            for it in r["files"]:
                w.writerow([r["scene"], it["status"] or r["status"], f"{it['size'] / GB:.3f}", r["products"],
                            f"s3://{r['bucket']}/{it['key']}", it["dest"], r["note"]])


def main():
    ap = argparse.ArgumentParser(description="Download Umbra open-data CPHD files for a list of scene names")
    ap.add_argument("--scenes", default="/in/completed_scenes.json", help="JSON or text file with the scene names")
    ap.add_argument("--out", default="/out", help="output folder; one sub-folder per scene")
    ap.add_argument("--index-dir", default=None, help="folder for the bucket listing and the report (default <out>/_index)")
    ap.add_argument("--refresh", action="store_true", help="list the bucket again instead of reusing the saved listing")
    ap.add_argument("--dry-run", action="store_true", help="show what would be downloaded and stop")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes of the file")
    ap.add_argument("--folder-name", choices=["scene", "json", "date"], default="scene",
                    help="sub-folder per scene: scene = 2025-10-29-03-46-55_UMBRA-10 (default), "
                         "json = as written in the input file, date = 2025-10-29-03-46-55")
    ap.add_argument("--jobs", type=int, default=3, help="files downloaded at the same time")
    ap.add_argument("--threads", type=int, default=8, help="connections per file")
    ap.add_argument("--list-threads", type=int, default=32, help="bucket folders listed at the same time")
    ap.add_argument("--list-depth", type=int, default=2, help="folder levels walked before listing in parallel")
    ap.add_argument("--force", action="store_true", help="download even if the disk looks too small")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between progress lines")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--prefix", nargs="+", default=PREFIXES, help="bucket folders to list")
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--endpoint-url", default=None, help="only for testing against another S3 endpoint")
    a = ap.parse_args()

    scenes = read_scenes(a.scenes)
    if not scenes:
        sys.exit(f"no scene names like 2025-10-29-03-46-55_UMBRA-10 found in {a.scenes}")
    say(f"{len(scenes)} scenes in {a.scenes}")
    if a.limit > 0:
        scenes = dict(list(scenes.items())[:a.limit])
        say(f"--limit: using the first {len(scenes)}")

    os.makedirs(a.out, exist_ok=True)
    index_dir = a.index_dir or os.path.join(a.out, "_index")
    os.makedirs(index_dir, exist_ok=True)
    inv_path = os.path.join(index_dir, "umbra_open_data_listing.tsv.gz")
    s3 = make_client(a.region, a.endpoint_url, max(a.list_threads, a.jobs * a.threads) + 4)

    fresh = a.refresh or not os.path.exists(inv_path)
    if fresh:
        keys = build_inventory(s3, a.bucket, a.prefix, a.list_threads, a.list_depth)
        save_inventory(inv_path, keys)
    else:
        keys = load_inventory(inv_path)
        age = (time.time() - os.path.getmtime(inv_path)) / 86400
        say(f"reusing the bucket listing in {inv_path}: {len(keys):,} files, {age:.1f} days old")

    rows = plan(scenes, keys, a.out, a.folder_name)
    for r in rows:
        r["bucket"] = a.bucket
    items = [it for r in rows for it in r["files"]]
    todo = []
    for it in items:
        if os.path.exists(it["dest"]) and os.path.getsize(it["dest"]) == it["size"]:
            it["status"] = "already present"
        else:
            todo.append(it)

    n_have = sum(1 for r in rows if r["files"])
    none = [r["scene"] for r in rows if r["status"] == "no CPHD published"]
    gone = [r["scene"] for r in rows if r["status"] == "not in bucket"]
    need = sum(it["size"] for it in todo)
    say(f"\nCPHD found for {n_have} of {len(rows)} scenes: {len(items)} files, {sum(i['size'] for i in items) / GB:.1f} GB "
        f"({len(items) - len(todo)} already on disk, {len(todo)} to download, {need / GB:.1f} GB)")
    if none:
        say(f"in the bucket but without a CPHD ({len(none)}): " + ", ".join(none))
    if gone:
        say(f"not in the bucket at all ({len(gone)}): " + ", ".join(gone))
        if not fresh:
            say("  the saved listing may be out of date; --refresh lists the bucket again")
    report = os.path.join(index_dir, "cphd_download_report.csv")

    if a.dry_run:
        for it in todo:
            say(f"  {it['size'] / GB:6.2f} GB  s3://{a.bucket}/{it['key']}")
        write_report(report, rows)
        say(f"\ndry run: nothing downloaded. Report: {report}")
        return 0

    free = shutil.disk_usage(a.out).free
    if need > free - 5 * GB and not a.force:
        write_report(report, rows)
        say(f"\nnot enough room in {a.out}: {need / GB:.1f} GB to download, {free / GB:.1f} GB free. "
            f"Use --limit, free some space, or pass --force. Nothing downloaded.")
        return 2

    failed = 0
    if todo:
        say(f"\ndownloading {len(todo)} files into {a.out}, {a.jobs} at a time")
        tcfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=a.threads,
                              use_threads=True)
        prog = Progress(need, len(todo), a.every)
        prog.thread.start()
        try:
            with ThreadPoolExecutor(a.jobs) as ex:
                futs = {ex.submit(fetch, s3, a.bucket, it, tcfg, prog): it for it in todo}
                for fut in as_completed(futs):
                    it = futs[fut]
                    it["status"] = fut.result()
                    prog.file_done()
                    failed += it["status"].startswith("failed")
                    say(f"  {it['status'].split(':')[0]:10s} {it['size'] / GB:6.2f} GB  {os.path.relpath(it['dest'], a.out)}"
                        + (f"\n      {it['status']}" if it["status"].startswith("failed") else ""))
                    write_report(report, rows)
        finally:
            prog.stop.set()
    for r in rows:
        if r["files"]:
            st = [it["status"] for it in r["files"]]
            r["status"] = "failed" if any(s.startswith("failed") for s in st) else \
                          "downloaded" if "downloaded" in st else "already present"
    write_report(report, rows)
    say(f"\ndone: {sum(it['status'] == 'downloaded' for it in items)} downloaded, "
        f"{sum(it['status'] == 'already present' for it in items)} already present, {failed} failed, "
        f"{len(none)} without a CPHD, {len(gone)} not in the bucket. Report: {report}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
