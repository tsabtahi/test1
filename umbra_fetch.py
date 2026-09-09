#!/usr/bin/env python3
"""
umbra_phase_fetch.py — build a small phase-data test set from Umbra's open-data bucket.

Bucket: s3://umbra-open-data-catalog (us-west-2, public, no credentials needed)
Phase-bearing products:
    *_SICD.nitf  single-look complex, phase preserved  (~200 MB/scene)
    *_CPHD.cphd  compensated phase history             (~600 MB/scene)
Also pulls *_METADATA.json (~4 KB) so incidence/azimuth/track are on disk.

Scene ID = Umbra's own naming: <YYYY-MM-DD-HH-MM-SS>_UMBRA-<NN>.
Output layout keeps that intact:

    <out>/
        2023-11-19-16-12-16_UMBRA-05/
            2023-11-19-16-12-16_UMBRA-05_SICD.nitf
            2023-11-19-16-12-16_UMBRA-05_METADATA.json
        ...
        _index.json      cached bucket listing (reused on later runs)
        manifest.csv     scene_id, task, product, key, bytes, path
        scenes.json      per-scene geometry pulled from METADATA.json

Examples:
    # dry run first — see what 30 scenes you'd get, and how big
    python umbra_phase_fetch.py --out ./umbra_phase --limit 30 --dry-run

    # fetch them
    python umbra_phase_fetch.py --out ./umbra_phase --limit 30

    # phase history too, one scene per task, only 2024+
    python umbra_phase_fetch.py --out ./umbra_phase --limit 30 \
        --products sicd,cphd --max-per-task 1 --after 2024-01-01

Requires: boto3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

BUCKET = "umbra-open-data-catalog"
REGION = "us-west-2"
PREFIX = "sar-data/tasks/"

# suffix -> product tag
PRODUCTS = {
    "_SICD.nitf": "SICD",
    "_CPHD.cphd": "CPHD",
    "_METADATA.json": "METADATA",
}

SCENE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_UMBRA-(\d+)$")


# ----------------------------------------------------------------------------- client
def s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 10, "mode": "standard"}),
    )


# ----------------------------------------------------------------------------- listing
def list_bucket(cache: Path, refresh: bool = False) -> list[dict]:
    """Full listing of sar-data/tasks/, cached to disk. ~20k objects, ~20 API calls."""
    if cache.exists() and not refresh:
        with cache.open() as f:
            objs = json.load(f)
        print(f"[index] {len(objs):,} objects from cache {cache}")
        return objs

    s3 = s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    objs: list[dict] = []
    t0 = time.time()
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for o in page.get("Contents", []):
            key = o["Key"]
            if key.endswith(tuple(PRODUCTS)):
                objs.append({"key": key, "size": o["Size"]})
        print(f"\r[index] scanned, {len(objs):,} product objects kept", end="", flush=True)
    print(f"\n[index] done in {time.time() - t0:.1f}s -> {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump(objs, f)
    return objs


def parse_key(key: str) -> tuple[str, str, str] | None:
    """key -> (scene_id, product, task). None if it isn't a recognisable scene file."""
    base = key.rsplit("/", 1)[-1]
    for suffix, product in PRODUCTS.items():
        if base.endswith(suffix):
            scene_id = base[: -len(suffix)]
            if not SCENE_RE.match(scene_id):
                return None
            parts = key.split("/")
            # sar-data/tasks/<task>/...  ; 'ad hoc' tasks nest one level deeper
            task = parts[2] if len(parts) > 3 else "unknown"
            if task.lower() == "ad hoc" and len(parts) > 4:
                task = f"ad hoc/{parts[3]}"
            return scene_id, product, task
    return None


def build_scenes(objs: list[dict]) -> dict[str, dict]:
    scenes: dict[str, dict] = {}
    for o in objs:
        parsed = parse_key(o["key"])
        if not parsed:
            continue
        scene_id, product, task = parsed
        sc = scenes.setdefault(
            scene_id,
            {"scene_id": scene_id, "task": task, "files": {}, "date": scene_id[:10]},
        )
        # keep the largest copy if a scene id appears under more than one task
        prev = sc["files"].get(product)
        if prev is None or o["size"] > prev["size"]:
            sc["files"][product] = {"key": o["key"], "size": o["size"]}
    return scenes


# ----------------------------------------------------------------------------- selection
def select(scenes: dict[str, dict], want: list[str], limit: int,
           max_per_task: int, task_filter: str | None,
           after: str | None, before: str | None) -> list[dict]:
    """Deterministic round-robin across tasks so the test set isn't 30 shots of one site."""
    pool = []
    for sc in scenes.values():
        if not all(p in sc["files"] for p in want):
            continue
        if task_filter and task_filter.lower() not in sc["task"].lower():
            continue
        if after and sc["date"] < after:
            continue
        if before and sc["date"] > before:
            continue
        pool.append(sc)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for sc in sorted(pool, key=lambda s: s["scene_id"]):
        by_task[sc["task"]].append(sc)

    chosen: list[dict] = []
    round_i = 0
    tasks = sorted(by_task)
    while len(chosen) < limit and round_i < max_per_task:
        added = False
        for t in tasks:
            if round_i < len(by_task[t]):
                chosen.append(by_task[t][round_i])
                added = True
                if len(chosen) == limit:
                    break
        if not added:
            break
        round_i += 1
    return chosen


# ----------------------------------------------------------------------------- download
def download_one(s3, key: str, dest: Path, expected: int) -> tuple[str, int, str]:
    """Atomic .tmp + rename; skip if already present at the right size."""
    if dest.exists() and dest.stat().st_size == expected:
        return key, 0, "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    cfg = TransferConfig(multipart_threshold=64 * 1024**2, multipart_chunksize=64 * 1024**2,
                         max_concurrency=4, use_threads=True)
    s3.download_file(BUCKET, key, str(tmp), Config=cfg)
    got = tmp.stat().st_size
    if expected and got != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(f"size mismatch {key}: got {got}, expected {expected}")
    os.replace(tmp, dest)
    return key, got, "ok"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


# ----------------------------------------------------------------------------- metadata
def summarize_metadata(scene_dir: Path, scene_id: str) -> dict:
    """Pull the geometry fields that matter for geolocation work out of METADATA.json."""
    mpath = scene_dir / f"{scene_id}_METADATA.json"
    if not mpath.exists():
        return {}
    try:
        md = json.loads(mpath.read_text())
    except Exception:
        return {}
    c = (md.get("collects") or [{}])[0]
    center = (c.get("sceneCenterPointLla") or {}).get("coordinates") or [None, None, None]
    return {
        "collect_id": c.get("id"),
        "task_id": c.get("taskId"),
        "start_utc": c.get("startAtUTC"),
        "satellite": md.get("umbraSatelliteName"),
        "imaging_mode": md.get("imagingMode"),
        "incidence_deg": c.get("angleIncidenceDegrees"),
        "azimuth_deg": c.get("angleAzimuthDegrees"),
        "grazing_deg": c.get("angleGrazingDegrees"),
        "squint_deg": c.get("angleSquintDegrees"),
        "track": c.get("satelliteTrack"),
        "look": c.get("observationDirection"),
        "pol": ",".join(c.get("polarizations") or []),
        "slant_range_m": c.get("slantRangeMeters"),
        "center_lon": center[0],
        "center_lat": center[1],
        "center_hae_m": center[2],
        "res_az_m": (c.get("maxGroundResolution") or {}).get("azimuthMeters"),
        "res_rg_m": (c.get("maxGroundResolution") or {}).get("rangeMeters"),
    }


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="output root")
    ap.add_argument("--limit", type=int, default=30, help="number of scenes (default 30)")
    ap.add_argument("--products", default="sicd",
                    help="comma list from sicd,cphd (METADATA always included). default: sicd")
    ap.add_argument("--max-per-task", type=int, default=2,
                    help="max scenes per task/location, for spatial diversity (default 2)")
    ap.add_argument("--task-filter", help="substring match on task name, e.g. 'Beet Piler'")
    ap.add_argument("--after", help="earliest scene date, YYYY-MM-DD")
    ap.add_argument("--before", help="latest scene date, YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=4, help="parallel scene downloads (default 4)")
    ap.add_argument("--dry-run", action="store_true", help="list the selection and size, download nothing")
    ap.add_argument("--refresh-index", action="store_true", help="re-walk the bucket instead of using cache")
    args = ap.parse_args()

    want = [p.strip().upper() for p in args.products.split(",") if p.strip()]
    bad = set(want) - {"SICD", "CPHD"}
    if bad:
        print(f"unknown product(s): {', '.join(sorted(bad))}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    objs = list_bucket(args.out / "_index.json", refresh=args.refresh_index)
    scenes = build_scenes(objs)
    print(f"[index] {len(scenes):,} distinct scenes in bucket")

    chosen = select(scenes, want, args.limit, args.max_per_task,
                    args.task_filter, args.after, args.before)
    if not chosen:
        print("no scenes matched the filters", file=sys.stderr)
        return 1

    fetch = want + ["METADATA"]
    total = sum(sc["files"][p]["size"] for sc in chosen for p in fetch if p in sc["files"])
    print(f"\n[plan] {len(chosen)} scenes across {len({s['task'] for s in chosen})} tasks, "
          f"products={'+'.join(fetch)}, {human(total)}\n")
    for sc in chosen:
        sz = sum(sc["files"][p]["size"] for p in fetch if p in sc["files"])
        print(f"  {sc['scene_id']}  {human(sz):>9}  {sc['task']}")

    if args.dry_run:
        print("\n[dry-run] nothing downloaded")
        return 0

    s3 = s3_client()
    jobs = []
    for sc in chosen:
        for p in fetch:
            if p in sc["files"]:
                f = sc["files"][p]
                dest = args.out / sc["scene_id"] / f["key"].rsplit("/", 1)[-1]
                jobs.append((sc, p, f, dest))

    print(f"\n[get] {len(jobs)} files -> {args.out}")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, s3, f["key"], dest, f["size"]): (sc, p, f, dest)
                for sc, p, f, dest in jobs}
        for fut in as_completed(futs):
            sc, p, f, dest = futs[fut]
            done += 1
            try:
                _, _, status = fut.result()
                print(f"  [{done}/{len(jobs)}] {status:4s} {sc['scene_id']}_{p}")
            except Exception as e:
                print(f"  [{done}/{len(jobs)}] FAIL {sc['scene_id']}_{p}: {e}", file=sys.stderr)

    # manifest + geometry table
    with (args.out / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene_id", "task", "product", "s3_key", "bytes", "local_path"])
        for sc, p, f, dest in jobs:
            w.writerow([sc["scene_id"], sc["task"], p, f["key"], f["size"],
                        str(dest.relative_to(args.out))])

    summaries = []
    for sc in chosen:
        s = summarize_metadata(args.out / sc["scene_id"], sc["scene_id"])
        if s:
            summaries.append({"scene_id": sc["scene_id"], "task": sc["task"], **s})
    (args.out / "scenes.json").write_text(json.dumps(summaries, indent=2))

    print(f"\n[done] manifest.csv + scenes.json written to {args.out}")
    print("       read complex data with: "
          "sarpy.io.complex.converter.open_complex('<scene>_SICD.nitf')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
