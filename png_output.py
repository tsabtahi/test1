#!/usr/bin/env python3
"""Render crop CPHDs as PNG images with their label boxes drawn on top, to check that the boxes sit on the ships.

Each crop is imaged by backprojection onto its own lat/lon grid (the same grid the labels use), shown in dB, and the
boxes from its .json are drawn from their lat/lon corners: green = ship, yellow = ignored (cut, < 30% visible).

    python png_output.py --cphd /crops/<scene>/<crop>.cphd                        # json next to it is found
    python png_output.py --cphd crop.cphd --json crop.json --out pngs/
    python png_output.py --crops /crops --n 20 --only positive --out pngs/        # a batch from the manifest
    python png_output.py --crops /crops --scene 2023-06-20 --n 50 --out pngs/

Uses the GPU if there is one (a 512 x 512 image takes well under a second there, tens of seconds on CPU).
"""
import argparse, json, os, random, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_detector as td


def image_crop(cphd_path, meta, pixel_m, device, chunk_elems, taper=True):
    """Amplitude image of a crop on a north-up lat/lon grid centred on the crop, and that grid."""
    d = td.read_crop(cphd_path)
    size_m = meta["grid"]["n"] * meta["grid"]["pixel_m"]
    n = int(round(size_m / pixel_m))
    g = td.make_grid(meta["centre_lat"], meta["centre_lon"], n, pixel_m)
    pix = td.grid_pixels_ecef(g, meta["h_ref"], d["srp"])
    P, S = d["X"].shape
    im = td.Imager(P, S, n_sub=1, oversample=2, chunk_elems=chunk_elems, learn_taper=False).to(device)
    if taper:                                                         # Hann taper: lower sidelobes, easier to read
        im.w_p.data = torch.hann_window(P, periodic=False, device=device)
        im.w_s.data = torch.hann_window(S, periodic=False, device=device)
    t = lambda a, dt=None: torch.as_tensor(a, dtype=dt)[None].to(device)
    with torch.no_grad():
        re, imag = im(t(d["X"]), t(d["tx"]), t(d["rc"]), t(d["f0"], torch.float64),
                      t(d["scss"], torch.float64), t(d["sgn"]), t(pix))
    amp = torch.sqrt(re[0, 0] ** 2 + imag[0, 0] ** 2).reshape(n, n).cpu().numpy()
    return amp, g


def geo_to_px(b, g):
    lon0, lat0, lon1, lat1 = b
    return [(lon0 - g["lon_w"]) / g["dlon"], (g["lat_n"] - lat1) / g["dlat"],
            (lon1 - g["lon_w"]) / g["dlon"], (g["lat_n"] - lat0) / g["dlat"]]


def render(amp, g, meta, lo, hi, zoom):
    ref = float(np.median(amp[amp > 0])) if np.any(amp > 0) else 1.0
    db = 20 * np.log10(np.maximum(amp, 1e-30) / ref)
    u8 = np.clip((db - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(u8).convert("RGB")
    if zoom != 1:
        img = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
    head = 34
    canvas = Image.new("RGB", (img.width, img.height + head), (0, 0, 0))
    canvas.paste(img, (0, head))
    dr = ImageDraw.Draw(canvas)
    n_box = 0
    for b in meta.get("boxes", []):
        x0, y0, x1, y1 = [v * zoom for v in geo_to_px(b["bbox_geo"], g)]
        colour = (255, 220, 0) if b.get("ignore") else (0, 255, 0)
        dr.rectangle([x0, y0 + head, x1, y1 + head], outline=colour, width=2)
        dr.text((max(x0, 2), max(y0 + head - 12, head)), f"{b.get('ann_id')} {b.get('visible', 1):.0%}", fill=colour)
        n_box += 1
    m = g["n"] * g["pixel_m"]
    dr.text((4, 2), f"{meta['crop_id']}", fill=(255, 255, 255))
    dr.text((4, 17), f"{meta['label']}  ships {meta['n_pos']}  ignored {meta['n_ignore']}  |  {m:.0f} m, "
                     f"{g['pixel_m']:g} m/px  |  dB {lo:g}..{hi:g}", fill=(180, 180, 180))
    return canvas, n_box


def pick(a):
    """(cphd, json) pairs from the arguments."""
    if a.cphd:
        js = a.json or os.path.splitext(a.cphd)[0] + ".json"
        return [(a.cphd, js)]
    rows = td.read_manifest(a.crops)
    if not rows:                                                      # no manifest yet: every crop json on disk
        import glob
        for j in sorted(glob.glob(os.path.join(a.crops, "*", "*.json"))):
            with open(j) as f:
                d = json.load(f)
            rows.append(dict(scene=d["scene"], label=d["label"], cphd=os.path.join(d["scene"], d["cphd"]),
                             json=os.path.relpath(j, a.crops)))
    if a.scene:
        rows = [r for r in rows if any(s in r["scene"] for s in a.scene)]
    if a.only != "all":
        rows = [r for r in rows if r["label"] == a.only]
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.n] if a.n else rows
    return [(os.path.join(a.crops, r["cphd"]), os.path.join(a.crops, r["json"])) for r in rows]


def main():
    ap = argparse.ArgumentParser(description="PNG of crop CPHDs with their label boxes")
    ap.add_argument("--cphd", help="one crop .cphd")
    ap.add_argument("--json", help="its label .json (default: same name next to the .cphd)")
    ap.add_argument("--crops", help="crop folder (with manifest.jsonl) for a batch")
    ap.add_argument("--scene", nargs="*", help="only scenes whose name contains one of these")
    ap.add_argument("--only", choices=["positive", "negative", "all"], default="all")
    ap.add_argument("--n", type=int, default=20, help="how many crops in a batch (0 = all)")
    ap.add_argument("--out", default="./crop_pngs")
    ap.add_argument("--pixel-m", type=float, default=1.0, help="image pixel; 2 is 4x faster")
    ap.add_argument("--db", nargs=2, type=float, default=[-5.0, 35.0], metavar=("LO", "HI"))
    ap.add_argument("--zoom", type=int, default=1)
    ap.add_argument("--no-taper", action="store_true")
    ap.add_argument("--chunk-elems", type=float, default=1.6e7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.cphd and not a.crops:
        ap.error("give --cphd or --crops")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    pairs = pick(a)
    print(f"{len(pairs)} crop(s) on {device} -> {a.out}")
    for cp, js in pairs:
        with open(js) as f:
            meta = json.load(f)
        amp, g = image_crop(cp, meta, a.pixel_m, device, a.chunk_elems, taper=not a.no_taper)
        img, nb = render(amp, g, meta, a.db[0], a.db[1], a.zoom)
        path = os.path.join(a.out, meta["crop_id"] + ".png")
        img.save(path)
        print(f"  {os.path.basename(path)}  {meta['label']}, {nb} box(es)")


if __name__ == "__main__":
    main()
