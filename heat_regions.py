"""Ship detections from the saved probability maps by watershed regions, compared with the detector's own peak picking.

Input: the folder written by  train_raw.py predict ... --save-heat <dir>  (one .npz per crop + index.json).

Methods, each swept over --thrs:
  peaks       the detector's decoding: 3x3 local maxima of the heat map >= thr (box from the size head)
  watershed   smooth the map (--sigma cells), seeds = local maxima >= thr at least --min-dist cells apart, grow each
              seed by watershed over the map down to thr x --low-frac; one detection per region:
              centre = heat-weighted centroid, box = region extent, score = region max
  watershed_size  watershed, then regions merged when the size head says they are one ship: the strongest region's
              predicted ship box (exp(size) at its peak) absorbs weaker regions whose centre lies inside it
  components  as watershed but without splitting: every connected area >= thr x --low-frac that holds a seed

Scores against the crop's ground truth (ignored ships neither hit nor miss):
  rec25 / prec25   a detection counts if its centre is within --tol-m of a ship centre (greedy by score)
  rec_iou          box IoU >= 0.3 with a ship box
  FA/km2           detections that matched nothing, per km2
Also: for every ship, the highest heat inside its box -> how many missed ships have activation at all.

Outputs in --out: table.csv, regions_<method>_thr<t>.geojson (detections in lat/lon), gt_heat.csv (per ship),
and --png-n example images per method: SAR image | image + ships + detections | probability map + boxes
(the SAR image needs predict --save-heat with --save-image-m > 0, the default 2 m).

    python heat_regions.py --heat /runs/fourier_adj_fixed/heat_val --out /runs/fourier_adj_fixed/heat_val_regions
Needs numpy, scipy, pillow.
"""
import argparse, csv, glob, json, math, os
import numpy as np
from scipy import ndimage as ndi


def load(heat_dir):
    idx = os.path.join(heat_dir, "index.json")
    files = ([os.path.join(heat_dir, r["npz"]) for r in json.load(open(idx))["crops"]] if os.path.exists(idx)
             else sorted(glob.glob(os.path.join(heat_dir, "*", "*.npz"))))
    for p in files:
        z = np.load(p)
        yield dict(path=p, scene=os.path.basename(os.path.dirname(p)), crop_id=os.path.basename(p)[:-4],
                   heat=z["heat"].astype(np.float32), off=z["off"].astype(np.float32), size=z["size"].astype(np.float32),
                   gt=z["gt_boxes"].reshape(-1, 4), ign=z["gt_ignore"].astype(bool), lat=float(z["lat"]),
                   lon=float(z["lon"]), cm=float(z["cell_m"]), cells=int(z["cells"]), geo=z["geo"],
                   image=z["image"] if "image" in z.files else None,
                   image_m=float(z["image_m"]) if "image_m" in z.files else None)


def cell_xy(r, c, cm, cells):
    """cell (row, col), fractional allowed -> crop-local metres east, north of the crop centre"""
    H = cells * cm / 2
    return (c + 0.5) * cm - H, H - (r + 0.5) * cm


# ------------------------------------------------------------------ the three methods: lists of (score, [e0,n0,e1,n1], (ce, cn))
def peaks(cr, thr):
    h = cr["heat"]
    pk = (h == ndi.maximum_filter(h, 3)) & (h >= thr)
    lab, n = ndi.label(pk, np.ones((3, 3)))                     # a flat top counts once
    out = []
    for sl_k, sl in enumerate(ndi.find_objects(lab), 1):
        rr, cc = np.nonzero(lab[sl] == sl_k)
        r, c = rr[0] + sl[0].start, cc[0] + sl[1].start
        ce, cn = cell_xy(r + cr["off"][1, r, c] - 0.5, c + cr["off"][0, r, c] - 0.5, cr["cm"], cr["cells"])
        w, hh = math.exp(cr["size"][0, r, c]), math.exp(cr["size"][1, r, c])
        out.append((float(h[r, c]), [ce - w / 2, cn - hh / 2, ce + w / 2, cn + hh / 2], (ce, cn)))
    return out


def regions(cr, thr, a, split=True, merge=False):
    h = ndi.gaussian_filter(cr["heat"], a.sigma) if a.sigma > 0 else cr["heat"]
    low = thr * a.low_frac
    seeds = (h == ndi.maximum_filter(h, 2 * a.min_dist + 1)) & (ndi.maximum_filter(cr["heat"], 3) >= thr)  # raw level
    if not seeds.any():
        return []
    mask = h >= low
    seed_lab, n_seed = ndi.label(seeds, np.ones((3, 3)))       # touching seed cells (a flat top) = one seed
    if split:
        mk = seed_lab.astype(np.int32)
        mk[~mask] = -1                                           # background marker below the low level
        img = np.round((1 - h / max(h.max(), 1e-6)) * 65535).astype(np.uint16)   # basins at the peaks
        lab = ndi.watershed_ift(img, mk)
        lab[(lab < 0) | ~mask] = 0
    else:
        lab, _ = ndi.label(mask)
        keep = np.unique(lab[seeds])
        lab[~np.isin(lab, keep[keep > 0])] = 0
    out = []
    w8 = cr["heat"]
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = lab[sl] == k
        if not m.any():
            continue
        rr, cc = np.nonzero(m)
        ww = w8[sl][m]
        r_c, c_c = (rr * ww).sum() / ww.sum() + sl[0].start, (cc * ww).sum() / ww.sum() + sl[1].start
        ce, cn = cell_xy(r_c, c_c, cr["cm"], cr["cells"])
        e0, n1 = cell_xy(sl[0].start - 0.5, sl[1].start - 0.5, cr["cm"], cr["cells"])
        e1, n0 = cell_xy(sl[0].stop - 0.5, sl[1].stop - 0.5, cr["cm"], cr["cells"])
        pr, pc = np.unravel_index(np.argmax(np.where(m, w8[sl], -1)), m.shape)
        out.append(dict(score=float(ww.max()), box=[e0, n0, e1, n1], c=(ce, cn), mass=float(ww.sum()),
                        peak=(pr + sl[0].start, pc + sl[1].start)))
    if merge:                                                  # repeat: a merged ship's new centre can reach further
        for _ in range(5):
            n0 = len(out)
            out = size_merge(out, cr, a.merge_pad)
            if len(out) == n0:
                break
    return [(d["score"], d["box"], d["c"]) for d in out]


def size_merge(regs, cr, pad=1.2):
    """Strongest region first: the size head at its peak says how big that ship is (exp(size) m east x north). Every
    weaker region whose centre falls inside that ship box is part of the same ship and is absorbed: centre = heat-weighted
    mean, box = union of the regions and the ship box. Splits a long hull's activation into one ship, but keeps two
    ships apart when the size head says each is small."""
    regs = sorted(regs, key=lambda d: -d["score"])
    out, used = [], [False] * len(regs)
    for i, d in enumerate(regs):
        if used[i]:
            continue
        pr, pc = d["peak"]
        w, h = math.exp(float(cr["size"][0, pr, pc])) * pad, math.exp(float(cr["size"][1, pr, pc])) * pad
        ce, cn = d["c"]
        sb = [ce - w / 2, cn - h / 2, ce + w / 2, cn + h / 2]
        grp = [d]
        for j in range(i + 1, len(regs)):
            if not used[j] and sb[0] <= regs[j]["c"][0] <= sb[2] and sb[1] <= regs[j]["c"][1] <= sb[3]:
                used[j] = True
                grp.append(regs[j])
        M = sum(g["mass"] for g in grp)
        c = (sum(g["c"][0] * g["mass"] for g in grp) / M, sum(g["c"][1] * g["mass"] for g in grp) / M)
        bx = [min([sb[0]] + [g["box"][0] for g in grp]), min([sb[1]] + [g["box"][1] for g in grp]),
              max([sb[2]] + [g["box"][2] for g in grp]), max([sb[3]] + [g["box"][3] for g in grp])]
        if len(grp) == 1:
            bx = d["box"]
        out.append(dict(score=d["score"], box=bx, c=c, mass=M, peak=d["peak"]))
    return out


# ------------------------------------------------------------------ scoring
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def score_crop(dets, cr, tol):
    gt, ign = cr["gt"], cr["ign"]
    gc = [(0.5 * (g[0] + g[2]), 0.5 * (g[1] + g[3])) for g in gt]
    used_c, used_i = set(), set()
    tp_c = tp_i = fp = 0
    for s, box, (ce, cn) in sorted(dets, key=lambda d: -d[0]):
        best, bd = None, tol
        for j, (x, y) in enumerate(gc):
            if j not in used_c and math.hypot(ce - x, cn - y) <= bd:
                best, bd = j, math.hypot(ce - x, cn - y)
        bi, bv = None, 0.3
        for j, g in enumerate(gt):
            if j not in used_i and iou(box, g) >= bv:
                bi, bv = j, iou(box, g)
        if best is not None and ign[best]:
            used_c.add(best); continue                           # on an ignored ship: neither hit nor false alarm
        if best is not None:
            used_c.add(best); tp_c += 1
        elif not any(g[0] <= ce <= g[2] and g[1] <= cn <= g[3] for g, i in zip(gt, ign) if i):
            fp += 1
        if bi is not None and not ign[bi]:
            used_i.add(bi); tp_i += 1
    return tp_c, tp_i, fp, int((~ign).sum())


def gt_heat(cr):
    """highest heat inside each ship box"""
    out = []
    H = cr["cells"] * cr["cm"] / 2
    for g, ig in zip(cr["gt"], cr["ign"]):
        c0, c1 = int(np.floor((g[0] + H) / cr["cm"])), int(np.ceil((g[2] + H) / cr["cm"]))
        r0, r1 = int(np.floor((H - g[3]) / cr["cm"])), int(np.ceil((H - g[1]) / cr["cm"]))
        sub = cr["heat"][max(r0, 0):max(min(r1, cr["cells"]), r0 + 1), max(c0, 0):max(min(c1, cr["cells"]), c0 + 1)]
        out.append((bool(ig), float(sub.max()) if sub.size else 0.0, float((g[2] - g[0]) * (g[3] - g[1]))))
    return out


def to_geo(cr, box):
    ml = math.pi / 180 * 6378137.0 * (1 - 6.69437999014e-3) / (1 - 6.69437999014e-3 * math.sin(math.radians(cr["lat"])) ** 2) ** 1.5
    mo = math.pi / 180 * 6378137.0 / math.sqrt(1 - 6.69437999014e-3 * math.sin(math.radians(cr["lat"])) ** 2) * math.cos(math.radians(cr["lat"]))
    e0, n0, e1, n1 = box
    return [cr["lon"] + e0 / mo, cr["lat"] + n0 / ml, cr["lon"] + e1 / mo, cr["lat"] + n1 / ml]


# ------------------------------------------------------------------ pictures
def png(cr, dets, path, title="", size=512):
    """Three panels, same extent, north up: SAR image | SAR image + ships (green, yellow = ignored) + detections (red,
    with score) | probability map + the same boxes. Without a stored image: the map panel only."""
    from PIL import Image, ImageDraw
    H = cr["cells"] * cr["cm"] / 2
    s = size / (2 * H)                                            # screen pixels per metre
    px = lambda e, n: ((e + H) * s, (H - n) * s)

    def boxes(im):
        d = ImageDraw.Draw(im)
        for g, ig in zip(cr["gt"], cr["ign"]):
            d.rectangle([*px(g[0], g[3]), *px(g[2], g[1])], outline=(255, 230, 0) if ig else (60, 255, 90), width=2)
        for sc, b, (ce, cn) in dets:
            d.rectangle([*px(b[0], b[3]), *px(b[2], b[1])], outline=(255, 50, 50), width=2)
            x, y = px(ce, cn)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 50, 50), width=2)
            d.text((x + 5, y - 12), f"{sc:.2f}", fill=(255, 90, 90))
        return im

    vmax = max(0.3, float(cr["heat"].max()))                      # fixed floor: a flat weak map stays dark
    h = np.clip(cr["heat"] / vmax, 0, 1)
    rgb = np.stack([np.clip(3 * h, 0, 1), np.clip(3 * h - 1, 0, 1),
                    np.clip(1 - 3 * h, 0, 1) * 0.6 + np.clip(3 * h - 2, 0, 1)], -1)
    heat = boxes(Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size), Image.NEAREST))
    ImageDraw.Draw(heat).text((4, 4), f"probability 0..{vmax:.2f}, max {cr['heat'].max():.2f}", fill=(255, 255, 255))
    panels = [heat]
    if cr.get("image") is not None:
        sar = Image.fromarray(cr["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
        raw = sar.copy()
        ImageDraw.Draw(raw).text((4, 4), f"SAR image ({cr['image_m']:g} m)", fill=(255, 255, 255))
        lab = boxes(sar)
        ImageDraw.Draw(lab).text((4, 4), "green = ship, yellow = ignored, red = detection", fill=(255, 255, 255))
        panels = [raw, lab, heat]
    W = len(panels) * size + (len(panels) - 1) * 6
    sheet = Image.new("RGB", (W, size + 20), (25, 25, 25))
    for k, pnl in enumerate(panels):
        sheet.paste(pnl, (k * (size + 6), 20))
    n_gt = int((~cr["ign"]).sum())
    ImageDraw.Draw(sheet).text((4, 4), f"{cr['crop_id']}   ships {n_gt}   detections {len(dets)}   {title}",
                               fill=(230, 230, 230))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heat", required=True, help="folder from train_raw.py predict --save-heat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thrs", type=float, nargs="*", default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--methods", nargs="*", default=["peaks", "watershed", "watershed_size", "components"])
    ap.add_argument("--sigma", type=float, default=1.0, help="smoothing of the map before regions, cells")
    ap.add_argument("--min-dist", type=int, default=3, help="seeds at least this many cells apart")
    ap.add_argument("--low-frac", type=float, default=0.5, help="regions grow down to thr x this")
    ap.add_argument("--merge-pad", type=float, default=1.2, help="watershed_size: ship box from the size head x this")
    ap.add_argument("--tol-m", type=float, default=25.0)
    ap.add_argument("--png-n", type=int, default=12, help="example images (crops with ships first)")
    ap.add_argument("--png-thr", type=float, default=0.1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    crops = list(load(a.heat))
    if not crops:
        raise SystemExit(f"no .npz under {a.heat}")
    print(f"{len(crops)} crops, {sum(int((~c['ign']).sum()) for c in crops)} ships")

    gh = [(c["scene"], c["crop_id"], *g) for c in crops for g in gt_heat(c)]
    with open(os.path.join(a.out, "gt_heat.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "crop_id", "ignore", "max_heat_in_box", "box_area_m2"]); w.writerows(gh)
    mh = np.array([g[3] for g in gh if not g[2]])
    if len(mh):
        print("ships by the highest heat inside their box: " +
              ", ".join(f">= {t:g}: {np.mean(mh >= t):.0%}" for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)))

    fn = dict(peaks=lambda c, t: peaks(c, t), watershed=lambda c, t: regions(c, t, a, True),
              components=lambda c, t: regions(c, t, a, False),
              watershed_size=lambda c, t: regions(c, t, a, True, merge=True))
    area = len(crops) * (crops[0]["cells"] * crops[0]["cm"] / 1000) ** 2
    rows = []
    for meth in a.methods:
        for t in a.thrs:
            tc = ti = fp = ng = nd = 0
            feats = []
            for c in crops:
                dets = fn[meth](c, t)
                x = score_crop(dets, c, a.tol_m)
                tc += x[0]; ti += x[1]; fp += x[2]; ng += x[3]; nd += len(dets)
                for s, b, _ in dets:
                    g = to_geo(c, b)
                    feats.append(dict(type="Feature", properties=dict(scene=c["scene"], crop_id=c["crop_id"], score=round(s, 4)),
                                      geometry=dict(type="Polygon", coordinates=[[[g[0], g[1]], [g[2], g[1]], [g[2], g[3]],
                                                                                 [g[0], g[3]], [g[0], g[1]]]])))
            with open(os.path.join(a.out, f"regions_{meth}_thr{t:g}.geojson"), "w") as f:
                json.dump(dict(type="FeatureCollection", features=feats), f)
            rows.append(dict(method=meth, thr=t, dets=nd, rec25=tc / max(ng, 1), prec25=tc / max(tc + fp, 1),
                             rec_iou=ti / max(ng, 1), fa_km2=fp / max(area, 1e-9)))
    print(f"{'method':>10s} {'thr':>6s} {'dets':>6s} {'rec25':>6s} {'prec25':>6s} {'recIoU':>6s} {'FA/km2':>7s}")
    for r in rows:
        print(f"{r['method']:>10s} {r['thr']:6g} {r['dets']:6d} {r['rec25']:6.3f} {r['prec25']:6.3f} {r['rec_iou']:6.3f} {r['fa_km2']:7.2f}")
    with open(os.path.join(a.out, "table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    if a.png_n > 0:
        os.makedirs(os.path.join(a.out, "png"), exist_ok=True)
        pick = sorted(crops, key=lambda c: -int((~c["ign"]).sum()))[:a.png_n]
        for c in pick:
            for meth in a.methods:
                png(c, fn[meth](c, a.png_thr), os.path.join(a.out, "png", f"{c['crop_id']}_{meth}.png"),
                    f"{meth}, threshold {a.png_thr:g}")
    print(f"-> {a.out}: table.csv, gt_heat.csv, regions_<method>_thr<t>.geojson" + (", png/" if a.png_n else ""))


if __name__ == "__main__":
    main()
