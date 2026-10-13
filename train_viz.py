"""Monitoring for train_raw.py: dataset statistics, TensorBoard, and image collages of predictions.

Collage: a fixed set of validation crops is imaged once by backprojection (for viewing only, the model never sees
these images). Every N epochs the model runs on the same crops and each tile shows
    left:  the crop image (dB) with true boxes (green), ignored boxes (yellow) and predicted boxes (red, with score)
    right: the predicted heat map (ship-centre score per 4 m cell), true centres marked green
"""
import json, math, os
import numpy as np
import torch
from PIL import Image, ImageDraw

import train_detector as td


# ------------------------------------------------------------------ dataset statistics
def _box_m(b, lat):
    ml, mo = td.m_per_deg(lat)
    return (b[2] - b[0]) * mo, (b[3] - b[1]) * ml


def dataset_stats(tr, va):
    """Counts per split and per scene, plus ship sizes, from the crop label files."""
    def split(ds):
        scenes, sizes, vis = {}, [], []
        n_box = n_ign = 0
        per_crop = []
        for m in ds.meta:
            s = scenes.setdefault(m["scene"], dict(crops=0, positive=0, negative=0, boxes=0))
            s["crops"] += 1
            s["positive" if m["label"] == "positive" else "negative"] += 1
            k = 0
            for b in m["boxes"]:
                if b.get("ignore"):
                    n_ign += 1
                    continue
                k += 1
                w, h = _box_m(b["bbox_geo"], m["centre_lat"])
                sizes.append((max(w, h), min(w, h)))
                vis.append(b.get("visible", 1.0))
            s["boxes"] += k
            n_box += k
            per_crop.append(k)
        sizes = np.array(sizes).reshape(-1, 2)
        pc = np.array(per_crop)
        q = lambda a: {p: round(float(np.percentile(a, p)), 1) for p in (5, 50, 95)} if len(a) else {}
        return dict(crops=len(ds.meta), positive=int(ds.positive.sum()), negative=int((~ds.positive).sum()),
                    scenes=len(scenes), boxes=n_box, ignored_boxes=n_ign,
                    boxes_per_positive_crop=round(float(pc[pc > 0].mean()), 2) if (pc > 0).any() else 0.0,
                    ship_length_m=q(sizes[:, 0]), ship_width_m=q(sizes[:, 1]), per_scene=scenes,
                    _lengths=sizes[:, 0].tolist(), _widths=sizes[:, 1].tolist())
    return dict(train=split(tr), val=split(va))


def stats_text(st):
    lines = []
    for name in ("train", "val"):
        s = st[name]
        lines.append(f"{name}: {s['crops']} crops ({s['positive']} with ships, {s['negative']} empty) from {s['scenes']} scenes; "
                     f"{s['boxes']} ship boxes ({s['boxes_per_positive_crop']} per ship crop), {s['ignored_boxes']} ignored")
        if s["ship_length_m"]:
            L, W = s["ship_length_m"], s["ship_width_m"]
            lines.append(f"  ship length m: 5% {L[5]}, median {L[50]}, 95% {L[95]};  width m: 5% {W[5]}, median {W[50]}, 95% {W[95]}")
    return "\n".join(lines)


def stats_markdown(st):
    md = ["| split | crops | with ships | empty | scenes | ship boxes | ignored | median length m | median width m |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in ("train", "val"):
        s = st[name]
        md.append(f"| {name} | {s['crops']} | {s['positive']} | {s['negative']} | {s['scenes']} | {s['boxes']} | "
                  f"{s['ignored_boxes']} | {s['ship_length_m'].get(50, '-')} | {s['ship_width_m'].get(50, '-')} |")
    md += ["", "| scene | split | crops | with ships | empty | ship boxes |", "|---|---|---|---|---|---|"]
    for name in ("train", "val"):
        for sc, v in sorted(st[name]["per_scene"].items()):
            md.append(f"| {sc} | {name} | {v['crops']} | {v['positive']} | {v['negative']} | {v['boxes']} |")
    return "\n".join(md)


def write_stats(st, out, writer=None):
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in st.items()}
    with open(os.path.join(out, "dataset_stats.json"), "w") as f:
        json.dump(clean, f, indent=1)
    if writer is not None:
        writer.add_text("dataset/summary", stats_markdown(st), 0)
        for name in ("train", "val"):
            if st[name]["_lengths"]:
                writer.add_histogram(f"dataset/{name}_ship_length_m", np.array(st[name]["_lengths"]), 0)
                writer.add_histogram(f"dataset/{name}_ship_width_m", np.array(st[name]["_widths"]), 0)


# ------------------------------------------------------------------ TensorBoard
def make_writer(out):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:                                             # tensorboard not installed: keep training
        print(f"TensorBoard off ({type(e).__name__}: {e}); metrics.jsonl is still written", flush=True)
        return None
    return SummaryWriter(os.path.join(out, "tb"))


# ------------------------------------------------------------------ collage
def _colormap(x):
    """0..1 -> RGB uint8, dark blue -> red -> yellow."""
    x = np.clip(x, 0, 1)
    r = np.clip(3 * x, 0, 1); g = np.clip(3 * x - 1, 0, 1); b = np.clip(1 - 3 * x, 0, 1) * 0.6 + np.clip(3 * x - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


class Collage:
    def __init__(self, ds, cfg, device, n=8, pixel_m=2.0, seed=0, chunk_elems=1.6e7):
        """Pick n crops of the dataset (half with ships when possible) and image them once."""
        self.ds, self.cfg, self.device, self.px = ds, cfg, device, pixel_m
        rng = np.random.default_rng(seed)
        pos, neg = np.nonzero(ds.positive)[0], np.nonzero(~ds.positive)[0]
        k = min(len(pos), max(n - min(len(neg), n // 4), n // 2))
        self.idx = list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, min(len(neg), n - k), replace=False))
        self.items = [ds.get(int(i), shift=(0.0, 0.0)) for i in self.idx]
        self.H = cfg["cells"] * cfg["cell_m"] / 2
        self.n = int(round(2 * self.H / pixel_m))
        self.images = [self._image(it, chunk_elems) for it in self.items]

    @torch.no_grad()
    def _image(self, it, chunk_elems):
        g = td.make_grid(it["lat"], it["lon"], self.n, self.px)
        h_ref = self._h_ref(it)
        centre = td.llh_to_ecef(it["lat"], it["lon"], h_ref)
        pix = torch.from_numpy(td.grid_pixels_ecef(g, h_ref, centre))[None].to(self.device)
        P, S = it["X"].shape
        im = td.Imager(P, S, n_sub=1, oversample=2, chunk_elems=chunk_elems, learn_taper=False).to(self.device)
        im.w_p.data = torch.hann_window(P, periodic=False, device=self.device)
        im.w_s.data = torch.hann_window(S, periodic=False, device=self.device)
        re, imag = im(it["X"][None].to(self.device), it["tx"][None].to(self.device), it["rc"][None].to(self.device),
                      it["f0"][None].to(self.device), it["scss"][None].to(self.device), torch.tensor([-1], device=self.device), pix)
        amp = torch.sqrt(re[0, 0] ** 2 + imag[0, 0] ** 2).reshape(self.n, self.n).cpu().numpy()
        db = 20 * np.log10(np.maximum(amp, 1e-30) / max(float(np.median(amp)), 1e-30))
        return np.clip((db + 5) / 40 * 255, 0, 255).astype(np.uint8)

    def _h_ref(self, it):
        for m in self.ds.meta:
            if m["crop_id"] == it["crop_id"]:
                return m["h_ref"]
        return 0.0

    def _px(self, b):                                                  # crop metres [e0, n0, e1, n1] -> image pixels
        return [(b[0] + self.H) / self.px, (self.H - b[3]) / self.px, (b[2] + self.H) / self.px, (self.H - b[1]) / self.px]

    @torch.no_grad()
    def render(self, model, decode, collate, to_device, epoch, score_thr=0.3):
        core = model.module if hasattr(model, "module") else model
        was = core.training
        core.eval()
        tiles = []
        for it, img in zip(self.items, self.images):
            b = to_device(collate([it]), self.device)
            out = core(b)
            (sc, bx), = decode(out, self.cfg, thr=score_thr)
            heat = torch.sigmoid(out["heat"][0, 0]).float().cpu().numpy()
            left = Image.fromarray(img).convert("RGB")
            d = ImageDraw.Draw(left)
            for gb, ig in zip(it["gt"], it["gt_ignore"]):
                d.rectangle(self._px(gb), outline=(255, 220, 0) if ig else (0, 255, 0), width=2)
            for s, pb in zip(sc, bx):
                x0, y0, x1, y1 = self._px(pb)
                d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=2)
                d.text((x0 + 2, max(y0 - 11, 0)), f"{s:.2f}", fill=(255, 80, 80))
            right = Image.fromarray(_colormap(heat)).resize((self.n, self.n), Image.NEAREST)
            dr = ImageDraw.Draw(right)
            for gb, ig in zip(it["gt"], it["gt_ignore"]):
                if not ig:
                    cx, cy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                    x, y = (cx + self.H) / self.px, (self.H - cy) / self.px
                    dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(0, 255, 0), width=2)
            dr.text((3, 3), f"max {heat.max():.2f}", fill=(255, 255, 255))
            tile = Image.new("RGB", (2 * self.n + 4, self.n + 16), (0, 0, 0))
            tile.paste(left, (0, 16)); tile.paste(right, (self.n + 4, 16))
            ImageDraw.Draw(tile).text((3, 2), f"{it['crop_id'][-28:]}  ships {int((~it['gt_ignore']).sum())}  pred {len(sc)}",
                                      fill=(220, 220, 220))
            tiles.append(tile)
        cols = 2
        rows = math.ceil(len(tiles) / cols)
        W, Hh = tiles[0].size
        sheet = Image.new("RGB", (cols * W + (cols - 1) * 6, rows * Hh + (rows - 1) * 6 + 22), (30, 30, 30))
        ImageDraw.Draw(sheet).text((4, 4), f"epoch {epoch}: green = ship, yellow = ignored, red = predicted (score >= {score_thr}); "
                                            f"right = heat map", fill=(255, 255, 255))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * (W + 6), 22 + (i // cols) * (Hh + 6)))
        if was:
            core.train()
        return sheet
