#!/usr/bin/env python3
"""Ship detector trained end to end on CPHD crops, with geo boxes in and geo boxes out.

    CPHD crop (1024 pulses x 1024 samples, complex)
      -> Imager: differentiable backprojection onto a lat/lon grid (torch), with a learnable range and pulse taper;
         2 sub-apertures (first and second half of the pulses) and their coherent sum
      -> 3 channels: full-aperture dB, sub-aperture 1 dB, sub-aperture 2 dB
      -> RetinaNet (torchvision, ResNet-FPN, focal loss, GIoU box loss)
      -> pixel boxes -> lat/lon boxes (the grid is regular in lat/lon, so this map is exact)

Labels come from the scene's annotations.json in lat/lon, placed on each training grid after a random shift of the
grid (augmentation), so the loss compares predicted and annotated boxes on the same geo grid. Grid pixels are square
in metres, so the box loss is in metres. Cut ships (visible < 30%) are ignored rather than counted as background.
The class skew is handled by focal loss plus a sampler that draws a set fraction of positive crops per batch.

    python train_detector.py train --crops /data/cphd_crops --ann-root /ann --out runs/r1
    torchrun --nproc_per_node=4 train_detector.py train ...           # several GPUs (DDP)
    python train_detector.py predict --ckpt runs/r1/best.pt --crops /data/cphd_crops --out preds.geojson
    python train_detector.py selftest                                 # synthetic end-to-end check on CPU
"""
import argparse, json, math, os, random, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import sarkit.cphd as skcphd

C = 299792458.0
_A, _E2 = 6378137.0, 6.69437999014e-3
SCENE_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_UMBRA-\d+")


# ------------------------------------------------------------------ geometry
def llh_to_ecef(lat, lon, h):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    h = np.asarray(h, float)
    n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo), (n * (1 - _E2) + h) * np.sin(la)], -1)


def m_per_deg(lat):
    s = math.sin(math.radians(lat))
    m_lat = math.pi / 180 * _A * (1 - _E2) / (1 - _E2 * s * s) ** 1.5
    m_lon = math.pi / 180 * _A / math.sqrt(1 - _E2 * s * s) * math.cos(math.radians(lat))
    return m_lat, m_lon


def make_grid(lat, lon, n, pixel_m):
    """North-up n x n grid regular in lat/lon, pixel_m square at its centre (same as GeoGrid in build_crops.py)."""
    m_lat, m_lon = m_per_deg(lat)
    dlat, dlon = pixel_m / m_lat, pixel_m / m_lon
    return dict(n=n, pixel_m=pixel_m, dlat=dlat, dlon=dlon, lon_w=lon - n * dlon / 2, lat_n=lat + n * dlat / 2)


def grid_bounds(g):
    return g["lon_w"], g["lat_n"] - g["n"] * g["dlat"], g["lon_w"] + g["n"] * g["dlon"], g["lat_n"]   # W, S, E, N


def grid_pixels_ecef(g, h, origin):
    lat = g["lat_n"] - (np.arange(g["n"]) + 0.5) * g["dlat"]
    lon = g["lon_w"] + (np.arange(g["n"]) + 0.5) * g["dlon"]
    lon, lat = np.meshgrid(lon, lat)
    return (llh_to_ecef(lat, lon, np.full(lat.shape, h)).reshape(-1, 3) - origin).astype(np.float32)


def px_to_geo(box, g):
    x0, y0, x1, y1 = [float(v) for v in box]
    return [g["lon_w"] + x0 * g["dlon"], g["lat_n"] - y1 * g["dlat"], g["lon_w"] + x1 * g["dlon"], g["lat_n"] - y0 * g["dlat"]]


def boxes_on_grid(anns, g, min_visible):
    """Annotation boxes (lat/lon) placed on grid g: pixel boxes, labels, ignore flags."""
    W, S, E, N = grid_bounds(g)
    out, ign = [], []
    for a in anns:
        x0, x1 = max(a["lon_min"], W), min(a["lon_max"], E)
        y0, y1 = max(a["lat_min"], S), min(a["lat_max"], N)
        if x1 <= x0 or y1 <= y0:
            continue
        area = (a["lon_max"] - a["lon_min"]) * (a["lat_max"] - a["lat_min"])
        vis = (x1 - x0) * (y1 - y0) / area if area > 0 else 1.0
        b = [(x0 - W) / g["dlon"], (N - y1) / g["dlat"], (x1 - W) / g["dlon"], (N - y0) / g["dlat"]]
        if b[2] - b[0] < 1 or b[3] - b[1] < 1:
            continue
        out.append(b)
        ign.append(bool(vis < min_visible or a.get("iscrowd", 0)))
    return np.array(out, np.float32).reshape(-1, 4), np.array(ign, bool)


def load_scene_annotations(ann_root):
    """{scene: [box dicts]} from <ann_root>/<scene>_GEC/annotations.json; bbox_geo = [lon_min, lat_min, lon_max, lat_max]."""
    out = {}
    if not ann_root or not os.path.isdir(ann_root):
        return out
    for d in sorted(os.listdir(ann_root)):
        for name in ("annotations.json", "annotation.json"):
            p = os.path.join(ann_root, d, name)
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                js = json.load(f)
            js = js[0] if isinstance(js, list) else js
            m = SCENE_RE.search(js.get("scene_name") or d)
            if not m:
                break
            anns = []
            for a in js.get("annotations", []):
                g = a.get("bbox_geo")
                if g and len(g) == 4:
                    lo0, la0, lo1, la1 = map(float, g)
                    anns.append(dict(id=a.get("id"), lon_min=min(lo0, lo1), lat_min=min(la0, la1), lon_max=max(lo0, lo1),
                                     lat_max=max(la0, la1), iscrowd=int(a.get("iscrowd", 0) or 0)))
            out[m.group(0)] = anns
            break
    return out


# ------------------------------------------------------------------ data
def read_crop(path):
    """Signal and per-pulse geometry of one crop CPHD. Positions are returned relative to the crop centre (SRP)."""
    with open(path, "rb") as f, skcphd.Reader(f) as r:
        xml = r.metadata.xmltree
        ch = xml.find("{*}Data/{*}Channel").findtext("{*}Identifier")
        sgn = int(xml.findtext("{*}Global/{*}SGN"))
        pv = r.read_pvps(ch)
        sig = r.read_signal(ch)
    if sig.dtype.names:
        x = np.empty(sig.shape, np.complex64)
        x.real, x.imag = sig["real"], sig["imag"]
    else:
        x = sig.astype(np.complex64)
    srp = np.asarray(pv["SRPPos"][0], np.float64)
    return dict(X=x, tx=(np.asarray(pv["TxPos"], np.float64) - srp).astype(np.float32),
                rc=(np.asarray(pv["RcvPos"], np.float64) - srp).astype(np.float32),
                f0=np.asarray(pv["SC0"], np.float64), scss=float(pv["SCSS"][0]), sgn=sgn, srp=srp)


class CropSet(torch.utils.data.Dataset):
    def __init__(self, root, entries, anns, cfg, train):
        self.root, self.entries, self.anns, self.cfg, self.train = root, entries, anns, cfg, train
        self.meta = []
        for e in entries:
            with open(os.path.join(root, e["json"])) as f:
                self.meta.append(json.load(f))
        self.positive = np.array([m["label"] == "positive" for m in self.meta])

    def __len__(self):
        return len(self.entries)

    def scene_anns(self, i):
        m = self.meta[i]
        if m["scene"] in self.anns:
            return self.anns[m["scene"]]
        return [dict(lon_min=b["bbox_geo"][0], lat_min=b["bbox_geo"][1], lon_max=b["bbox_geo"][2], lat_max=b["bbox_geo"][3],
                     iscrowd=0) for b in m["boxes"]]           # fallback: boxes stored with the crop (core only)

    def grid(self, i, shift=True):
        m, cfg = self.meta[i], self.cfg
        lat, lon = m["centre_lat"], m["centre_lon"]
        s = cfg["shift_m"] if (self.train and shift and m["scene"] in self.anns) else 0.0
        if s > 0:
            ml, mo = m_per_deg(lat)
            lat += random.uniform(-s, s) / ml
            lon += random.uniform(-s, s) / mo
        return make_grid(lat, lon, cfg["image_px"], cfg["pixel_m"])

    def __getitem__(self, i):
        m = self.meta[i]
        d = read_crop(os.path.join(self.root, self.entries[i]["cphd"]))
        g = self.grid(i)
        boxes, ign = boxes_on_grid(self.scene_anns(i), g, self.cfg["min_visible"])
        return dict(X=torch.from_numpy(d["X"]), tx=torch.from_numpy(d["tx"]), rc=torch.from_numpy(d["rc"]),
                    f0=torch.from_numpy(d["f0"]), scss=torch.tensor(d["scss"], dtype=torch.float64),
                    sgn=torch.tensor(d["sgn"]), pix=torch.from_numpy(grid_pixels_ecef(g, m["h_ref"], d["srp"])),
                    boxes=torch.from_numpy(boxes), ignore=torch.from_numpy(ign), grid=g, idx=i,
                    crop_id=m["crop_id"], scene=m["scene"])


def collate(items):
    out = {k: torch.stack([it[k] for it in items]) for k in ("X", "tx", "rc", "f0", "scss", "sgn", "pix")}
    for k in ("boxes", "ignore", "grid", "idx", "crop_id", "scene"):
        out[k] = [it[k] for it in items]
    return out


class BalancedSampler(torch.utils.data.Sampler):
    """Each draw is a positive crop with probability pos_frac, else a negative one. Different stream per rank and epoch."""
    def __init__(self, positive, pos_frac, num_samples, seed=0, rank=0, world=1):
        self.pos, self.neg = np.nonzero(positive)[0], np.nonzero(~positive)[0]
        self.pos_frac = pos_frac if len(self.pos) and len(self.neg) else float(len(self.pos) > 0)
        self.n, self.seed, self.rank, self.epoch = num_samples // world, seed, rank, 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000 * self.epoch + self.rank)
        take_pos = rng.random(self.n) < self.pos_frac
        idx = np.where(take_pos, rng.choice(self.pos, self.n) if len(self.pos) else 0,
                       rng.choice(self.neg, self.n) if len(self.neg) else 0)
        return iter(idx.tolist())

    def __len__(self):
        return self.n


# ------------------------------------------------------------------ differentiable imaging
class Imager(nn.Module):
    """Backprojection of a crop's phase history onto a lat/lon pixel grid, written in torch so gradients flow back to
    the signal and to a learnable taper over pulses and samples.

    FX-domain model: sample n of pulse p is at f = SC0_p + n*SCSS and a scatterer adds exp(SGN*j*2*pi*f*dtau), dtau being
    its two-way delay minus that of the crop centre. Per pulse: inverse FFT gives a range profile; each pixel reads it at
    its own dtau (linear interpolation), the carrier phase is removed, and pulses are summed. Delays use
    R(p) - R(0) = (|p|^2 - 2 t.p) / (|t - p| + |t|) with t the antenna position relative to the crop centre, which keeps
    float32 accurate to about 1e-5 m."""
    def __init__(self, pulses, samples, n_sub=2, oversample=2, chunk_elems=1.6e7, learn_taper=True, db_lo=-10.0, db_hi=40.0):
        super().__init__()
        self.P, self.S, self.n_sub, self.os = pulses, samples, n_sub, oversample
        self.chunk_elems, self.db_lo, self.db_hi = chunk_elems, db_lo, db_hi
        self.w_p = nn.Parameter(torch.ones(pulses), requires_grad=learn_taper)
        self.w_s = nn.Parameter(torch.ones(samples), requires_grad=learn_taper)

    @staticmethod
    def _delta(t, p, pp):
        tp = torch.einsum("bkd,bmd->bkm", t, p)
        tt = (t * t).sum(-1)[:, :, None]
        num = pp[:, None, :] - 2 * tp
        return num / (torch.sqrt(torch.clamp(tt + num, min=0)) + torch.sqrt(tt))

    def _chunk(self, pr, pi, tx, rc, f0, kscale, pix, pp):
        """Sum over one chunk of pulses. pr/pi: (B, k, K) range profiles; returns (B, M) real and imaginary parts."""
        K = pr.shape[-1]
        dtau = (self._delta(tx, pix, pp) + self._delta(rc, pix, pp)) / C          # (B, k, M) seconds
        kk = dtau * kscale[:, None, None] + K // 2
        k0 = torch.floor(kk)
        fr = kk - k0
        ok = (k0 >= 0) & (k0 < K - 1)
        ki = k0.clamp(0, K - 2).long()
        g = lambda a, idx: torch.gather(a, 2, idx)
        vr = g(pr, ki) * (1 - fr) + g(pr, ki + 1) * fr
        vi = g(pi, ki) * (1 - fr) + g(pi, ki + 1) * fr
        ph = 2 * math.pi * (f0[:, :, None] * dtau)
        c, s = torch.cos(ph), torch.sin(ph)
        okf = ok.to(vr.dtype)
        return ((vr * c - vi * s) * okf).sum(1), ((vr * s + vi * c) * okf).sum(1)

    def forward(self, X, tx, rc, f0, scss, sgn, pix):
        """X (B,P,S) complex64; tx, rc (B,P,3); f0 (B,P) float64; scss (B,); sgn (B,); pix (B,M,3).
        Returns complex sub-aperture images (B, n_sub, M) as (real, imag)."""
        B, P, S = X.shape
        X = X * (self.w_p[:, None] * self.w_s[None, :]).to(X.dtype)
        X = torch.where((sgn > 0)[:, None, None], X.conj(), X)
        K = S * self.os
        prof = torch.fft.fftshift(torch.fft.ifft(X, n=K, dim=-1), dim=-1)          # (B, P, K)
        pr, pi = prof.real.contiguous(), prof.imag.contiguous()
        kscale = (K * scss).to(torch.float32)
        f0 = f0.to(torch.float32)
        pp = (pix * pix).sum(-1)
        M = pix.shape[1]
        per = P // self.n_sub
        k = max(1, min(per, int(self.chunk_elems // (B * M))))
        while per % k:
            k -= 1
        re = [torch.zeros(B, M, device=X.device) for _ in range(self.n_sub)]
        im = [torch.zeros(B, M, device=X.device) for _ in range(self.n_sub)]
        for a in range(0, P, k):
            j = a // per
            args = (pr[:, a:a + k], pi[:, a:a + k], tx[:, a:a + k], rc[:, a:a + k], f0[:, a:a + k], kscale, pix, pp)
            if torch.is_grad_enabled() and (self.w_p.requires_grad or X.requires_grad):
                r_, i_ = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                r_, i_ = self._chunk(*args)
            re[j] = re[j] + r_
            im[j] = im[j] + i_
        return torch.stack(re, 1) / P, torch.stack(im, 1) / P

    def features(self, re, im, n):
        """3 channels: full-aperture dB, then each sub-aperture dB, relative to the median of the full aperture."""
        fr, fi = re.sum(1, keepdim=True), im.sum(1, keepdim=True)
        pw = torch.cat([fr * fr + fi * fi, re * re + im * im], 1)                    # (B, 1 + n_sub, M)
        ref = pw[:, :1].detach().median(dim=-1, keepdim=True).values.clamp(min=1e-20)
        db = 10 * torch.log10(pw / ref + 1e-6)
        x = (db - self.db_lo) / (self.db_hi - self.db_lo) * 2 - 1
        x = torch.clamp(x, -1.5, 1.5)
        return x.reshape(x.shape[0], x.shape[1], n, n)


# ------------------------------------------------------------------ detector
def build_detector(cfg):
    import torchvision
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops import boxes as box_ops
    from torchvision.ops.feature_pyramid_network import LastLevelP6P7

    class GeoRetinaNet(RetinaNet):
        """RetinaNet whose anchors matched to an ignore box count neither as ship nor as background."""
        def compute_loss(self, targets, head_outputs, anchors):
            matched = []
            for a, t in zip(anchors, targets):
                if t["boxes"].numel() == 0:
                    matched.append(torch.full((a.size(0),), -1, dtype=torch.int64, device=a.device))
                    continue
                m = self.proposal_matcher(box_ops.box_iou(t["boxes"], a))
                ig = t.get("ignore")
                if ig is not None and bool(ig.any()):
                    hit = m >= 0
                    m[hit & ig[m.clamp(min=0)]] = -2
                matched.append(m)
            return self.head.compute_loss(targets, head_outputs, anchors, matched)

    weights = "DEFAULT" if cfg["pretrained"] else None
    backbone = resnet_fpn_backbone(backbone_name=cfg["backbone"], weights=weights, trainable_layers=5,
                                   returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    base = [16, 32, 64, 128, 256]
    sizes = tuple(tuple(int(round(b * 2 ** (i / 3))) for i in range(3)) for b in base)
    ratios = ((0.2, 0.5, 1.0, 2.0, 5.0),) * len(base)                               # ships are long and thin
    n = cfg["image_px"]
    model = GeoRetinaNet(backbone, num_classes=2, anchor_generator=AnchorGenerator(sizes, ratios),
                         min_size=n, max_size=n, image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0],
                         score_thresh=0.05, nms_thresh=0.4, detections_per_img=100, topk_candidates=1000)
    model.head.regression_head._loss_type = "giou"
    return model


class CphdShipDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.imager = Imager(cfg["pulses"], cfg["samples"], n_sub=2, oversample=cfg["oversample"],
                             chunk_elems=cfg["chunk_elems"], learn_taper=cfg["learn_taper"])
        self.det = build_detector(cfg)

    def features(self, b):
        re, im = self.imager(b["X"], b["tx"], b["rc"], b["f0"], b["scss"], b["sgn"], b["pix"])
        return self.imager.features(re, im, self.cfg["image_px"])

    def forward(self, b, targets=None):
        x = self.features(b)
        if self.cfg["amp"] and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.det(list(x), targets)
        return self.det(list(x), targets)


def to_device(b, device):
    out = dict(b)
    for k in ("X", "tx", "rc", "f0", "scss", "sgn", "pix"):
        out[k] = b[k].to(device, non_blocking=True)
    return out


def targets_of(b, device):
    return [dict(boxes=bx.to(device), labels=torch.ones(len(bx), dtype=torch.int64, device=device), ignore=ig.to(device))
            for bx, ig in zip(b["boxes"], b["ignore"])]


# ------------------------------------------------------------------ evaluation
def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter)


def match(rec, thr, min_score=0.0):
    """Greedy matching by score. Returns (scores, is_tp) for counted detections, n_gt, centre errors of TPs (px)."""
    sc, db, gb, ig = rec["scores"], rec["boxes"], rec["gt"], rec["ignore"]
    keep = sc >= min_score
    sc, db = sc[keep], db[keep]
    iou = _iou(db, gb)
    used = np.zeros(len(gb), bool)
    out_s, out_t, err = [], [], []
    for d in np.argsort(-sc):
        cand = np.nonzero(iou[d] >= thr)[0] if len(gb) else np.array([], int)
        real = [j for j in cand if not ig[j] and not used[j]]
        if real:
            j = max(real, key=lambda j: iou[d, j])
            used[j] = True
            out_s.append(sc[d]); out_t.append(1)
            cd = 0.5 * (db[d, :2] + db[d, 2:]) - 0.5 * (gb[j, :2] + gb[j, 2:])
            err.append(float(np.hypot(*cd)))
        elif any(ig[j] for j in cand):
            continue                                                          # on an ignored box: not counted
        else:
            out_s.append(sc[d]); out_t.append(0)
    return np.array(out_s), np.array(out_t), int((~ig).sum()) if len(ig) else 0, err


def average_precision(records, thr):
    s, t, n_gt = [], [], 0
    for r in records:
        a, b, n, _ = match(r, thr)
        s.append(a); t.append(b); n_gt += n
    s, t = np.concatenate(s) if s else np.zeros(0), np.concatenate(t) if t else np.zeros(0)
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-s)
    tp = np.cumsum(t[order]); fp = np.cumsum(1 - t[order])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@torch.no_grad()
def evaluate(model, loader, device, cfg, score_thr=0.5):
    core = model.module if hasattr(model, "module") else model
    core.eval()
    records = []
    for b in loader:
        b = to_device(b, device)
        out = core(b)
        for k, o in enumerate(out):
            records.append(dict(scores=o["scores"].float().cpu().numpy(), boxes=o["boxes"].float().cpu().numpy(),
                                gt=b["boxes"][k].numpy(), ignore=b["ignore"][k].numpy(), grid=b["grid"][k],
                                crop_id=b["crop_id"][k], scene=b["scene"][k]))
    core.train()
    tp = fp = n_gt = 0
    errs, neg_hit, n_neg = [], 0, 0
    for r in records:
        s, t, n, e = match(r, 0.3, score_thr)
        tp += int(t.sum()); fp += int(len(t) - t.sum()); n_gt += n; errs += e
        if n == 0 and not r["ignore"].any():
            n_neg += 1
            neg_hit += int((r["scores"] >= score_thr).any())
    area_km2 = len(records) * (cfg["image_px"] * cfg["pixel_m"] / 1000) ** 2
    m = dict(crops=len(records), gt=n_gt, ap30=average_precision(records, 0.3), ap50=average_precision(records, 0.5),
             precision=tp / max(tp + fp, 1), recall=tp / max(n_gt, 1), false_alarms_per_km2=fp / max(area_km2, 1e-9),
             centre_error_m=float(np.mean(errs)) * cfg["pixel_m"] if errs else float("nan"),
             negative_crops_with_detection=neg_hit / max(n_neg, 1), score_thr=score_thr)
    return m, records


def detections_geojson(records, path, score_thr=0.3, nms_iou=0.3):
    """Detections from all crops as lat/lon boxes, merged across overlapping crops per scene (NMS in metres)."""
    import torchvision
    by_scene = {}
    for r in records:
        for s, bx in zip(r["scores"], r["boxes"]):
            if s >= score_thr:
                by_scene.setdefault(r["scene"], []).append((float(s), px_to_geo(bx, r["grid"]), r["crop_id"]))
    feats = []
    for scene, dets in by_scene.items():
        lat0 = np.mean([0.5 * (d[1][1] + d[1][3]) for d in dets]); lon0 = np.mean([0.5 * (d[1][0] + d[1][2]) for d in dets])
        ml, mo = m_per_deg(lat0)
        bm = torch.tensor([[(g[0] - lon0) * mo, (lat0 - g[3]) * ml, (g[2] - lon0) * mo, (lat0 - g[1]) * ml] for _, g, _ in dets],
                          dtype=torch.float32)
        keep = torchvision.ops.nms(bm, torch.tensor([d[0] for d in dets], dtype=torch.float32), nms_iou).tolist()
        for i in keep:
            s, (w, so, e, n), cid = dets[i]
            feats.append(dict(type="Feature", properties=dict(scene=scene, score=round(s, 4), crop_id=cid,
                                                              bbox_geo=[w, so, e, n]),
                              geometry=dict(type="Polygon", coordinates=[[[w, so], [e, so], [e, n], [w, n], [w, so]]])))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(type="FeatureCollection", features=feats), f)
    return len(feats)


# ------------------------------------------------------------------ training
def setup_ddp():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return rank, world, device


def read_manifest(root):
    with open(os.path.join(root, "manifest.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_scenes(entries, val_scenes, val_frac, seed):
    scenes = sorted({e["scene"] for e in entries})
    if val_scenes:
        val = {s for s in scenes if any(v in s for v in val_scenes)}
    elif len(scenes) >= 2:
        rng = random.Random(seed)
        order = scenes[:]
        rng.shuffle(order)
        val = set(order[:max(1, int(round(val_frac * len(scenes))))])
    else:
        val = set()
    tr = [e for e in entries if e["scene"] not in val]
    va = [e for e in entries if e["scene"] in val]
    if not va:                                              # one scene only: hold out every 5th crop (overlaps leak)
        va, tr = entries[::5], [e for i, e in enumerate(entries) if i % 5]
    return tr, va, sorted(val)


def train(a):
    rank, world, device = setup_ddp()
    main = rank == 0
    say = (lambda *x: print(*x, flush=True)) if main else (lambda *x: None)
    torch.manual_seed(a.seed + rank); random.seed(a.seed + rank); np.random.seed(a.seed + rank)
    cfg = cfg_of(a)
    entries = read_manifest(a.crops)
    anns = load_scene_annotations(a.ann_root)
    tr_e, va_e, val_scenes = split_scenes(entries, a.val_scenes, a.val_frac, a.seed)
    if a.overfit:                                           # sanity check: validate on the training crops
        tr_e = va_e = entries
        val_scenes = ["(training crops)"]
    tr = CropSet(a.crops, tr_e, anns, cfg, train=True)
    va = CropSet(a.crops, va_e, anns, cfg, train=False)
    missing = sorted({m["scene"] for m in tr.meta + va.meta} - set(anns))
    say(f"crops: {len(tr)} train ({int(tr.positive.sum())} positive), {len(va)} val ({int(va.positive.sum())} positive); "
        f"val scenes {val_scenes or 'none (crop split)'}; {world} process(es) on {device}")
    if missing:
        say(f"warning: no annotations.json under --ann-root for {len(missing)} scenes; using the boxes stored with "
            f"each crop and no grid shift for those")
    os.makedirs(a.out, exist_ok=True)
    if main:
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump(dict(cfg, args=vars(a), val_scenes=val_scenes), f, indent=1, default=str)
    sampler = BalancedSampler(tr.positive, a.pos_frac, a.samples_per_epoch or len(tr), a.seed, rank, world)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, sampler=sampler, num_workers=a.workers, collate_fn=collate,
                                     pin_memory=device.type == "cuda", persistent_workers=a.workers > 0, drop_last=True)
    vdl = torch.utils.data.DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
    model = CphdShipDetector(cfg).to(device)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location="cpu", weights_only=False)["model"])
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    core = model.module if world > 1 else model
    opt = torch.optim.AdamW([dict(params=core.det.parameters(), lr=a.lr),
                             dict(params=[p for p in core.imager.parameters() if p.requires_grad], lr=a.lr_taper,
                                  weight_decay=0.0)], weight_decay=1e-4)
    steps = a.max_iters or a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    best, step, hist = -1.0, 0, []
    log_f = open(os.path.join(a.out, "metrics.jsonl"), "a") if main else None
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        t0, seen = time.time(), 0
        for b in dl:
            b = to_device(b, device)
            losses = model(b, targets_of(b, device))
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); sched.step()
            step += 1; seen += len(b["X"]) * world
            hist.append(loss.item())
            if main and (step % a.log_every == 0 or step == 1):
                rec = dict(step=step, epoch=epoch, loss=loss.item(), **{k: v.item() for k, v in losses.items()},
                           lr=sched.get_last_lr()[0], crops_per_s=seen / (time.time() - t0),
                           taper_grad=float(core.imager.w_p.grad.abs().mean()) if core.imager.w_p.grad is not None else 0.0)
                say(f"  step {step}: loss {rec['loss']:.4f} (cls {rec['classification']:.4f}, box {rec['bbox_regression']:.4f}), "
                    f"{rec['crops_per_s']:.2f} crops/s")
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if a.max_iters and step >= a.max_iters:
                break
        if world > 1:
            dist.barrier()
        if main and ((epoch + 1) % a.eval_every == 0 or epoch == a.epochs - 1 or (a.max_iters and step >= a.max_iters)):
            m, records = evaluate(model, vdl, device, cfg)
            m.update(epoch=epoch, step=step)
            say(f"epoch {epoch}: AP30 {m['ap30']:.3f}, AP50 {m['ap50']:.3f}, precision {m['precision']:.3f}, recall {m['recall']:.3f}, "
                f"{m['false_alarms_per_km2']:.2f} false alarms per km2, centre error {m['centre_error_m']:.1f} m")
            log_f.write(json.dumps(dict(eval=m)) + "\n"); log_f.flush()
            state = dict(model=core.state_dict(), cfg=cfg, epoch=epoch, step=step, metrics=m)
            torch.save(state, os.path.join(a.out, "last.pt"))
            score = m["ap30"] if m["ap30"] == m["ap30"] else -m["false_alarms_per_km2"]
            if score > best:
                best = score
                torch.save(state, os.path.join(a.out, "best.pt"))
                detections_geojson(records, os.path.join(a.out, "val_detections.geojson"))
        if world > 1:
            dist.barrier()
        if a.max_iters and step >= a.max_iters:
            break
    if world > 1:
        dist.destroy_process_group()
    return hist


@torch.no_grad()
def predict(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    model = CphdShipDetector(cfg).to(device)
    model.load_state_dict(st["model"])
    entries = read_manifest(a.crops)
    if a.scenes:
        entries = [e for e in entries if any(s in e["scene"] for s in a.scenes)]
    ds = CropSet(a.crops, entries, load_scene_annotations(a.ann_root), cfg, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, collate_fn=collate)
    m, records = evaluate(model, dl, device, cfg, a.score_thr)
    n = detections_geojson(records, a.out, a.score_thr)
    print(f"{len(ds)} crops, {n} detections after merging overlaps -> {a.out}")
    if ds.anns:
        print(json.dumps(m, indent=1))
    return m


def cfg_of(a):
    return dict(image_px=a.image_px, pixel_m=a.pixel_m, shift_m=a.shift_m, min_visible=a.min_visible, pulses=a.pulses,
                samples=a.samples, oversample=a.oversample, chunk_elems=a.chunk_elems, learn_taper=not a.fixed_taper,
                backbone=a.backbone, pretrained=not a.no_pretrained, amp=a.amp)


# ------------------------------------------------------------------ self-test
def selftest(work):
    """Synthetic scenes with line-shaped ships -> crops -> imager check -> short training -> evaluation -> GeoJSON."""
    import shutil
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.join(here, "..", "cphd-crops")):
        if os.path.isfile(os.path.join(p, "build_crops.py")):
            sys.path.insert(0, p)
    import build_crops as bc
    shutil.rmtree(work, ignore_errors=True)
    ann_root, cphd_root, crops = os.path.join(work, "ann"), os.path.join(work, "cphd"), os.path.join(work, "crops")
    ok = True
    t0 = time.time()
    ships = [((300.0, 200.0), 0), ((-250.0, -150.0), 90), ((50.0, 350.0), 0), ((-50.0, -350.0), 0)]   # (east, north), heading
    for scene in ("2023-06-20-12-06-09_UMBRA-05", "2023-09-17-15-40-17_UMBRA-06"):
        pts = []
        for (e, n), hd in ships:
            for s in np.linspace(-30, 30, 9):                       # 60 m ship as 9 scatterers
                pts.append((e + s, n) if hd == 0 else (e, n + s))
        truth, centre = bc.make_synthetic(os.path.join(cphd_root, scene, f"{scene}_CPHD.cphd"), pts)
        lat0, lon0 = centre
        ml, mo = m_per_deg(lat0)
        gb = [lon0 - 700 / mo, lat0 - 700 / ml, lon0 + 700 / mo, lat0 + 700 / ml]
        anns = []
        for i, ((e, n), hd) in enumerate(ships):
            k = i * 9 + 4
            la, lo = truth[k]["lat"], truth[k]["lon"]
            hx, hy = (34, 8) if hd == 0 else (8, 34)
            anns.append(dict(bbox=[0, 0, 1, 1], area=1, bbox_geo=[lo - hx / mo, la - hy / ml, lo + hx / mo, la + hy / ml],
                             category_id=1, id=i + 1, iscrowd=0))
        os.makedirs(os.path.join(ann_root, f"{scene}_GEC"), exist_ok=True)
        with open(os.path.join(ann_root, f"{scene}_GEC", "annotations.json"), "w") as f:
            json.dump(dict(scene_name=f"{scene}_GEC", image=dict(geo_bounds=gb, crs="EPSG:4326"), annotations=anns), f)
    print(f"synthetic scenes: {time.time() - t0:.0f} s")
    args = argparse.Namespace(ann_root=ann_root, cphd_root=cphd_root, out=crops, dry_run=False, overwrite=False,
                              scenes=None, limit=0)
    bc.run(args, dict(bc.DEFAULTS, workers=2, neg_ratio=1.0), log=lambda *x: None)
    man = read_manifest(crops)
    print(f"crops built: {len(man)} ({sum(m['label'] == 'positive' for m in man)} positive), {time.time() - t0:.0f} s")

    # 1. the torch imager against the numpy backprojection of build_crops
    e = next(m for m in man if m["kind"] == "ship_centred")
    with open(os.path.join(crops, e["json"])) as f:
        meta = json.load(f)
    d = read_crop(os.path.join(crops, e["cphd"]))
    g = make_grid(meta["centre_lat"], meta["centre_lon"], 48, 2.0)
    pix = grid_pixels_ecef(g, meta["h_ref"], d["srp"])
    im = Imager(1024, 1024, n_sub=2, oversample=4, learn_taper=False)
    re_, im_ = im(torch.from_numpy(d["X"])[None], torch.from_numpy(d["tx"])[None], torch.from_numpy(d["rc"])[None],
                  torch.from_numpy(d["f0"])[None], torch.tensor([d["scss"]], dtype=torch.float64), torch.tensor([d["sgn"]]),
                  torch.from_numpy(pix)[None])
    a_t = (re_.sum(1) + 1j * im_.sum(1))[0].numpy()
    c = bc.CPHD(os.path.join(crops, e["cphd"]))
    a_n = bc.backproject(c, pix.astype(np.float64) + d["srp"])
    corr = float(np.abs(np.vdot(a_t, a_n)) / (np.linalg.norm(a_t) * np.linalg.norm(a_n)))
    good = corr > 0.99
    ok &= good
    print(f"torch imager vs numpy backprojection on a crop: correlation {corr:.4f} | {'ok' if good else 'FAIL'}")

    # 2. short training run: gradients reach the taper, the loss falls, evaluation and GeoJSON are written
    out = os.path.join(work, "run")
    p = parser()
    a = p.parse_args(["train", "--crops", crops, "--ann-root", ann_root, "--out", out, "--image-px", "128", "--pixel-m", "4",
                      "--backbone", "resnet18", "--no-pretrained", "--batch", "2", "--epochs", "1", "--max-iters", "20",
                      "--samples-per-epoch", "40", "--lr", "1e-3", "--workers", "2", "--log-every", "5",
                      "--chunk-elems", "4e6", "--warmup", "5", "--pos-frac", "0.8"])
    hist = train(a)
    with open(os.path.join(out, "metrics.jsonl")) as f:
        recs = [json.loads(l) for l in f]
    tg = [r["taper_grad"] for r in recs if "taper_grad" in r]
    good = bool(np.isfinite(hist).all()) and max(tg) > 0
    ok &= good
    print(f"training: {len(hist)} steps, losses finite ({hist[0]:.3f} -> {hist[-1]:.3f}), gradient reaches the imager taper "
          f"({max(tg):.2e}) | {'ok' if good else 'FAIL'}")
    ev = [r["eval"] for r in recs if "eval" in r]
    good = bool(ev) and os.path.exists(os.path.join(out, "best.pt"))
    ok &= good
    print(f"evaluation written: {ev[-1] if ev else None} | {'ok' if good else 'FAIL'}")
    pa = p.parse_args(["predict", "--ckpt", os.path.join(out, "best.pt"), "--crops", crops, "--ann-root", ann_root,
                       "--out", os.path.join(out, "pred.geojson"), "--score-thr", "0.05", "--workers", "0"])
    predict(pa)
    with open(os.path.join(out, "pred.geojson")) as f:
        gj = json.load(f)
    good = gj["type"] == "FeatureCollection"
    ok &= good
    print(f"predict: {len(gj['features'])} merged detections in GeoJSON | {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def parser():
    ap = argparse.ArgumentParser(description="CPHD ship detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--crops", required=True, help="output folder of build_crops.py (with manifest.jsonl)")
    t.add_argument("--ann-root", default="/data3/sandbox/jhu/satlock/annotation_tool/test_aligned")
    t.add_argument("--out", required=True)
    t.add_argument("--val-scenes", nargs="*", help="scenes held out for validation (default: a fraction of the scenes)")
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--max-iters", type=int, default=0)
    t.add_argument("--samples-per-epoch", type=int, default=0, help="default: number of training crops")
    t.add_argument("--batch", type=int, default=4, help="crops per GPU")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lr-taper", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--pos-frac", type=float, default=0.5, help="share of positive crops per batch")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--eval-every", type=int, default=1)
    t.add_argument("--resume", default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--overfit", action="store_true", help="train and validate on the same crops (pipeline sanity check)")
    for s in (t,):
        s.add_argument("--image-px", type=int, default=512)
        s.add_argument("--pixel-m", type=float, default=1.0)
        s.add_argument("--shift-m", type=float, default=64.0, help="random grid shift for training, metres")
        s.add_argument("--min-visible", type=float, default=0.3)
        s.add_argument("--pulses", type=int, default=1024)
        s.add_argument("--samples", type=int, default=1024)
        s.add_argument("--oversample", type=int, default=2)
        s.add_argument("--chunk-elems", type=float, default=1.6e7, help="pulse x pixel elements per imaging step (GPU memory)")
        s.add_argument("--fixed-taper", action="store_true", help="do not learn the range and pulse taper")
        s.add_argument("--backbone", default="resnet50")
        s.add_argument("--no-pretrained", action="store_true")
        s.add_argument("--amp", action="store_true", help="bfloat16 autocast for the detector")
    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--crops", required=True)
    p.add_argument("--ann-root", default=None, help="if given, metrics are printed too")
    p.add_argument("--out", required=True, help="GeoJSON of detections")
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("selftest")
    s.add_argument("--dir", default="./cphd_detector_selftest")
    return ap


def main():
    a = parser().parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "predict":
        predict(a)
    else:
        sys.exit(0 if selftest(a.dir) else 1)


if __name__ == "__main__":
    main()
