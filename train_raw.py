#!/usr/bin/env python3
"""Ship detector that reads the CPHD phase history directly (no image formation), geo boxes out.

    CPHD crop, 1024 pulses x 1024 frequency samples, complex
      -> 4 channels: real, imag (scaled by the crop RMS), and the ground wavenumbers kE, kN of every sample
         (where each sample sits in k-space on the ground; this ties the array to ground geometry for any scene)
      -> ViT encoder: 32 x 32 patches = 1024 tokens (each patch is a short sub-aperture x sub-band), plus a token-wide
         embedding of the look direction
      -> decoder with one fixed query per ground cell (default 128 x 128 cells of 4 m = the 512 m crop); each query
         carries its cell's east/north position and cross-attends to the tokens
      -> small conv head on the cell grid: heat (ship centre here), offset (centre inside the cell), log size (m)
      -> peaks of the heat map -> boxes in crop-local metres -> lat/lon (exact, from the crop centre)

Losses (CenterNet style, no matching step; every term is differentiable):
    heat   penalty-reduced focal loss against a Gaussian around each ship centre; cells of cut ships get zero weight
    offset L1 at ship-centre cells (cells)
    size   L1 on log width / log height (m) at ship-centre cells, weight 0.1
    giou   generalised IoU of the decoded box at ship-centre cells, weight 1 (set --giou-weight 0 to drop)

Augmentation: the crop centre is moved by up to --shift-m metres by an exact phase ramp on the data (re-referencing),
so labels and data stay consistent. Skew: focal loss plus the same balanced sampler as train_detector.py.

    python train_raw.py train --crops /crops --ann-root /ann --out runs/raw1
    torchrun --nproc_per_node=4 train_raw.py train ...
    python train_raw.py predict --ckpt runs/raw1/best.pt --crops /crops --out preds.geojson
    python train_raw.py selftest
"""
import argparse, glob, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_detector as td                                           # data reading, sampler, metrics, GeoJSON
from fourier_net import FourierShipNet
import train_viz as tv                                                # dataset stats, TensorBoard, collages

C = 299792458.0


# ------------------------------------------------------------------ targets
def gaussian_radius(h, w, min_overlap=0.7):
    """CenterNet radius (cells) so a box shifted by it still overlaps the truth by min_overlap."""
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return max(0.0, min(r1, r2, r3))


def build_targets(boxes, ignore, cells, cell_m, max_objs):
    """boxes (K,4) crop-local metres [e0, n0, e1, n1] (already clipped). Row 0 is the north edge."""
    H = cells * cell_m / 2
    heat = np.zeros((cells, cells), np.float32)
    ign = np.zeros((cells, cells), np.float32)
    idx = np.zeros(max_objs, np.int64); off = np.zeros((max_objs, 2), np.float32)
    size = np.zeros((max_objs, 2), np.float32); box = np.zeros((max_objs, 4), np.float32)
    mask = np.zeros(max_objs, np.float32)
    ys, xs = np.mgrid[0:cells, 0:cells]
    k = 0
    for b, ig in zip(boxes, ignore):
        e0, n0, e1, n1 = b
        c0, c1 = int(np.floor((e0 + H) / cell_m)), int(np.ceil((e1 + H) / cell_m))
        r0, r1 = int(np.floor((H - n1) / cell_m)), int(np.ceil((H - n0) / cell_m))
        if ig:
            ign[max(r0, 0):min(r1, cells), max(c0, 0):min(c1, cells)] = 1
            continue
        u, v = (0.5 * (e0 + e1) + H) / cell_m, (H - 0.5 * (n0 + n1)) / cell_m
        iu, iv = min(int(u), cells - 1), min(int(v), cells - 1)
        w_c, h_c = (e1 - e0) / cell_m, (n1 - n0) / cell_m
        sig = (2 * max(0, int(gaussian_radius(h_c, w_c))) + 1) / 6
        heat = np.maximum(heat, np.exp(-((xs - iu) ** 2 + (ys - iv) ** 2) / (2 * sig * sig)).astype(np.float32))
        heat[iv, iu] = 1.0
        if k < max_objs:
            idx[k] = iv * cells + iu
            off[k] = (u - iu, v - iv)
            size[k] = (math.log(max(e1 - e0, 0.5)), math.log(max(n1 - n0, 0.5)))
            box[k] = b
            mask[k] = 1
            k += 1
    ign[heat > 0] = 0                                                  # never silence a real ship's cells
    return dict(heat=heat, ign=ign, idx=idx, off=off, size=size, box=box, mask=mask)


# ------------------------------------------------------------------ data
class RawSet(torch.utils.data.Dataset):
    def __init__(self, root, entries, anns, cfg, train):
        self.root, self.entries, self.anns, self.cfg, self.train = root, entries, anns, cfg, train
        self.meta = []
        for e in entries:
            with open(os.path.join(root, e["json"])) as f:
                self.meta.append(json.load(f))
        self.positive = np.array([m["label"] == "positive" for m in self.meta], dtype=bool)

    def __len__(self):
        return len(self.entries)

    def scene_anns(self, i):
        m = self.meta[i]
        if m["scene"] in self.anns:
            return self.anns[m["scene"]], True
        return [dict(lon_min=b["bbox_geo"][0], lat_min=b["bbox_geo"][1], lon_max=b["bbox_geo"][2], lat_max=b["bbox_geo"][3],
                     iscrowd=0) for b in m["boxes"]], False

    def get(self, i, shift=None):
        m, cfg = self.meta[i], self.cfg
        d = td.read_crop(os.path.join(self.root, self.entries[i]["cphd"]))
        anns, full = self.scene_anns(i)
        lat, lon = m["centre_lat"], m["centre_lon"]
        E, N, U = (td_enu(lat, lon))
        X, tx, rc = d["X"], d["tx"].astype(np.float64), d["rc"].astype(np.float64)
        f = d["f0"][:, None] + np.arange(X.shape[1])[None, :] * d["scss"]
        if shift is None:
            s = cfg["shift_m"] if (self.train and full) else 0.0
            shift = (random.uniform(-s, s), random.uniform(-s, s)) if s > 0 else (0.0, 0.0)
        dx, dy = shift
        if dx or dy:                                                   # move the reference point: exact phase ramp
            D = dx * E + dy * N
            def dr(t):
                return (D @ D - 2 * t @ D) / (np.linalg.norm(t - D, axis=1) + np.linalg.norm(t, axis=1))
            dtau = (dr(tx) + dr(rc)) / C
            X = (X * np.exp((-d["sgn"] * 2j * np.pi) * (f * dtau[:, None]))).astype(np.complex64)
            tx, rc = tx - D, rc - D
            ml, mo = td.m_per_deg(lat)
            lat, lon = lat + dy / ml, lon + dx / mo
        if d["sgn"] > 0:                                               # one sign convention for the network
            X = np.conj(X)
        if self.train and cfg.get("aug"):
            X = augment(X, cfg)
        X = X / max(float(np.sqrt(np.mean(np.abs(X) ** 2))), 1e-12)
        u = 0.5 * (tx + rc)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        ml, mo = td.m_per_deg(lat)
        Hm = cfg["cells"] * cfg["cell_m"] / 2
        bx, ig = [], []
        for a in anns:
            e0, e1 = (a["lon_min"] - lon) * mo, (a["lon_max"] - lon) * mo
            n0, n1 = (a["lat_min"] - lat) * ml, (a["lat_max"] - lat) * ml
            ce0, ce1, cn0, cn1 = max(e0, -Hm), min(e1, Hm), max(n0, -Hm), min(n1, Hm)
            if ce1 - ce0 < 1 or cn1 - cn0 < 1:
                continue
            vis = (ce1 - ce0) * (cn1 - cn0) / max((e1 - e0) * (n1 - n0), 1e-9)
            bx.append([ce0, cn0, ce1, cn1])
            ig.append(bool(vis < cfg["min_visible"] or a.get("iscrowd", 0)))
        bx = np.array(bx, np.float32).reshape(-1, 4)
        ig = np.array(ig, bool)
        t = build_targets(bx, ig, cfg["cells"], cfg["cell_m"], cfg["max_objs"])
        return dict(X=torch.from_numpy(np.ascontiguousarray(X)), u=torch.from_numpy(u.astype(np.float32)),
                    f0=torch.from_numpy(d["f0"]), scss=torch.tensor(d["scss"], dtype=torch.float64),
                    E=torch.from_numpy(E.astype(np.float32)), N=torch.from_numpy(N.astype(np.float32)),
                    **{k: torch.from_numpy(v) for k, v in t.items()},
                    gt=bx, gt_ignore=ig, lat=lat, lon=lon, crop_id=m["crop_id"], scene=m["scene"],
                    tx=torch.from_numpy(tx.astype(np.float32)), rc=torch.from_numpy(rc.astype(np.float32)))

    def __getitem__(self, i):
        return self.get(i)


def augment(X, cfg, rng=random):
    """Signal-only augmentation (labels unchanged), each part with probability aug_p:
      sub-aperture  keep a random run of aug_min_frac..1 of the pulses (lower cross-range resolution); the run keeps at
                    least a quarter of itself in each half, so both half-aperture features stay valid
      sub-band      keep a random run of aug_min_frac..1 of the frequencies (lower range resolution)
      noise         complex white noise at a random SNR in [aug_snr_db, 30] dB (relative to the crop's power)
      phase         one random global phase (physically meaningless: the network must ignore it)"""
    P, S = X.shape
    pr, mf = cfg.get("aug_p", 0.5), cfg.get("aug_min_frac", 0.6)
    X = X.copy()
    if rng.random() < pr:                                              # sub-aperture
        w = int(round(P * rng.uniform(mf, 1.0)))
        lo, hi = max(0, int(math.ceil(P / 2 - 0.75 * w))), min(P - w, int(P / 2 - 0.25 * w))
        if w < P and hi >= lo:
            s0 = rng.randint(lo, hi)
            X[:s0] = 0
            X[s0 + w:] = 0
    if rng.random() < pr:                                              # sub-band
        w = int(round(S * rng.uniform(mf, 1.0)))
        if w < S:
            s0 = rng.randint(0, S - w)
            X[:, :s0] = 0
            X[:, s0 + w:] = 0
    if rng.random() < pr:                                              # noise
        snr = rng.uniform(cfg.get("aug_snr_db", 0.0), 30.0)
        pw = float(np.mean(np.abs(X[X != 0]) ** 2)) if np.any(X) else 1.0
        sd = math.sqrt(pw * 10 ** (-snr / 10) / 2)
        nz = (np.random.standard_normal(X.shape) + 1j * np.random.standard_normal(X.shape)) * sd
        X = X + (nz * (X != 0)).astype(np.complex64)                   # only where there is signal (keeps the windows)
    X = X * np.complex64(np.exp(1j * rng.uniform(0, 2 * math.pi)))     # phase
    return X.astype(np.complex64)


def td_enu(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return (np.array([-math.sin(lo), math.cos(lo), 0.0]),
            np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)]),
            np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]))


TENSORS = ("X", "u", "f0", "scss", "E", "N", "heat", "ign", "idx", "off", "size", "box", "mask")


def collate(items):
    out = {k: torch.stack([it[k] for it in items]) for k in TENSORS}
    for k in ("gt", "gt_ignore", "lat", "lon", "crop_id", "scene"):
        out[k] = [it[k] for it in items]
    return out


# ------------------------------------------------------------------ model
class CrossBlock(nn.Module):
    def __init__(self, d_mem, d, heads):
        super().__init__()
        self.h = heads
        self.ln_q, self.ln_m = nn.LayerNorm(d), nn.LayerNorm(d_mem)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d_mem, 2 * d), nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, mem):
        B, Q, D = q.shape
        k, v = self.kv(self.ln_m(mem)).chunk(2, -1)
        sh = lambda x: x.reshape(B, x.shape[1], self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(self.q(self.ln_q(q))), sh(k), sh(v))
        q = q + self.o(a.transpose(1, 2).reshape(B, Q, D))
        return q + self.mlp(self.ln2(q))


class RawShipNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D, dq, ps, cells = cfg["dim"], cfg["dec_dim"], cfg["patch"], cfg["cells"]
        T = (cfg["pulses"] // ps) * (cfg["samples"] // ps)
        self.patch = nn.Conv2d(4, D, ps, ps)
        self.pos = nn.Parameter(torch.zeros(1, T, D))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.geo = nn.Sequential(nn.Linear(3, D), nn.GELU(), nn.Linear(D, D))
        layer = nn.TransformerEncoderLayer(D, cfg["heads"], 4 * D, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg["depth"], enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(D)
        H = cells * cfg["cell_m"] / 2
        c = (torch.arange(cells) + 0.5) * cfg["cell_m"]
        n, e = torch.meshgrid(H - c, c - H, indexing="ij")             # row 0 = north edge
        fr = 2.0 ** torch.arange(8) * math.pi / (2 * H)
        pe = torch.cat([torch.sin(e.reshape(-1, 1) * fr), torch.cos(e.reshape(-1, 1) * fr),
                        torch.sin(n.reshape(-1, 1) * fr), torch.cos(n.reshape(-1, 1) * fr)], 1)
        self.register_buffer("cell_pe", pe, persistent=False)          # (cells^2, 32)
        self.qmlp = nn.Sequential(nn.Linear(pe.shape[1], dq), nn.GELU(), nn.Linear(dq, dq))
        self.qgeo = nn.Linear(D, dq)
        self.dec = nn.ModuleList(CrossBlock(D, dq, cfg["dec_heads"]) for _ in range(cfg["dec_layers"]))
        conv = lambda: nn.Sequential(nn.Conv2d(dq, dq, 3, padding=1), nn.GroupNorm(8, dq), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(conv(), conv())
        self.heat = nn.Conv2d(dq, 1, 1)
        self.off = nn.Conv2d(dq, 2, 1)
        self.size = nn.Conv2d(dq, 2, 1)
        nn.init.constant_(self.heat.bias, -2.19)
        nn.init.constant_(self.size.bias, math.log(40.0))

    def channels(self, b):
        """real, imag and the ground wavenumbers kE, kN of every sample (scaled to [-1, 1] per crop)."""
        X = b["X"]
        S = X.shape[-1]
        f = b["f0"][:, :, None] + torch.arange(S, device=X.device, dtype=torch.float64)[None, None, :] * b["scss"][:, None, None]
        u = b["u"].double()
        out = [X.real, X.imag]
        for ax in (b["E"], b["N"]):
            k = (2 / C) * f * (u * ax.double()[:, None, :]).sum(-1)[:, :, None]
            k = k - k.mean(dim=(1, 2), keepdim=True)
            out.append((k / k.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)).float())
        return torch.stack(out, 1)

    def forward(self, b):
        x = self.channels(b)
        B = x.shape[0]
        g = self.geo(b["u"].mean(1))                                   # look direction, includes grazing angle
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (self.cfg["amp"] and x.is_cuda) else _null()
        with ctx:
            t = self.patch(x).flatten(2).transpose(1, 2) + self.pos + g[:, None, :]
            mem = self.enc_norm(self.enc(t))
            q = self.qmlp(self.cell_pe)[None].expand(B, -1, -1) + self.qgeo(g)[:, None, :]
            for blk in self.dec:
                q = blk(q, mem)
            cells = self.cfg["cells"]
            y = self.trunk(q.transpose(1, 2).reshape(B, -1, cells, cells))
            return dict(heat=self.heat(y).float(), off=self.off(y).float(), size=self.size(y).float())


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------------ losses
def gather(t, idx):
    """t (B, C, H, W), idx (B, K) flat cell index -> (B, K, C)."""
    B, Cc = t.shape[:2]
    return t.flatten(2).gather(2, idx[:, None, :].expand(B, Cc, idx.shape[1])).transpose(1, 2)


def decode_at(off, size, idx, cells, cell_m):
    H = cells * cell_m / 2
    iu, iv = (idx % cells).float(), (idx // cells).float()
    ce = (iu + off[..., 0]) * cell_m - H
    cn = H - (iv + off[..., 1]) * cell_m
    w, h = torch.exp(size[..., 0]), torch.exp(size[..., 1])
    return torch.stack([ce - w / 2, cn - h / 2, ce + w / 2, cn + h / 2], -1)


def compute_loss(out, b, cfg):
    """CenterNet losses. With --tol-m R > 0 the position only has to be right to within R metres (labels that may be
    shifted by tens of metres, e.g. a height / alignment offset between the image and the GEC boxes):
      heat    the positive term scores the BEST cell within R of each ship centre (multiple-instance), and every cell
              within R of a centre is left out of the negative term, so firing anywhere near the ship costs nothing
      offset  epsilon-insensitive L1: errors smaller than R (in cells) cost nothing
      size    L1 on log width / height, unchanged (a shift does not change the size)
      giou    on the box SHAPE only: predicted size placed at the true centre, so position cannot hurt it
    With R = 0 these are the plain CenterNet losses."""
    from torchvision.ops import generalized_box_iou_loss
    p = torch.sigmoid(out["heat"][:, 0]).clamp(1e-4, 1 - 1e-4)
    t, ign = b["heat"], b["ign"]
    pos = (t >= 1.0).float()
    n_pos = pos.sum().clamp(min=1.0)
    kc = int(round(cfg.get("tol_m", 0.0) / cfg["cell_m"]))
    if kc > 0:
        win = 2 * kc + 1
        p_best = F.max_pool2d(p[:, None], win, 1, kc)[:, 0]                 # best score within R of each cell
        near = F.max_pool2d(pos[:, None], win, 1, kc)[:, 0]                 # cells within R of a ship centre
        l_pos = -(torch.log(p_best) * (1 - p_best) ** 2 * pos).sum()
        l_neg = -(torch.log(1 - p) * p ** 2 * (1 - t) ** 4 * (1 - near) * (1 - ign)).sum()
    else:
        l_pos = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
        l_neg = -(torch.log(1 - p) * p ** 2 * (1 - t) ** 4 * (1 - pos) * (1 - ign)).sum()
    heat = (l_pos + l_neg) / n_pos
    m = b["mask"][..., None]
    k = m.sum().clamp(min=1.0)
    off_p, size_p = gather(out["off"], b["idx"]), gather(out["size"], b["idx"])
    off_err = torch.abs(off_p - b["off"])
    if kc > 0:
        off_err = F.relu(off_err - kc)                                       # dead zone of R
    off = (off_err * m).sum() / k
    size = (torch.abs(size_p - b["size"]) * m).sum() / k
    losses = dict(heat=heat, offset=off, size=0.1 * size)
    if cfg["giou_weight"] > 0:
        bp = decode_at(b["off"] if kc > 0 else off_p, size_p, b["idx"], cfg["cells"], cfg["cell_m"])
        sel = b["mask"] > 0
        g = generalized_box_iou_loss(bp[sel], b["box"][sel], reduction="sum") / k if sel.any() else heat * 0
        losses["giou"] = cfg["giou_weight"] * g
    return losses


@torch.no_grad()
def decode(out, cfg, k=100, thr=0.05):
    """Peaks of the heat map -> boxes in crop-local metres [e0, n0, e1, n1] and scores, per crop."""
    heat = torch.sigmoid(out["heat"][:, 0])
    peak = heat == F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0]
    sc, idx = (heat * peak).flatten(1).topk(min(k, heat[0].numel()), dim=1)
    boxes = decode_at(gather(out["off"], idx).clamp(0, 1), gather(out["size"], idx), idx, cfg["cells"], cfg["cell_m"])
    res = []
    for s, bx in zip(sc, boxes):
        keep = s >= thr
        res.append((s[keep].cpu().numpy(), bx[keep].cpu().numpy()))
    return res


# ------------------------------------------------------------------ evaluation (same metrics as train_detector.py)
def to_records(dets, b, cfg):
    """Crop-local metres -> boxes on a 1 m lat/lon grid centred on the crop, so train_detector's metrics and GeoJSON apply."""
    Hm = cfg["cells"] * cfg["cell_m"] / 2
    n = int(round(2 * Hm))
    tr = lambda x: np.stack([x[:, 0] + Hm, Hm - x[:, 3], x[:, 2] + Hm, Hm - x[:, 1]], 1) if len(x) else np.zeros((0, 4))
    recs = []
    for (s, bx), gt, ig, lat, lon, cid, scene in zip(dets, b["gt"], b["gt_ignore"], b["lat"], b["lon"], b["crop_id"], b["scene"]):
        recs.append(dict(scores=s, boxes=tr(bx), gt=tr(gt), ignore=ig, grid=td.make_grid(lat, lon, n, 1.0),
                         crop_id=cid, scene=scene))
    return recs


def centre_match(records, tol_m, score_thr):
    """Position-tolerant counts: a detection is a hit if its centre is within tol_m of an unmatched ship centre
    (records are on a 1 m grid, so pixels = metres). Detections near ignored ships are not counted."""
    tp = fp = n_gt = 0
    for r in records:
        keep = r["scores"] >= score_thr
        sc, db, gb, ig = r["scores"][keep], r["boxes"][keep], r["gt"], r["ignore"]
        dc = 0.5 * (db[:, :2] + db[:, 2:]) if len(db) else np.zeros((0, 2))
        gc = 0.5 * (gb[:, :2] + gb[:, 2:]) if len(gb) else np.zeros((0, 2))
        used = np.zeros(len(gb), bool)
        n_gt += int((~ig).sum()) if len(ig) else 0
        for d in np.argsort(-sc):
            dist = np.hypot(*(gc - dc[d]).T) if len(gc) else np.zeros(0)
            cand = [j for j in np.argsort(dist) if dist[j] <= tol_m]
            real = [j for j in cand if not ig[j] and not used[j]]
            if real:
                used[real[0]] = True
                tp += 1
            elif not any(ig[j] for j in cand):
                fp += 1
    return tp, fp, n_gt


def summarise(records, cfg, score_thr=0.5):
    tp = fp = n_gt = 0
    errs, neg_hit, n_neg = [], 0, 0
    for r in records:
        s, t, n, e = td.match(r, 0.3, score_thr)
        tp += int(t.sum()); fp += int(len(t) - t.sum()); n_gt += n; errs += e
        if n == 0 and not r["ignore"].any():
            n_neg += 1
            neg_hit += int((r["scores"] >= score_thr).any())
    area = len(records) * (cfg["cells"] * cfg["cell_m"] / 1000) ** 2
    return dict(crops=len(records), gt=n_gt, ap30=td.average_precision(records, 0.3), ap50=td.average_precision(records, 0.5),
                precision=tp / max(tp + fp, 1), recall=tp / max(n_gt, 1), false_alarms_per_km2=fp / max(area, 1e-9),
                centre_error_m=float(np.mean(errs)) if errs else float("nan"),
                negative_crops_with_detection=neg_hit / max(n_neg, 1), score_thr=score_thr,
                **_centre_metrics(records, cfg, score_thr))


def _centre_metrics(records, cfg, score_thr):
    tol = float(cfg.get("eval_tol_m", 25.0))
    tp, fp, n_gt = centre_match(records, tol, score_thr)
    return {f"recall_within_{tol:.0f}m": tp / max(n_gt, 1), f"precision_within_{tol:.0f}m": tp / max(tp + fp, 1)}


def to_device(b, device):
    out = dict(b)
    for k in TENSORS:
        out[k] = b[k].to(device, non_blocking=True)
    return out


@torch.no_grad()
def evaluate(model, loader, device, cfg, score_thr=0.5, min_score=0.05, keep=None):
    core = model.module if hasattr(model, "module") else model
    core.eval()
    records = []
    for b in loader:
        b = to_device(b, device)
        out = core(b)
        if keep is not None:
            keep(out, b)
        records += to_records(decode(out, cfg, thr=min_score), b, cfg)
    core.train()
    return summarise(records, cfg, score_thr), records


# ------------------------------------------------------------------ training
def cfg_of(a):
    return dict(cells=a.cells, cell_m=a.cell_m, shift_m=a.shift_m, min_visible=a.min_visible, max_objs=a.max_objs,
                pulses=a.pulses, samples=a.samples, patch=a.patch, dim=a.dim, depth=a.depth, heads=a.heads,
                dec_dim=a.dec_dim, dec_heads=a.dec_heads, dec_layers=a.dec_layers, giou_weight=a.giou_weight, amp=a.amp, tol_m=a.tol_m,
                eval_tol_m=a.eval_tol_m, aug=a.aug, aug_p=a.aug_p, aug_min_frac=a.aug_min_frac, aug_snr_db=a.aug_snr_db,
                arch=a.arch, pretrained=not a.no_pretrained, fixed_dft=a.fixed_dft)


def build_model(cfg):
    return FourierShipNet(cfg) if cfg.get("arch", "vit") == "fourier" else RawShipNet(cfg)


def adjusted_scenes(ann_root):
    """Scenes whose annotations.json under ann_root carries a box_translation (written by box_adjust_server.py Apply)."""
    out = set()
    for p in glob.glob(os.path.join(ann_root or "", "*", "annotation*.json")):
        if p.endswith("_pre_adjustment.json"):
            continue
        try:
            with open(p) as f:
                js = json.load(f)
        except Exception:
            continue
        js = js[0] if isinstance(js, list) else js
        if js.get("box_translation"):
            m = td.SCENE_RE.search(js.get("scene_name") or os.path.basename(os.path.dirname(p)))
            if m:
                out.add(m.group(0))
    return out


def train_eval_subset(tr_e, tr_pos, va_pos_frac, n, seed):
    """Fixed training crops for a training-set AP, with the validation set's share of positives so the two compare."""
    rng = np.random.default_rng(seed + 12345)
    pos, neg = np.flatnonzero(tr_pos), np.flatnonzero(~tr_pos)
    n_pos = min(len(pos), int(round(n * va_pos_frac)) if va_pos_frac == va_pos_frac else n // 2)
    n_neg = min(len(neg), n - n_pos)
    idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]))
    return [tr_e[i] for i in idx]


def train(a):
    rank, world, device = td.setup_ddp()
    main = rank == 0
    say = (lambda *x: print(*x, flush=True)) if main else (lambda *x: None)
    torch.manual_seed(a.seed + rank); random.seed(a.seed + rank); np.random.seed(a.seed + rank)
    cfg = cfg_of(a)
    entries = td.read_manifest(a.crops)
    anns = td.load_scene_annotations(a.ann_root)
    if not entries:
        sys.exit(f"no crops listed in {os.path.join(a.crops, 'manifest.jsonl')}: run build_crops.py (without --dry-run) first")
    if a.only_adjusted:                                 # only scenes whose boxes were fixed with box_adjust_server.py
        keep = adjusted_scenes(a.ann_root)
        all_sc = sorted({e["scene"] for e in entries})
        dropped = [s for s in all_sc if s not in keep]
        entries = [e for e in entries if e["scene"] in keep]
        say(f"--only-adjusted: {len(all_sc) - len(dropped)} of {len(all_sc)} scenes kept; left out: {dropped or 'none'}")
        if not entries:
            sys.exit(f"no adjusted scenes in {a.ann_root} (looking for annotations.json with box_translation)")
    if a.scenes:
        entries = [e for e in entries if any(s in e["scene"] for s in a.scenes)]
    if a.exclude_scenes:
        entries = [e for e in entries if not any(s in e["scene"] for s in a.exclude_scenes)]
    tr_e, va_e, val_scenes = td.split_scenes(entries, a.val_scenes, a.val_frac, a.seed)
    if a.overfit:
        tr_e = va_e = entries
        val_scenes = ["(training crops)"]
    tr, va = RawSet(a.crops, tr_e, anns, cfg, True), RawSet(a.crops, va_e, anns, cfg, False)
    say(f"crops: {len(tr)} train ({int(tr.positive.sum())} positive), {len(va)} val ({int(va.positive.sum())} positive); "
        f"val scenes {val_scenes or 'none (crop split)'}; {world} process(es) on {device}")
    tre = None                                          # training-set AP: fixed crops, no augmentation
    if a.train_eval_n > 0 and len(tr):
        te_e = train_eval_subset(tr_e, tr.positive, float(va.positive.mean()) if len(va) else float("nan"),
                                 min(a.train_eval_n, len(tr)), a.seed)
        tre = RawSet(a.crops, te_e, anns, cfg, False)
        say(f"training-set AP on {len(tre)} fixed training crops ({int(tre.positive.sum())} positive)")
    os.makedirs(a.out, exist_ok=True)
    writer = None
    if main:
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump(dict(cfg, args=vars(a), val_scenes=val_scenes, train_scenes=sorted({e["scene"] for e in tr_e})), f,
                      indent=1, default=str)
        st = tv.dataset_stats(tr, va)
        say("dataset statistics:\n" + tv.stats_text(st))
        writer = tv.make_writer(a.out) if not a.no_tensorboard else None
        tv.write_stats(st, a.out, writer)
        if writer is not None:
            writer.add_text("config", "```\n" + json.dumps(dict(cfg, val_scenes=val_scenes), indent=1, default=str) + "\n```", 0)
    sampler = td.BalancedSampler(tr.positive, a.pos_frac, a.samples_per_epoch or len(tr), a.seed, rank, world)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, sampler=sampler, num_workers=a.workers, collate_fn=collate,
                                     pin_memory=device.type == "cuda", persistent_workers=a.workers > 0, drop_last=True)
    vdl = torch.utils.data.DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
    tdl = (torch.utils.data.DataLoader(tre, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
           if tre is not None else None)
    model = build_model(cfg).to(device)
    say(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location="cpu", weights_only=False)["model"])
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    core = model.module if world > 1 else model
    front = set(id(p) for p in core.front_parameters()) if hasattr(core, "front_parameters") else set()
    groups = [dict(params=[p for p in model.parameters() if id(p) not in front and p.requires_grad])]
    fp = [p for p in model.parameters() if id(p) in front and p.requires_grad]
    if fp:                                                  # the learnable transform moves slowly, no weight decay
        groups.append(dict(params=fp, lr=a.lr * a.front_lr_scale, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.05)
    steps = a.max_iters or a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    collage = None
    if main and a.collage_every > 0 and len(va):
        try:
            collage = tv.Collage(va, cfg, device, n=a.collage_n, pixel_m=a.collage_pixel_m, seed=a.seed)
            say(f"collage: {len(collage.items)} validation crops, every {a.collage_every} epochs -> {os.path.join(a.out, 'collages')}")
        except Exception as e:
            say(f"collage off ({type(e).__name__}: {e})")
    best, step, hist = -1e9, 0, []
    log_f = open(os.path.join(a.out, "metrics.jsonl"), "a") if main else None
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        t0, seen = time.time(), 0
        for b in dl:
            b = to_device(b, device)
            losses = compute_loss(model(b), b, cfg)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            opt.step(); sched.step()
            step += 1; seen += len(b["X"]) * world
            hist.append(loss.item())
            if main and (step % a.log_every == 0 or step == 1):
                rec = dict(step=step, epoch=epoch, loss=loss.item(), **{k: v.item() for k, v in losses.items()},
                           lr=sched.get_last_lr()[0], grad_norm=gn, crops_per_s=seen / (time.time() - t0))
                say(f"  step {step}: loss {rec['loss']:.4f} (" + ", ".join(f"{k} {rec[k]:.4f}" for k in losses) +
                    f"), {rec['crops_per_s']:.2f} crops/s")
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                if writer is not None:
                    for k in losses:
                        writer.add_scalar(f"loss/{k}", rec[k], step)
                    writer.add_scalar("loss/total", rec["loss"], step)
                    for k in ("lr", "grad_norm", "crops_per_s"):
                        writer.add_scalar(f"train/{k}", rec[k], step)
            if a.max_iters and step >= a.max_iters:
                break
        if world > 1:
            dist.barrier()
        last = epoch == a.epochs - 1 or (a.max_iters and step >= a.max_iters)
        if main and ((epoch + 1) % a.eval_every == 0 or last):
            m, records = evaluate(model, vdl, device, cfg)
            m.update(epoch=epoch, step=step)
            say(f"epoch {epoch}: AP30 {m['ap30']:.3f}, AP50 {m['ap50']:.3f}, precision {m['precision']:.3f}, recall {m['recall']:.3f}, "
                f"{m['false_alarms_per_km2']:.2f} false alarms per km2, centre error {m['centre_error_m']:.1f} m")
            log_f.write(json.dumps(dict(eval=m)) + "\n"); log_f.flush()
            if writer is not None:
                for k, v in m.items():
                    if isinstance(v, (int, float)) and k not in ("epoch", "step", "score_thr") and v == v:
                        writer.add_scalar(f"eval/{k}", v, step)
            if tdl is not None:
                mt, _ = evaluate(model, tdl, device, cfg)
                mt.update(epoch=epoch, step=step)
                say(f"epoch {epoch} TRAIN set: AP30 {mt['ap30']:.3f}, AP50 {mt['ap50']:.3f}, precision {mt['precision']:.3f}, "
                    f"recall {mt['recall']:.3f}, centre error {mt['centre_error_m']:.1f} m")
                log_f.write(json.dumps(dict(eval_train=mt)) + "\n"); log_f.flush()
                if writer is not None:
                    for k, v in mt.items():
                        if isinstance(v, (int, float)) and k not in ("epoch", "step", "score_thr") and v == v:
                            writer.add_scalar(f"eval_train/{k}", v, step)
            state = dict(model=core.state_dict(), cfg=cfg, epoch=epoch, step=step, metrics=m)
            torch.save(state, os.path.join(a.out, "last.pt"))
            score = m["ap30"] if m["ap30"] == m["ap30"] else -m["false_alarms_per_km2"]
            if score > best:
                best = score
                torch.save(state, os.path.join(a.out, "best.pt"))
                td.detections_geojson(records, os.path.join(a.out, "val_detections.geojson"))
        if main and collage is not None and ((epoch + 1) % a.collage_every == 0 or last):
            try:
                sheet = collage.render(model, decode, collate, to_device, epoch)
                os.makedirs(os.path.join(a.out, "collages"), exist_ok=True)
                sheet.save(os.path.join(a.out, "collages", f"epoch_{epoch + 1:03d}.png"))
                if writer is not None:
                    writer.add_image("collage", np.asarray(sheet), step, dataformats="HWC")
            except Exception as e:
                say(f"collage failed at epoch {epoch} ({type(e).__name__}: {e})")
        if writer is not None:
            writer.flush()
        if world > 1:
            dist.barrier()
        if a.max_iters and step >= a.max_iters:
            break
    if writer is not None:
        writer.close()
    if world > 1:
        dist.destroy_process_group()
    return hist


@torch.no_grad()
def predict(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(st["model"])
    entries = td.read_manifest(a.crops)
    run_cfg = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "config.json")
    if a.split != "all":                                # the run's own split, from its config.json
        if not os.path.exists(run_cfg):
            sys.exit(f"--split {a.split} needs {run_cfg}; use --split all or --scenes")
        with open(run_cfg) as f:
            rc = json.load(f)
        val = set(rc.get("val_scenes") or [])
        seen = set(rc.get("train_scenes") or [])
        if a.split == "val":
            entries = [e for e in entries if e["scene"] in val]
        else:                                           # train: scenes the run trained on (adjusted ones if it used it)
            keep = seen or ((adjusted_scenes(rc["args"].get("ann_root")) if rc["args"].get("only_adjusted") else
                             {e["scene"] for e in entries}) - val)
            entries = [e for e in entries if e["scene"] in keep]
        print(f"--split {a.split}: {len({e['scene'] for e in entries})} scenes, {len(entries)} crops")
    if a.scenes:
        entries = [e for e in entries if any(s in e["scene"] for s in a.scenes)]
    if not entries:
        sys.exit("no crops to run on")
    ds = RawSet(a.crops, entries, td.load_scene_annotations(a.ann_root), cfg, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, collate_fn=collate)
    thrs = sorted(a.score_thrs) if a.score_thrs else [a.score_thr]
    keep = HeatSaver(a.save_heat, cfg) if a.save_heat else None
    m0, records = evaluate(model, dl, device, cfg, thrs[0], min_score=min(thrs), keep=keep)   # one pass, all thresholds
    if keep is not None:
        keep.close(a.ckpt, st.get("epoch"))
    if not a.score_thrs:
        n = td.detections_geojson(records, a.out, a.score_thr)
        print(f"{len(ds)} crops, {n} detections after merging overlaps -> {a.out}")
        if ds.anns:
            print(json.dumps(m0, indent=1))
        return m0
    os.makedirs(a.out, exist_ok=True)                   # --out is a folder when several thresholds are given
    rows = []
    collage = None
    if a.collage_n > 0:
        try:
            collage = tv.Collage(ds, cfg, device, n=a.collage_n, pixel_m=a.collage_pixel_m, seed=0)
        except Exception as e:
            print(f"collage off ({type(e).__name__}: {e})")
    for t in thrs:
        m = summarise(records, cfg, t)
        n = td.detections_geojson(records, os.path.join(a.out, f"detections_thr{t:g}.geojson"), t)
        rows.append(dict(score_thr=t, detections=n, **{k: v for k, v in m.items() if k != "score_thr"}))
        if collage is not None:
            collage.render(model, decode, collate, to_device, st.get("epoch", -1), score_thr=t).save(
                os.path.join(a.out, f"collage_thr{t:g}.png"))
    keys = ["score_thr", "detections", "precision", "recall", "recall_within_25m", "precision_within_25m",
            "false_alarms_per_km2", "negative_crops_with_detection", "centre_error_m"]
    keys = [k for k in keys if k in rows[0]]
    print(f"{len(ds)} crops, {rows[0]['gt']} ships; AP30 {m0['ap30']:.3f}, AP50 {m0['ap50']:.3f} (AP does not depend on the threshold)")
    short = dict(score_thr="thr", detections="dets", precision="prec", recall="recall", recall_within_25m="rec25m",
                 precision_within_25m="prec25m", false_alarms_per_km2="FA/km2", negative_crops_with_detection="neg_hit",
                 centre_error_m="ctr_err_m")
    print(" ".join(f"{short[k]:>9s}" for k in keys))
    for r in rows:
        print(" ".join(f"{r[k]:9g}" if k in ("score_thr", "detections") else f"{r[k]:9.3f}" for k in keys))
    import csv
    with open(os.path.join(a.out, "thresholds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "thresholds.json"), "w") as f:
        json.dump(dict(ckpt=a.ckpt, epoch=st.get("epoch"), ap30=m0["ap30"], ap50=m0["ap50"], rows=rows), f, indent=1,
                  default=float)
    print(f"-> {a.out}: thresholds.csv, detections_thr<t>.geojson" + (", collage_thr<t>.png" if collage else ""))
    return rows


class HeatSaver:
    """Writes the network's probability map of every crop to <dir>/<scene>/<crop_id>.npz for experiments outside the
    detector (watershed, thresholds, ...):
      heat        (cells, cells) float32, sigmoid of the heat head; row 0 = north edge, column 0 = west edge
      off, size   (2, cells, cells) float16: the offset / log-size heads (box size at a cell = exp(size) metres E, N)
      gt_boxes    (K, 4) crop-local metres [e0, n0, e1, n1] (east / north of the crop centre), gt_ignore (K,)
      lat, lon    crop centre; cell_m, cells
      geo         [lon_w, lat_n, dlon, dlat]: cell (r, c) covers lon_w + c*dlon .. +dlon, lat_n - r*dlat .. -dlat
    index.json lists every file."""
    def __init__(self, out, cfg):
        self.out, self.cfg, self.rows = out, cfg, []
        os.makedirs(out, exist_ok=True)

    def __call__(self, out, b):
        heat = torch.sigmoid(out["heat"][:, 0]).float().cpu().numpy()
        off, size = out["off"].float().cpu().numpy(), out["size"].float().cpu().numpy()
        cells, cm = self.cfg["cells"], self.cfg["cell_m"]
        H = cells * cm / 2
        for i in range(len(heat)):
            lat, lon = float(b["lat"][i]), float(b["lon"][i])
            ml, mo = td.m_per_deg(lat)
            geo = [lon - H / mo, lat + H / ml, cm / mo, cm / ml]
            d = os.path.join(self.out, b["scene"][i])
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, b["crop_id"][i] + ".npz")
            np.savez_compressed(p, heat=heat[i].astype(np.float32), off=off[i].astype(np.float16),
                                size=size[i].astype(np.float16), gt_boxes=np.asarray(b["gt"][i], np.float32),
                                gt_ignore=np.asarray(b["gt_ignore"][i], bool), lat=lat, lon=lon, cell_m=cm, cells=cells,
                                geo=np.array(geo))
            self.rows.append(dict(scene=b["scene"][i], crop_id=b["crop_id"][i], npz=os.path.relpath(p, self.out),
                                  n_gt=int((~np.asarray(b["gt_ignore"][i], bool)).sum()), heat_max=float(heat[i].max())))

    def close(self, ckpt, epoch):
        with open(os.path.join(self.out, "index.json"), "w") as f:
            json.dump(dict(ckpt=ckpt, epoch=epoch, cells=self.cfg["cells"], cell_m=self.cfg["cell_m"],
                           crops=self.rows), f, indent=1)
        print(f"probability maps of {len(self.rows)} crops -> {self.out}")


# ------------------------------------------------------------------ self-test
def selftest(work, reuse=False):
    crops, ann_root, man, bc = td.make_synthetic_dataset(work, reuse=reuse)
    print(f"synthetic crops: {len(man)} ({sum(m['label'] == 'positive' for m in man)} positive)")
    ok = True
    p = parser()
    small = ["--cells", "64", "--cell-m", "8", "--dim", "192", "--depth", "4", "--heads", "4", "--dec-dim", "96",
             "--dec-heads", "4", "--dec-layers", "2"]
    a = p.parse_args(["train", "--crops", crops, "--ann-root", ann_root, "--out", os.path.join(work, "raw"), *small])
    cfg = cfg_of(a)
    anns = td.load_scene_annotations(ann_root)
    ds = RawSet(crops, man, anns, cfg, train=True)

    # 1. moving the reference point by a phase ramp keeps the scene where it is on the ground
    i = next(k for k, m in enumerate(man) if m["kind"] == "ship_centred")
    d0, d1 = ds.get(i, shift=(0.0, 0.0)), ds.get(i, shift=(40.0, -25.0))
    m0 = ds.meta[i]
    g = td.make_grid(m0["centre_lat"], m0["centre_lon"], 64, 1.0)
    imgs = []
    for d in (d0, d1):
        Dv = td.llh_to_ecef(d["lat"], d["lon"], m0["h_ref"])
        pix = td.grid_pixels_ecef(g, m0["h_ref"], Dv)
        im = td.Imager(d["X"].shape[0], d["X"].shape[1], n_sub=1, oversample=4, learn_taper=False)
        re_, im_ = im(d["X"][None], d["tx"][None], d["rc"][None], d["f0"][None], d["scss"][None], torch.tensor([-1]),
                      torch.from_numpy(pix)[None])
        imgs.append((re_[0, 0] + 1j * im_[0, 0]).numpy())
    a0, a1 = np.abs(imgs[0]).reshape(64, 64), np.abs(imgs[1]).reshape(64, 64)
    corr = float(np.corrcoef(a0.ravel(), a1.ravel())[0, 1])
    yy, xx = np.mgrid[0:64, 0:64]
    cen = lambda a: np.array([(a * yy).sum(), (a * xx).sum()]) / a.sum()
    moved = float(np.hypot(*(cen(a0 ** 2) - cen(a1 ** 2))))
    good = corr > 0.98 and moved < 0.5
    ok &= good
    print(f"reference point moved by (40, -25) m with a phase ramp: image of the same ground correlates {corr:.4f}, "
          f"bright centroid moved {moved:.2f} m | {'ok' if good else 'FAIL'}")

    # 2. targets: each visible ship gives one centre cell with heat 1, and decoding the target returns the box
    b = collate([d1])
    k = int(b["mask"].sum())
    dec = decode_at(b["off"], b["size"], b["idx"], cfg["cells"], cfg["cell_m"])[b["mask"] > 0]
    err = float((dec - b["box"][b["mask"] > 0]).abs().max()) if k else 0.0
    good = int((b["heat"] >= 1).sum()) == k == int((~d1["gt_ignore"]).sum()) and err < 0.6
    ok &= good
    print(f"targets: {k} ship centre(s), heat peaks {int((b['heat'] >= 1).sum())}, box from target values within "
          f"{err:.3f} m | {'ok' if good else 'FAIL'}")

    # 3. forward, losses and gradients
    model = RawShipNet(cfg)
    losses = compute_loss(model(b), b, cfg)
    sum(losses.values()).backward()
    gp = float(model.patch.weight.grad.abs().mean())
    good = all(torch.isfinite(v) for v in losses.values()) and gp > 0
    ok &= good
    print(f"losses {', '.join(f'{k} {v.item():.3f}' for k, v in losses.items())}; gradient reaches the patch embedding "
          f"({gp:.2e}) | {'ok' if good else 'FAIL'}")

    # 4. can it memorise the training crops? (CPU, small model)
    out = os.path.join(work, "raw")
    a = p.parse_args(["train", "--crops", crops, "--ann-root", ann_root, "--out", out, *small, "--overfit", "--batch", "4",
                      "--epochs", "1", "--max-iters", "300", "--samples-per-epoch", "1200", "--lr", "5e-4", "--warmup", "30",
                      "--workers", "2", "--log-every", "50", "--pos-frac", "0.8", "--shift-m", "16"])
    hist = train(a)
    with open(os.path.join(out, "metrics.jsonl")) as f:
        ev = [json.loads(l)["eval"] for l in f if '"eval"' in l]
    first, last = float(np.mean(hist[:20])), float(np.mean(hist[-20:]))
    good = last < first and bool(ev)
    ok &= good
    print(f"overfit run: mean loss {first:.3f} (first 20 steps) -> {last:.3f} (last 20); AP30 on the training crops "
          f"{ev[-1]['ap30']:.3f}, recall {ev[-1]['recall']:.2f} at score 0.5 | {'ok' if good else 'FAIL'}")
    pa = p.parse_args(["predict", "--ckpt", os.path.join(out, "best.pt"), "--crops", crops, "--ann-root", ann_root,
                       "--out", os.path.join(out, "pred.geojson"), "--score-thr", "0.3", "--workers", "2"])
    predict(pa)
    good = os.path.exists(os.path.join(out, "pred.geojson"))
    ok &= good
    print(f"predict wrote GeoJSON | {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def parser():
    ap = argparse.ArgumentParser(description="CPHD ship detector on raw phase history")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--crops", required=True)
    t.add_argument("--ann-root", default="/data3/sandbox/jhu/satlock/annotation_tool/test_aligned")
    t.add_argument("--out", required=True)
    t.add_argument("--val-scenes", nargs="*")
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--only-adjusted", action="store_true",
                   help="use only scenes whose --ann-root annotations.json was adjusted with box_adjust_server.py")
    t.add_argument("--scenes", nargs="*", help="use only these scenes (substring match)")
    t.add_argument("--exclude-scenes", nargs="*", help="leave these scenes out (substring match)")
    t.add_argument("--train-eval-n", type=int, default=400,
                   help="training-set AP each eval on this many fixed training crops (0 = off)")
    t.add_argument("--overfit", action="store_true", help="train and validate on the same crops (sanity check)")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--max-iters", type=int, default=0)
    t.add_argument("--samples-per-epoch", type=int, default=0)
    t.add_argument("--batch", type=int, default=8, help="crops per GPU")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--pos-frac", type=float, default=0.5)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--eval-every", type=int, default=1)
    t.add_argument("--collage-every", type=int, default=10, help="image collage of predictions every N epochs (0 = off)")
    t.add_argument("--collage-n", type=int, default=8, help="validation crops in the collage")
    t.add_argument("--collage-pixel-m", type=float, default=2.0, help="pixel size of the collage images")
    t.add_argument("--no-tensorboard", action="store_true")
    t.add_argument("--resume", default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--cells", type=int, default=128, help="output grid cells per side (128 x 4 m = 512 m crop)")
    t.add_argument("--cell-m", type=float, default=4.0)
    t.add_argument("--shift-m", type=float, default=64.0, help="random move of the crop centre, metres (phase ramp)")
    t.add_argument("--min-visible", type=float, default=0.3)
    t.add_argument("--max-objs", type=int, default=64)
    t.add_argument("--pulses", type=int, default=1024)
    t.add_argument("--samples", type=int, default=1024)
    t.add_argument("--patch", type=int, default=32)
    t.add_argument("--dim", type=int, default=384, help="encoder width (ViT-S)")
    t.add_argument("--depth", type=int, default=12)
    t.add_argument("--heads", type=int, default=6)
    t.add_argument("--dec-dim", type=int, default=128)
    t.add_argument("--dec-heads", type=int, default=4)
    t.add_argument("--dec-layers", type=int, default=2)
    t.add_argument("--giou-weight", type=float, default=1.0)
    t.add_argument("--tol-m", type=float, default=0.0,
                   help="position tolerance in metres for the losses (e.g. 24): boxes may be off by this much")
    t.add_argument("--eval-tol-m", type=float, default=25.0,
                   help="extra metrics: a detection counts if its centre is within this many metres of a ship")
    t.add_argument("--amp", action="store_true", help="bfloat16 autocast on GPU")
    t.add_argument("--aug", action="store_true",
                   help="signal augmentation in training: random sub-aperture, sub-band, noise, global phase")
    t.add_argument("--aug-p", type=float, default=0.5, help="probability of each augmentation")
    t.add_argument("--aug-min-frac", type=float, default=0.6, help="shortest kept run of pulses / frequencies")
    t.add_argument("--aug-snr-db", type=float, default=0.0, help="lowest SNR of the added noise, dB")
    t.add_argument("--arch", choices=["vit", "fourier"], default="fourier",
                   help="fourier: learnable DFT front end + CNN (default); vit: patch ViT encoder + cell decoder")
    t.add_argument("--no-pretrained", action="store_true", help="fourier: ResNet-18 stem from scratch")
    t.add_argument("--fixed-dft", action="store_true", help="fourier: keep the transform and taper fixed")
    t.add_argument("--front-lr-scale", type=float, default=0.1, help="fourier: learning-rate factor for the transform")
    p = sub.add_parser("predict")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--crops", required=True)
    p.add_argument("--ann-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--score-thrs", type=float, nargs="*",
                   help="several thresholds: metrics table + a GeoJSON (+ collage) per threshold; --out is then a folder")
    p.add_argument("--split", choices=["val", "train", "all"], default="all",
                   help="val / train = that run's own scene split (config.json next to --ckpt)")
    p.add_argument("--collage-n", type=int, default=8, help="crops per collage with --score-thrs (0 = none)")
    p.add_argument("--collage-pixel-m", type=float, default=2.0)
    p.add_argument("--save-heat", default=None,
                   help="folder: save every crop's probability map (+ geo, ground truth) as <scene>/<crop_id>.npz")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("selftest")
    s.add_argument("--dir", default="./cphd_raw_selftest")
    s.add_argument("--reuse", action="store_true", help="reuse synthetic crops already in --dir")
    return ap


def main():
    a = parser().parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "predict":
        predict(a)
    else:
        sys.exit(0 if selftest(a.dir, a.reuse) else 1)


if __name__ == "__main__":
    main()
