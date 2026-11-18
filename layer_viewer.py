"""Layer viewer: look inside a trained train_raw.py model (--arch fourier or --arch det), stage by stage, in a browser.

For any crop of the run's validation (or training) scenes it shows every stage:
    phase history |X|                         pulses x frequencies
    learned front end                         tapers, and how far each DFT row moved from the plain DFT
    Fourier features (5 ch)                   logZ logZ1 logZ2 coh_re coh_im, in transform bins (radar frame)
    features on the ground                    the same 5 channels resampled north-up (what --arch det sees)
    fourier:  conv1  layer1  layer2  fuse  (radar frame)  ->  sampled to cells  trunk  heat  offset  size
    det:      conv1  layer1..layer4  FPN levels  RPN objectness  (ground frame)  ->  detections
each as channel thumbnails (sortable) plus a large view with the ship boxes (ground frame) or the ship centres mapped
into the radar bins (radar frame).

"Val analysis" runs the model over N crops and, for every stage, measures how well it separates ship from background:
    per channel:  AUC of ship cells vs background cells (on a common 4 m north-up grid; radar-frame stages are
                  resampled there with the crop's geometry, exactly as the model does)
    per stage:    best single channel (separation = max(AUC, 1-AUC)), median channel, channels above 0.7,
                  and a linear probe: logistic regression on all channels, trained on half the crops and tested on
                  the other half (test AUC) = how much ship information that stage carries, linearly readable
Results are kept in <work>/layer_stats.json and the thumbnails can then be sorted by val AUC.

Run on the server (same image as training):
    docker run -d --name layers --gpus '"device=1"' --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \\
      -p 127.0.0.1:8770:8770 -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann:ro \\
      -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \\
      cphd-gpu /app/layer_viewer.py --ckpt /runs/det_frcnn/best.pt --crops /crops --ann-root /ann \\
      --split val --host 0.0.0.0
then on the laptop: ssh -L 8770:localhost:8770 tabtahi@geohub3 (or forward 8770 in VS Code), open http://localhost:8770
"""
import argparse, io, json, math, os, sys, threading, time, traceback, types
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import train_detector as td
import train_raw as tr
from fourier_net import FourierShipNet, LearnableDFT

CMAP = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21],
                 [250, 193, 39], [252, 255, 164]], float)          # inferno-like
FCH = ["logZ", "logZ1", "logZ2", "coh_re", "coh_im"]


def colour(a):
    lo, hi = np.nanpercentile(a, [1, 99.5])
    x = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * (len(CMAP) - 1)
    i = np.minimum(x.astype(int), len(CMAP) - 2)
    f = (x - i)[..., None]
    return (CMAP[i] * (1 - f) + CMAP[i + 1] * f).astype(np.uint8)


# ------------------------------------------------------------------ the model and its stages
class Viewer:
    def __init__(self, a):
        self.a = a
        self.device = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
        st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        self.cfg = cfg = dict(st["cfg"], amp=False)
        self.epoch = st.get("epoch")
        self.model = tr.build_model(cfg).to(self.device).eval()
        self.model.load_state_dict(st["model"])
        self.arch = cfg.get("arch", "vit")
        if self.arch not in ("fourier", "det"):
            sys.exit("layer_viewer supports --arch fourier and --arch det checkpoints")
        self.front = self.model.front if self.arch == "det" else self.model
        entries = td.read_manifest(a.crops)
        rc_path = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "config.json")
        if a.split != "all" and os.path.exists(rc_path):
            rc = json.load(open(rc_path))
            val = set(rc.get("val_scenes") or [])
            if a.split == "val":
                entries = [e for e in entries if e["scene"] in val]
            else:
                tr_sc = set(rc.get("train_scenes") or []) or ({e["scene"] for e in entries} - val)
                entries = [e for e in entries if e["scene"] in tr_sc]
        if a.scenes:
            entries = [e for e in entries if any(s in e["scene"] for s in a.scenes)]
        self.ds = tr.RawSet(a.crops, entries, td.load_scene_annotations(a.ann_root), cfg, train=False)
        self.order = sorted(range(len(self.ds)), key=lambda i: (not self.ds.positive[i], self.ds.meta[i]["crop_id"]))
        self.H = cfg["cells"] * cfg["cell_m"] / 2
        self.G = cfg["cells"]                                     # stats grid: cells x cells at cell_m
        self.work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "layer_viewer")
        os.makedirs(self.work, exist_ok=True)
        self.stats_path = os.path.join(self.work, "layer_stats.json")
        self.stats = json.load(open(self.stats_path)) if os.path.exists(self.stats_path) else None
        self.cache = OrderedDict()
        self.lock, self.fwd_lock = threading.Lock(), threading.Lock()
        self.job = dict(running=False, done=0, total=0, msg="")
        print(f"{self.arch} model, epoch {self.epoch}, {len(self.ds)} crops ({int(self.ds.positive.sum())} with ships) "
              f"on {self.device}; work folder {self.work}")

    # ---- geometry helpers
    def grid_for(self, b, en):
        """normalised radar-bin coordinates (as grid_sample wants) of ground points en (h, w, 2) metres"""
        ns = types.SimpleNamespace(cell_en=torch.as_tensor(en, dtype=torch.float64, device=self.device))
        return FourierShipNet.sample_grid(ns, b)

    def ground_cells(self):
        cm = self.cfg["cell_m"]
        c = (torch.arange(self.G, dtype=torch.float64) + 0.5) * cm
        n_, e_ = torch.meshgrid(self.H - c, c - self.H, indexing="ij")
        return torch.stack([e_, n_], -1)

    # ---- one forward pass, every stage captured
    def taps(self, i):
        with self.lock:
            if i in self.cache:
                self.cache.move_to_end(i)
                return self.cache[i]
        with self.fwd_lock:                                       # one forward at a time: the hooks are shared
            rec = self._taps(i)
        with self.lock:
            self.cache[i] = rec
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        return rec

    @torch.no_grad()
    def _taps(self, i):
        it = self.ds.get(i, shift=(0.0, 0.0))
        b = tr.to_device(tr.collate([it]), self.device)
        cap = OrderedDict()
        add = lambda name, t, frame, desc: cap.__setitem__(name, dict(t=t.detach().float()[0].cpu(), frame=frame, desc=desc))
        X = b["X"]
        add("phase history |X|", torch.log10(X.abs() ** 2 + 1e-12)[:, None], "pulse",
            "log10 power of the input phase history: rows = pulses, columns = frequency samples")
        Z, Z1, Z2 = FourierShipNet.transform(self.front, X.to(torch.complex64))
        feat = FourierShipNet.features(Z, Z1, Z2).float()
        add("Fourier features", feat, "radar",
            "5 channels in transform bins (rows = cross-range bins, columns = range bins): " + ", ".join(FCH))
        n1 = int(round(2 * self.H))                               # 1 m ground image of the features
        c = (torch.arange(n1, dtype=torch.float64) + 0.5)
        n_, e_ = torch.meshgrid(self.H - c, c - self.H, indexing="ij")
        g1 = self.grid_for(b, torch.stack([e_, n_], -1))
        add("features on the ground", F.grid_sample(feat, g1, mode="bilinear", align_corners=False), "ground",
            "the 5 feature channels resampled north-up at 1 m with the crop's geometry" +
            (" (= the detector's input)" if self.arch == "det" else ""))
        hooks, m = [], self.model
        hk = lambda name, frame, desc, f=lambda o: o: (lambda mod, inp, out: add(name, f(out), frame, desc))
        if self.arch == "fourier":
            hooks += [m.relu.register_forward_hook(hk("conv1", "radar", "conv1 + BN + ReLU, stride 2 (64 ch)")),
                      m.layer1.register_forward_hook(hk("layer1", "radar", "ResNet-18 layer1, stride 4 (64 ch)")),
                      m.layer2.register_forward_hook(hk("layer2", "radar", "ResNet-18 layer2, stride 8 (128 ch)")),
                      m.fuse.register_forward_hook(hk("fuse", "radar", "layer1 + upsampled layer2, 1x1 conv, stride 4")),
                      m.trunk.register_forward_pre_hook(lambda mod, inp: add("sampled to cells", inp[0], "ground",
                                                        "fuse features sampled at each 4 m ground cell's bin")),
                      m.trunk.register_forward_hook(hk("trunk", "ground", "two 3x3 conv blocks on the cell grid"))]
            out = m(b)
            add("heat", torch.sigmoid(out["heat"]), "ground", "ship probability per 4 m cell")
            add("offset", out["off"], "ground", "centre offset within the cell (east, north)")
            add("size", out["size"], "ground", "log box size, metres (east, north)")
            dets = tr.decode(out, self.cfg, thr=0.05)[0]
        else:
            body, det = m.det.backbone.body, m.det
            hooks += [body.relu.register_forward_hook(hk("conv1", "ground", "conv1 + BN + ReLU, stride 2 (64 ch)"))]
            for k in ("layer1", "layer2", "layer3", "layer4"):
                hooks.append(getattr(body, k).register_forward_hook(hk(k, "ground", f"ResNet {k}")))
            lev = {}
            hooks.append(det.backbone.register_forward_hook(lambda mod, inp, out: lev.update(out)))
            rpn = {}
            if hasattr(det, "rpn"):
                hooks.append(det.rpn.head.register_forward_hook(lambda mod, inp, out: rpn.update(obj=out[0])))
            out = m(b)
            for k, v in lev.items():
                add(f"FPN {k}", v, "ground", f"feature pyramid level {k} (256 ch)")
            for k, v in zip(lev.keys(), rpn.get("obj", [])):
                add(f"RPN objectness {k}", torch.sigmoid(v), "ground",
                    f"RPN objectness at FPN level {k}, one channel per anchor aspect ratio (0.25, 0.5, 1, 2, 4)")
            s, bx = out["dets"][0]
            dets = (s.cpu().numpy(), bx.cpu().numpy())
        for h in hooks:
            h.remove()
        gt = np.asarray(it["gt"], np.float32).reshape(-1, 4)
        ign = np.asarray(it["gt_ignore"], bool).reshape(-1)
        cen = torch.as_tensor(np.stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2], -1)[:, None]
                              if len(gt) else np.zeros((0, 1, 2)))
        radar_c = self.grid_for(b, cen)[0, :, 0].cpu().numpy() if len(gt) else np.zeros((0, 2))
        return dict(stages=cap, gt=gt, ign=ign, radar_c=radar_c, dets=dets, b=b,
                    crop_id=self.ds.meta[i]["crop_id"], scene=self.ds.meta[i]["scene"])

    # ---- everything on the common 4 m ground grid (for statistics)
    def to_ground(self, rec, name):
        s = rec["stages"][name]
        t = s["t"][None].to(self.device)
        if s["frame"] == "ground":
            return F.interpolate(t, size=(self.G, self.G), mode="bilinear", align_corners=False)[0]
        if s["frame"] == "radar":
            g = self.grid_for(rec["b"], self.ground_cells())
            return F.grid_sample(t, g, mode="bilinear", align_corners=False)[0]
        return None

    def masks(self, rec):
        cm, H, G = self.cfg["cell_m"], self.H, self.G
        c = (np.arange(G) + 0.5) * cm
        E, N = np.meshgrid(c - H, H - c)
        ship, near = np.zeros((G, G), bool), np.zeros((G, G), bool)
        for g, ig in zip(rec["gt"], rec["ign"]):
            inside = (E >= g[0]) & (E <= g[2]) & (N >= g[1]) & (N <= g[3])
            pad = (E >= g[0] - cm) & (E <= g[2] + cm) & (N >= g[1] - cm) & (N <= g[3] + cm)
            near |= pad
            if not ig:
                ship |= inside
        return ship, ~near

    def contrast(self, rec, name):
        """this crop: AUC of ship cells vs background cells, per channel (0.5 = no separation; far from 0.5 = separates)"""
        g = self.to_ground(rec, name)
        if g is None:
            return None
        ship, bg = self.masks(rec)
        if ship.sum() < 2 or bg.sum() < 2:
            return None
        g = g.cpu().numpy().reshape(g.shape[0], -1)
        rng = np.random.default_rng(0)
        bi = np.flatnonzero(bg.ravel())
        bi = rng.choice(bi, min(len(bi), 2000), replace=False)
        return auc_per_channel(g[:, ship.ravel()].T, g[:, bi].T)

    # ---- pictures
    def render(self, rec, name, ch, size, boxes=True):
        s = rec["stages"][name]
        t = s["t"]
        if ch == "mean":
            a = t.mean(0).numpy()
        elif ch == "max":
            a = t.max(0).values.numpy()
        else:
            a = t[int(ch)].numpy()
        im = Image.fromarray(colour(a))
        W = size
        Hh = max(1, int(round(size * a.shape[0] / a.shape[1]))) if s["frame"] == "pulse" else size
        im = im.resize((W, Hh), Image.NEAREST if a.shape[0] < W else Image.BILINEAR)
        if boxes and s["frame"] != "pulse":
            d = ImageDraw.Draw(im)
            if s["frame"] == "ground":
                sc = W / (2 * self.H)
                for g, ig in zip(rec["gt"], rec["ign"]):
                    d.rectangle([(g[0] + self.H) * sc, (self.H - g[3]) * sc, (g[2] + self.H) * sc, (self.H - g[1]) * sc],
                                outline=(255, 230, 0) if ig else (60, 255, 90), width=max(1, W // 300))
                if name.startswith(("heat", "features on", "RPN", "FPN")) or name == "trunk":
                    for sco, bx in zip(*rec["dets"]):
                        if sco >= 0.3:
                            d.rectangle([(bx[0] + self.H) * sc, (self.H - bx[3]) * sc, (bx[2] + self.H) * sc,
                                         (self.H - bx[1]) * sc], outline=(80, 200, 255), width=1)
            else:                                                 # radar frame: ship centres in bins
                for (gx, gy), ig in zip(rec["radar_c"], rec["ign"]):
                    x, y = (gx + 1) / 2 * W, (gy + 1) / 2 * Hh
                    r = max(4, W // 60)
                    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 230, 0) if ig else (60, 255, 90), width=2)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def front_plot(self, size=900):
        """tapers and how far each DFT row moved from its initial value"""
        f = self.front
        rows = [("taper over pulses", f.taper_p.detach().cpu().numpy(), None),
                ("taper over frequencies", f.taper_s.detach().cpu().numpy(), None)]
        for nm, dft in (("pulse DFT (cross-range)", f.dft_p), ("frequency DFT (range)", f.dft_s)):
            w0 = LearnableDFT(dft.wr.shape[0], False)
            dev = (torch.hypot(dft.wr.detach().cpu() - w0.wr, dft.wi.detach().cpu() - w0.wi).norm(dim=1) /
                   torch.hypot(w0.wr, w0.wi).norm(dim=1)).numpy()
            rows.append((f"{nm}: relative change per output bin", dev, None))
        h = 150
        im = Image.new("RGB", (size, h * len(rows)), (20, 22, 26))
        d = ImageDraw.Draw(im)
        for k, (title, y, _) in enumerate(rows):
            y0 = k * h
            lo, hi = float(np.min(y)), float(np.max(y))
            flat = hi - lo < 1e-7
            if flat:
                hi = lo + 1e-9
            pts = [(40 + j / (len(y) - 1) * (size - 60), y0 + h - 25 - (v - lo) / (hi - lo) * (h - 50)) for j, v in enumerate(y)]
            d.line(pts, fill=(250, 170, 60), width=2)
            d.text((40, y0 + 6), f"{title}   " + (f"unchanged ({lo:.3g})" if flat else f"min {lo:.4g}  max {hi:.4g}"),
                   fill=(230, 230, 230))
            d.line([(40, y0 + h - 25), (size - 20, y0 + h - 25)], fill=(90, 90, 90))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    # ---- val analysis
    def analyse(self, n):
        try:
            idx = [i for i in self.order if self.ds.positive[i]][:n]
            negs = [i for i in self.order if not self.ds.positive[i]][:max(2, n // 4)]
            idx = idx + negs
            self.job.update(running=True, done=0, total=len(idx), msg="running")
            rng = np.random.default_rng(0)
            pos, neg = {}, {}
            for k, i in enumerate(idx):
                rec = self.taps(i)
                ship, bg = self.masks(rec)
                si, bi = np.flatnonzero(ship.ravel()), np.flatnonzero(bg.ravel())
                si = rng.choice(si, min(len(si), 200), replace=False) if len(si) else si
                bi = rng.choice(bi, min(len(bi), 400), replace=False) if len(bi) else bi
                for name in rec["stages"]:
                    g = self.to_ground(rec, name)
                    if g is None:
                        continue
                    g = g.reshape(g.shape[0], -1).T.cpu().numpy().astype(np.float32)
                    pos.setdefault(name, []).append((k, g[si]))
                    neg.setdefault(name, []).append((k, g[bi]))
                self.job["done"] = k + 1
            out = []
            for name in pos:
                P = np.concatenate([v for _, v in pos[name]])
                N = np.concatenate([v for _, v in neg[name]])
                if len(P) < 10 or len(N) < 10:
                    continue
                auc = auc_per_channel(P, N)
                sep = np.maximum(auc, 1 - auc)
                probe = linear_probe(pos[name], neg[name])
                out.append(dict(stage=name, channels=int(P.shape[1]), best_sep=float(sep.max()), best_ch=int(sep.argmax()),
                                median_sep=float(np.median(sep)), n_above_07=int((sep > 0.7).sum()), probe_auc=probe,
                                auc=[round(float(x), 4) for x in auc]))
            self.stats = dict(crops=len(idx), ship_crops=len(idx) - len(negs), epoch=self.epoch, ckpt=self.a.ckpt,
                              stages=out, time=time.strftime("%Y-%m-%d %H:%M"))
            json.dump(self.stats, open(self.stats_path, "w"))
            self.job.update(running=False, msg=f"done: {len(idx)} crops")
        except Exception as e:
            traceback.print_exc()
            self.job.update(running=False, msg=f"failed: {type(e).__name__}: {e}")


def auc_per_channel(P, N):
    """Mann-Whitney AUC per column: P (n1, C) ship values, N (n0, C) background values."""
    x = torch.from_numpy(np.concatenate([P, N])).float()
    r = torch.empty_like(x)
    order = x.argsort(0)
    r.scatter_(0, order, torch.arange(1, len(x) + 1, dtype=torch.float32)[:, None].expand_as(x).contiguous())
    n1, n0 = len(P), len(N)
    return ((r[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)).numpy()


def linear_probe(pos, neg):
    """logistic regression on all channels: train on even-numbered crops, test AUC on odd-numbered ones"""
    tr_p = [v for k, v in pos if k % 2 == 0]; te_p = [v for k, v in pos if k % 2 == 1]
    tr_n = [v for k, v in neg if k % 2 == 0]; te_n = [v for k, v in neg if k % 2 == 1]
    if not (tr_p and te_p and tr_n and te_n):
        return None
    Xp, Xn = np.concatenate(tr_p), np.concatenate(tr_n)
    Tp, Tn = np.concatenate(te_p), np.concatenate(te_n)
    if min(len(Xp), len(Xn), len(Tp), len(Tn)) < 5:
        return None
    X = torch.from_numpy(np.concatenate([Xp, Xn])).float()
    y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pw = torch.tensor(len(Xn) / max(len(Xp), 1))
    opt = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(X @ w + b, y, pos_weight=pw) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        sp = ((torch.from_numpy(Tp).float() - mu) / sd) @ w + b
        sn = ((torch.from_numpy(Tn).float() - mu) / sd) @ w + b
    return float(auc_per_channel(sp[:, None].numpy(), sn[:, None].numpy())[0])


# ------------------------------------------------------------------ web
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Layer viewer</title>
<style>
:root{--bg:#15171a;--panel:#1d2024;--line:#30343a;--text:#e6e6e6;--muted:#9aa0a6;--acc:#4cc38a;--warn:#e5a50a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,sans-serif;height:100vh;display:flex}
#left{width:270px;border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
#left h3,#right h3{margin:10px 12px 6px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.list{overflow:auto;flex:1;min-height:0;border-top:1px solid var(--line)}
.row{padding:5px 12px;cursor:pointer;border-bottom:1px solid #23262b;display:flex;gap:6px}.row:hover{background:#23272c}.row.sel{background:#2a3a33}
.row .n{margin-left:auto;color:var(--muted)}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#tabs{padding:8px 10px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:4px}
.tab{padding:3px 8px;border:1px solid #3a4047;border-radius:4px;cursor:pointer;background:#22262b}.tab.sel{background:#24563f;border-color:#2f6e51}
#body{flex:1;overflow:auto;padding:10px;display:flex;gap:12px;align-items:flex-start}
#big{flex:0 0 auto}#big img{width:520px;image-rendering:pixelated;border:1px solid var(--line);background:#000}
#thumbs{display:flex;flex-wrap:wrap;gap:6px;align-content:flex-start;flex:1}
.th{cursor:pointer;text-align:center;font-size:11px;color:var(--muted)}.th img{width:120px;height:120px;object-fit:contain;border:1px solid var(--line);background:#000;display:block}
.th.sel img{border-color:var(--acc)}
#right{width:400px;border-left:1px solid var(--line);overflow:auto;padding-bottom:20px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px;margin:8px 10px}
button,select,input{background:#2b3036;color:var(--text);border:1px solid #3a4047;border-radius:5px;padding:4px 8px;font:inherit}
table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
td,th{padding:3px 4px;border-bottom:1px solid #2a2e33;text-align:right}th{color:var(--muted);font-weight:500}td:first-child,th:first-child{text-align:left}
tr.click{cursor:pointer}tr.click:hover{background:#262a2f}
.bar{display:inline-block;height:8px;background:var(--acc);vertical-align:middle;border-radius:2px}
.muted{color:var(--muted)}#desc{color:var(--muted);margin:6px 0 0}
</style></head><body>
<div id="left"><h3>Crops <span id="ncrops" class="muted"></span></h3><div class="list" id="crops"></div></div>
<div id="main">
  <div id="tabs"></div>
  <div style="padding:6px 10px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    <b id="title"></b><span class="muted" id="shape"></span>
    <span style="margin-left:auto" class="muted">sort channels
      <select id="sort"><option value="contrast">ship AUC, this crop</option><option value="auc">val AUC (after analysis)</option><option value="index">index</option></select>
      show <select id="nth"><option>24</option><option selected>48</option><option>96</option><option>256</option></select>
      <label><input type="checkbox" id="ov" checked> boxes</label></span>
  </div>
  <div id="body"><div id="big"><img id="bigimg"><div id="bigcap" class="muted"></div><div id="desc"></div></div><div id="thumbs"></div></div>
</div>
<div id="right">
  <div class="box"><h3 style="margin:0 0 6px">Model</h3><div id="model" class="muted"></div></div>
  <div class="box"><h3 style="margin:0 0 6px">Val analysis</h3>
    <div>ship crops <input id="nan" type="number" value="40" style="width:60px"> <button id="run">Run</button> <span id="prog" class="muted"></span></div>
    <div class="muted" style="margin-top:6px">Per stage, ship vs background cells on a 4 m ground grid. <b>probe</b> = linear probe test AUC (train/test on different crops); <b>best ch</b> = best single channel max(AUC, 1-AUC); <b>&gt;0.7</b> = channels separating that well.</div>
    <div id="stats" style="margin-top:8px"></div></div>
  <div class="box muted">Green = ship (yellow = ignored); in radar-frame stages circles mark ship centres mapped into the transform bins. Blue boxes = detections &ge; 0.3. Colours are scaled per map (1st-99.5th percentile).</div>
</div>
<script>
const $=id=>document.getElementById(id);
let S={crops:[],crop:null,stages:[],stage:null,ch:'mean',stats:null};
async function api(p){const r=await fetch(p);const j=await r.json();if(!r.ok||j.error)throw new Error(j.error||r.statusText);return j}
async function init(){const m=await api('/api/info');$('model').innerHTML=m.html;S.crops=m.crops;$('ncrops').textContent=`(${m.crops.length})`;
  const el=$('crops');el.innerHTML='';m.crops.forEach(c=>{const d=document.createElement('div');d.className='row';d.dataset.i=c.i;
    const m=c.crop_id.match(/^(\d{4}-\d{2}-\d{2}).*?UMBRA-\d+_(.*)$/);const nm=m?`${m[1]} ${m[2]}`:c.crop_id.slice(-26);
    d.innerHTML=`<span>${nm}</span><span class="n">${c.n_pos} ship${c.n_pos==1?'':'s'}</span>`;d.title=c.crop_id;d.onclick=()=>pick(c.i);el.appendChild(d)});
  S.stats=m.stats;renderStats();if(m.crops.length)pick(m.crops[0].i)}
async function pick(i){S.crop=i;document.querySelectorAll('#crops .row').forEach(r=>r.classList.toggle('sel',+r.dataset.i===i));
  $('title').textContent='running the model...';const j=await api('/api/crop?i='+i);S.stages=j.stages;
  if(!S.stage||!S.stages.find(s=>s.name===S.stage))S.stage=S.stages[0].name;tabs();showStage()}
function tabs(){const el=$('tabs');el.innerHTML='';[{name:'learned front end'}].concat(S.stages).forEach(s=>{const b=document.createElement('span');
  b.className='tab'+(s.name===S.stage?' sel':'');b.textContent=s.name;b.onclick=()=>{S.stage=s.name;S.ch='mean';tabs();showStage()};el.appendChild(b)})}
function url(ch,size){return `/api/img?i=${S.crop}&stage=${encodeURIComponent(S.stage)}&ch=${ch}&size=${size}&ov=${$('ov').checked?1:0}`}
async function showStage(){
  if(S.stage==='learned front end'){$('title').textContent='learned front end';$('shape').textContent='';$('bigimg').src='/api/front';$('bigimg').style.width='900px';
    $('bigcap').textContent='';$('desc').textContent='Tapers over pulses and frequencies, and how much each output bin of the two DFT matrices moved from the plain DFT (0 = unchanged; all 0 with --fixed-dft).';$('thumbs').innerHTML='';return}
  $('bigimg').style.width='520px';
  const st=S.stages.find(s=>s.name===S.stage);$('title').textContent=st.name;$('shape').textContent=`${st.c} ch x ${st.h} x ${st.w}  (${st.frame} frame)`;$('desc').textContent=st.desc;
  let order=[...Array(st.c).keys()];const mode=$('sort').value;
  if(mode==='contrast'&&st.contrast)order.sort((a,b)=>Math.abs(st.contrast[b]-.5)-Math.abs(st.contrast[a]-.5));
  if(mode==='auc'&&S.stats){const s=S.stats.stages.find(x=>x.stage===st.name);if(s)order.sort((a,b)=>Math.abs(s.auc[b]-.5)-Math.abs(s.auc[a]-.5))}
  const n=+$('nth').value;const el=$('thumbs');el.innerHTML='';
  const vs=S.stats&&S.stats.stages.find(x=>x.stage===st.name);
  ['mean','max'].concat(order.slice(0,n)).forEach(ch=>{const d=document.createElement('div');d.className='th'+(String(ch)===String(S.ch)?' sel':'');
    let lab=typeof ch==='number'?(st.names?st.names[ch]:'ch '+ch):ch;
    if(typeof ch==='number'){if(st.contrast)lab+=` <span title="ship AUC, this crop">c ${st.contrast[ch].toFixed(2)}</span>`;if(vs)lab+=` <span title="val AUC">a ${vs.auc[ch].toFixed(2)}</span>`}
    d.innerHTML=`<img loading="lazy" src="${url(ch,120)}"><div>${lab}</div>`;d.onclick=()=>{S.ch=ch;big();document.querySelectorAll('.th').forEach(t=>t.classList.remove('sel'));d.classList.add('sel')};el.appendChild(d)});
  big()}
function big(){const st=S.stages.find(s=>s.name===S.stage);$('bigimg').src=url(S.ch,520);
  let cap=typeof S.ch==='number'?(st.names?st.names[S.ch]:'channel '+S.ch):S.ch+' over channels';
  if(typeof S.ch==='number'&&st.contrast)cap+=` | ship AUC this crop ${st.contrast[S.ch].toFixed(3)}`;
  const vs=S.stats&&S.stats.stages.find(x=>x.stage===st.name);if(typeof S.ch==='number'&&vs)cap+=` | val AUC ${vs.auc[S.ch].toFixed(3)}`;$('bigcap').textContent=cap}
$('sort').onchange=showStage;$('nth').onchange=showStage;$('ov').onchange=showStage;
function renderStats(){const el=$('stats');if(!S.stats){el.innerHTML='<span class="muted">not run yet</span>';return}
  let h=`<div class="muted">${S.stats.ship_crops} ship crops + ${S.stats.crops-S.stats.ship_crops} empty, epoch ${S.stats.epoch}, ${S.stats.time}</div><table><tr><th>stage</th><th>ch</th><th>probe</th><th>best ch</th><th>median</th><th>&gt;0.7</th></tr>`;
  S.stats.stages.forEach(s=>{const p=s.probe_auc==null?'-':s.probe_auc.toFixed(3);
    h+=`<tr class="click" data-s="${s.stage}"><td>${s.stage}</td><td>${s.channels}</td><td>${p} <span class="bar" style="width:${s.probe_auc?Math.max(0,(s.probe_auc-.5)*120):0}px"></span></td><td>${s.best_sep.toFixed(3)}</td><td>${s.median_sep.toFixed(3)}</td><td>${s.n_above_07}</td></tr>`});
  el.innerHTML=h+'</table>';el.querySelectorAll('tr.click').forEach(r=>r.onclick=()=>{S.stage=r.dataset.s;S.ch='mean';$('sort').value='auc';tabs();showStage()})}
$('run').onclick=async()=>{await api('/api/analyse?n='+$('nan').value);poll()};
async function poll(){const j=await api('/api/job');$('prog').textContent=j.running?`${j.done}/${j.total} crops`:j.msg;
  if(j.running)setTimeout(poll,1500);else{const m=await api('/api/info');S.stats=m.stats;renderStats();if(S.crop!=null)showStage()}}
init();
</script></body></html>"""


def make_handler(v):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: x[0] for k, x in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
                if u.path == "/api/info":
                    c = v.cfg
                    html = (f"arch <b>{v.arch}</b>" + (f" ({c.get('detector')}, {c.get('det_backbone')})" if v.arch == "det" else "") +
                            f", epoch {v.epoch}<br>{os.path.basename(os.path.dirname(os.path.abspath(v.a.ckpt)))}/"
                            f"{os.path.basename(v.a.ckpt)}<br>DFT {'fixed' if c.get('fixed_dft') else 'learnable'}; split {v.a.split}")
                    crops = [dict(i=i, crop_id=v.ds.meta[i]["crop_id"], n_pos=int(v.ds.meta[i].get("n_pos", 0))) for i in v.order]
                    return self.send(200, dict(html=html, crops=crops, stats=v.stats))
                if u.path == "/api/crop":
                    rec = v.taps(int(q["i"]))
                    st = []
                    for name, s in rec["stages"].items():
                        con = v.contrast(rec, name)
                        st.append(dict(name=name, frame=s["frame"], desc=s["desc"], c=int(s["t"].shape[0]),
                                       h=int(s["t"].shape[1]), w=int(s["t"].shape[2]),
                                       contrast=[round(float(x), 3) for x in con] if con is not None else None,
                                       names=FCH if name in ("Fourier features", "features on the ground") else
                                       (["east", "north"] if name in ("offset", "size") else None)))
                    return self.send(200, dict(stages=st))
                if u.path == "/api/img":
                    rec = v.taps(int(q["i"]))
                    return self.send(200, v.render(rec, q["stage"], q["ch"], int(q.get("size", 256)), q.get("ov", "1") == "1"),
                                     "image/png")
                if u.path == "/api/front":
                    return self.send(200, v.front_plot(), "image/png")
                if u.path == "/api/analyse":
                    if not v.job["running"]:
                        threading.Thread(target=v.analyse, args=(int(q.get("n", 40)),), daemon=True).start()
                        time.sleep(0.2)
                    return self.send(200, v.job)
                if u.path == "/api/job":
                    return self.send(200, v.job)
                return self.send(404, {"error": "not found"})
            except Exception as e:
                traceback.print_exc()
                return self.send(400, {"error": f"{type(e).__name__}: {e}"})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--ann-root", default=None)
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="the run's own scene split, from config.json next to --ckpt")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--work", default=None, help="analysis results (default <run>/layer_viewer)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    v = Viewer(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(v))
    print(f"open http://localhost:{a.port}  (from the laptop: ssh -L {a.port}:localhost:{a.port} <server>)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
