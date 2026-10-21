"""Box adjust server: fix the label/image offset scene by scene, in a web browser, on the server where the crops live.

Idea: within one scene the labels are off by one translation (a height offset moves everything the same way), so
  1. open a positive chip of a scene: its image is formed from the chip CPHD (on the crop's h_ref plane, as in training)
  2. drag one box onto its ship (all boxes move together)  ->  "Save measurement"           (key s)
  3. open another chip of the same scene: the boxes start at the scene translation. Fits? -> "Save measurement" again
     (or nudge first). The table shows every measurement and how far it is from the scene median.
  4. convinced -> "Apply to scene": every crop JSON of the scene AND the scene's annotations.json (training reads the
     boxes from --ann-root, not from the crop JSONs) get the boxes moved by the scene median. Crop JSONs are rewritten
     in place, old ones kept as <name>_pre_adjustment.json (manifest.jsonl -> manifest_pre_adjustment.jsonl). The
     master annotations (--ann-root) are only read: the moved copy goes to --ann-out/<scene>_GEC/annotations.json.
     At start every scene's master is copied there unchanged, so  train_raw.py --ann-root <ann-out>  sees all scenes.
     Boxes are re-clipped to each crop, so visible / ignore / n_pos / label are recomputed; a ship that moves into a
     crop is added. "Undo" puts the _pre_adjustment files back.

The translation is stored and applied as metres east / north (the same for every box). The panel also shows the part
along the radar, the height offset it implies (along / tan(graze)) and the across-range part height cannot explain.

Run on the server (inside the cphd-gpu container, or any Python with numpy sarkit pillow; torch makes images fast):
    python box_adjust_server.py --crops /data/crops --ann-root /data/ann --ann-out /data/ann_adjusted --device cuda:0
then on your laptop:  ssh -L 8765:localhost:8765 user@server   and open  http://localhost:8765
Docker: add  -p 127.0.0.1:8765:8765  and  --host 0.0.0.0, mount the crops and --ann-out read-write, the master --ann-root read-only (:ro).

State (measurements, applied translations) and the image cache go to <crops>/_box_adjust/ (--work to change).
"""
import argparse, copy, glob, io, json, math, os, re, sys, threading, time, traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
if int(np.__version__.split(".")[0]) < 2:              # sarkit needs np.asarray(copy=...): add it on numpy 1.x
    _np_asarray = np.asarray

    def _asarray_compat(a, dtype=None, order=None, *, copy=None, **kw):
        if copy:
            return np.array(a, dtype=dtype, order=order, copy=True, **kw)
        return _np_asarray(a, dtype=dtype, order=order, **kw)

    np.asarray = _asarray_compat
from PIL import Image

A, E2, C = 6378137.0, 6.69437999014e-3, 299792458.0
SCENE_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_UMBRA-\d+")
PNG_DB = (-10.0, 40.0)                                   # dB range stored in the cached PNG (the page re-maps it)
BACKUP = "_pre_adjustment"


def m_per_deg(lat):
    s = math.sin(math.radians(lat))
    return (math.pi / 180 * A * (1 - E2) / (1 - E2 * s * s) ** 1.5,
            math.pi / 180 * A / math.sqrt(1 - E2 * s * s) * math.cos(math.radians(lat)))


def llh_to_ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


def read_json(p):
    with open(p) as f:
        return json.load(f)


def write_json(p, d):
    with open(p + ".part", "w") as f:
        json.dump(d, f, indent=1, default=float)
    os.replace(p + ".part", p)


# ------------------------------------------------------------------ image formation (backprojection on the h_ref plane)
def read_cphd(path):
    import sarkit.cphd as skcphd
    with open(path, "rb") as f, skcphd.Reader(f) as r:
        xml = r.metadata.xmltree
        ch = xml.find("{*}Data/{*}Channel").findtext("{*}Identifier")
        sgn = int(xml.findtext("{*}Global/{*}SGN"))
        pv = r.read_pvps(ch)
        sig = r.read_signal(ch)
    X = (sig["real"] + 1j * sig["imag"]) if sig.dtype.names else sig.astype(np.complex128)
    srp = np.asarray(pv["SRPPos"][0], float)
    return dict(X=np.asarray(X, np.complex128), sgn=sgn, srp=srp, tx=np.asarray(pv["TxPos"], float) - srp,
                rc=np.asarray(pv["RcvPos"], float) - srp, f0=np.asarray(pv["SC0"], float), scss=float(pv["SCSS"][0]))


def view_geometry(d, lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
    up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    m = len(d["tx"]) // 2
    u = d["tx"][m] + d["rc"][m]
    u /= np.linalg.norm(u)                                 # crop centre -> radar
    return math.degrees(math.asin(u @ up)), math.degrees(math.atan2(u @ east, u @ north)) % 360


def backproject(d, pix, device):
    """|image| at ECEF points pix (N, 3, relative to the SRP). torch (GPU or CPU) when available, else numpy."""
    X = np.conj(d["X"]) if d["sgn"] > 0 else d["X"].copy()
    P, S = X.shape
    X *= np.hanning(P)[:, None] * np.hanning(S)[None, :]
    K = 4 * S
    prof = np.fft.fftshift(np.fft.ifft(X, n=K, axis=1), axes=1)
    try:
        import torch
        dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
        T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)
        px, tx, rc, f0 = T(pix), T(d["tx"]), T(d["rc"]), T(d["f0"])
        pr, pi = T(prof.real), T(prof.imag)
        pp = (px * px).sum(1)
        acc_r = torch.zeros(len(pix), dtype=torch.float64, device=dev)
        acc_i = torch.zeros_like(acc_r)
        B = 32 if dev.type == "cuda" else 8

        def dd(t):                                          # path to pixel minus path to SRP, stable form, (B, N)
            tt = (t * t).sum(1, keepdim=True)
            num = pp[None] - 2 * (t @ px.T)
            return num / (torch.sqrt(tt + num) + torch.sqrt(tt))
        for p0 in range(0, P, B):
            sl = slice(p0, min(P, p0 + B))
            dtau = (dd(tx[sl]) + dd(rc[sl])) / C
            k = dtau * K * d["scss"] + K // 2
            k0 = torch.clamp(torch.floor(k).long(), 0, K - 2)
            w = k - k0
            ok = ((k >= 0) & (k < K - 1)).double()
            vr = pr[sl].gather(1, k0) * (1 - w) + pr[sl].gather(1, k0 + 1) * w
            vi = pi[sl].gather(1, k0) * (1 - w) + pi[sl].gather(1, k0 + 1) * w
            ph = 2 * math.pi * f0[sl, None] * dtau
            c, s = torch.cos(ph), torch.sin(ph)
            acc_r += ((vr * c - vi * s) * ok).sum(0)
            acc_i += ((vr * s + vi * c) * ok).sum(0)
        return torch.sqrt(acc_r ** 2 + acc_i ** 2).cpu().numpy()
    except ImportError:
        pass
    pp = (pix ** 2).sum(1)

    def dd(t):
        num = pp - 2 * (pix @ t)
        return num / (np.sqrt(t @ t + num) + np.sqrt(t @ t))
    img = np.zeros(len(pix), complex)
    for p in range(P):
        dtau = (dd(d["tx"][p]) + dd(d["rc"][p])) / C
        k = dtau * K * d["scss"] + K // 2
        k0 = np.clip(np.floor(k).astype(int), 0, K - 2)
        w = k - k0
        val = prof[p, k0] * (1 - w) + prof[p, k0 + 1] * w
        img += np.where((k >= 0) & (k < K - 1), val * np.exp(2j * np.pi * d["f0"][p] * dtau), 0)
    return np.abs(img)


# ------------------------------------------------------------------ labels (same rule as build_crops.crop_labels)
def crop_labels(g, anns, min_visible):
    W, S, E, N = g["bounds_geo"]
    boxes = []
    for a in anns:
        x0, x1 = max(a["lon_min"], W), min(a["lon_max"], E)
        y0, y1 = max(a["lat_min"], S), min(a["lat_max"], N)
        if x1 <= x0 or y1 <= y0:
            continue
        area = (a["lon_max"] - a["lon_min"]) * (a["lat_max"] - a["lat_min"])
        vis = (x1 - x0) * (y1 - y0) / area if area > 0 else 1.0
        b = dict(ann_id=a["id"], category_id=a["category_id"],
                 bbox_geo=[a["lon_min"], a["lat_min"], a["lon_max"], a["lat_max"]],
                 bbox_geo_clipped=[x0, y0, x1, y1],
                 bbox_px=[(x0 - g["lon_w"]) / g["dlon"], (g["lat_n"] - y1) / g["dlat"],
                          (x1 - g["lon_w"]) / g["dlon"], (g["lat_n"] - y0) / g["dlat"]],
                 visible=float(vis), ignore=bool(vis < min_visible or a["iscrowd"]))
        if a.get("orig"):
            b["bbox_geo_original"] = a["orig"]
        boxes.append(b)
    return boxes


def ann_list(js):
    out = []
    for a in js.get("annotations", []):
        g = a.get("bbox_geo")
        if g and len(g) == 4:
            lo0, la0, lo1, la1 = map(float, g)
            out.append(dict(id=a.get("id"), category_id=a.get("category_id", 1), iscrowd=int(a.get("iscrowd", 0) or 0),
                            lon_min=min(lo0, lo1), lat_min=min(la0, la1), lon_max=max(lo0, lo1), lat_max=max(la0, la1),
                            orig=a.get("bbox_geo_original")))
    return out


def shift_geo(b, east_m, north_m):
    lo0, la0, lo1, la1 = b
    ml, mo = m_per_deg(0.5 * (la0 + la1))
    return [lo0 + east_m / mo, la0 + north_m / ml, lo1 + east_m / mo, la1 + north_m / ml]


def metrics(east_m, north_m, graze, az):
    a = math.radians(az)
    along = east_m * math.sin(a) + north_m * math.cos(a)
    across = -east_m * math.cos(a) + north_m * math.sin(a)
    return dict(along_radar_m=along, across_m=across, height_offset_m=along / math.tan(math.radians(graze)))


# ------------------------------------------------------------------ the data and its state
class Store:
    def __init__(self, a):
        self.crops, self.ann_root, self.a = os.path.abspath(a.crops), a.ann_root, a
        self.ann_out = os.path.abspath(a.ann_out)            # adjusted copies; the master annotations are only read
        if self.ann_root and os.path.abspath(self.ann_root) == self.ann_out:
            raise SystemExit("--ann-out must not be the master --ann-root")
        self.work = a.work or os.path.join(self.crops, "_box_adjust")
        self.cache = os.path.join(self.work, "cache")
        os.makedirs(self.cache, exist_ok=True)
        self.state_path = os.path.join(self.work, "translations.json")
        self.state = read_json(self.state_path) if os.path.exists(self.state_path) else {}
        self.lock, self.render_lock = threading.RLock(), threading.Lock()
        self._meta = {}

    def save_state(self):
        write_json(self.state_path, self.state)

    def sdir(self, scene):
        p = os.path.join(self.crops, scene)
        if os.path.dirname(os.path.abspath(p)) != self.crops or not os.path.isdir(p):
            raise ValueError(f"no scene folder {scene}")
        return p

    def crop_jsons(self, scene):
        out = []
        for p in sorted(glob.glob(os.path.join(self.sdir(scene), "*.json"))):
            b = os.path.basename(p)
            if b.startswith("manifest") or b.endswith(BACKUP + ".json"):
                continue
            out.append(p)
        return out

    def meta(self, p):
        mt = os.path.getmtime(p)
        hit = self._meta.get(p)
        if hit and hit[0] == mt:
            return hit[1]
        d = read_json(p)
        d = d if isinstance(d, dict) and "grid" in d and "crop_id" in d else None
        self._meta[p] = (mt, d)
        return d

    def applied(self, scene):
        return bool(glob.glob(os.path.join(self.sdir(scene), "*" + BACKUP + ".json")))

    def scene_translation(self, scene):
        ms = list(self.state.get(scene, {}).get("measurements", {}).values())
        if not ms:
            return None
        e, n = np.array([m["east_m"] for m in ms]), np.array([m["north_m"] for m in ms])
        return dict(east_m=float(np.median(e)), north_m=float(np.median(n)), n=len(ms),
                    spread_m=float(np.hypot(e - np.median(e), n - np.median(n)).max()))

    def scenes(self):
        out = []
        for d in sorted(os.listdir(self.crops)):
            p = os.path.join(self.crops, d)
            if d.startswith("_") or not os.path.isdir(p):
                continue
            metas = [m for m in (self.meta(j) for j in self.crop_jsons(d)) if m]
            if not metas:
                continue
            out.append(dict(scene=d, chips=len(metas), positive=sum(m["label"] == "positive" for m in metas),
                            ships=len({b["ann_id"] for m in metas for b in m["boxes"]}), applied=self.applied(d),
                            translation=self.scene_translation(d),
                            applied_translation=self.state.get(d, {}).get("applied")))
        return out

    def chips(self, scene):
        ms = self.state.get(scene, {}).get("measurements", {})
        out = []
        for j in self.crop_jsons(scene):
            m = self.meta(j)
            if m:
                out.append(dict(crop_id=m["crop_id"], kind=m.get("kind"), label=m["label"], n_pos=m.get("n_pos", 0),
                                n_ignore=m.get("n_ignore", 0), measured=m["crop_id"] in ms,
                                cached=os.path.exists(self.cache_png(m["crop_id"]))))
        out.sort(key=lambda c: (c["label"] != "positive", -c["n_pos"], c["crop_id"]))
        return out

    def cache_png(self, crop_id):
        return os.path.join(self.cache, f"{crop_id}_px{self.a.pixel_m:g}_m{self.a.margin_m:g}.png")

    def chip(self, scene, crop_id):
        jp = os.path.join(self.sdir(scene), crop_id + ".json")
        m = self.meta(jp)
        if m is None:
            raise ValueError(f"not a crop JSON: {jp}")
        png, geo = self.cache_png(crop_id), self.cache_png(crop_id)[:-4] + ".json"
        if not (os.path.exists(png) and os.path.exists(geo)):
            with self.render_lock:
                if not (os.path.exists(png) and os.path.exists(geo)):
                    self.render(scene, m, png, geo)
        g = read_json(geo)
        ml, mo = m_per_deg(m["centre_lat"])

        def to_px(b):
            lo0, la0, lo1, la1 = b
            return [(lo0 - g["lon_w"]) * mo / g["pixel_m"], (g["lat_n"] - la1) * ml / g["pixel_m"],
                    (lo1 - g["lon_w"]) * mo / g["pixel_m"], (g["lat_n"] - la0) * ml / g["pixel_m"]]
        applied = self.applied(scene)
        boxes = [dict(ann_id=b.get("ann_id"), ignore=bool(b.get("ignore")), visible=b.get("visible", 1.0),
                      px=to_px(b["bbox_geo"]), px_orig=to_px(b["bbox_geo_original"]) if b.get("bbox_geo_original") else None)
                 for b in m["boxes"]]
        c0 = (m["grid"]["lon_w"] - g["lon_w"]) * mo / g["pixel_m"]
        r0 = (g["lat_n"] - m["grid"]["lat_n"]) * ml / g["pixel_m"]
        side = m["grid"]["n"] * m["grid"]["pixel_m"] / g["pixel_m"]
        core = [c0, r0, c0 + side, r0 + side]                # the labelled 512 m core inside the wider image
        st = self.state.get(scene, {})
        return dict(scene=scene, crop_id=crop_id, kind=m.get("kind"), label=m["label"], n_pos=m.get("n_pos"),
                    h_ref=m.get("h_ref"), centre_lat=m["centre_lat"], centre_lon=m["centre_lon"], geo=g, boxes=boxes,
                    core_px=core, applied=applied, applied_translation=st.get("applied"),
                    measurement=st.get("measurements", {}).get(crop_id), translation=self.scene_translation(scene),
                    image=f"/api/image?scene={scene}&crop_id={crop_id}&v={int(os.path.getmtime(png))}")

    def render(self, scene, m, png, geo):
        t0 = time.time()
        d = read_cphd(os.path.join(self.sdir(scene), m["cphd"]))
        g = m["grid"]
        ml, mo = m_per_deg(m["centre_lat"])
        px, mg = self.a.pixel_m, self.a.margin_m
        n = int(round((g["n"] * g["pixel_m"] + 2 * mg) / px))
        lon_w, lat_n = g["lon_w"] - mg / mo, g["lat_n"] + mg / ml
        lats = lat_n - (np.arange(n) + 0.5) * px / ml
        lons = lon_w + (np.arange(n) + 0.5) * px / mo
        LON, LAT = np.meshgrid(lons, lats)
        h = float(m["h_ref"])
        pix = llh_to_ecef(LAT, LON, np.full(LAT.shape, h)).reshape(-1, 3) - d["srp"]
        amp = backproject(d, pix, self.a.device).reshape(n, n)
        db = 20 * np.log10(np.maximum(amp, 1e-30) / max(np.median(amp), 1e-30))
        u8 = np.clip((db - PNG_DB[0]) / (PNG_DB[1] - PNG_DB[0]) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(u8).save(png + ".part.png")
        os.replace(png + ".part.png", png)
        graze, az = view_geometry(d, m["centre_lat"], m["centre_lon"])
        write_json(geo, dict(n=n, pixel_m=px, lon_w=lon_w, lat_n=lat_n, h_plane=h, graze_deg=graze, radar_az_deg=az,
                             png_db=PNG_DB))
        print(f"  formed {m['crop_id']} ({n} x {n} px, {time.time() - t0:.1f} s)", flush=True)

    # ---- measurements
    def measure(self, scene, crop_id, east_m, north_m):
        with self.lock:
            if self.applied(scene):
                raise ValueError("this scene is already applied: Undo first to change it")
            c = self.chip(scene, crop_id)
            g = c["geo"]
            rec = dict(east_m=float(east_m), north_m=float(north_m), graze_deg=g["graze_deg"],
                       radar_az_deg=g["radar_az_deg"], h_ref=c["h_ref"], time=time.strftime("%Y-%m-%d %H:%M:%S"),
                       **metrics(float(east_m), float(north_m), g["graze_deg"], g["radar_az_deg"]))
            self.state.setdefault(scene, {}).setdefault("measurements", {})[crop_id] = rec
            self.save_state()
            return self.measurements(scene)

    def unmeasure(self, scene, crop_id):
        with self.lock:
            self.state.get(scene, {}).get("measurements", {}).pop(crop_id, None)
            self.save_state()
            return self.measurements(scene)

    def measurements(self, scene):
        t = self.scene_translation(scene)
        ms = self.state.get(scene, {}).get("measurements", {})
        rows = []
        for k, v in sorted(ms.items()):
            r = dict(crop_id=k, **v)
            if t:
                r["residual_m"] = math.hypot(v["east_m"] - t["east_m"], v["north_m"] - t["north_m"])
            rows.append(r)
        out = dict(scene=scene, translation=t, rows=rows, applied=self.applied(scene),
                   applied_translation=self.state.get(scene, {}).get("applied"))
        if t and rows:
            out["translation_metrics"] = metrics(t["east_m"], t["north_m"], np.median([r["graze_deg"] for r in rows]),
                                                 np.median([r["radar_az_deg"] for r in rows]))
            out["h_ref"] = rows[0].get("h_ref")
        return out

    # ---- apply / undo
    def scene_ann_path(self, scene, metas):
        cands = [m.get("annotations") for m in metas[:1]]
        if self.ann_root:
            cands += [os.path.join(self.ann_root, f"{scene}_GEC", n) for n in ("annotations.json", "annotation.json")]
            cands += sorted(glob.glob(os.path.join(self.ann_root, f"*{scene}*", "annotation*.json")))
        for p in cands:
            if p and os.path.isfile(p) and not p.endswith(BACKUP + ".json"):
                return p
        return None

    def adjusted_path(self, master):
        """<ann-out>/<same scene folder name as the master>/<same file name>"""
        return os.path.join(self.ann_out, os.path.basename(os.path.dirname(master)), os.path.basename(master))

    def master_for(self, scene):
        js = self.crop_jsons(scene)
        m = self.meta(js[0]) if js else None
        ap = self.scene_ann_path(scene, [m] if m else [])
        if ap and os.path.abspath(ap).startswith(self.ann_out + os.sep):
            raise ValueError(f"the master annotation path {ap} is inside --ann-out")
        return ap

    def mirror(self):
        """Copy every scene's master annotations (unchanged) into --ann-out if it has none yet, so that training with
        --ann-root <ann-out> sees all scenes: adjusted ones moved, the others as in the master."""
        n = 0
        for d in sorted(os.listdir(self.crops)):
            if d.startswith("_") or not os.path.isdir(os.path.join(self.crops, d)):
                continue
            try:
                ap = self.master_for(d)
            except Exception as e:
                print(f"  {d}: {e}"); continue
            if ap and not os.path.exists(self.adjusted_path(ap)):
                os.makedirs(os.path.dirname(self.adjusted_path(ap)), exist_ok=True)
                write_json(self.adjusted_path(ap), read_json(ap))
                n += 1
        return n

    def apply(self, scene):
        with self.lock:
            if self.applied(scene):
                raise ValueError("already applied (there are _pre_adjustment files): Undo first")
            t = self.scene_translation(scene)
            if not t:
                raise ValueError("no measurements for this scene yet")
            E_, N_ = t["east_m"], t["north_m"]
            jps = self.crop_jsons(scene)
            metas = [(p, self.meta(p)) for p in jps]
            metas = [(p, m) for p, m in metas if m]
            rec = dict(east_m=E_, north_m=N_, from_measurements=t["n"], spread_m=t["spread_m"],
                       time=time.strftime("%Y-%m-%d %H:%M:%S"),
                       note="every bbox_geo moved by east_m / north_m (metres); originals in bbox_geo_original and in "
                            "the *_pre_adjustment files")
            ap = self.master_for(scene)
            done, out_ann = [], None
            if ap:                                         # 1. scene annotations: read the master, write the copy
                js = read_json(ap)
                root = js[0] if isinstance(js, list) else js
                for a in root.get("annotations", []):
                    if a.get("bbox_geo") and len(a["bbox_geo"]) == 4:
                        a["bbox_geo_original"] = list(a["bbox_geo"])
                        a["bbox_geo"] = shift_geo(a["bbox_geo"], E_, N_)
                root["box_translation"] = dict(rec, master_annotations=ap)
                out_ann = self.adjusted_path(ap)
                os.makedirs(os.path.dirname(out_ann), exist_ok=True)
                write_json(out_ann, js)                    # the master file is never written
                done.append(out_ann)
                anns = ann_list(root)
            else:                                          # no annotations.json found: use every box seen in the crops
                seen = {}
                for _, m in metas:
                    for b in m["boxes"]:
                        seen.setdefault(b["ann_id"], b)
                anns = []
                for k, b in seen.items():
                    lo0, la0, lo1, la1 = shift_geo(b["bbox_geo"], E_, N_)
                    anns.append(dict(id=k, category_id=b.get("category_id", 1), iscrowd=0, lon_min=lo0, lat_min=la0,
                                     lon_max=lo1, lat_max=la1, orig=b["bbox_geo"]))
            ships = [a for a in anns if not a["iscrowd"]]
            for p, m in metas:                             # 2. every crop JSON: boxes re-clipped from the moved labels
                new = copy.deepcopy(m)
                new["boxes"] = crop_labels(m["grid"], anns, self.a.min_visible)
                new["n_pos"] = sum(not b["ignore"] for b in new["boxes"])
                new["n_ignore"] = sum(b["ignore"] for b in new["boxes"])
                new["label"] = "positive" if new["n_pos"] else "negative"
                if ships:
                    ml, mo = m_per_deg(m["centre_lat"])
                    new["nearest_ship_m"] = float(min(math.hypot((0.5 * (a["lat_min"] + a["lat_max"]) - m["centre_lat"]) * ml,
                                                                 (0.5 * (a["lon_min"] + a["lon_max"]) - m["centre_lon"]) * mo)
                                                      for a in ships))
                new["box_translation"] = dict(rec, label_before=m["label"], n_pos_before=m.get("n_pos"),
                                              annotations_adjusted=out_ann)
                os.replace(p, p[:-5] + BACKUP + ".json")
                write_json(p, new)
                done.append(p)
            self.write_manifests(scene, backup=True)
            self.state.setdefault(scene, {})["applied"] = dict(rec, master_annotations=ap, annotations=out_ann,
                                                               files=len(done))
            self.save_state()
            changed = sum(1 for p, m in metas if self.meta(p)["label"] != m["label"])
            return dict(scene=scene, files=len(done), annotations=out_ann, label_changed=changed,
                        warning=None if ap else "no master annotations.json found (set --ann-root): only the crop "
                                                "JSONs were changed")

    def undo(self, scene):
        with self.lock:
            sd = self.sdir(scene)
            bks = glob.glob(os.path.join(sd, "*" + BACKUP + ".json"))
            if not bks:
                raise ValueError("nothing to undo")
            for b in bks:
                os.replace(b, b[: -len(BACKUP + ".json")] + ".json")
            ap = self.master_for(scene)                    # adjusted copy back to the master's boxes
            if ap:
                write_json(self.adjusted_path(ap), read_json(ap))
            mb = os.path.join(sd, "manifest" + BACKUP + ".jsonl")
            if os.path.exists(mb):
                os.replace(mb, os.path.join(sd, "manifest.jsonl"))
            self.write_manifests(scene, backup=False, scene_too=False)
            self.state.get(scene, {}).pop("applied", None)
            self.save_state()
            return dict(scene=scene, restored=len(bks))

    def write_manifests(self, scene, backup, scene_too=True):
        sd = self.sdir(scene)
        keys = ("crop_id", "scene", "kind", "label", "n_pos", "n_ignore", "centre_lat", "centre_lon", "nearest_ship_m")
        lines = []
        for p in self.crop_jsons(scene):
            d = self.meta(p)
            if d:
                lines.append({k: d.get(k) for k in keys} | dict(cphd=os.path.join(scene, d["cphd"]),
                                                                  json=os.path.join(scene, d["crop_id"] + ".json")))
        sm = os.path.join(sd, "manifest.jsonl")
        if scene_too:
            if backup and os.path.exists(sm):
                os.replace(sm, os.path.join(sd, "manifest" + BACKUP + ".jsonl"))
            with open(sm + ".part", "w") as f:
                f.writelines(json.dumps(l) + "\n" for l in lines)
            os.replace(sm + ".part", sm)
        gm = os.path.join(self.crops, "manifest.jsonl")   # whole-run manifest: replace this scene's lines
        if os.path.exists(gm):
            gb = os.path.join(self.crops, "manifest" + BACKUP + ".jsonl")
            if backup and not os.path.exists(gb):
                with open(gm) as f, open(gb, "w") as g:
                    g.write(f.read())
            with open(gm) as f:
                old = [json.loads(l) for l in f if l.strip()]
            new, put = [], False
            for l in old:
                if l.get("scene") == scene:
                    if not put:
                        new += lines; put = True
                else:
                    new.append(l)
            if not put:
                new += lines
            with open(gm + ".part", "w") as f:
                f.writelines(json.dumps(l) + "\n" for l in new)
            os.replace(gm + ".part", gm)


# ------------------------------------------------------------------ web
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Box adjust</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#15171a;--panel:#1d2024;--line:#30343a;--text:#e6e6e6;--muted:#9aa0a6;--acc:#4cc38a;--warn:#e5a50a;--bad:#e5484d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,sans-serif;height:100vh;display:flex}
#side{width:330px;border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
#side h3{margin:10px 12px 6px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.list{overflow:auto;flex:1;min-height:0;border-top:1px solid var(--line)}
.row{padding:6px 12px;cursor:pointer;border-bottom:1px solid #23262b;display:flex;gap:6px;align-items:center}
.row:hover{background:#23272c}.row.sel{background:#2a3a33}
.row .n{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{font-size:11px;padding:0 5px;border-radius:3px;background:#30343a;color:var(--muted)}
.tag.ok{background:#1f3d2f;color:var(--acc)}.tag.m{background:#3d351f;color:var(--warn)}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#top{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#title{font-weight:600}
#wrap{flex:1;position:relative;min-height:0;background:#000}
canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
#busy{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#ddd;background:#0008;font-size:15px}
#right{width:390px;border-left:1px solid var(--line);overflow:auto;padding:10px 12px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px;margin-bottom:10px}
.box h4{margin:0 0 6px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-variant-numeric:tabular-nums}
.kv b{font-weight:500;color:var(--muted)}
button{background:#2b3036;color:var(--text);border:1px solid #3a4047;border-radius:5px;padding:5px 10px;cursor:pointer;font:inherit}
button:hover{background:#353b42}button.pri{background:#24563f;border-color:#2f6e51}button.pri:hover{background:#2b6649}
button.dan{background:#5a2326;border-color:#7a2f33}button:disabled{opacity:.4;cursor:default}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12px}
td,th{padding:3px 4px;border-bottom:1px solid #2a2e33;text-align:right}th{color:var(--muted);font-weight:500}
td:first-child,th:first-child{text-align:left}tr.click{cursor:pointer}tr.click:hover{background:#262a2f}
input[type=number]{width:56px;background:#111;color:var(--text);border:1px solid #3a4047;border-radius:4px;padding:2px 4px}
.muted{color:var(--muted)}.warn{color:var(--warn)}.bad{color:var(--bad)}.good{color:var(--acc)}
#msg{min-height:18px}
.legend span{display:inline-block;width:18px;height:0;border-top:2px solid;vertical-align:middle;margin:0 4px 0 10px}
</style></head><body>
<div id="side">
  <h3>Scenes</h3><div class="list" id="scenes" style="flex:0 0 38%"></div>
  <h3>Chips <label class="muted" style="text-transform:none;float:right"><input type="checkbox" id="posonly" checked> positives only</label></h3>
  <div class="list" id="chips"></div>
</div>
<div id="main">
  <div id="top"><span id="title" class="muted">pick a scene, then a chip</span>
    <span class="legend muted"><span style="border-color:#ff9f1a;border-top-style:dashed"></span>labels as they are<span style="border-color:#3ee07a"></span>moved</span>
    <span style="margin-left:auto" class="muted">dB <input type="number" id="lo" value="-5"> to <input type="number" id="hi" value="35"></span>
    <label class="muted"><input type="checkbox" id="showorig" checked> show originals</label>
  </div>
  <div id="wrap"><canvas id="cv"></canvas><div id="busy">forming image from the chip CPHD...</div></div>
</div>
<div id="right">
  <div class="box"><h4>This chip</h4>
    <div class="kv" id="cur"></div>
    <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
      <button class="pri" id="bsave" title="key s">Save measurement</button>
      <button id="bscene" title="key t">Scene translation</button>
      <button id="bzero" title="key r">Zero</button>
      <button id="bfit" title="key f">Fit view</button>
    </div>
    <div class="muted" style="margin-top:8px">Drag any box: all move together. Wheel = zoom, right-drag = pan, arrows = nudge 0.25 m (shift: 2 m), n / p = next / previous chip.</div>
  </div>
  <div class="box"><h4>Scene</h4><div id="scn" class="muted">-</div>
    <table id="meas" style="margin-top:6px"></table>
    <div style="margin-top:8px;display:flex;gap:6px"><button class="pri" id="bapply" disabled>Apply to scene...</button><button class="dan" id="bundo" disabled>Undo apply</button></div>
    <div id="confirm" style="display:none;margin-top:8px" class="box"></div>
  </div>
  <div id="msg"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const cv=$('cv'),ctx=cv.getContext('2d');
let S={scene:null,chip:null,chips:[],img:null,disp:null,off:{e:0,n:0},view:{s:1,x:0,y:0},drag:null,pan:null,meas:null};
const f1=v=>(v>=0?'+':'')+v.toFixed(1), f2=v=>(v>=0?'+':'')+v.toFixed(2);
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
  const j=await r.json();if(!r.ok||j.error)throw new Error(j.error||r.statusText);return j}
function msg(t,cls){$('msg').className=cls||'muted';$('msg').textContent=t}

async function loadScenes(){const sc=await api('/api/scenes');const el=$('scenes');el.innerHTML='';
  for(const s of sc){const d=document.createElement('div');d.className='row'+(s.scene===S.scene?' sel':'');
    const st=s.applied?'<span class="tag ok">applied</span>':(s.translation?`<span class="tag m">${s.translation.n} meas</span>`:'');
    d.innerHTML=`<span>${s.scene}</span>${st}<span class="n">${s.positive}/${s.chips}</span>`;d.title=`${s.positive} positive of ${s.chips} chips, ${s.ships} ships`;
    d.onclick=()=>pickScene(s.scene);el.appendChild(d)}}
async function pickScene(sc){S.scene=sc;await loadScenes();await loadChips();await loadMeas();
  const first=S.chips.find(c=>c.label==='positive')||S.chips[0];if(first)openChip(first.crop_id)}
async function loadChips(){S.chips=await api('/api/chips?scene='+encodeURIComponent(S.scene));renderChips()}
function renderChips(){const el=$('chips');el.innerHTML='';const po=$('posonly').checked;
  for(const c of S.chips){if(po&&c.label!=='positive')continue;const d=document.createElement('div');
    d.className='row'+(S.chip&&c.crop_id===S.chip.crop_id?' sel':'');
    d.innerHTML=`<span>${c.crop_id.replace(S.scene+'_','')}</span>${c.measured?'<span class="tag m">meas</span>':''}${c.cached?'':'<span class="tag" title="image not formed yet">new</span>'}<span class="n">${c.n_pos} ship${c.n_pos==1?'':'s'}</span>`;
    d.onclick=()=>openChip(c.crop_id);el.appendChild(d)}}
$('posonly').onchange=renderChips;

async function openChip(id){$('busy').style.display='flex';
  try{const c=await api(`/api/chip?scene=${encodeURIComponent(S.scene)}&crop_id=${encodeURIComponent(id)}`);
    S.chip=c;const im=new Image();im.src=c.image;await im.decode();S.img=im;makeDisp();
    if(c.applied){S.off={e:0,n:0}}
    else if(c.measurement){S.off={e:c.measurement.east_m,n:c.measurement.north_m}}
    else if(c.translation){S.off={e:c.translation.east_m,n:c.translation.north_m}}
    else S.off={e:0,n:0};
    fit();renderChips();const ci=S.chips.find(x=>x.crop_id===id);if(ci)ci.cached=true;renderChips();
    $('title').className='';$('title').textContent=`${c.crop_id}  (${c.kind}, ${c.n_pos} ship${c.n_pos==1?'':'s'})`+(c.applied?'  [scene applied: boxes shown as saved]':'');
  }catch(e){msg(e.message,'bad')}finally{$('busy').style.display='none'}}

function makeDisp(){if(!S.img)return;const c=document.createElement('canvas');c.width=S.img.width;c.height=S.img.height;
  const x=c.getContext('2d');x.drawImage(S.img,0,0);const d=x.getImageData(0,0,c.width,c.height);const p=d.data;
  const [a,b]=S.chip.geo.png_db, lo=+$('lo').value, hi=+$('hi').value;const lut=new Uint8ClampedArray(256);
  for(let i=0;i<256;i++){const db=a+(b-a)*i/255;lut[i]=Math.max(0,Math.min(255,(db-lo)/(hi-lo)*255))}
  for(let i=0;i<p.length;i+=4){const v=lut[p[i]];p[i]=p[i+1]=p[i+2]=v}x.putImageData(d,0,0);S.disp=c;draw()}
$('lo').onchange=$('hi').onchange=makeDisp;$('showorig').onchange=()=>draw();

function size(){const r=cv.getBoundingClientRect();cv.width=r.width*devicePixelRatio;cv.height=r.height*devicePixelRatio;draw()}
new ResizeObserver(size).observe($('wrap'));
function fit(){if(!S.img)return;const r=cv.getBoundingClientRect();const s=Math.min(r.width,r.height)/S.img.width;
  S.view={s,x:(r.width-S.img.width*s)/2,y:(r.height-S.img.height*s)/2};draw()}
const scr=(c,r)=>[S.view.x+c*S.view.s,S.view.y+r*S.view.s], img=(x,y)=>[(x-S.view.x)/S.view.s,(y-S.view.y)/S.view.s];
function dpx(){const pm=S.chip.geo.pixel_m;return [S.off.e/pm,-S.off.n/pm]}
function draw(){ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);const r=cv.getBoundingClientRect();
  ctx.fillStyle='#000';ctx.fillRect(0,0,r.width,r.height);if(!S.disp||!S.chip)return;
  ctx.imageSmoothingEnabled=S.view.s<1;ctx.drawImage(S.disp,S.view.x,S.view.y,S.disp.width*S.view.s,S.disp.height*S.view.s);
  const k=S.chip.core_px;let [x0,y0]=scr(k[0],k[1]),[x1,y1]=scr(k[2],k[3]);ctx.setLineDash([2,4]);ctx.strokeStyle='#6af';ctx.lineWidth=1;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
  const [dc,dr]=dpx();
  for(const b of S.chip.boxes){
    const o=S.chip.applied?(b.px_orig||b.px):b.px, m=S.chip.applied?b.px:[b.px[0]+dc,b.px[1]+dr,b.px[2]+dc,b.px[3]+dr];
    if($('showorig').checked){[x0,y0]=scr(o[0],o[1]);[x1,y1]=scr(o[2],o[3]);ctx.setLineDash([5,4]);ctx.strokeStyle='#ff9f1a';ctx.lineWidth=1.2;ctx.strokeRect(x0,y0,x1-x0,y1-y0)}
    [x0,y0]=scr(m[0],m[1]);[x1,y1]=scr(m[2],m[3]);ctx.setLineDash([]);ctx.strokeStyle=b.ignore?'#f5e663':'#3ee07a';ctx.lineWidth=2;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
    ctx.fillStyle=ctx.strokeStyle;ctx.font='11px system-ui';ctx.fillText(`${b.ann_id}`,x0,y0-3)}
  const a=S.chip.geo.radar_az_deg*Math.PI/180, ax=50, ay=r.height-50;
  ctx.strokeStyle='#ffd400';ctx.fillStyle='#ffd400';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(ax,ay);const ex=ax+40*Math.sin(a),ey=ay-40*Math.cos(a);ctx.lineTo(ex,ey);ctx.stroke();
  ctx.beginPath();ctx.arc(ex,ey,4,0,7);ctx.fill();ctx.font='12px system-ui';ctx.fillText('to radar',ax-24,ay+20);
  // scale bar 50 m
  const L=50/S.chip.geo.pixel_m*S.view.s;ctx.fillStyle='#fff';ctx.fillRect(r.width-20-L,r.height-30,L,3);ctx.fillText('50 m',r.width-20-L,r.height-36);
  info()}

function met(e,n){const g=S.chip.geo,a=g.radar_az_deg*Math.PI/180;const along=e*Math.sin(a)+n*Math.cos(a),across=-e*Math.cos(a)+n*Math.sin(a);
  return {along,across,dh:along/Math.tan(g.graze_deg*Math.PI/180)}}
function info(){const c=S.chip,g=c.geo;let e=S.off.e,n=S.off.n;if(c.applied&&c.applied_translation){e=c.applied_translation.east_m;n=c.applied_translation.north_m}
  const m=met(e,n);
  $('cur').innerHTML=`<b>shift east</b><span>${f2(e)} m</span><b>shift north</b><span>${f2(n)} m</span>
   <b>along radar</b><span>${f2(m.along)} m (towards +)</span><b>height offset</b><span>${f2(m.dh)} m &rarr; ships at ${(g.h_plane+m.dh).toFixed(2)} m HAE</span>
   <b>across</b><span class="${Math.abs(m.across)>3?'warn':''}">${f2(m.across)} m</span>
   <b>image plane</b><span>${g.h_plane.toFixed(2)} m HAE (h_ref)</span><b>graze / radar az</b><span>${g.graze_deg.toFixed(1)}&deg; / ${g.radar_az_deg.toFixed(0)}&deg;</span>
   ${c.measurement&&!c.applied?`<b>saved here</b><span>${f2(c.measurement.east_m)}, ${f2(c.measurement.north_m)} m</span>`:''}`;
  $('bsave').disabled=$('bscene').disabled=$('bzero').disabled=!!c.applied}

cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{if(!S.chip)return;const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  if(e.button===2||e.button===1){S.pan=[x,y,S.view.x,S.view.y];return}
  if(S.chip.applied)return;const [c,rr]=img(x,y),[dc,dr]=dpx(),pad=4/S.view.s;
  for(const b of S.chip.boxes){const p=b.px;if(c>=p[0]+dc-pad&&c<=p[2]+dc+pad&&rr>=p[1]+dr-pad&&rr<=p[3]+dr+pad){S.drag=[c,rr,S.off.e,S.off.n];return}}});
window.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  if(S.pan){S.view.x=S.pan[2]+x-S.pan[0];S.view.y=S.pan[3]+y-S.pan[1];draw()}
  else if(S.drag){const [c,rr]=img(x,y),pm=S.chip.geo.pixel_m;S.off={e:S.drag[2]+(c-S.drag[0])*pm,n:S.drag[3]-(rr-S.drag[1])*pm};draw()}});
window.addEventListener('mouseup',()=>{S.drag=null;S.pan=null});
cv.addEventListener('wheel',e=>{e.preventDefault();if(!S.img)return;const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
  const [c,rr]=img(x,y),f=e.deltaY<0?1.25:0.8;S.view.s*=f;S.view.x=x-c*S.view.s;S.view.y=y-rr*S.view.s;draw()},{passive:false});
window.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||!S.chip)return;const st=e.shiftKey?2:0.25;
  const mv={ArrowLeft:[-st,0],ArrowRight:[st,0],ArrowUp:[0,st],ArrowDown:[0,-st]}[e.key];
  if(mv&&!S.chip.applied){S.off.e+=mv[0];S.off.n+=mv[1];draw();e.preventDefault();return}
  if(e.key==='s')save();else if(e.key==='r')zero();else if(e.key==='f')fit();else if(e.key==='t')useScene();
  else if(e.key==='n'||e.key==='p'){const vis=S.chips.filter(c=>!$('posonly').checked||c.label==='positive');
    const i=vis.findIndex(c=>c.crop_id===S.chip.crop_id);const j=i+(e.key==='n'?1:-1);if(vis[j])openChip(vis[j].crop_id)}});
function zero(){if(S.chip.applied)return;S.off={e:0,n:0};draw()}
function useScene(){const t=S.meas&&S.meas.translation;if(t&&!S.chip.applied){S.off={e:t.east_m,n:t.north_m};draw()}}
$('bzero').onclick=zero;$('bfit').onclick=fit;$('bscene').onclick=useScene;$('bsave').onclick=save;
async function save(){if(!S.chip||S.chip.applied)return;try{S.meas=await api('/api/measure',{scene:S.scene,crop_id:S.chip.crop_id,east_m:S.off.e,north_m:S.off.n});
  S.chip.measurement={east_m:S.off.e,north_m:S.off.n};const ci=S.chips.find(x=>x.crop_id===S.chip.crop_id);if(ci)ci.measured=true;
  renderChips();renderMeas();loadScenes();info();msg(`saved measurement for ${S.chip.crop_id}`,'good')}catch(e){msg(e.message,'bad')}}

async function loadMeas(){S.meas=await api('/api/measurements?scene='+encodeURIComponent(S.scene));renderMeas()}
function renderMeas(){const M=S.meas,el=$('scn');$('confirm').style.display='none';
  if(M.applied){const a=M.applied_translation||{};el.innerHTML=`<span class="good">Applied</span> ${a.east_m!=null?`E ${f2(a.east_m)} m, N ${f2(a.north_m)} m`:''} to ${a.files||'?'} files${a.time?' on '+a.time:''}.<br>Old files are kept as *_pre_adjustment.json.`}
  else if(M.translation){const t=M.translation,m=M.translation_metrics;
    el.innerHTML=`Scene translation (median of ${t.n}): <b>E ${f2(t.east_m)} m, N ${f2(t.north_m)} m</b><br>
     along ${f2(m.along_radar_m)} m &rarr; height offset <b>${f2(m.height_offset_m)} m</b> (ships at ${(M.h_ref+m.height_offset_m).toFixed(2)} m HAE), across ${f2(m.across_m)} m<br>
     largest distance of a measurement from the median: <span class="${t.spread_m>3?'warn':'good'}">${t.spread_m.toFixed(1)} m</span>`}
  else el.textContent='No measurements yet: drag the boxes onto the ships of a chip, then Save measurement.';
  const tb=$('meas');tb.innerHTML=M.rows.length?'<tr><th>chip</th><th>E</th><th>N</th><th>dh</th><th>across</th><th>off median</th><th></th></tr>':'';
  for(const r of M.rows){const tr=document.createElement('tr');tr.className='click';
    tr.innerHTML=`<td>${r.crop_id.replace(S.scene+'_','')}</td><td>${f1(r.east_m)}</td><td>${f1(r.north_m)}</td><td>${f1(r.height_offset_m)}</td><td>${f1(r.across_m)}</td><td class="${r.residual_m>3?'warn':''}">${(r.residual_m||0).toFixed(1)}</td><td>${M.applied?'':'<button title="delete">x</button>'}</td>`;
    tr.onclick=ev=>{if(ev.target.tagName==='BUTTON'){ev.stopPropagation();del(r.crop_id)}else openChip(r.crop_id)};tb.appendChild(tr)}
  $('bapply').disabled=M.applied||!M.translation;$('bundo').disabled=!M.applied}
async function del(id){S.meas=await api('/api/unmeasure',{scene:S.scene,crop_id:id});const ci=S.chips.find(x=>x.crop_id===id);if(ci)ci.measured=false;
  if(S.chip&&S.chip.crop_id===id)S.chip.measurement=null;renderChips();renderMeas();loadScenes()}
$('bapply').onclick=()=>{const t=S.meas.translation,c=$('confirm');c.style.display='block';
  c.innerHTML=`Move every box of <b>${S.scene}</b> by <b>E ${f2(t.east_m)} m, N ${f2(t.north_m)} m</b> in all ${S.chips.length} crop JSONs (old ones renamed *_pre_adjustment.json), and write the moved scene annotations to the adjusted folder. The master annotations are not touched. Labels are re-clipped, so some chips can switch positive/negative.<div style="margin-top:8px;display:flex;gap:6px"><button class="pri" id="yes">Yes, apply</button><button id="no">Cancel</button></div>`;
  $('no').onclick=()=>c.style.display='none';
  $('yes').onclick=async()=>{try{const r=await api('/api/apply',{scene:S.scene});msg(`applied: ${r.files} files rewritten${r.label_changed?`, ${r.label_changed} chips changed label`:''}`+(r.warning?' - '+r.warning:''),r.warning?'warn':'good');
    await loadMeas();await loadScenes();await loadChips();if(S.chip)openChip(S.chip.crop_id)}catch(e){msg(e.message,'bad')}}};
$('bundo').onclick=async()=>{try{const r=await api('/api/undo',{scene:S.scene});msg(`restored ${r.restored} files`,'good');await loadMeas();await loadScenes();await loadChips();if(S.chip)openChip(S.chip.crop_id)}catch(e){msg(e.message,'bad')}};
size();loadScenes();
</script></body></html>"""


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype == "application/json" else "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def route(self, method):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            try:
                p = u.path
                if p == "/":
                    return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
                if p == "/api/scenes":
                    return self.send(200, store.scenes())
                if p == "/api/chips":
                    return self.send(200, store.chips(q["scene"]))
                if p == "/api/chip":
                    return self.send(200, store.chip(q["scene"], q["crop_id"]))
                if p == "/api/image":
                    store.sdir(q["scene"])
                    png = store.cache_png(os.path.basename(q["crop_id"]))
                    with open(png, "rb") as f:
                        return self.send(200, f.read(), "image/png")
                if p == "/api/measurements":
                    return self.send(200, store.measurements(q["scene"]))
                if p == "/api/measure" and method == "POST":
                    return self.send(200, store.measure(body["scene"], body["crop_id"], body["east_m"], body["north_m"]))
                if p == "/api/unmeasure" and method == "POST":
                    return self.send(200, store.unmeasure(body["scene"], body["crop_id"]))
                if p == "/api/apply" and method == "POST":
                    r = store.apply(body["scene"])
                    print(f"applied {body['scene']}: {r}", flush=True)
                    return self.send(200, r)
                if p == "/api/undo" and method == "POST":
                    r = store.undo(body["scene"])
                    print(f"undo {body['scene']}: {r}", flush=True)
                    return self.send(200, r)
                return self.send(404, {"error": "not found"})
            except Exception as e:
                traceback.print_exc()
                return self.send(400, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True, help="crop root: <crops>/<scene>/<crop_id>.cphd + .json")
    ap.add_argument("--ann-root", default="/data3/sandbox/jhu/satlock/annotation_tool/test_aligned",
                    help="<ann-root>/<scene>_GEC/annotations.json (used if the path in the crop JSON does not exist)")
    ap.add_argument("--ann-out", default="/home/tabtahi/SATLOCK/dataset/annotations_adjusted",
                    help="adjusted annotations go here as <ann-out>/<scene>_GEC/annotations.json (train with "
                         "--ann-root <ann-out>); the master --ann-root is only read")
    ap.add_argument("--work", default=None, help="state + image cache (default <crops>/_box_adjust)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda:0", help="torch device for image formation (cpu works, slower)")
    ap.add_argument("--pixel-m", type=float, default=1.0, help="display pixel (0.5 = sharper, 4x slower)")
    ap.add_argument("--margin-m", type=float, default=64.0, help="image this much beyond the crop core")
    ap.add_argument("--min-visible", type=float, default=0.3, help="as build_crops: visible below this = ignore")
    ap.add_argument("--prerender", action="store_true", help="form the images of all positive chips in the background")
    a = ap.parse_args()
    store = Store(a)
    sc = store.scenes()
    print(f"master annotations (read only): {a.ann_root}\nadjusted annotations: {store.ann_out} "
          f"({store.mirror()} scenes copied unchanged as a start)")
    print(f"{len(sc)} scenes, {sum(s['chips'] for s in sc)} chips under {store.crops}; state in {store.work}")
    if a.prerender:
        def pre():
            for s in sc:
                for c in store.chips(s["scene"]):
                    if c["label"] == "positive" and not c["cached"]:
                        try:
                            store.chip(s["scene"], c["crop_id"])
                        except Exception as e:
                            print(f"  prerender {c['crop_id']} failed: {e}")
        threading.Thread(target=pre, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(store))
    print(f"open http://localhost:{a.port}  (from your laptop: ssh -L {a.port}:localhost:{a.port} <server>)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
