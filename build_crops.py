#!/usr/bin/env python3
"""Fixed-shape CPHD crops with geo ship labels, for training a detector on phase history.

For every scene that has both an annotations.json (GEC labels in lat/lon) and a CPHD:
  1. Crop centres: a global lat/lon lattice (stride 256 m) inside the GEC footprint, plus one crop centred on every
     annotated ship (with random jitter). Each crop's labelled core is a 512 x 512 grid at 1 m (EPSG:4326 GeoGrid).
  2. Labels: annotation boxes clipped to the core, with pixel boxes in the 512 grid. Boxes that keep less than
     --min-visible of their area are marked ignore. Crops with no box at all are negatives; they are subsampled to
     --neg-ratio x the positive crops (half nearest the ships, half random), so the dataset skew is set here.
  3. Phase history: one pass over the CPHD per group of crops. Each pulse is re-referenced to the crop centre, range
     gated, and its band cut to a fixed width (ground-range resolution --res-rg). Across pulses, the crop's Doppler band
     is low-pass filtered, decimated, and resampled to P pulses uniform in look angle spanning lambda / (2 * res_az),
     centred on the middle of the dwell. Every crop is therefore P x S = 1024 x 1024 complex64, whatever the scene.
  4. Output: <out>/<scene>/<crop_id>.cphd (valid CPHD, reference point = crop centre) and <crop_id>.json (geometry,
     signal description, labels), plus manifest.jsonl per scene and for the whole run.

    python build_crops.py --ann-root /data3/sandbox/jhu/satlock/annotation_tool/test_aligned \\
        --cphd-root /home/tabtahi/SATLOCK/dataset/cphd --out /home/tabtahi/SATLOCK/dataset/cphd_crops --dry-run
    python build_crops.py ... --limit 1          # one scene
    python build_crops.py --selftest             # synthetic end-to-end check, exits 0 on pass
"""
import argparse, copy, csv, json, math, os, re, shutil, sys, time, traceback
import tempfile
# CuPy keeps compiled kernels under ~/.cupy; in a container run with --user the home folder is often "/" and not
# writable, so point the cache at a writable temp folder unless one is already set.
if not os.environ.get("CUPY_CACHE_DIR"):
    os.environ["CUPY_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), f"cupy_cache_{os.getuid()}")
import numpy as np
import scipy.fft as sp_fft
from scipy.signal import firwin, upfirdn
import sarkit.cphd as skcphd

C = 299792458.0
_A, _E2 = 6378137.0, 6.69437999014e-3                       # WGS-84
SCENE_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_UMBRA-\d+")

DEFAULTS = dict(core_m=512.0, pixel_m=1.0, pulses=1024, samples=1024, res_az=1.0, res_rg=1.0, margin=0.25, guard=0.25,
                stride_m=256.0, jitter_m=64.0, min_visible=0.3, neg_ratio=3.0, neg_empty=8, interp_taps=24,
                mem_gb=8.0, h_ref=None, seed=0, workers=4, device="cpu", resample="auto", chunk=256)


# ------------------------------------------------------------------ coordinates
def llh_to_ecef(lat, lon, h):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    h = np.asarray(h, float)
    n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo), (n * (1 - _E2) + h) * np.sin(la)], -1)


def ecef_to_llh(xyz):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    xyz = np.asarray(xyz, float)
    lon, lat, h = t.transform(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    return np.asarray(lat), np.asarray(lon), np.asarray(h)


def enu_basis(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([[-math.sin(lo), math.cos(lo), 0.0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def m_per_deg(lat):
    s = math.sin(math.radians(lat))
    m_lat = math.pi / 180 * _A * (1 - _E2) / (1 - _E2 * s * s) ** 1.5
    m_lon = math.pi / 180 * _A / math.sqrt(1 - _E2 * s * s) * math.cos(math.radians(lat))
    return m_lat, m_lon


def two_way(tx, rc, p):
    return np.linalg.norm(tx - p, axis=-1) + np.linalg.norm(rc - p, axis=-1)


class GeoGrid:
    """North-up square grid, regular in longitude and latitude (EPSG:4326), at one ellipsoid height."""
    def __init__(self, lat, lon, size_m, pixel_m, h_ref):
        m_lat, m_lon = m_per_deg(lat)
        n = max(8, int(round(size_m / pixel_m)))
        self.n, self.px, self.h_ref = n, float(pixel_m), float(h_ref)
        self.dlat, self.dlon = pixel_m / m_lat, pixel_m / m_lon
        self.lon_w, self.lat_n = lon - n * self.dlon / 2, lat + n * self.dlat / 2
        self.bounds = [self.lon_w, self.lat_n - n * self.dlat, self.lon_w + n * self.dlon, self.lat_n]   # W, S, E, N

    def ll_to_rc(self, lat, lon):                                     # pixel-centre convention
        return (self.lat_n - np.asarray(lat)) / self.dlat - 0.5, (np.asarray(lon) - self.lon_w) / self.dlon - 0.5

    def ecef(self):
        lat = self.lat_n - (np.arange(self.n) + 0.5) * self.dlat
        lon = self.lon_w + (np.arange(self.n) + 0.5) * self.dlon
        lon, lat = np.meshgrid(lon, lat)
        return llh_to_ecef(lat, lon, np.full(lat.shape, self.h_ref)).reshape(-1, 3)


# ------------------------------------------------------------------ CPHD access (same reader as cphd_viewer.py)
class CPHD:
    """Header fields and per-pulse parameters of one CPHD channel. Pulses with unusable parameters are skipped."""
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f, skcphd.Reader(f) as r:
            xml = r.metadata.xmltree
            self.domain = xml.findtext("{*}Global/{*}DomainType")
            self.sgn = int(xml.findtext("{*}Global/{*}SGN"))
            self.fmt = xml.findtext("{*}Data/{*}SignalArrayFormat")
            ids = [c.findtext("{*}Identifier") for c in xml.findall("{*}Data/{*}Channel")]
            ref = xml.findtext("{*}Channel/{*}RefChId")
            self.ch = ref if ref in ids else ids[0]
            node = xml.find(f"{{*}}Data/{{*}}Channel[{{*}}Identifier='{self.ch}']")
            self.nv, self.ns = int(node.findtext("{*}NumVectors")), int(node.findtext("{*}NumSamples"))
            if self.domain != "FX" or node.findtext("{*}CompressedSignalSize") is not None:
                raise NotImplementedError("only uncompressed FX-domain CPHD is handled")
            pv = r.read_pvps(self.ch)
        self.pv = pv
        g = lambda k: np.ascontiguousarray(pv[k]).astype(np.float64)
        self.tx_time, self.tx_pos, self.rcv_pos, self.srp = g("TxTime"), g("TxPos"), g("RcvPos"), g("SRPPos")
        self.sc0, self.scss, fx1, fx2 = g("SC0"), g("SCSS"), g("FX1"), g("FX2")
        self.amp_sf = g("AmpSF") if "AmpSF" in pv.dtype.names else None
        flagged = (np.asarray(pv["SIGNAL"]) != 0) if "SIGNAL" in pv.dtype.names else np.ones(self.nv, bool)
        finite = (np.isfinite(self.tx_time) & np.isfinite(self.tx_pos).all(1) & np.isfinite(self.rcv_pos).all(1)
                  & np.isfinite(self.srp).all(1) & np.isfinite(self.sc0) & np.isfinite(self.scss) & (self.scss != 0))
        if self.amp_sf is not None:
            finite &= np.isfinite(self.amp_sf)
        self.valid = flagged & finite
        self.n_unusable = int((~self.valid).sum())
        self.good = np.nonzero(self.valid)[0]
        if len(self.good) < 16:
            raise ValueError(f"only {len(self.good)} usable pulses in {path}")
        self.srp0 = np.median(self.srp[self.good], axis=0)
        self.lat0, self.lon0, self.h0 = [float(v) for v in ecef_to_llh(self.srp0)]
        cand = self.good[np.isfinite(fx1[self.good]) & np.isfinite(fx2[self.good])]
        m = int(cand[len(cand) // 2]) if len(cand) else int(self.good[len(self.good) // 2])
        self.m_ref = m
        lo = (fx1[m] - self.sc0[m]) / self.scss[m] if len(cand) else 0.0
        hi = (fx2[m] - self.sc0[m]) / self.scss[m] if len(cand) else float(self.ns - 1)
        self.n_lo = int(max(0, math.ceil(min(lo, hi) - 1e-6)))
        self.n_hi = int(min(self.ns, math.floor(max(lo, hi) + 1e-6) + 1))
        if self.n_hi - self.n_lo < 16:
            self.n_lo, self.n_hi = 0, self.ns
        self.bandwidth = (self.n_hi - self.n_lo) * abs(self.scss[m])
        self.fc = self.sc0[m] + 0.5 * (self.n_lo + self.n_hi) * self.scss[m]

    def read(self, reader, v0, v1):
        s = reader.read_signal(self.ch, start_vector=v0, stop_vector=v1)
        if s.dtype.names:                                   # CI2 / CI4: integer (real, imag) pairs
            out = np.empty(s.shape, np.complex64)
            out.real, out.imag = s["real"], s["imag"]
        else:
            out = s.astype(np.complex64)
        if self.amp_sf is not None:
            out *= np.nan_to_num(self.amp_sf[v0:v1, None]).astype(np.float32)
        return out


# ------------------------------------------------------------------ annotations and crop layout
def load_annotations(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        d = d[0]
    img = d.get("image", {}) or {}
    name = d.get("scene_name") or img.get("file_name") or path
    m = SCENE_RE.search(name) or SCENE_RE.search(path)
    if m is None:
        raise ValueError(f"no scene name like 2023-06-20-12-06-09_UMBRA-05 in {path}")
    anns = []
    for a in d.get("annotations", []):
        g = a.get("bbox_geo")
        if not g or len(g) != 4:
            continue
        lo0, la0, lo1, la1 = map(float, g)                  # [lon_min, lat_min, lon_max, lat_max]
        anns.append(dict(id=a.get("id"), category_id=a.get("category_id", 1), iscrowd=int(a.get("iscrowd", 0) or 0),
                         lon_min=min(lo0, lo1), lat_min=min(la0, la1), lon_max=max(lo0, lo1), lat_max=max(la0, la1)))
    gb = img.get("geo_bounds")
    if gb is None and anns:                                 # no footprint stated: the boxes plus 1 km
        dla, dlo = 1000 / m_per_deg(anns[0]["lat_min"])[0], 1000 / m_per_deg(anns[0]["lat_min"])[1]
        gb = [min(a["lon_min"] for a in anns) - dlo, min(a["lat_min"] for a in anns) - dla,
              max(a["lon_max"] for a in anns) + dlo, max(a["lat_max"] for a in anns) + dla]
    return dict(scene=m.group(0), geo_bounds=[float(v) for v in gb] if gb else None, anns=anns, path=path)


def find_annotation_files(ann_root):
    out = []
    for d in sorted(os.listdir(ann_root)):
        p = os.path.join(ann_root, d)
        if os.path.isdir(p):
            for name in ("annotations.json", "annotation.json"):
                if os.path.isfile(os.path.join(p, name)):
                    out.append(os.path.join(p, name))
                    break
    return out


def index_cphd(cphd_root):
    idx = {}
    for root, dirs, files in os.walk(cphd_root):
        if root[len(cphd_root):].count(os.sep) >= 3:
            dirs[:] = []
        for f in files:
            m = SCENE_RE.search(f)
            if m and f.lower().endswith(".cphd") and not f.endswith(".part"):
                idx.setdefault(m.group(0), os.path.join(root, f))
    return idx


def lattice_centres(gb, cfg):
    """Centres of a global lattice (same everywhere inside a 1-degree latitude band) whose core fits in the footprint."""
    W, S, E, N = gb
    m_lat, m_lon = m_per_deg(math.floor(0.5 * (S + N)) + 0.5)
    dphi, dlam = cfg["stride_m"] / m_lat, cfg["stride_m"] / m_lon
    ml, mo = m_per_deg(0.5 * (S + N))
    hl, ho = 0.5 * cfg["core_m"] / ml, 0.5 * cfg["core_m"] / mo
    ks = range(math.ceil((S + hl) / dphi - 0.5), math.floor((N - hl) / dphi - 0.5) + 1)
    js = range(math.ceil((W + ho) / dlam - 0.5), math.floor((E - ho) / dlam - 0.5) + 1)
    return [((k + 0.5) * dphi, (j + 0.5) * dlam, f"{k}_{j}") for k in ks for j in js]


def crop_labels(grid, anns, min_visible):
    W, S, E, N = grid.bounds
    boxes = []
    for a in anns:
        x0, x1 = max(a["lon_min"], W), min(a["lon_max"], E)
        y0, y1 = max(a["lat_min"], S), min(a["lat_max"], N)
        if x1 <= x0 or y1 <= y0:
            continue
        area = (a["lon_max"] - a["lon_min"]) * (a["lat_max"] - a["lat_min"])
        vis = (x1 - x0) * (y1 - y0) / area if area > 0 else 1.0
        boxes.append(dict(ann_id=a["id"], category_id=a["category_id"],
                          bbox_geo=[a["lon_min"], a["lat_min"], a["lon_max"], a["lat_max"]],
                          bbox_geo_clipped=[x0, y0, x1, y1],
                          bbox_px=[(x0 - W) / grid.dlon, (N - y1) / grid.dlat, (x1 - W) / grid.dlon, (N - y0) / grid.dlat],
                          visible=float(vis), ignore=bool(vis < min_visible or a["iscrowd"])))
    return boxes


def choose_crops(ann, cfg, h_ref, rng):
    """All crop centres with labels, then the negatives subsampled. Returns (selected, counts)."""
    gb = ann["geo_bounds"]
    W, S, E, N = gb
    ml, mo = m_per_deg(0.5 * (S + N))
    ships = [a for a in ann["anns"] if not a["iscrowd"]]
    ship_xy = np.array([[0.5 * (a["lat_min"] + a["lat_max"]), 0.5 * (a["lon_min"] + a["lon_max"])] for a in ships]).reshape(-1, 2)
    hl, ho = 0.5 * cfg["core_m"] / ml, 0.5 * cfg["core_m"] / mo
    cands = [("L", lat, lon, key) for lat, lon, key in lattice_centres(gb, cfg)]
    for i, a in enumerate(ships):                           # one crop centred on every ship, jittered, kept in the footprint
        jy, jx = rng.uniform(-cfg["jitter_m"], cfg["jitter_m"], 2)
        lat = 0.5 * (a["lat_min"] + a["lat_max"]) + jy / ml
        lon = 0.5 * (a["lon_min"] + a["lon_max"]) + jx / mo
        lat = min(max(lat, S + hl), N - hl) if N - S > 2 * hl else 0.5 * (S + N)
        lon = min(max(lon, W + ho), E - ho) if E - W > 2 * ho else 0.5 * (W + E)
        cands.append(("S", lat, lon, f"ship{a['id']}"))
    pos, neg, ign = [], [], 0
    for kind, lat, lon, key in cands:
        grid = GeoGrid(lat, lon, cfg["core_m"], cfg["pixel_m"], h_ref)
        boxes = crop_labels(grid, ann["anns"], cfg["min_visible"])
        d = (np.hypot((ship_xy[:, 0] - lat) * ml, (ship_xy[:, 1] - lon) * mo).min() if len(ship_xy) else float("inf"))
        rec = dict(kind=kind, key=key, lat=lat, lon=lon, grid=grid, boxes=boxes, nearest_ship_m=float(d),
                   n_pos=sum(not b["ignore"] for b in boxes), n_ignore=sum(b["ignore"] for b in boxes))
        if rec["n_pos"]:
            pos.append(rec)
        elif boxes:
            ign += 1                                        # only cut ships: neither positive nor a clean negative
        else:
            neg.append(rec)
    quota = int(round(cfg["neg_ratio"] * len(pos))) if pos else int(cfg["neg_empty"])
    quota = min(quota, len(neg))
    near = sorted(neg, key=lambda r: r["nearest_ship_m"])[: quota // 2]
    rest = [r for r in neg if r not in near]
    far = [rest[i] for i in rng.permutation(len(rest))[: quota - len(near)]] if rest else []
    chosen = pos + near + far
    for r in near + far:
        r["neg_pick"] = "near" if r in near else "random"
    return chosen, dict(candidates=len(cands), positive=len(pos), negative_available=len(neg), negative_kept=quota,
                        ignore_only_dropped=ign)


# ------------------------------------------------------------------ scene and crop planning
class Scene:
    """Scene-wide choices: fast-time band, aperture span, reference height."""
    def __init__(self, cphd, cfg):
        c = self.c = cphd
        self.cfg = cfg
        self.h_ref = c.h0 if cfg["h_ref"] is None else float(cfg["h_ref"])
        E, N, U = enu_basis(c.lat0, c.lon0)
        self.arp = 0.5 * (c.tx_pos + c.rcv_pos)
        m = int(c.good[len(c.good) // 2])
        los = c.srp[m] - self.arp[m]
        self.graze = math.asin(-(los @ U) / np.linalg.norm(los))
        self.lam = C / c.fc
        self.dtheta = self.lam / (2 * cfg["res_az"])                 # aperture span, radians
        scss = abs(c.scss[c.m_ref])
        B_want = C / (2 * cfg["res_rg"] * math.cos(self.graze))
        N_av = c.n_hi - c.n_lo
        self.N_t = int(min(N_av, max(16, round(B_want / scss))))
        n_c = (c.n_lo + c.n_hi) // 2
        self.n_s = int(min(max(n_c - self.N_t // 2, c.n_lo), c.n_hi - self.N_t))
        self.B_t = self.N_t * scss
        self.res_rg = C / (2 * self.B_t * math.cos(self.graze))
        off = np.zeros(c.nv)
        off[c.good] = (c.sc0[c.m_ref] - c.sc0[c.good]) / c.scss[c.good]     # per-pulse shift if SC0 moves
        self.offs = np.round(off).astype(np.int64)
        self.sc0_residual = float(np.abs(off - self.offs)[c.good].max())     # fraction of a sample, ideally 0
        self.f0 = np.full(c.nv, np.nan)
        self.f0[c.good] = c.sc0[c.good] + (self.n_s + self.offs[c.good]) * c.scss[c.good]   # band start per pulse
        self.S, self.P = int(cfg["samples"]), int(cfg["pulses"])
        self.scss_out = self.N_t * c.scss[c.m_ref] / self.S
        u = self.arp[c.good] - c.srp0
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        self.full_deg = math.degrees(math.acos(float(np.clip(u[0] @ u[-1], -1, 1))))

    def summary(self):
        return dict(graze_deg=math.degrees(self.graze), aperture_full_deg=self.full_deg,
                    aperture_crop_deg=math.degrees(self.dtheta), band_mhz=self.B_t / 1e6, res_rg_m=self.res_rg,
                    res_az_m=self.cfg["res_az"], samples_in_band=self.N_t, sc0_residual_samples=self.sc0_residual,
                    h_ref=self.h_ref, unusable_pulses=self.c.n_unusable)

    def plan(self, lat, lon):
        """Which input pulses, delays and filter a crop centred at (lat, lon) needs. Geometry only."""
        c, cfg, P = self.c, self.cfg, self.P
        cen = llh_to_ecef(lat, lon, self.h_ref)
        g = c.good
        u = self.arp[g] - cen
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        e_r = u[len(g) // 2]
        d = u[-1] - u[0]
        e_a = d - (d @ e_r) * e_r
        e_a /= np.linalg.norm(e_a)
        th = np.arctan2(u @ e_a, u @ e_r)                           # look angle to the crop centre, increasing
        th_k = 0.5 * (th[0] + th[-1]) + (np.arange(P) - (P - 1) / 2) * (self.dtheta / P)
        if th_k[0] <= th[0] or th_k[-1] >= th[-1]:
            return None, f"aperture: dwell spans {math.degrees(th[-1] - th[0]):.2f} deg, crop needs {math.degrees(self.dtheta):.2f}"
        t_k = np.interp(th_k, th, g.astype(np.float64))             # fractional original pulse index of each output pulse
        a0, a1 = int(math.floor(t_k[0])), int(math.ceil(t_k[-1])) + 1
        ga = g[(g >= a0) & (g < a1)]
        ext = cfg["core_m"] * (1 + cfg["margin"])
        w, s_, e, n_ = GeoGrid(lat, lon, ext, ext / 10, self.h_ref).bounds
        la, lo = np.meshgrid(np.linspace(s_, n_, 11), np.linspace(w, e, 11))
        pts = llh_to_ecef(la.ravel(), lo.ravel(), np.full(la.size, self.h_ref))
        tx, rc = c.tx_pos[ga], c.rcv_pos[ga]
        r_c = two_way(tx, rc, cen)
        rel = np.stack([(two_way(tx, rc, q) - r_c) / C for q in pts])            # points x pulses, seconds
        W = float(np.abs(rel).max()) + 4.0 / self.B_t
        if W > 0.5 / abs(c.scss[c.m_ref]):
            return None, "swath: crop reaches outside the delay span the CPHD holds"
        f_hi = np.abs(self.f0[ga] + (self.N_t - 1) * c.scss[ga])
        dph = np.diff(rel * f_hi[None, :], axis=1) / np.diff(ga)[None, :]
        B_d = float(np.abs(dph).max())                                           # cycles per input pulse
        gd = cfg["guard"]
        cut, stop = B_d * (1 + gd), B_d * (1 + 2 * gd)
        if stop >= 0.5:
            return None, f"PRF too low for this crop extent (Doppler {B_d:.2f} cycles/pulse)"
        Da = max(1, int(math.floor(0.25 / stop)))
        L = min(int(math.ceil(4.6 / (B_d * gd))) | 1, 8001)
        dt_out = float(np.median(np.diff(t_k)))
        b_out = B_d * dt_out                                      # crop band at the output rate, cycles per output pulse
        pad = L // 2 + Da * (cfg["interp_taps"] // 2 + 2)
        n0, n1 = a0 - pad, a1 + pad
        partial = n0 < g[0] or n1 > g[-1] + 1
        n0, n1 = max(n0, int(g[0])), min(n1, int(g[-1]) + 1)
        sl = slice(n0, n1)
        ok = c.valid[sl]
        tau = (two_way(c.tx_pos[sl], c.rcv_pos[sl], cen) - two_way(c.tx_pos[sl], c.rcv_pos[sl], c.srp[sl])) / C
        tau = np.where(ok, tau, 0.0)
        return dict(lat=lat, lon=lon, cen=cen, t_k=t_k, n0=n0, n1=n1, tau_c=tau, W=W, B_d=B_d, cut=cut, Da=Da, L=L,
                    b_out=b_out, alias_free=b_out * (2 + 2 * gd) <= 1.0, partial_filter=bool(partial),
                    ext_bounds=[w, s_, e, n_], bytes=(n1 - n0) * self.S * 8), None

    def gate_bins(self, W):
        return int(max(1, min(math.floor(W * self.N_t * abs(self.c.scss[self.c.m_ref])), self.S // 2 - 1, (self.N_t - 1) // 2)))


def get_xp(cfg):
    """numpy on the CPU, cupy on the GPU; the same code runs on either."""
    if cfg.get("device") == "gpu":
        import cupy
        return cupy
    return np


def _fft(xp, a, inverse, workers):
    if xp is np:
        return (sp_fft.ifft if inverse else sp_fft.fft)(a, axis=1, workers=workers)
    return (xp.fft.ifft if inverse else xp.fft.fft)(a, axis=1)


def extract_group(sc, plans, workers=4, chunk=256, log=print, xp=np):
    """One pass over the pulses that a group of crops needs. Returns {i: (pulses x S) complex64 on the device}."""
    c, S, N_t = sc.c, sc.S, sc.N_t
    lo, hi = min(p["n0"] for p in plans.values()), max(p["n1"] for p in plans.values())
    buf = {i: xp.zeros((p["n1"] - p["n0"], S), xp.complex64) for i, p in plans.items()}
    tau = {i: xp.asarray(p["tau_c"]) for i, p in plans.items()}
    hb = {i: sc.gate_bins(p["W"]) for i, p in plans.items()}
    nidx = np.arange(N_t)
    t0, t_say = time.time(), time.time()
    with open(c.path, "rb") as f, skcphd.Reader(f) as reader:
        for a in range(lo, hi, chunk):
            b = min(hi, a + chunk)
            idx = np.clip(sc.n_s + sc.offs[a:b, None] + nidx[None, :], 0, c.ns - 1)
            sig = xp.asarray(np.take_along_axis(c.read(reader, a, b), idx, axis=1))
            freq = xp.asarray(c.sc0[a:b, None] + idx * c.scss[a:b, None])
            bad = xp.asarray(~c.valid[a:b])
            for i, p in plans.items():
                i0, i1 = max(a, p["n0"]), min(b, p["n1"])
                if i0 >= i1:
                    continue
                r = slice(i0 - a, i1 - a)
                tc = tau[i][i0 - p["n0"]:i1 - p["n0"]]
                s = sig[r] * xp.exp((-c.sgn * 2j * np.pi) * (freq[r] * tc[:, None])).astype(xp.complex64)
                s[bad[r]] = 0
                Q = _fft(xp, s, True, workers)                       # range profiles; delay 0 = crop centre
                h = hb[i]
                Qc = xp.zeros((i1 - i0, S), xp.complex64)
                Qc[:, :h + 1] = Q[:, :h + 1]                         # keep |delay| <= W, place on S bins
                Qc[:, S - h:] = Q[:, N_t - h:]
                buf[i][i0 - p["n0"]:i1 - p["n0"]] = _fft(xp, Qc, False, workers)
            if time.time() - t_say > 30:
                log(f"    pulses {b - lo:,} of {hi - lo:,}, {time.time() - t0:.0f} s")
                t_say = time.time()
    return buf


def resample_slow_time(sc, p, Y, xp=np, method="auto"):
    """Low-pass the crop's Doppler band and resample to the P output angles. Returns P x S complex64 (host).

    upfirdn: polyphase low-pass and decimation, then a short windowed-sinc interpolation (CPU default).
    direct:  one windowed-sinc low-pass evaluated at each output's fractional pulse position (GPU default)."""
    if method == "auto":
        method = "upfirdn" if xp is np else "direct"
    if method == "direct":
        return resample_direct(sc, p, Y, xp)
    T, beta = sc.cfg["interp_taps"], 8.0
    if p["cut"] < 0.5:
        h = firwin(p["L"], p["cut"], window=("kaiser", 7.0), fs=1.0)
        Z = upfirdn(h, Y, up=1, down=p["Da"], axis=0)
        u = (p["t_k"] - p["n0"] + (p["L"] - 1) // 2) / p["Da"]
    else:
        Z, u = Y, p["t_k"] - p["n0"]
    J = np.floor(u).astype(np.int64)[:, None] - (T // 2 - 1) + np.arange(T)[None, :]
    d = u[:, None] - J
    w = np.sinc(d) * np.i0(beta * np.sqrt(np.clip(1 - (2 * d / T) ** 2, 0, None))) / np.i0(beta)
    return _apply_taps(J, w, Z, np)


def resample_direct(sc, p, Y, xp):
    """out[k] = sum_n h(t_k - n) Y[n], h a Kaiser-windowed sinc low-pass at the crop's Doppler cut-off."""
    tk = p["t_k"] - p["n0"]
    if p["cut"] < 0.5:
        cut, half, beta = p["cut"], (p["L"] - 1) // 2 + 1, 7.0
    else:
        cut, half, beta = 0.5, sc.cfg["interp_taps"] // 2, 8.0
    J = np.floor(tk).astype(np.int64)[:, None] - (half - 1) + np.arange(2 * half)[None, :]
    d = tk[:, None] - J
    w = 2 * cut * np.sinc(2 * cut * d) * np.i0(beta * np.sqrt(np.clip(1 - (d / half) ** 2, 0, None))) / np.i0(beta)
    return _apply_taps(J, w, Y, xp)


def _apply_taps(J, w, Z, xp):
    """Rows of taps (indices J, weights w, P x T, host) applied to Z (n x S, on the device). Returns host array."""
    inside = (J >= 0) & (J < Z.shape[0])
    w = np.where(inside, w, 0.0)
    w /= w.sum(1, keepdims=True)
    J = np.clip(J, 0, Z.shape[0] - 1)
    P, T = J.shape
    rows = max(1, int(2.5e8 // (T * Z.shape[1] * 8)))               # keep the gathered block near 250 MB
    out = xp.empty((P, Z.shape[1]), xp.complex64)
    for k0 in range(0, P, rows):
        k1 = min(P, k0 + rows)
        out[k0:k1] = xp.einsum("pt,pts->ps", xp.asarray(w[k0:k1].astype(np.complex64)), Z[xp.asarray(J[k0:k1])])
    return out if xp is np else xp.asnumpy(out)


def interp_pvp(sc, t_k):
    """PVP rows of the output pulses: positions and times interpolated at the fractional pulse index."""
    c = sc.c
    g = c.good.astype(np.float64)
    pv = c.pv[np.clip(np.round(t_k).astype(int), 0, c.nv - 1)].copy()
    for name in ("TxTime", "RcvTime", "TxPos", "RcvPos", "TxVel", "RcvVel"):
        if name not in pv.dtype.names:
            continue
        src = np.asarray(c.pv[name], np.float64)
        if src.ndim == 1:
            pv[name] = np.interp(t_k, g, src[c.good])
        else:
            pv[name] = np.stack([np.interp(t_k, g, src[c.good, j]) for j in range(src.shape[1])], -1)
    return pv


def write_crop(sc, meta, p, Z, out_path, core_bounds):
    """Write a crop as a CPHD: the original header with sizes, swath, band, timeline and scene reference set to the
    crop, and per-pulse parameters of the resampled pulses re-referenced to the crop centre."""
    c = sc.c
    tree = copy.deepcopy(meta["tree"])
    root = tree.getroot()
    data = root.find("{*}Data")

    def drop(node):
        if node is not None:
            node.getparent().remove(node)

    def set_text(path, value, base=root):
        node = base.find(path)
        if node is not None:
            node.text = str(value)

    for chan in data.findall("{*}Channel"):
        if chan.findtext("{*}Identifier") != c.ch:
            data.remove(chan)
    for prm in root.findall("{*}Channel/{*}Parameters"):
        if prm.findtext("{*}Identifier") != c.ch:
            drop(prm)
    set_text("{*}Data/{*}SignalArrayFormat", "CF8")
    set_text("{*}Data/{*}NumCPHDChannels", 1)
    drop(root.find("{*}Data/{*}SignalCompressionID"))
    chan = data.find("{*}Channel")
    for tag, val in (("NumVectors", sc.P), ("NumSamples", sc.S), ("SignalArrayByteOffset", 0), ("PVPArrayByteOffset", 0)):
        set_text("{*}" + tag, val, chan)
    drop(chan.find("{*}CompressedSignalSize"))
    Wh = min(p["W"], 0.5 / abs(sc.scss_out))
    f0 = np.interp(p["t_k"], c.good.astype(float), sc.f0[c.good])
    fx_lo, fx_hi = float(np.min(f0)), float(np.max(f0 + (sc.S - 1) * sc.scss_out))
    set_text("{*}Global/{*}TOASwath/{*}TOAMin", -Wh)
    set_text("{*}Global/{*}TOASwath/{*}TOAMax", Wh)
    set_text("{*}Global/{*}FxBand/{*}FxMin", fx_lo)
    set_text("{*}Global/{*}FxBand/{*}FxMax", fx_hi)
    for prm in root.findall("{*}Channel/{*}Parameters"):
        set_text("{*}TOASaved", 2 * Wh, prm)
        set_text("{*}FxC", 0.5 * (fx_lo + fx_hi), prm)
        set_text("{*}FxBW", fx_hi - fx_lo, prm)
        drop(prm.find("{*}ImageArea"))
    pv = interp_pvp(sc, p["t_k"])
    set_text("{*}Global/{*}Timeline/{*}TxTime1", float(pv["TxTime"][0]))
    set_text("{*}Global/{*}Timeline/{*}TxTime2", float(pv["TxTime"][-1]))
    scn = root.find("{*}SceneCoordinates")
    if scn is not None:
        for ax, v in zip("XYZ", p["cen"]):
            set_text(f"{{*}}IARP/{{*}}ECF/{{*}}{ax}", float(v), scn)
        set_text("{*}IARP/{*}LLH/{*}Lat", p["lat"], scn)
        set_text("{*}IARP/{*}LLH/{*}Lon", p["lon"], scn)
        set_text("{*}IARP/{*}LLH/{*}HAE", sc.h_ref, scn)
        half = 0.5 * sc.cfg["core_m"] * (1 + sc.cfg["margin"])
        for tag, v in (("X1Y1", -half), ("X2Y2", half)):
            set_text(f"{{*}}ImageArea/{{*}}{tag}/{{*}}X", v, scn)
            set_text(f"{{*}}ImageArea/{{*}}{tag}/{{*}}Y", v, scn)
        drop(scn.find("{*}ImageArea/{*}Polygon"))
        drop(scn.find("{*}ImageGrid"))
        iacp = scn.find("{*}ImageAreaCornerPoints")
        if iacp is not None:
            W, S, E, N = core_bounds
            for node, (la, lo) in zip(iacp.findall("{*}IACP"), [(N, W), (N, E), (S, E), (S, W)]):
                set_text("{*}Lat", la, node)
                set_text("{*}Lon", lo, node)
    for ax, v in zip("XYZ", p["cen"]):
        set_text(f"{{*}}ReferenceGeometry/{{*}}SRP/{{*}}ECF/{{*}}{ax}", float(v))
        set_text(f"{{*}}ReferenceGeometry/{{*}}SRP/{{*}}IAC/{{*}}{ax}", 0.0)
    pv["SRPPos"] = p["cen"]
    pv["SC0"], pv["SCSS"] = f0, sc.scss_out
    pv["FX1"], pv["FX2"] = np.minimum(f0, f0 + (sc.S - 1) * sc.scss_out), np.maximum(f0, f0 + (sc.S - 1) * sc.scss_out)
    pv["TOA1"], pv["TOA2"] = -Wh, Wh
    if "SIGNAL" in pv.dtype.names:
        pv["SIGNAL"] = 1
    if "AmpSF" in pv.dtype.names:
        pv["AmpSF"] = 1.0
    tmp = out_path + ".part"
    with open(tmp, "wb") as fh, skcphd.Writer(fh, skcphd.Metadata(xmltree=tree)) as w:
        w.write_signal(c.ch, Z.astype(np.complex64))
        w.write_pvp(c.ch, pv)
        for sid, arr in meta["support"].items():
            w.write_support_array(sid, arr)
    os.replace(tmp, out_path)


# ------------------------------------------------------------------ one scene
def process_scene(ann, cphd_path, out_dir, cfg, dry_run=False, overwrite=False, log=print):
    t_scene = time.time()
    scene = ann["scene"]
    c = CPHD(cphd_path)
    sc = Scene(c, cfg)
    info = sc.summary()
    log(f"  CPHD {os.path.basename(cphd_path)}: {c.nv:,} pulses ({c.n_unusable} unusable), {c.ns:,} samples, "
        f"graze {info['graze_deg']:.1f} deg, dwell {info['aperture_full_deg']:.2f} deg; crop aperture "
        f"{info['aperture_crop_deg']:.2f} deg, band {info['band_mhz']:.0f} MHz ({info['res_rg_m']:.2f} m ground range)")
    if info["sc0_residual_samples"] > 0.05:
        log(f"  warning: SC0 moves by a fraction of a sample between pulses ({info['sc0_residual_samples']:.2f}); "
            f"the band is aligned to the nearest sample")
    if ann["geo_bounds"] is None:
        return dict(scene=scene, status="no geo_bounds and no boxes", **info), []
    rng = np.random.default_rng(cfg["seed"] + int(re.sub(r"\D", "", scene)[:12]) % 100000)
    chosen, counts = choose_crops(ann, cfg, sc.h_ref, rng)
    log(f"  crops: {counts['candidates']} candidates, {counts['positive']} positive, {counts['negative_kept']} of "
        f"{counts['negative_available']} negatives kept, {counts['ignore_only_dropped']} with only cut ships dropped")
    plans, skipped = {}, {}
    n_l = n_s = 0
    for r in chosen:
        if r["kind"] == "L":
            r["crop_id"] = f"{scene}_L{r['key']}"
        else:
            r["crop_id"] = f"{scene}_S{n_s:04d}"
            n_s += 1
        p, why = sc.plan(r["lat"], r["lon"])
        if p is None:
            skipped[why.split(":")[0]] = skipped.get(why.split(":")[0], 0) + 1
            r["skip"] = why
            continue
        plans[r["crop_id"]] = (r, p)
    if skipped:
        log(f"  skipped crops: {skipped}")
    if plans:
        bad = sum(not p["alias_free"] for _, p in plans.values())
        if bad:
            log(f"  warning: {bad} crops have a Doppler band too wide for {sc.P} pulses at {cfg['res_az']} m; "
                f"filter skirts fold back near the crop edges (raise --pulses or lower --margin)")
    est_gb = len(plans) * (sc.P * sc.S * 8 + sc.P * c.pv.dtype.itemsize) / 1e9
    summary = dict(scene=scene, status="planned" if dry_run else "", cphd=cphd_path, crops=len(plans),
                   positive_planned=sum(r["n_pos"] > 0 for r, _ in plans.values()), skipped=sum(skipped.values()),
                   est_gb=round(est_gb, 2), **counts, **info)
    log(f"  {len(plans)} crops to write, about {est_gb:.1f} GB")
    if dry_run or not plans:
        summary["status"] = summary["status"] or "nothing to write"
        return summary, []

    sdir = os.path.join(out_dir, scene)
    os.makedirs(sdir, exist_ok=True)
    todo = {k: v for k, v in plans.items()
            if overwrite or not os.path.exists(os.path.join(sdir, k + ".json"))}
    if len(todo) < len(plans):
        log(f"  {len(plans) - len(todo)} crops already written; {len(todo)} left")
    with open(cphd_path, "rb") as f, skcphd.Reader(f) as r:
        meta = dict(tree=copy.deepcopy(r.metadata.xmltree))
        meta["support"] = {s.findtext("{*}Identifier"): r.read_support_array(s.findtext("{*}Identifier"))
                           for s in meta["tree"].findall("{*}Data/{*}SupportArray")}
    xp = get_xp(cfg)
    budget = cfg["mem_gb"] * 1e9 * 0.5
    if xp is not np:                                         # GPU: buffers live in device memory
        free, total = xp.cuda.runtime.memGetInfo()
        budget = min(budget, 0.35 * free)
        log(f"  GPU {xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)['name'].decode()}: "
            f"{free / 1e9:.1f} of {total / 1e9:.1f} GB free, buffer budget {budget / 1e9:.1f} GB")
    groups, cur, size = [], {}, 0
    for k in sorted(todo, key=lambda k: todo[k][1]["n0"]):
        b = todo[k][1]["bytes"]
        if cur and size + b > budget:
            groups.append(cur); cur, size = {}, 0
        cur[k] = todo[k]; size += b
    if cur:
        groups.append(cur)
    for gi, grp in enumerate(groups):
        log(f"  group {gi + 1} of {len(groups)}: {len(grp)} crops, {sum(v[1]['bytes'] for v in grp.values()) / 1e9:.1f} GB buffer")
        bufs = extract_group(sc, {k: v[1] for k, v in grp.items()}, workers=cfg["workers"], log=log, xp=xp,
                             chunk=cfg["chunk"] if xp is np else max(cfg["chunk"], 1024))
        for k, (rec, p) in grp.items():
            Z = resample_slow_time(sc, p, bufs.pop(k), xp, cfg["resample"])
            path = os.path.join(sdir, k + ".cphd")
            write_crop(sc, meta, p, Z, path, rec["grid"].bounds)
            g = rec["grid"]
            label = dict(
                crop_id=k, scene=scene, kind="lattice" if rec["kind"] == "L" else "ship_centred", lattice_key=rec["key"],
                cphd=os.path.basename(path), source_cphd=cphd_path, annotations=ann["path"],
                centre_lat=rec["lat"], centre_lon=rec["lon"], h_ref=sc.h_ref,
                grid=dict(n=g.n, pixel_m=g.px, lon_w=g.lon_w, lat_n=g.lat_n, dlon=g.dlon, dlat=g.dlat,
                          bounds_geo=g.bounds, crs="EPSG:4326"),
                extent_geo=p["ext_bounds"],
                signal=dict(pulses=sc.P, samples=sc.S, format="CF8", aperture_deg=math.degrees(sc.dtheta),
                            res_az_m=cfg["res_az"], res_rg_m=sc.res_rg, band_hz=sc.B_t, scss_hz=float(abs(sc.scss_out)),
                            delay_gate_us=p["W"] * 1e6, sgn=c.sgn),
                slow_time=dict(orig_pulse_first=float(p["t_k"][0]), orig_pulse_last=float(p["t_k"][-1]),
                               orig_pulses_per_output=float(np.median(np.diff(p["t_k"]))), doppler_cycles_per_pulse=p["B_d"],
                               decimation=p["Da"], filter_taps=p["L"], alias_free=p["alias_free"],
                               partial_filter=p["partial_filter"]),
                label="positive" if rec["n_pos"] else "negative", n_pos=rec["n_pos"], n_ignore=rec["n_ignore"],
                negative_pick=rec.get("neg_pick"), nearest_ship_m=rec["nearest_ship_m"], boxes=rec["boxes"])
            with open(path[:-5] + ".json.part", "w") as fh:
                json.dump(label, fh, indent=1, default=float)
            os.replace(path[:-5] + ".json.part", path[:-5] + ".json")
        del bufs
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
        log(f"    written, {time.time() - t_scene:.0f} s so far")
    lines = []
    for k in sorted(plans):
        jp = os.path.join(sdir, k + ".json")
        if os.path.exists(jp):
            with open(jp) as fh:
                d = json.load(fh)
            lines.append({key: d[key] for key in ("crop_id", "scene", "kind", "label", "n_pos", "n_ignore",
                                                   "centre_lat", "centre_lon", "nearest_ship_m")}
                         | dict(cphd=os.path.join(scene, d["cphd"]), json=os.path.join(scene, k + ".json")))
    with open(os.path.join(sdir, "manifest.jsonl"), "w") as fh:
        fh.writelines(json.dumps(l) + "\n" for l in lines)
    summary.update(status="done", written=len(lines), seconds=round(time.time() - t_scene, 1))
    return summary, lines


# ------------------------------------------------------------------ run over all scenes
def run(args, cfg, log=print):
    files = find_annotation_files(args.ann_root)
    log(f"{len(files)} annotation files under {args.ann_root}")
    cidx = index_cphd(args.cphd_root)
    log(f"{len(cidx)} CPHD files under {args.cphd_root}")
    anns = []
    for f in files:
        try:
            anns.append(load_annotations(f))
        except Exception as e:
            log(f"  cannot read {f}: {e}")
    if args.scenes:
        anns = [a for a in anns if any(s in a["scene"] for s in args.scenes)]
    if args.limit:
        anns = anns[:args.limit]
    shard = getattr(args, "shard", None)
    if shard:                                               # i/N: largest CPHDs spread first, greedy by size
        i, n = map(int, shard.split("/"))
        size = lambda a: os.path.getsize(cidx[a["scene"]]) if a["scene"] in cidx else 0
        load, mine = [0] * n, []
        for a in sorted(anns, key=size, reverse=True):
            j = int(np.argmin(load))
            load[j] += size(a) + 1
            if j == i:
                mine.append(a)
        anns = sorted(mine, key=lambda a: a["scene"])
        log(f"shard {i}/{n}: {len(anns)} scenes, {load[i] / 1e9:.1f} GB of CPHD")
    os.makedirs(args.out, exist_ok=True)
    rows, all_lines = [], []
    for i, a in enumerate(anns):
        log(f"\n[{i + 1}/{len(anns)}] {a['scene']}: {len(a['anns'])} boxes")
        if a["scene"] not in cidx:
            log("  no CPHD for this scene")
            rows.append(dict(scene=a["scene"], status="no CPHD", boxes=len(a["anns"])))
            continue
        try:
            s, lines = process_scene(a, cidx[a["scene"]], args.out, cfg, args.dry_run, args.overwrite, log)
            s["boxes"] = len(a["anns"])
            rows.append(s)
            all_lines += lines
        except Exception as e:
            traceback.print_exc()
            rows.append(dict(scene=a["scene"], status=f"failed: {type(e).__name__}: {e}", boxes=len(a["anns"])))
    tag = f"_shard{shard.split('/')[0]}" if shard else ""
    report = os.path.join(args.out, ("crop_report_dry_run" if args.dry_run else "crop_report") + tag + ".csv")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(report, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if not args.dry_run and not shard:                      # whole-run manifest from every scene folder on disk
        write_run_manifest(args.out, log)
    tot = sum(r.get("crops", 0) for r in rows)
    gb = sum(r.get("est_gb", 0) for r in rows)
    log(f"{len(rows)} scenes, {sum(r.get('status') in ('done', 'planned') for r in rows)} usable, {tot} crops, "
        f"about {gb:.1f} GB. Report: {report}")
    return rows


def write_run_manifest(out, log=print):
    if True:
        lines = []
        for d in sorted(os.listdir(out)):
            mp = os.path.join(out, d, "manifest.jsonl")
            if os.path.isfile(mp):
                with open(mp) as fh:
                    lines += fh.readlines()
        with open(os.path.join(out, "manifest.jsonl"), "w") as fh:
            fh.writelines(lines)
        n_pos = sum('"label": "positive"' in l for l in lines)
        log(f"\nmanifest: {len(lines)} crops ({n_pos} positive, {len(lines) - n_pos} negative)")


# ------------------------------------------------------------------ synthetic data and self-test
def make_synthetic(path, targets, sgn=-1, nan_head=0, lat0=35.51, lon0=129.44):
    """A spotlight CPHD with point targets at given east/north offsets (m) from the scene centre, plus noise."""
    import lxml.etree
    rng = np.random.default_rng(1)
    fc, bw, scss, prf, dwell, r0, graze, vplat = 9.6e9, 300e6, 100e3, 2000.0, 2.4, 838e3, math.radians(35.0), 7600.0
    ns, nv = int(bw / scss), int(prf * dwell)
    srp = llh_to_ecef(lat0, lon0, 0.0)
    E, N, U = enu_basis(lat0, lon0)
    t_tx = np.arange(nv) / prf
    pos = lambda t: srp + r0 * (-math.cos(graze) * E + math.sin(graze) * U) + vplat * (np.asarray(t)[:, None] - dwell / 2) * N
    tx = pos(t_tx)
    t_rc = t_tx + 2 * np.linalg.norm(tx - srp, axis=1) / C
    rc = pos(t_rc)
    r_srp = two_way(tx, rc, srp)
    f = fc - bw / 2 + scss * np.arange(ns)
    sig = ((rng.standard_normal((nv, ns), dtype=np.float32) + 1j * rng.standard_normal((nv, ns), dtype=np.float32)) * 20).astype(np.complex64)
    truth = []
    for e, n in targets:
        p = srp + e * E + n * N
        dtau = ((two_way(tx, rc, p) - r_srp) / C).astype(np.float64)
        for a in range(0, nv, 600):
            sig[a:a + 600] += (5.0 * np.exp(sgn * 2j * np.pi * f[None, :] * dtau[a:a + 600, None])).astype(np.complex64)
        la, lo, _ = ecef_to_llh(p)
        truth.append(dict(east_m=e, north_m=n, lat=float(la), lon=float(lo)))
    fields = [("TxTime", 1, "F8"), ("TxPos", 3, "X=F8;Y=F8;Z=F8;"), ("TxVel", 3, "X=F8;Y=F8;Z=F8;"), ("RcvTime", 1, "F8"),
              ("RcvPos", 3, "X=F8;Y=F8;Z=F8;"), ("RcvVel", 3, "X=F8;Y=F8;Z=F8;"), ("SRPPos", 3, "X=F8;Y=F8;Z=F8;"),
              ("aFDOP", 1, "F8"), ("aFRR1", 1, "F8"), ("aFRR2", 1, "F8"), ("FX1", 1, "F8"), ("FX2", 1, "F8"), ("TOA1", 1, "F8"),
              ("TOA2", 1, "F8"), ("TDTropoSRP", 1, "F8"), ("SC0", 1, "F8"), ("SCSS", 1, "F8"), ("SIGNAL", 1, "I8")]
    off, nodes = 0, ""
    for name, size, fmt in fields:
        nodes += f"<{name}><Offset>{off}</Offset><Size>{size}</Size><Format>{fmt}</Format></{name}>"
        off += size
    xml = (f'<CPHD xmlns="http://api.nsgreg.nga.mil/schema/cphd/1.0.1"><CollectionID><CollectorName>SYNTH</CollectorName>'
           f'<CoreName>SYNTH</CoreName><CollectType>MONOSTATIC</CollectType><RadarMode><ModeType>SPOTLIGHT</ModeType></RadarMode>'
           f'<Classification>UNCLASSIFIED</Classification><ReleaseInfo>UNRESTRICTED</ReleaseInfo></CollectionID>'
           f'<Global><DomainType>FX</DomainType><SGN>{sgn}</SGN><Timeline><CollectionStart>2023-06-20T12:06:09.000000Z</CollectionStart>'
           f'<TxTime1>0.0</TxTime1><TxTime2>{t_tx[-1]}</TxTime2></Timeline></Global>'
           f'<SceneCoordinates><ImageArea><X1Y1><X>-700</X><Y>-700</Y></X1Y1><X2Y2><X>700</X><Y>700</Y></X2Y2></ImageArea></SceneCoordinates>'
           f'<Data><SignalArrayFormat>CF8</SignalArrayFormat><NumBytesPVP>{off * 8}</NumBytesPVP><NumCPHDChannels>1</NumCPHDChannels>'
           f'<Channel><Identifier>CH1</Identifier><NumVectors>{nv}</NumVectors><NumSamples>{ns}</NumSamples>'
           f'<SignalArrayByteOffset>0</SignalArrayByteOffset><PVPArrayByteOffset>0</PVPArrayByteOffset></Channel>'
           f'<NumSupportArrays>0</NumSupportArrays></Data><Channel><RefChId>CH1</RefChId></Channel><PVP>{nodes}</PVP></CPHD>')
    tree = lxml.etree.ElementTree(lxml.etree.fromstring(xml))
    pv = np.zeros(nv, dtype=skcphd.get_pvp_dtype(tree))
    pv["TxTime"], pv["TxPos"], pv["RcvTime"], pv["RcvPos"], pv["SRPPos"], pv["SIGNAL"] = t_tx, tx, t_rc, rc, srp, 1
    pv["TxVel"], pv["RcvVel"] = vplat * N, vplat * N
    pv["FX1"], pv["FX2"], pv["SC0"], pv["SCSS"] = f[0], f[-1], f[0], scss
    if nan_head:
        for name in ("TxTime", "TxPos", "RcvTime", "RcvPos", "SRPPos"):
            pv[name][:nan_head] = np.nan
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh, skcphd.Writer(fh, skcphd.Metadata(xmltree=tree)) as w:
        w.write_signal("CH1", sig)
        w.write_pvp("CH1", pv)
    return truth, (lat0, lon0)


def write_synthetic_annotations(path, scene, truth, centre, half_m=700.0, box_m=(60.0, 20.0)):
    lat0, lon0 = centre
    ml, mo = m_per_deg(lat0)
    gb = [lon0 - half_m / mo, lat0 - half_m / ml, lon0 + half_m / mo, lat0 + half_m / ml]
    px = 3.0e-6                                             # GEC-like pixel in degrees
    anns = []
    for i, t in enumerate(truth):
        bg = [t["lon"] - box_m[0] / 2 / mo, t["lat"] - box_m[1] / 2 / ml, t["lon"] + box_m[0] / 2 / mo, t["lat"] + box_m[1] / 2 / ml]
        bb = [round((bg[0] - gb[0]) / px), round((gb[3] - bg[3]) / px), round((bg[2] - bg[0]) / px), round((bg[3] - bg[1]) / px)]
        anns.append(dict(area=bb[2] * bb[3], bbox=bb, bbox_geo=bg, category_id=1, id=i + 1, iscrowd=0))
    d = dict(scene_name=f"{scene}_GEC", image=dict(file_name=f"{scene}_GEC.tif", width=round((gb[2] - gb[0]) / px),
             height=round((gb[3] - gb[1]) / px), geo_bounds=gb, crs="EPSG:4326"), annotations=anns)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)


def backproject(c, pix, oversample=4, chunk=64):
    """Plain backprojection of every usable pulse of a CPHD onto ECEF pixels (full band, no taper)."""
    nb = c.n_hi - c.n_lo
    nfft = 1 << int(math.ceil(math.log2(nb * oversample)))
    acc = np.zeros(len(pix), np.complex128)
    with open(c.path, "rb") as f, skcphd.Reader(f) as reader:
        for a in range(0, c.nv, chunk):
            b = min(c.nv, a + chunk)
            sig = c.read(reader, a, b)[:, c.n_lo:c.n_hi]
            if c.sgn > 0:
                sig = np.conj(sig)
            prof = np.fft.fftshift(np.fft.ifft(sig, n=nfft, axis=1), axes=1)
            for i in range(a, b):
                if not c.valid[i]:
                    continue
                dtau = (two_way(c.tx_pos[i], c.rcv_pos[i], pix) - two_way(c.tx_pos[i], c.rcv_pos[i], c.srp[i])) / C
                kk = dtau * nfft * c.scss[i] + nfft // 2
                k0 = np.floor(kk)
                fr = kk - k0
                ki = np.clip(k0, 0, nfft - 2).astype(np.int64)
                p = prof[i - a]
                val = (p[ki] * (1 - fr) + p[ki + 1] * fr) * np.exp(2j * np.pi * (c.sc0[i] + c.n_lo * c.scss[i]) * dtau)
                acc += np.where((k0 >= 0) & (k0 < nfft - 1), val, 0)
    return acc


def selftest(work, workers=2):
    shutil.rmtree(work, ignore_errors=True)
    ann_root, cphd_root, out = os.path.join(work, "ann"), os.path.join(work, "cphd"), os.path.join(work, "crops")
    targets = [(350.0, 250.0), (-60.0, -40.0)]                      # two annotated ships, 500 m apart
    ok = True
    scenes = [("2023-06-20-12-06-09_UMBRA-05", -1, 9), ("2023-07-25-02-40-15_UMBRA-05", +1, 0)]
    truths = {}
    t0 = time.time()
    for scene, sgn, nan_head in scenes:
        truth, centre = make_synthetic(os.path.join(cphd_root, scene, f"{scene}_CPHD.cphd"), targets, sgn, nan_head)
        write_synthetic_annotations(os.path.join(ann_root, f"{scene}_GEC", "annotations.json"), scene, truth, centre)
        truths[scene] = truth
    print(f"synthetic scenes written in {time.time() - t0:.0f} s (SGN -1 with 9 unusable pulses, SGN +1)")
    args = argparse.Namespace(ann_root=ann_root, cphd_root=cphd_root, out=out, dry_run=False, overwrite=False,
                              scenes=None, limit=0)
    cfg = dict(DEFAULTS, stride_m=512.0, neg_ratio=1.0, workers=workers, mem_gb=2.0)
    rows = run(args, cfg, log=lambda *a: print(*a))
    good = all(r.get("status") == "done" for r in rows)
    ok &= good
    print(f"\nboth scenes processed | {'ok' if good else 'FAIL'}")
    with open(os.path.join(out, "manifest.jsonl")) as fh:
        man = [json.loads(l) for l in fh]
    n_pos = sum(m["label"] == "positive" for m in man)
    good = n_pos >= 2 and len(man) > n_pos and all(os.path.exists(os.path.join(out, m["cphd"])) for m in man)
    ok &= good
    print(f"manifest: {len(man)} crops, {n_pos} positive, all files present | {'ok' if good else 'FAIL'}")
    for scene, sgn, _ in scenes:
        truth = truths[scene]
        lab = None
        for m in man:
            if m["scene"] == scene and m["kind"] == "ship_centred":
                with open(os.path.join(out, m["json"])) as fh:
                    d = json.load(fh)
                if any(b["ann_id"] == 1 and b["visible"] > 0.99 for b in d["boxes"]):
                    lab = d
                    break
        if lab is None:
            print(f"{scene}: no crop centred on ship 1 | FAIL"); ok = False; continue
        c = CPHD(os.path.join(out, scene, lab["cphd"]))
        g = GeoGrid(lab["centre_lat"], lab["centre_lon"], 512, 1.0, lab["h_ref"])
        b1 = next(b for b in lab["boxes"] if b["ann_id"] == 1)
        rr, cc = [float(v) for v in g.ll_to_rc(truth[0]["lat"], truth[0]["lon"])]
        x0, y0, x1, y1 = b1["bbox_px"]
        lab_ok = (c.nv, c.ns) == (1024, 1024) and abs(0.5 * (x0 + x1) - (cc + 0.5)) < 0.6 and abs(0.5 * (y0 + y1) - (rr + 0.5)) < 0.6 \
            and abs((x1 - x0) - 60) < 1 and abs((y1 - y0) - 20) < 1 and all(b["ann_id"] != 2 for b in lab["boxes"])
        ok &= lab_ok
        print(f"{scene} (SGN {sgn:+d}) crop {lab['crop_id']}: {c.nv} x {c.ns}, ship 1 box {x1 - x0:.1f} x {y1 - y0:.1f} px "
              f"centred on the target, ship 2 not in the core | {'ok' if lab_ok else 'FAIL'}")
        fine = GeoGrid(truth[0]["lat"], truth[0]["lon"], 16, 0.125, lab["h_ref"])
        amp = np.abs(backproject(c, fine.ecef())).reshape(fine.n, fine.n)
        pr, pc = np.unravel_index(np.argmax(amp), amp.shape)
        r0, c0 = [float(v) for v in fine.ll_to_rc(truth[0]["lat"], truth[0]["lon"])]
        off_m = math.hypot(pr - r0, pc - c0) * 0.125
        def irw(profile):
            prof = profile / profile.max()
            k = int(np.argmax(prof))
            lft = k - np.argmax(prof[k::-1] < 0.7071)
            rgt = k + np.argmax(prof[k:] < 0.7071)
            return (rgt - lft) * 0.125
        irw_n, irw_e = irw(amp[:, pc]), irw(amp[pr, :])          # north = cross-range, east = ground range here
        peak = amp.max()
        g2 = GeoGrid(truth[1]["lat"], truth[1]["lon"], 8, 1.0, lab["h_ref"])
        leak = 20 * math.log10(np.abs(backproject(c, g2.ecef())).max() / peak)
        good = off_m <= 0.5 and 0.7 <= irw_n <= 1.3 and 0.7 <= irw_e <= 1.3 and leak < -30
        ok &= good
        print(f"  imaged from the crop CPHD: ship 1 within {off_m:.2f} m of truth, 3 dB width {irw_n:.2f} m cross-range x "
              f"{irw_e:.2f} m range; ship 2 (outside the crop) at {leak:.1f} dB | {'ok' if good else 'FAIL'}")
    scene = scenes[0][0]                                    # the GPU resampler (direct taps) against the CPU one (upfirdn)
    sc = Scene(CPHD(os.path.join(cphd_root, scene, f"{scene}_CPHD.cphd")), cfg)
    p, _ = sc.plan(truths[scene][0]["lat"], truths[scene][0]["lon"])
    Y = extract_group(sc, {"x": p}, workers=workers)["x"]
    za, zb = resample_slow_time(sc, p, Y, np, "upfirdn"), resample_slow_time(sc, p, Y, np, "direct")
    corr = float(np.abs(np.vdot(za, zb)) / (np.linalg.norm(za) * np.linalg.norm(zb)))
    good = corr > 0.999
    ok &= good
    print(f"direct-tap resampler (GPU path) vs upfirdn: correlation {corr:.5f} | {'ok' if good else 'FAIL'}")
    neg = [m for m in man if m["label"] == "negative"]
    good = bool(neg) and all(m["n_pos"] == 0 for m in neg)
    ok &= good
    print(f"negatives: {len(neg)}, none with a box | {'ok' if good else 'FAIL'}")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def launch(a, argv):
    """One child process per GPU (or per --procs), each taking a share of the scenes; then one manifest."""
    if a.gpus:
        slots = [dict(CUDA_VISIBLE_DEVICES=g.strip()) for g in a.gpus.split(",") if g.strip()]
        extra = ["--device", "gpu"]
    else:
        slots, extra = [dict() for _ in range(a.procs)], []
    skip, rest = False, []
    for x in argv:                                          # pass every flag through except the launcher's own
        if skip:
            skip = False
            continue
        if x in ("--gpus", "--procs"):
            skip = True
            continue
        if x.startswith(("--gpus=", "--procs=")):
            continue
        rest.append(x)
    os.makedirs(os.path.join(a.out, "_logs"), exist_ok=True)
    import subprocess
    procs = []
    for i, env in enumerate(slots):
        log_path = os.path.join(a.out, "_logs", f"shard{i}.log")
        cmd = [sys.executable, os.path.abspath(__file__)] + rest + extra + ["--shard", f"{i}/{len(slots)}"]
        print(f"shard {i}: {' '.join(f'{k}={v}' for k, v in env.items()) or 'cpu'}, log {log_path}", flush=True)
        procs.append((i, subprocess.Popen(cmd, env=dict(os.environ, **env), stdout=open(log_path, "w"),
                                          stderr=subprocess.STDOUT)))
    failed = 0
    for i, p in procs:
        rc = p.wait()
        failed += rc != 0
        print(f"shard {i} finished with exit code {rc}", flush=True)
    if not a.dry_run:
        write_run_manifest(a.out)
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description="Fixed-shape CPHD crops with geo ship labels")
    ap.add_argument("--ann-root", default="/data3/sandbox/jhu/satlock/annotation_tool/test_aligned",
                    help="folder with one <scene>_GEC/annotations.json per scene")
    ap.add_argument("--cphd-root", default="/home/tabtahi/SATLOCK/dataset/cphd", help="folder searched for <scene>*_CPHD.cphd")
    ap.add_argument("--out", default="/home/tabtahi/SATLOCK/dataset/cphd_crops")
    ap.add_argument("--scenes", nargs="*", help="only scenes whose name contains one of these")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes")
    ap.add_argument("--dry-run", action="store_true", help="plan crops and sizes, write nothing but the report")
    ap.add_argument("--overwrite", action="store_true", help="rewrite crops that already exist")
    for k, v in DEFAULTS.items():
        if k in ("h_ref",):
            ap.add_argument("--h-ref", type=float, default=None, help="plane height above the ellipsoid, m (default: CPHD reference point)")
            continue
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids, e.g. 0,1,2,3: one process per GPU, scenes shared out")
    ap.add_argument("--procs", type=int, default=1, help="CPU processes, scenes shared out (ignored with --gpus)")
    ap.add_argument("--shard", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-dir", default="./cphd_crops_selftest")
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    if a.selftest:
        sys.exit(0 if selftest(a.selftest_dir, a.workers) else 1)
    if (a.gpus or a.procs > 1) and not a.shard:
        sys.exit(launch(a, sys.argv[1:]))
    run(a, cfg)


if __name__ == "__main__":
    main()
