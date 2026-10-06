#!/usr/bin/env python3
"""CPHD ground-projection viewer: one file, served over HTTP.

Give it a CPHD (FX domain, spotlight). In the browser choose the full scene or a sub-area of a size you set, and a
slow-time window t1..t2. The pulses in that window are backprojected onto a latitude/longitude grid (EPSG:4326) at one
ellipsoid height, and the image is shown and can be downloaded as a GeoTIFF.

    python cphd_viewer.py --data /data --cache /cache --port 8095          # pick a .cphd under /data in the browser
    python cphd_viewer.py --cphd /data/scene_CPHD.cphd --port 8095         # or name the file up front
    python cphd_viewer.py --selftest                                       # synthetic end-to-end check, exits 0 on pass

How it stays fast: the collect is cut into blocks of pulses. Only blocks inside the requested window are projected,
in parallel over worker processes, and each finished block is kept in memory and on disk. Projection is a sum over
pulses, so any window made of finished blocks is drawn at once.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                      # one thread per worker process; parallelism comes from the pool

import argparse, hashlib, io, json, math, multiprocessing as mp, shutil, sys, threading, time, traceback
import numpy as np
from scipy.signal.windows import taylor

C = 299792458.0
N_BLOCKS = 48
OVERSAMPLE = 4
_A, _E2 = 6378137.0, 6.69437999014e-3                   # WGS-84


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


# ------------------------------------------------------------------ CPHD access
class CPHD:
    """Header fields and per-pulse parameters of one CPHD channel. Pulses with unusable parameters are skipped."""
    def __init__(self, path):
        import sarkit.cphd as skcphd, pandas as pd
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
            ts = pd.Timestamp(xml.findtext("{*}Global/{*}Timeline/{*}CollectionStart"))
            self.t0 = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts
            area = []
            for tag in ("X1Y1", "X2Y2"):                    # scene extent in the image plane, metres, if the file states it
                for ax in ("X", "Y"):
                    v = xml.findtext(f"{{*}}SceneCoordinates/{{*}}ImageArea/{{*}}{tag}/{{*}}{ax}")
                    area.append(float(v) if v is not None else float("nan"))
            pv = r.read_pvps(self.ch)
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
        self.n_flagged, self.n_nonfinite = int((~flagged).sum()), int((~finite).sum())
        self.good = np.nonzero(self.valid)[0]
        if len(self.good) < 16:
            raise ValueError(f"only {len(self.good)} usable pulses in {path}")
        self.t = np.interp(np.arange(self.nv), self.good, self.tx_time[self.good]) - self.tx_time[self.good[0]]
        self.t_end = float(self.t[self.good[-1]])
        self.prf = (len(self.good) - 1) / self.t_end
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
        self.full = self.geometry(0, self.nv)
        graze = math.radians(self.full["graze_deg"])
        swath = C / (2 * abs(self.scss[m])) / math.cos(graze)          # unambiguous ground-range extent of the samples
        stated = max(abs(area[2] - area[0]), abs(area[3] - area[1])) if np.all(np.isfinite(area)) else float("nan")
        self.scene_size_m = float(stated) if np.isfinite(stated) and 50 < stated < 1e5 else float(min(swath, 2e4))
        self.scene_size_from = "file" if np.isfinite(stated) and 50 < stated < 1e5 else "range swath"

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

    def geometry(self, v0, v1):
        """Viewing geometry and resolution of the usable pulses in [v0, v1)."""
        vi = self.good[(self.good >= v0) & (self.good < v1)]
        if len(vi) < 2:
            return None
        E, N, U = enu_basis(self.lat0, self.lon0)
        arp = 0.5 * (self.tx_pos + self.rcv_pos)
        k = len(vi) // 2
        m, first, last = int(vi[k]), int(vi[0]), int(vi[-1])
        los = self.srp[m] - arp[m]
        rng = float(np.linalg.norm(los))
        a, b = arp[first] - self.srp[first], arp[last] - self.srp[last]
        dtheta = math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
        graze = math.asin(-(los @ U) / rng)
        i, j = int(vi[max(0, k - 5)]), int(vi[min(len(vi) - 1, k + 5)])
        vel = (self.tx_pos[j] - self.tx_pos[i]) / (self.tx_time[j] - self.tx_time[i])
        look = math.degrees(math.atan2(los @ E, los @ N)) % 360
        head = math.degrees(math.atan2(vel @ E, vel @ N)) % 360
        return dict(n_pulses=int(len(vi)), duration_s=float(self.t[last] - self.t[first]), slant_range_km=rng / 1e3,
                    graze_deg=math.degrees(graze), look_azimuth_deg=look, platform_heading_deg=head,
                    look_side="right" if ((look - head) % 360) < 180 else "left",
                    cross_range_res_m=(C / self.fc) / (2 * dtheta) if dtheta > 0 else float("inf"),
                    ground_range_res_m=C / (2 * self.bandwidth * math.cos(graze)))


class GeoGrid:
    """North-up square grid, regular in longitude and latitude (EPSG:4326), at one ellipsoid height."""
    def __init__(self, lat, lon, size_m, pixel_m, h_ref):
        s = math.sin(math.radians(lat))
        m_lat = math.pi / 180 * _A * (1 - _E2) / (1 - _E2 * s * s) ** 1.5
        m_lon = math.pi / 180 * _A / math.sqrt(1 - _E2 * s * s) * math.cos(math.radians(lat))
        n = max(8, int(round(size_m / pixel_m)))
        self.params = (float(lat), float(lon), float(size_m), float(pixel_m), float(h_ref))
        self.n, self.px, self.h_ref = n, float(pixel_m), float(h_ref)
        self.dlat, self.dlon = pixel_m / m_lat, pixel_m / m_lon
        self.lon_w, self.lat_n = lon - n * self.dlon / 2, lat + n * self.dlat / 2
        self.bounds = [self.lon_w, self.lat_n - n * self.dlat, self.lon_w + n * self.dlon, self.lat_n]   # W, S, E, N

    def ll_to_rc(self, lat, lon):
        return (self.lat_n - np.asarray(lat)) / self.dlat - 0.5, (np.asarray(lon) - self.lon_w) / self.dlon - 0.5

    def ecef(self):
        lat = self.lat_n - (np.arange(self.n) + 0.5) * self.dlat
        lon = self.lon_w + (np.arange(self.n) + 0.5) * self.dlon
        lon, lat = np.meshgrid(lon, lat)
        return llh_to_ecef(lat, lon, np.full(lat.shape, self.h_ref)).reshape(-1, 3)


# ------------------------------------------------------------------ projection
def backproject(cphd, v0, v1, pix, band_frac, oversample=OVERSAMPLE, chunk=32):
    """Sum of the usable pulses in [v0, v1) projected onto the pixels `pix` (N x 3 ECEF). Returns (complex64 N, count).

    FX-domain CPHD: sample n of a pulse is at frequency f_n = SC0 + n*SCSS and a scatterer adds
    exp(SGN * j*2*pi * f_n * dTOA), dTOA being its two-way delay minus that of the pulse's reference point (SRP).
    Per pulse: inverse FFT over frequency gives a range profile; each pixel reads it at its own dTOA, the carrier
    phase is removed, and the result is added up."""
    import sarkit.cphd as skcphd
    n_use = cphd.n_hi - cphd.n_lo
    nb = max(16, int(round(n_use * min(1.0, band_frac))))
    n0 = cphd.n_lo + (n_use - nb) // 2
    nfft = 1 << int(math.ceil(math.log2(nb * oversample)))
    w_rg = taylor(nb, nbar=4, sll=30).astype(np.float32)
    px_, py_, pz_ = pix[:, 0], pix[:, 1], pix[:, 2]
    acc, count = np.zeros(pix.shape[0], np.complex128), 0
    with open(cphd.path, "rb") as f, skcphd.Reader(f) as reader:
        for a in range(v0, v1, chunk):
            b = min(v1, a + chunk)
            if not cphd.valid[a:b].any():
                continue
            sig = cphd.read(reader, a, b)[:, n0:n0 + nb]
            if cphd.sgn > 0:
                sig = np.conj(sig)
            prof = np.fft.fftshift(np.fft.ifft(sig * w_rg[None, :], n=nfft, axis=1), axes=1) * (nfft / nb)
            for i in range(a, b):
                if not cphd.valid[i]:
                    continue
                tx, rc, srp = cphd.tx_pos[i], cphd.rcv_pos[i], cphd.srp[i]
                r_srp = np.linalg.norm(tx - srp) + np.linalg.norm(rc - srp)
                d = np.sqrt((px_ - tx[0]) ** 2 + (py_ - tx[1]) ** 2 + (pz_ - tx[2]) ** 2)
                d += np.sqrt((px_ - rc[0]) ** 2 + (py_ - rc[1]) ** 2 + (pz_ - rc[2]) ** 2)
                dtau = (d - r_srp) / C
                kk = dtau * (nfft * cphd.scss[i]) + nfft // 2
                k0 = np.floor(kk)
                frac = kk - k0
                ok = (k0 >= 0) & (k0 < nfft - 1)
                ki = np.clip(k0, 0, nfft - 2).astype(np.int64)
                p = prof[i - a]
                val = (p[ki] * (1.0 - frac) + p[ki + 1] * frac) * np.exp(2j * math.pi * (cphd.sc0[i] + n0 * cphd.scss[i]) * dtau)
                acc += np.where(ok, val, 0)
                count += 1
    return acc.astype(np.complex64), count


_W = {}                                                 # per-worker cache: the open CPHD description and the pixel positions

def _task(args):
    path, grid_params, band_frac, k, a, b = args
    key = (path, grid_params)
    if _W.get("key") != key:
        _W.clear()
        _W.update(key=key, cphd=CPHD(path), pix=GeoGrid(*grid_params).ecef())
    img, n = backproject(_W["cphd"], a, b, _W["pix"], band_frac)
    return k, a, b, img, n


# ------------------------------------------------------------------ engine: blocks, cache, jobs
class Engine:
    def __init__(self, cache_dir, workers, max_side=1500, cache_gb=30.0):
        self.cache_dir, self.workers, self.max_side, self.cache_gb = cache_dir, int(workers), int(max_side), float(cache_gb)
        os.makedirs(cache_dir, exist_ok=True)
        self.lock = threading.RLock()
        self.pool = None
        self.cphd = None
        self.gkey, self.grid, self.band_frac, self.blocks, self.counts = None, None, None, {}, {}
        self.job = dict(state="idle", done=0, total=0, t0=0.0, message="", error="")
        self.result = None
        self.version = 0
        self.rate = 0.6e7 * self.workers                 # pixel-pulses per second; replaced by a measurement after a run
        self.cancel = threading.Event()

    def start_pool(self):
        if self.pool is None:
            self.pool = mp.get_context("spawn").Pool(self.workers)

    def stop_pool(self):
        if self.pool is not None:
            self.pool.terminate(); self.pool.join(); self.pool = None

    # ---- file
    def load(self, path):
        cphd = CPHD(path)
        with self.lock:
            self.cphd = cphd
            va, vb = int(cphd.good[0]), int(cphd.good[-1]) + 1
            self.n_blocks = int(min(N_BLOCKS, max(1, (vb - va) // 8)))
            self.edges = np.linspace(va, vb, self.n_blocks + 1).round().astype(int)
            self.t_edges = np.append(cphd.t[self.edges[:-1]], cphd.t_end)
            self.step = float(cphd.t_end / self.n_blocks)
            self.gkey, self.grid, self.blocks, self.counts, self.result = None, None, {}, {}, None
        return self.meta()

    def meta(self):
        c = self.cphd
        if c is None:
            return None
        f = c.full
        return dict(path=c.path, name=os.path.basename(c.path), size_gb=os.path.getsize(c.path) / 1e9, n_pulses=c.nv,
                    n_usable=int(len(c.good)), n_flagged=c.n_flagged, n_nonfinite=c.n_nonfinite, n_samples=c.ns, fmt=c.fmt,
                    prf=c.prf, duration_s=c.t_end, fc_ghz=c.fc / 1e9, bandwidth_mhz=c.bandwidth / 1e6, lat0=c.lat0, lon0=c.lon0,
                    h0=c.h0, start_utc=str(c.t0), scene_size_m=c.scene_size_m, scene_size_from=c.scene_size_from,
                    graze_deg=f["graze_deg"], look_azimuth_deg=f["look_azimuth_deg"], look_side=f["look_side"],
                    slant_range_km=f["slant_range_km"], cross_range_res_m=f["cross_range_res_m"],
                    ground_range_res_m=f["ground_range_res_m"], n_blocks=self.n_blocks, t_edges=[float(t) for t in self.t_edges])

    # ---- planning
    def plan(self, p):
        """Turn the request into a grid, a block range and a cost estimate. Nothing is computed here."""
        c = self.cphd
        if c is None:
            raise ValueError("load a CPHD first")
        full = p.get("mode", "full") == "full"
        size = c.scene_size_m if full else float(p.get("size_m") or 0)
        if not (10 <= size <= 1e5):
            raise ValueError("size must be between 10 m and 100 km")
        lat = c.lat0 if full or p.get("center_lat") in (None, "") else float(p["center_lat"])
        lon = c.lon0 if full or p.get("center_lon") in (None, "") else float(p["center_lon"])
        finest = min(c.full["cross_range_res_m"], c.full["ground_range_res_m"]) / 1.5
        if p.get("pixel_m") in (None, "", 0):
            pixel = math.ceil(max(size / self.max_side, finest) * 100) / 100       # as fine as the pixel budget allows
        else:
            pixel = float(p["pixel_m"])
            if pixel <= 0 or size / pixel > 6000:
                raise ValueError("pixel size gives more than 6000 pixels a side; raise it or shrink the area")
        h_ref = c.h0 if p.get("h_ref") in (None, "") else float(p["h_ref"])
        grid = GeoGrid(lat, lon, size, pixel, h_ref)
        graze = math.radians(c.full["graze_deg"])
        band_frac = min(1.0, C / (2 * 1.5 * pixel * math.cos(graze)) / c.bandwidth)
        t_match = c.t_end * c.full["cross_range_res_m"] / (1.5 * pixel)
        if not np.isfinite(t_match) or t_match <= 0:
            t_match = c.t_end / 4
        tmode = p.get("time", "auto")
        if tmode == "full":
            k1, k2 = 0, self.n_blocks
        elif tmode == "custom":
            k1, k2 = self.k_range(float(p["t1"]), float(p["t2"]))
        else:                                               # the window whose cross-range resolution matches the pixel
            n0 = max(1, min(self.n_blocks, int(round(t_match / self.step))))
            k1 = (self.n_blocks - n0) // 2
            k2 = k1 + n0
        gkey = self._gkey(grid, band_frac)
        have = self._available(gkey)
        todo = [k for k in range(k1, k2) if k not in have]
        work = grid.n * grid.n * sum(int(self.edges[k + 1] - self.edges[k]) for k in todo)
        geo = c.geometry(int(self.edges[k1]), int(self.edges[k2])) or {}
        return dict(grid=grid, gkey=gkey, band_frac=band_frac, k1=k1, k2=k2, todo=todo, work=work,
                    public=dict(n=grid.n, pixel_m=pixel, size_m=size, center_lat=lat, center_lon=lon, h_ref=h_ref,
                                band_pct=100 * band_frac, range_res_m=C / (2 * c.bandwidth * band_frac * math.cos(graze)),
                                cross_range_res_m=geo.get("cross_range_res_m"), t1=float(self.t_edges[k1]),
                                t2=float(self.t_edges[k2]), k1=k1, k2=k2, n_window_blocks=k2 - k1, n_todo=len(todo),
                                t_match=float(t_match), est_s=work / self.rate, ready=sorted(int(k) for k in have),
                                mem_gb=(k2 - k1) * grid.n * grid.n * 8 / 1e9, n_pulses=geo.get("n_pulses", 0)))

    def k_range(self, t1, t2):
        k1 = int(np.clip(np.argmin(np.abs(self.t_edges - t1)), 0, self.n_blocks - 1))
        k2 = int(np.clip(np.argmin(np.abs(self.t_edges - t2)), k1 + 1, self.n_blocks))
        return k1, k2

    def _gkey(self, grid, band_frac):
        c = self.cphd
        st = os.stat(c.path)
        return hashlib.md5(repr((os.path.basename(c.path), st.st_size, int(st.st_mtime), grid.params, round(band_frac, 6),
                                 self.edges.tolist(), OVERSAMPLE)).encode()).hexdigest()[:14]

    def _dir(self, gkey):
        return os.path.join(self.cache_dir, f"blocks_{gkey}")

    def _available(self, gkey):
        have = set(self.blocks) if gkey == self.gkey else set()
        d = self._dir(gkey)
        if os.path.isdir(d):
            have |= {int(f[:4]) for f in os.listdir(d) if f.endswith(".npz") and f[:4].isdigit()}
        return have

    def _trim_cache(self, keep):
        """Delete the oldest block folders when the cache grows past the limit."""
        dirs = []
        for name in os.listdir(self.cache_dir):
            d = os.path.join(self.cache_dir, name)
            if name.startswith("blocks_") and os.path.isdir(d):
                size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
                dirs.append((os.path.getmtime(d), size, d))
        total = sum(s for _, s, _ in dirs)
        for _, size, d in sorted(dirs):
            if total <= self.cache_gb * 1e9:
                break
            if os.path.abspath(d) != os.path.abspath(keep):
                shutil.rmtree(d, ignore_errors=True); total -= size

    # ---- running
    def run(self, p):
        with self.lock:
            if self.job["state"] == "running":
                raise RuntimeError("a reconstruction is already running")
            plan = self.plan(p)
            total = sum(int(self.edges[k + 1] - self.edges[k]) for k in plan["todo"])
            self.job = dict(state="running", done=0, total=total, t0=time.time(), message="starting", error="")
            self.cancel.clear()
        threading.Thread(target=self._run, args=(plan,), daemon=True).start()
        return plan["public"]

    def _run(self, plan):
        try:
            t_start = time.time()
            grid, gkey, bf, k1, k2 = plan["grid"], plan["gkey"], plan["band_frac"], plan["k1"], plan["k2"]
            d = self._dir(gkey)
            os.makedirs(d, exist_ok=True)
            os.utime(d, None)
            with self.lock:
                if gkey != self.gkey:                       # a different grid: drop the blocks held in memory
                    self.gkey, self.grid, self.band_frac, self.blocks, self.counts = gkey, grid, bf, {}, {}
            for k in range(k1, k2):                         # finished blocks from disk
                f = os.path.join(d, f"{k:04d}.npz")
                if k not in self.blocks and os.path.exists(f):
                    z = np.load(f)
                    self.blocks[k], self.counts[k] = z["img"], int(z["n"])
            todo = [k for k in range(k1, k2) if k not in self.blocks]
            if todo:
                self._trim_cache(keep=d)
                self.start_pool()
                n_pulses = sum(int(self.edges[k + 1] - self.edges[k]) for k in todo)
                piece = int(np.clip(math.ceil(n_pulses / (4 * self.workers)), 16, 256))
                tasks, left = [], {}
                for k in todo:
                    a0, b0 = int(self.edges[k]), int(self.edges[k + 1])
                    parts = [(a, min(b0, a + piece)) for a in range(a0, b0, piece)]
                    left[k] = len(parts)
                    tasks += [(self.cphd.path, grid.params, bf, k, a, b) for a, b in parts]
                acc = {k: np.zeros(grid.n * grid.n, np.complex64) for k in todo}
                cnt = {k: 0 for k in todo}
                self.job["message"] = f"projecting {len(todo)} blocks, {n_pulses} pulses, on {self.workers} workers"
                t_run = time.time()
                for k, a, b, img, n in self.pool.imap_unordered(_task, tasks):
                    if self.cancel.is_set():
                        self.stop_pool()
                        self.job.update(state="cancelled", message="cancelled; finished blocks were kept")
                        return
                    acc[k] += img; cnt[k] += n; left[k] -= 1
                    self.job["done"] += b - a
                    if left[k] == 0:                        # block complete: keep it in memory and on disk
                        self.blocks[k], self.counts[k] = acc.pop(k).reshape(grid.n, grid.n), cnt[k]
                        np.savez(os.path.join(d, f"{k:04d}.npz"), img=self.blocks[k], n=cnt[k])
                self.rate = grid.n * grid.n * n_pulses / max(time.time() - t_run, 1e-3)
            img = self.window_image(k1, k2)
            geo = self.cphd.geometry(int(self.edges[k1]), int(self.edges[k2])) or {}
            with self.lock:
                self.version += 1
                self.result = dict(img=img, grid=grid, k1=k1, k2=k2, version=self.version, public=dict(
                    plan["public"], version=self.version, bounds=grid.bounds, seconds=time.time() - t_start,
                    n_pulses=geo.get("n_pulses", 0), cross_range_res_m=geo.get("cross_range_res_m"),
                    pulse_first=int(self.edges[k1]), pulse_last=int(self.edges[k2]) - 1, computed_blocks=len(todo)))
                self.job.update(state="done", message="done")
        except Exception as e:
            traceback.print_exc()
            self.job.update(state="error", error=f"{type(e).__name__}: {e}", message="failed")

    def window_image(self, k1, k2, taper=True):
        """Coherent image from finished blocks k1..k2-1, with a block-level taper against cross-range sidelobes."""
        w = taylor(k2 - k1, nbar=4, sll=30) if (taper and k2 - k1 >= 4) else np.ones(k2 - k1)
        img = np.zeros(self.blocks[k1].shape, np.complex64)
        norm = 0.0
        for j, k in enumerate(range(k1, k2)):
            img += np.float32(w[j]) * self.blocks[k]
            norm += w[j] * self.counts[k]
        return img / np.float32(max(norm, 1.0))

    def status(self):
        j = dict(self.job)
        el = time.time() - j["t0"] if j["state"] == "running" else 0.0
        j["elapsed"] = el
        j["eta"] = (el / j["done"] * (j["total"] - j["done"])) if j["state"] == "running" and j["done"] else None
        j["result"] = self.result["public"] if self.result else None
        j["ready"] = sorted(int(k) for k in self._available(self.gkey)) if self.gkey else []
        return j

    # ---- outputs
    def png(self, lo, hi):
        from PIL import Image
        r = self.result
        if r is None:
            return None
        power = np.abs(r["img"]).astype(np.float32) ** 2
        ref = float(np.median(power[power > 0])) if np.any(power > 0) else 1.0
        db = 10 * np.log10(np.maximum(power, 1e-30) / ref)
        u8 = np.clip((db - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8).save(buf, format="PNG", compress_level=3)
        return buf.getvalue()

    def geotiff(self, kind="complex", lo=-5.0, hi=30.0):
        import rasterio
        from rasterio.transform import Affine
        r = self.result
        if r is None:
            return None
        g, pub = r["grid"], r["public"]
        name = f"{os.path.splitext(os.path.basename(self.cphd.path))[0]}_t{pub['t1']:.2f}-{pub['t2']:.2f}s_{kind}_epsg4326.tif"
        path = os.path.join(self.cache_dir, name)
        kw = dict(driver="GTiff", height=g.n, width=g.n, count=1, crs="EPSG:4326",
                  transform=Affine(g.dlon, 0, g.lon_w, 0, -g.dlat, g.lat_n), compress="deflate")
        if kind == "complex":
            with rasterio.open(path, "w", dtype="complex64", **kw) as dst:
                dst.write(r["img"].astype(np.complex64), 1)
        else:
            power = np.abs(r["img"]).astype(np.float32) ** 2
            ref = float(np.median(power[power > 0])) if np.any(power > 0) else 1.0
            db = 10 * np.log10(np.maximum(power, 1e-30) / ref)
            with rasterio.open(path, "w", dtype="uint8", **kw) as dst:
                dst.write(np.clip((db - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), 1)
        return path


# ------------------------------------------------------------------ synthetic data and self-test
def make_synthetic(outdir, nan_head=0):
    """A small CPHD with four fixed point targets at known positions, one moving target, and noise."""
    import sarkit.cphd as skcphd, lxml.etree
    rng = np.random.default_rng(0)
    lat0, lon0 = 22.60, 120.15
    fc, bw, scss, prf, dwell, r0, graze, vplat = 9.6e9, 300e6, 200e3, 600.0, 4.0, 838e3, math.radians(35.0), 7600.0
    ns, nv = int(bw / scss), int(prf * dwell)
    srp = llh_to_ecef(lat0, lon0, 0.0)
    E, N, U = enu_basis(lat0, lon0)
    t_tx = np.arange(nv) / prf
    pos = lambda t: srp + r0 * (-math.cos(graze) * E + math.sin(graze) * U) + vplat * (np.asarray(t)[:, None] - dwell / 2) * N
    tx = pos(t_tx)
    t_rc = t_tx + 2 * np.linalg.norm(tx - srp, axis=1) / C
    rc = pos(t_rc)
    r_srp = np.linalg.norm(tx - srp, axis=1) + np.linalg.norm(rc - srp, axis=1)
    f, n_idx = fc - bw / 2 + scss * np.arange(ns), np.arange(ns)
    sig = ((rng.standard_normal((nv, ns), dtype=np.float32) + 1j * rng.standard_normal((nv, ns), dtype=np.float32)) * 20).astype(np.complex64)
    def add(p, amp=5.0):
        dtau = (np.linalg.norm(tx - p, axis=1) + np.linalg.norm(rc - p, axis=1) - r_srp) / C
        sig[...] += (amp * np.exp(-2j * np.pi * f[0] * dtau)[:, None] * np.exp(-2j * np.pi * scss * dtau[:, None] * n_idx[None, :])).astype(np.complex64)
    truth = []
    for e, n in [(0.0, 0.0), (150.0, 100.0), (-200.0, -50.0), (80.0, -220.0)]:
        p = srp + e * E + n * N
        add(p)
        la, lo, _ = ecef_to_llh(p)
        truth.append(dict(east_m=e, north_m=n, lat=float(la), lon=float(lo)))
    tt = (t_tx - dwell / 2)[:, None]
    add(srp + (-60.0 + 1.0 * tt) * E + (160.0 + 6.0 * tt) * N)                    # the moving target
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
           f'<Global><DomainType>FX</DomainType><SGN>-1</SGN><Timeline><CollectionStart>2023-04-15T01:17:55.000000Z</CollectionStart>'
           f'<TxTime1>0.0</TxTime1><TxTime2>{t_tx[-1]}</TxTime2></Timeline></Global>'
           f'<SceneCoordinates><ImageArea><X1Y1><X>-350</X><Y>-350</Y></X1Y1><X2Y2><X>350</X><Y>350</Y></X2Y2></ImageArea></SceneCoordinates>'
           f'<Data><SignalArrayFormat>CF8</SignalArrayFormat><NumBytesPVP>{off * 8}</NumBytesPVP><NumCPHDChannels>1</NumCPHDChannels>'
           f'<Channel><Identifier>CH1</Identifier><NumVectors>{nv}</NumVectors><NumSamples>{ns}</NumSamples>'
           f'<SignalArrayByteOffset>0</SignalArrayByteOffset><PVPArrayByteOffset>0</PVPArrayByteOffset></Channel>'
           f'<NumSupportArrays>0</NumSupportArrays></Data><Channel><RefChId>CH1</RefChId></Channel><PVP>{nodes}</PVP></CPHD>')
    tree = lxml.etree.ElementTree(lxml.etree.fromstring(xml))
    pv = np.zeros(nv, dtype=skcphd.get_pvp_dtype(tree))
    pv["TxTime"], pv["TxPos"], pv["RcvTime"], pv["RcvPos"], pv["SRPPos"], pv["SIGNAL"] = t_tx, tx, t_rc, rc, srp, 1
    pv["FX1"], pv["FX2"], pv["SC0"], pv["SCSS"] = f[0], f[-1] + scss, f[0], scss
    if nan_head:                                            # leading pulses with unusable parameters, as real files can have
        for name in ("TxTime", "TxPos", "RcvTime", "RcvPos", "SRPPos"):
            pv[name][:nan_head] = np.nan
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "synthetic_CPHD.cphd")
    with open(path, "wb") as fh, skcphd.Writer(fh, skcphd.Metadata(xmltree=tree)) as w:
        w.write_signal("CH1", sig)
        w.write_pvp("CH1", pv)
    return path, truth


def selftest(cache_dir, workers):
    """Synthetic end-to-end check through the same code path the web app uses."""
    work = os.path.join(cache_dir, "selftest")
    shutil.rmtree(work, ignore_errors=True)
    path, truth = make_synthetic(work, nan_head=9)
    eng = Engine(os.path.join(work, "cache"), workers)
    ok = True
    try:
        meta = eng.load(path)
        print(f"loaded: {meta['n_pulses']} pulses, {meta['n_usable']} usable, {meta['n_nonfinite']} with non-finite parameters, "
              f"scene {meta['scene_size_m']:.0f} m ({meta['scene_size_from']})")
        ok &= meta["n_nonfinite"] == 9 and abs(meta["scene_size_m"] - 700) < 1
        def run(p):
            eng.run(p)
            while eng.status()["state"] == "running":
                time.sleep(0.2)
            s = eng.status()
            if s["state"] != "done":
                raise RuntimeError(s["error"] or s["state"])
            return s["result"]
        def check(label):
            nonlocal ok
            r = eng.result
            amp, g, worst, seen = np.abs(r["img"]), r["grid"], 0.0, 0
            for t in truth:
                rr, cc = [float(v) for v in g.ll_to_rc(t["lat"], t["lon"])]
                r0, c0 = int(round(rr)), int(round(cc))
                if not (6 <= r0 < g.n - 6 and 6 <= c0 < g.n - 6):
                    continue
                win = amp[r0 - 6:r0 + 7, c0 - 6:c0 + 7]
                pr, pc = np.unravel_index(np.argmax(win), win.shape)
                worst = max(worst, math.hypot(r0 - 6 + pr - rr, c0 - 6 + pc - cc)); seen += 1
                ok &= 20 * math.log10(win.max() / np.median(amp)) > 12
            good = worst <= 1.0 and seen >= 2 and bool(np.isfinite(amp).all())
            ok &= good
            pub = r["public"]
            print(f"{label}: {pub['n']} px at {pub['pixel_m']} m, t {pub['t1']:.2f}-{pub['t2']:.2f} s, {seen} targets in view, "
                  f"worst offset {worst:.2f} px, {pub['computed_blocks']} blocks computed in {pub['seconds']:.1f} s | {'ok' if good else 'FAIL'}")
        run(dict(mode="full", pixel_m=1.0, time="auto")); check("full scene, matched window")
        run(dict(mode="sub", size_m=400, center_lat=truth[1]["lat"], center_lon=truth[1]["lon"], pixel_m=0.5, time="custom", t1=1.5, t2=2.5))
        check("sub-area, t 1.5-2.5 s")
        r1 = run(dict(mode="full", pixel_m=1.0, time="custom", t1=1.6, t2=2.3))            # inside finished blocks: nothing to compute
        ok &= r1["computed_blocks"] == 0
        print(f"re-windowing inside finished blocks computed {r1['computed_blocks']} blocks | {'ok' if r1['computed_blocks'] == 0 else 'FAIL'}")
        k1, k2 = eng.result["k1"], eng.result["k2"]                                        # parallel pieces vs one direct pass
        direct, n = backproject(eng.cphd, int(eng.edges[k1]), int(eng.edges[k2]), eng.grid.ecef(), eng.band_frac)
        summed = sum(eng.blocks[k] for k in range(k1, k2)).ravel()
        rel = float(np.abs(direct - summed).max() / np.abs(direct).max())
        ok &= rel < 1e-4
        print(f"parallel blocks vs one direct projection: largest relative difference {rel:.1e} | {'ok' if rel < 1e-4 else 'FAIL'}")
        png = eng.png(-5, 30); tif = eng.geotiff("complex")
        import rasterio
        with rasterio.open(tif) as src:
            good = str(src.crs) == "EPSG:4326" and src.dtypes[0] == "complex64" and png[:4] == b"\x89PNG"
        ok &= good
        print(f"PNG and EPSG:4326 complex GeoTIFF written | {'ok' if good else 'FAIL'}")
    finally:
        eng.stop_pool()
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# ------------------------------------------------------------------ web app
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CPHD ground projection</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--ink:#e9e4d4;--dim:#9b9684;--faint:#5d5a4e;--bg:#12140f;--panel:#1a1d16;--line:#2e3227;--amber:#f2b134;--amber2:#7a5a16;--ok:#9fc46a;--bad:#e2644b;
  --mono:"IBM Plex Mono","JetBrains Mono","DejaVu Sans Mono","Menlo","Consolas",monospace;--cond:"Barlow Condensed","Oswald","DIN Condensed","Roboto Condensed","Arial Narrow",sans-serif}
*{box-sizing:border-box}html,body{height:100%;margin:0}[hidden]{display:none!important}
body{background:var(--bg);color:var(--ink);font:12.5px/1.45 var(--mono);display:grid;grid-template-columns:348px 1fr;overflow:hidden;
  background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);background-size:64px 64px;background-position:-1px -1px}
aside{background:var(--panel);border-right:1px solid var(--line);overflow-y:auto;padding:18px 18px 28px}
h1{font:600 25px/1 var(--cond);letter-spacing:.09em;text-transform:uppercase;margin:0 0 3px}
h1 b{color:var(--amber);font-weight:600}
.sub{color:var(--dim);margin-bottom:16px}
h2{font:600 13px/1 var(--cond);letter-spacing:.2em;text-transform:uppercase;color:var(--amber);margin:20px 0 9px;display:flex;align-items:center;gap:9px}
h2:after{content:"";flex:1;height:1px;background:var(--line)}
h2 i{font-style:normal;color:var(--faint)}
label{display:block;color:var(--dim);margin:8px 0 3px}
input[type=text],input[type=number],select{width:100%;background:#0d0f0b;border:1px solid var(--line);color:var(--ink);font:inherit;padding:6px 8px;border-radius:0;outline:none}
input:focus,select:focus{border-color:var(--amber)}
input:disabled{color:var(--faint);background:transparent}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.seg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;border:1px solid var(--line)}
.seg label{margin:0;padding:6px 4px;text-align:center;cursor:pointer;color:var(--dim);border-right:1px solid var(--line)}
.seg label:last-child{border-right:0}
.seg input{display:none}.seg input:checked+span{color:var(--bg)}
.seg label:has(input:checked){background:var(--amber);color:var(--bg)}
button{font:600 14px/1 var(--cond);letter-spacing:.16em;text-transform:uppercase;background:transparent;color:var(--ink);border:1px solid var(--faint);padding:9px 12px;cursor:pointer}
button:hover{border-color:var(--amber);color:var(--amber)}
button.go{background:var(--amber);color:var(--bg);border-color:var(--amber);width:100%;padding:12px;font-size:16px;margin-top:14px}
button.go:hover{background:#ffc94f;color:var(--bg)}
button:disabled{opacity:.4;cursor:default}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:0}
dt{color:var(--dim)}dd{margin:0;text-align:right}
.plan{border-left:2px solid var(--amber);padding:6px 0 6px 10px;margin-top:12px;color:var(--ink);min-height:54px}
.plan.err{border-color:var(--bad);color:var(--bad)}
.hint{color:var(--faint);margin-top:4px}
main{display:grid;grid-template-rows:auto 1fr auto;min-width:0;min-height:0}
.bar{display:flex;gap:18px;align-items:center;padding:9px 16px;border-bottom:1px solid var(--line);background:rgba(18,20,15,.9);white-space:nowrap;overflow:hidden}
.bar .grow{flex:1;white-space:normal;line-height:1.35;min-width:0}
.bar input{width:58px}
.bar a{color:var(--amber);text-decoration:none;border-bottom:1px dotted var(--amber2)}
#stage{position:relative;overflow:hidden;cursor:crosshair;min-height:0}
#img{position:absolute;left:0;top:0;transform-origin:0 0;image-rendering:pixelated;outline:1px solid var(--amber2);user-select:none;-webkit-user-drag:none}
#empty{position:absolute;inset:0;display:grid;place-items:center;color:var(--faint);text-align:center;font:500 22px/1.5 var(--cond);letter-spacing:.14em;text-transform:uppercase}
#read{position:absolute;left:14px;bottom:12px;background:rgba(13,15,11,.86);border:1px solid var(--line);padding:5px 9px;pointer-events:none}
#prog{position:absolute;left:0;right:0;top:0;height:3px;background:transparent}
#prog div{height:100%;width:0;background:var(--amber);transition:width .3s linear}
#tl{border-top:1px solid var(--line);padding:9px 16px 13px;background:rgba(18,20,15,.92)}
#tl .cap{display:flex;justify-content:space-between;color:var(--dim);margin-bottom:5px}
#blocks{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:2px;height:26px;cursor:ew-resize;user-select:none}
#blocks div{background:#22261c;border-top:3px solid transparent}
#blocks div.ready{background:#4b5239}
#blocks div.win{border-top-color:var(--amber)}
#blocks div.win.ready{background:#8a8f63}
#blocks div.drag{outline:1px solid var(--amber)}
.ok{color:var(--ok)}.bad{color:var(--bad)}
</style></head><body>
<aside>
  <h1>CPHD <b>/</b> ground</h1>
  <div class="sub">Slow-time window to lat/lon image</div>

  <h2><i>01</i> File</h2>
  <select id="file"></select>
  <input type="text" id="path" placeholder="or type a path on the server" style="margin-top:6px">
  <button id="load" style="margin-top:8px;width:100%">Load</button>
  <dl id="meta" style="margin-top:12px"></dl>

  <h2><i>02</i> Area</h2>
  <div class="seg"><label><input type="radio" name="mode" value="full" checked><span>Full scene</span></label>
    <label><input type="radio" name="mode" value="sub"><span>Sub-area</span></label></div>
  <label>Side of the square, metres</label><input type="number" id="size" value="400" min="10" step="10">
  <div class="row"><div><label>Centre latitude</label><input type="number" id="clat" step="0.000001"></div>
    <div><label>Centre longitude</label><input type="number" id="clon" step="0.000001"></div></div>
  <div class="hint">Double-click the image to put the centre there.</div>
  <div class="row"><div><label>Pixel, metres</label><input type="number" id="pixel" placeholder="auto" min="0.05" step="0.05"></div>
    <div><label>Plane height, metres</label><input type="number" id="href" placeholder="scene ref" step="1"></div></div>

  <h2><i>03</i> Slow time</h2>
  <div class="seg"><label><input type="radio" name="time" value="auto" checked><span>Matched</span></label>
    <label><input type="radio" name="time" value="full"><span>Whole collect</span></label>
    <label><input type="radio" name="time" value="custom"><span>t1 to t2</span></label></div>
  <div class="row"><div><label>t1, seconds</label><input type="number" id="t1" step="0.05" min="0"></div>
    <div><label>t2, seconds</label><input type="number" id="t2" step="0.05" min="0"></div></div>
  <div class="hint">Matched = the window whose cross-range resolution fits the pixel. Or drag across the timeline.</div>

  <div class="plan" id="plan">Load a CPHD to begin.</div>
  <button class="go" id="go" disabled>Reconstruct</button>
  <button id="cancel" style="width:100%;margin-top:8px" disabled>Cancel</button>
</aside>
<main>
  <div class="bar"><span class="grow" id="info">No image yet.</span>
    <span>dB <input type="number" id="lo" value="-5" step="1"> to <input type="number" id="hi" value="30" step="1"></span>
    <button id="fit">Fit</button>
    <a id="dlc" href="#" hidden>GeoTIFF complex</a><a id="dla" href="#" hidden>GeoTIFF 8-bit</a></div>
  <div id="stage"><div id="prog"><div></div></div><img id="img" alt="" hidden draggable="false">
    <div id="empty">Choose an area and a time window,<br>then reconstruct</div><div id="read" hidden></div></div>
  <div id="tl"><div class="cap"><span id="tlcap">Slow time</span><span>pale = projected and kept &middot; amber = current window</span></div><div id="blocks"></div></div>
</main>
<script>
const $=id=>document.getElementById(id), q=(s)=>document.querySelector(s);
let meta=null, plan=null, result=null, view={s:1,x:0,y:0}, timer=null, polling=null;
const api=async(u,body)=>{const r=await fetch(u,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j};
const f=(v,d=2)=>v==null||!isFinite(v)?'n/a':Number(v).toFixed(d);
const dur=s=>s==null?'':s<90?Math.round(s)+' s':(s/60).toFixed(1)+' min';
function params(){return{mode:q('input[name=mode]:checked').value,size_m:$('size').value,center_lat:$('clat').value,center_lon:$('clon').value,pixel_m:$('pixel').value,h_ref:$('href').value,time:q('input[name=time]:checked').value,t1:$('t1').value,t2:$('t2').value}}
function showMeta(){const m=meta;if(!m){$('meta').innerHTML='';return}
  const rows=[['pulses',m.n_usable+' of '+m.n_pulses+' usable'],['collect',f(m.duration_s)+' s at '+f(m.prf,0)+' Hz'],['samples',m.n_samples+' '+m.fmt],['bandwidth',f(m.bandwidth_mhz,0)+' MHz at '+f(m.fc_ghz,3)+' GHz'],['scene centre',f(m.lat0,5)+', '+f(m.lon0,5)],['scene size',f(m.scene_size_m,0)+' m ('+m.scene_size_from+')'],['geometry',f(m.graze_deg,1)+'° grazing, '+m.look_side+'-looking'],['full resolution',f(m.cross_range_res_m)+' x '+f(m.ground_range_res_m)+' m']];
  $('meta').innerHTML=rows.map(r=>'<dt>'+r[0]+'</dt><dd>'+r[1]+'</dd>').join('');
  if(!$('clat').value){$('clat').value=m.lat0.toFixed(6);$('clon').value=m.lon0.toFixed(6)}
  if(!$('t1').value){$('t1').value=(m.duration_s*0.4).toFixed(2);$('t2').value=(m.duration_s*0.6).toFixed(2)}
  $('t1').max=$('t2').max=m.duration_s.toFixed(2);drawBlocks([])}
function enable(){const sub=q('input[name=mode]:checked').value==='sub',cus=q('input[name=time]:checked').value==='custom';
  $('size').disabled=$('clat').disabled=$('clon').disabled=!sub;$('t1').disabled=$('t2').disabled=!cus}
async function replan(){enable();if(!meta)return;try{plan=await api('/api/plan',params());const p=plan;
  $('plan').className='plan';$('plan').innerHTML=p.n+' x '+p.n+' px at '+f(p.pixel_m)+' m, '+f(p.size_m,0)+' m across<br>window '+f(p.t1)+' to '+f(p.t2)+' s: '+p.n_pulses+' pulses, cross-range '+f(p.cross_range_res_m)+' m<br>'+(p.n_todo?p.n_todo+' of '+p.n_window_blocks+' blocks to project, about '+dur(p.est_s):'<span class="ok">all '+p.n_window_blocks+' blocks ready: instant</span>');
  $('go').disabled=false;drawBlocks(p.ready,p.k1,p.k2)}catch(e){$('plan').className='plan err';$('plan').textContent=e.message;$('go').disabled=true}}
const soon=()=>{clearTimeout(timer);timer=setTimeout(replan,250)};
function drawBlocks(ready,k1,k2){if(!meta)return;const n=meta.n_blocks,set=new Set(ready||[]);let h='';
  for(let k=0;k<n;k++)h+='<div data-k="'+k+'" class="'+(set.has(k)?'ready ':'')+(k1!=null&&k>=k1&&k<k2?'win':'')+'" title="'+f(meta.t_edges[k])+' to '+f(meta.t_edges[k+1])+' s"></div>';
  $('blocks').innerHTML=h;$('tlcap').textContent='Slow time 0 to '+f(meta.duration_s)+' s, '+n+' blocks of '+f(meta.duration_s/n*1000,0)+' ms'}
let drag=null;
$('blocks').addEventListener('mousedown',e=>{const k=e.target.dataset.k;if(k==null)return;drag=[+k,+k];mark()});
$('blocks').addEventListener('mousemove',e=>{if(!drag)return;const k=e.target.dataset.k;if(k==null)return;drag[1]=+k;mark()});
window.addEventListener('mouseup',()=>{if(!drag)return;const a=Math.min(...drag),b=Math.max(...drag)+1;drag=null;
  q('input[name=time][value=custom]').checked=true;$('t1').value=meta.t_edges[a].toFixed(2);$('t2').value=meta.t_edges[b].toFixed(2);replan()});
function mark(){const a=Math.min(...drag),b=Math.max(...drag);[...$('blocks').children].forEach((d,i)=>d.classList.toggle('drag',i>=a&&i<=b))}
function apply(){$('img').style.transform='translate('+view.x+'px,'+view.y+'px) scale('+view.s+')'}
function fit(){if(!result)return;const st=$('stage').getBoundingClientRect(),s=Math.min(st.width,st.height)/result.n*0.96;view={s,x:(st.width-result.n*s)/2,y:(st.height-result.n*s)/2};apply()}
function loadImg(){if(!result)return;$('img').src='/api/image.png?lo='+$('lo').value+'&hi='+$('hi').value+'&v='+result.version;
  $('dlc').href='/api/geotiff?kind=complex';$('dla').href='/api/geotiff?kind=amplitude&lo='+$('lo').value+'&hi='+$('hi').value;$('dlc').hidden=$('dla').hidden=false}
function showResult(r,refit){const first=!result||refit;result=r;$('img').hidden=false;$('empty').hidden=true;loadImg();if(first)fit();
  $('info').textContent='t '+f(r.t1)+' to '+f(r.t2)+' s | pulses '+r.pulse_first+'..'+r.pulse_last+' | '+r.n+' px at '+f(r.pixel_m)+' m | cross-range '+f(r.cross_range_res_m)+' m, range '+f(r.range_res_m)+' m | '+(r.computed_blocks?r.computed_blocks+' blocks in '+dur(r.seconds):'from finished blocks')}
async function poll(){try{const s=await api('/api/status');const bar=q('#prog div');
  if(s.state==='running'){bar.style.width=(s.total?100*s.done/s.total:100)+'%';$('plan').className='plan';$('plan').innerHTML=s.message+'<br>'+s.done+' of '+s.total+' pulses, '+dur(s.elapsed)+' elapsed'+(s.eta!=null?', about '+dur(s.eta)+' left':'');drawBlocks(s.ready,plan&&plan.k1,plan&&plan.k2)}
  else{clearInterval(polling);polling=null;bar.style.width='0';$('go').disabled=false;$('cancel').disabled=true;
    if(s.state==='done'&&s.result){const moved=!result||result.center_lat!==s.result.center_lat||result.center_lon!==s.result.center_lon||result.size_m!==s.result.size_m||result.n!==s.result.n;showResult(s.result,moved)}
    if(s.state==='error'){$('plan').className='plan err';$('plan').textContent=s.error}else replan()}}catch(e){}}
$('go').onclick=async()=>{try{$('go').disabled=true;$('cancel').disabled=false;plan=await api('/api/run',params());if(!polling)polling=setInterval(poll,500);poll()}catch(e){$('plan').className='plan err';$('plan').textContent=e.message;$('go').disabled=false;$('cancel').disabled=true}};
$('cancel').onclick=()=>api('/api/cancel',{});
$('load').onclick=async()=>{const p=$('path').value.trim()||$('file').value;if(!p)return;$('load').disabled=true;$('load').textContent='Loading';
  try{meta=await api('/api/load',{path:p});result=null;$('img').hidden=true;$('empty').hidden=false;$('read').hidden=$('dlc').hidden=$('dla').hidden=true;$('info').textContent='No image yet.';$('clat').value=$('clon').value=$('t1').value=$('t2').value='';showMeta();replan()}
  catch(e){$('plan').className='plan err';$('plan').textContent=e.message}$('load').disabled=false;$('load').textContent='Load'};
document.querySelectorAll('aside input').forEach(e=>{if(e.id!=='path')e.addEventListener('input',soon)});
$('lo').onchange=$('hi').onchange=loadImg;$('fit').onclick=fit;
const stage=$('stage');let pan=null;
function ll(e){const b=result.bounds,r=stage.getBoundingClientRect(),u=(e.clientX-r.left-view.x)/view.s/result.n,v=(e.clientY-r.top-view.y)/view.s/result.n;return[b[3]-v*(b[3]-b[1]),b[0]+u*(b[2]-b[0]),u>=0&&u<=1&&v>=0&&v<=1]}
stage.addEventListener('wheel',e=>{if(!result)return;e.preventDefault();const r=stage.getBoundingClientRect(),k=e.deltaY<0?1.25:0.8,mx=e.clientX-r.left,my=e.clientY-r.top;
  view.x=mx-(mx-view.x)*k;view.y=my-(my-view.y)*k;view.s*=k;apply()},{passive:false});
stage.addEventListener('mousedown',e=>{if(result)pan=[e.clientX-view.x,e.clientY-view.y]});
window.addEventListener('mousemove',e=>{if(pan){view.x=e.clientX-pan[0];view.y=e.clientY-pan[1];apply()}
  if(result&&stage.contains(e.target)){const p=ll(e);$('read').hidden=!p[2];$('read').textContent='lat '+p[0].toFixed(6)+'  lon '+p[1].toFixed(6)}});
window.addEventListener('mouseup',()=>pan=null);
stage.addEventListener('dblclick',e=>{if(!result)return;const p=ll(e);if(!p[2])return;q('input[name=mode][value=sub]').checked=true;$('clat').value=p[0].toFixed(6);$('clon').value=p[1].toFixed(6);replan()});
window.addEventListener('resize',()=>result&&fit());
(async()=>{const s=await api('/api/state');$('file').innerHTML=s.files.length?s.files.map(p=>'<option value="'+p+'">'+p.replace(s.data_dir+'/','')+'</option>').join(''):'<option value="">no .cphd files under '+s.data_dir+'</option>';
  if(s.meta){meta=s.meta;$('file').value=meta.path;showMeta();replan()}enable();
  const st=await api('/api/status');if(st.result)showResult(st.result,true);if(st.state==='running'){$('go').disabled=true;$('cancel').disabled=false;polling=setInterval(poll,500)}})();
</script></body></html>"""


def create_app(eng, data_dir):
    from flask import Flask, Response, jsonify, request, send_file
    import logging
    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)      # the page polls for status; keep the log readable

    def fail(e, code=400):
        return jsonify(error=str(e)), code

    def clean(o):                                           # JSON cannot carry inf or nan
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return float(o) if math.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        return o

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/state")
    def state():
        files = []
        if data_dir and os.path.isdir(data_dir):
            for root, dirs, names in os.walk(data_dir):
                if root[len(data_dir):].count(os.sep) >= 4:
                    dirs[:] = []
                files += [os.path.join(root, n) for n in names if n.lower().endswith(".cphd")]
        return jsonify(clean(dict(files=sorted(files)[:500], data_dir=data_dir or "", meta=eng.meta(), workers=eng.workers)))

    @app.post("/api/load")
    def load():
        try:
            path = (request.get_json(force=True) or {}).get("path", "")
            if not os.path.isfile(path):
                return fail(f"no such file on the server: {path}")
            if eng.job["state"] == "running":
                return fail("a reconstruction is running; cancel it first", 409)
            return jsonify(clean(eng.load(path)))
        except Exception as e:
            traceback.print_exc()
            return fail(f"{type(e).__name__}: {e}")

    @app.post("/api/plan")
    def plan():
        try:
            return jsonify(clean(eng.plan(request.get_json(force=True) or {})["public"]))
        except Exception as e:
            return fail(e)

    @app.post("/api/run")
    def run():
        try:
            return jsonify(clean(eng.run(request.get_json(force=True) or {})))
        except Exception as e:
            return fail(e, 409 if isinstance(e, RuntimeError) else 400)

    @app.post("/api/cancel")
    def cancel():
        eng.cancel.set()
        return jsonify(ok=True)

    @app.get("/api/status")
    def status():
        return jsonify(clean(eng.status()))

    @app.get("/api/image.png")
    def image():
        png = eng.png(float(request.args.get("lo", -5)), float(request.args.get("hi", 30)))
        if png is None:
            return fail("no image yet", 404)
        return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/geotiff")
    def geotiff():
        path = eng.geotiff(request.args.get("kind", "complex"), float(request.args.get("lo", -5)), float(request.args.get("hi", 30)))
        if path is None:
            return fail("no image yet", 404)
        return send_file(path, as_attachment=True, download_name=os.path.basename(path))

    return app


def main():
    ap = argparse.ArgumentParser(description="CPHD ground-projection viewer")
    ap.add_argument("--cphd", help="CPHD file to load at start")
    ap.add_argument("--data", default="/data", help="folder searched for .cphd files to offer in the browser")
    ap.add_argument("--cache", default=os.environ.get("CPHD_CACHE", "/cache"), help="folder for finished blocks and exports")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--workers", type=int, default=max(1, min(64, (os.cpu_count() or 2) - 2)), help="worker processes")
    ap.add_argument("--max-side", type=int, default=1500, help="largest image side, in pixels, when the pixel size is automatic")
    ap.add_argument("--cache-gb", type=float, default=30.0, help="oldest block folders are deleted beyond this size")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic end-to-end check and exit")
    a = ap.parse_args()
    try:
        os.makedirs(a.cache, exist_ok=True)
    except OSError:
        a.cache = os.path.abspath("./cphd_cache"); os.makedirs(a.cache, exist_ok=True)
    if a.selftest:
        sys.exit(0 if selftest(a.cache, min(a.workers, 8)) else 1)
    eng = Engine(a.cache, a.workers, a.max_side, a.cache_gb)
    eng.start_pool()                                        # workers start before the web server's threads exist
    if a.cphd:
        m = eng.load(a.cphd)
        print(f"loaded {m['name']}: {m['n_usable']} usable pulses, {m['duration_s']:.2f} s, scene {m['scene_size_m']:.0f} m")
    print(f"CPHD viewer on http://{a.host}:{a.port}  |  {a.workers} workers  |  cache {a.cache}  |  data {a.data}")
    create_app(eng, a.data).run(host=a.host, port=a.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
