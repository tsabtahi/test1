#!/usr/bin/env python3
"""Cut a CPHD chip for a lat/lon box at a chosen ground resolution.

Set the inputs in the block below and run:  python chip_cphd.py
(Command-line flags still work and override the block, e.g. --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX --gsd 2.)

The chip is a valid CPHD whose reference point is the box centre: pulses re-referenced to that point, range gated to
the box (plus --margin), band cut for --gsd ground-range resolution, Doppler low-passed and resampled to pulses
uniform in look angle spanning lambda / (2 gsd). It is the same processing as build_crops.py (copied in, so this
file stands alone), for one box of any size instead of the fixed 512 m training crops. A JSON with the geometry is written beside it
(it also works as the JSON for crop_to_png.py, to look at the chip).

Pulses / samples are chosen from the box size: (box side x (1 + margin) / gsd) x --oversample, rounded up to a multiple
of 64, unless --pulses / --samples are given. The box is processed as a square of its longer side.
Needs: pip install numpy scipy sarkit lxml   (sarkit needs Python >= 3.12; cupy only for DEVICE = "gpu")
"""
import argparse, json, math, os, sys, time
import numpy as np

# ------------------------------------------------------------------ set these
CPHD_PATH = r"/home/tabtahi/SATLOCK/dataset/cphd/2023-08-25-15-09-21_UMBRA-06/2023-08-25-15-09-21_UMBRA-06_CPHD.cphd"
BBOX = [-81.7050, 41.5010, -81.7000, 41.5040]   # lon_min, lat_min, lon_max, lat_max (degrees); None = random box
RANDOM_BOX_M = 256.0     # with BBOX = None: side of the random square box, metres, placed inside the scene footprint
SEED = None              # with BBOX = None: random seed (None = different box each run; a number repeats it)
GSD_M = 1.0              # ground resolution, metres (range and cross-range)
OUT_PATH = r"/home/tabtahi/SATLOCK/dataset/chips/chip1.cphd"   # a .json is written next to it
MARGIN = 0.25            # extra extent around the box, fraction of its side
OVERSAMPLE = 1.6         # samples per resolution cell across the extent
PULSES = 0               # output pulses; 0 = from the box size
SAMPLES = 0              # output frequency samples; 0 = from the box size
H_REF = None             # ellipsoid height of the chip plane, metres; None = the CPHD reference height
DEVICE = "cpu"           # "cpu" or "gpu"
MEM_GB = 8.0
# ------------------------------------------------------------------

# ------------------------------------------------------------------ CPHD processing (self-contained copy of the
# pieces of build_crops.py this needs: reader, scene geometry, crop planning, fast-time gating, slow-time resampling,
# CPHD writer). Runs on Windows, macOS and Linux; GPU (CuPy) is optional.
import copy
if int(np.__version__.split(".")[0]) < 2:
    # sarkit calls np.asarray(..., copy=False), which numpy 1.x does not accept. Teach numpy 1.x that keyword so this
    # runs without upgrading numpy (copy=False / None = no forced copy, copy=True = copy).
    _np_asarray = np.asarray

    def _asarray_compat(a, dtype=None, order=None, *, copy=None, **kw):
        if copy:
            return np.array(a, dtype=dtype, order=order, copy=True, **kw)
        return _np_asarray(a, dtype=dtype, order=order, **kw)

    np.asarray = _asarray_compat
import scipy.fft as sp_fft
from scipy.signal import firwin, upfirdn
import sarkit.cphd as skcphd

C = 299792458.0
_A, _E2 = 6378137.0, 6.69437999014e-3                       # WGS-84


DEFAULTS = dict(core_m=512.0, pixel_m=1.0, pulses=1024, samples=1024, res_az=1.0, res_rg=1.0, margin=0.25, guard=0.25,
                stride_m=256.0, jitter_m=64.0, min_visible=0.3, neg_ratio=3.0, neg_empty=8, interp_taps=24,
                mem_gb=8.0, h_ref=None, seed=0, workers=4, device="cpu", resample="auto", chunk=256)


def llh_to_ecef(lat, lon, h):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    h = np.asarray(h, float)
    n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo), (n * (1 - _E2) + h) * np.sin(la)], -1)


def ecef_to_llh(xyz):
    """ECEF (..., 3) -> lat, lon (degrees), height above the WGS-84 ellipsoid (m). Iterative, no pyproj needed."""
    xyz = np.asarray(xyz, float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(10):
        n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - _E2 * n / (n + h)))
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


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



def scene_footprint(cphd_path, c):
    """Scene footprint [lon_min, lat_min, lon_max, lat_max] from the CPHD header: the image-area corner points if given,
    else the image area rectangle around the reference point, else 500 m around the reference point."""
    with open(cphd_path, "rb") as f, skcphd.Reader(f) as r:
        sc = r.metadata.xmltree.find("{*}SceneCoordinates")
    if sc is not None:
        pts = [(float(p.findtext("{*}Lat")), float(p.findtext("{*}Lon"))) for p in sc.findall("{*}ImageAreaCornerPoints/{*}IACP")]
        if len(pts) >= 3:
            la, lo = zip(*pts)
            return [min(lo), min(la), max(lo), max(la)], "image-area corner points"
    lat0, lon0 = c.lat0, c.lon0
    if sc is not None and sc.findtext("{*}IARP/{*}LLH/{*}Lat"):
        lat0, lon0 = float(sc.findtext("{*}IARP/{*}LLH/{*}Lat")), float(sc.findtext("{*}IARP/{*}LLH/{*}Lon"))
    E, N, U = enu_basis(lat0, lon0)
    ux, uy = E, N
    if sc is not None:
        def vec(path):
            v = sc.find(path)
            return None if v is None else np.array([float(v.findtext("{*}" + k)) for k in "XYZ"])
        vx, vy = vec("{*}ReferenceSurface/{*}Planar/{*}uIAX"), vec("{*}ReferenceSurface/{*}Planar/{*}uIAY")
        if vx is not None and vy is not None:
            ux, uy = vx, vy
        x1, y1 = sc.findtext("{*}ImageArea/{*}X1Y1/{*}X"), sc.findtext("{*}ImageArea/{*}X1Y1/{*}Y")
        x2, y2 = sc.findtext("{*}ImageArea/{*}X2Y2/{*}X"), sc.findtext("{*}ImageArea/{*}X2Y2/{*}Y")
        if None not in (x1, y1, x2, y2):
            ml, mo = m_per_deg(lat0)
            cs = [(x, y) for x in (float(x1), float(x2)) for y in (float(y1), float(y2))]
            de = [x * (ux @ E) + y * (uy @ E) for x, y in cs]
            dn = [x * (ux @ N) + y * (uy @ N) for x, y in cs]
            return [lon0 + min(de) / mo, lat0 + min(dn) / ml, lon0 + max(de) / mo, lat0 + max(dn) / ml], "image area"
    ml, mo = m_per_deg(lat0)
    return [lon0 - 500 / mo, lat0 - 500 / ml, lon0 + 500 / mo, lat0 + 500 / ml], "500 m around the reference point"


def random_box(a, tries=30):
    """A random square box of a.random_box_m inside the scene footprint that the CPHD can actually chip."""
    rng = np.random.default_rng(a.seed)
    c = CPHD(a.cphd)
    (W, S, E_, N_), how = scene_footprint(a.cphd, c)
    lat_c = 0.5 * (S + N_)
    ml, mo = m_per_deg(lat_c)
    half = 0.5 * a.random_box_m * (1 + a.margin)
    lo_a, lo_b, la_a, la_b = W + half / mo, E_ - half / mo, S + half / ml, N_ - half / ml
    if lo_a > lo_b or la_a > la_b:
        raise SystemExit(f"a {a.random_box_m:.0f} m box does not fit in the scene footprint ({how}); lower --random-box-m")
    side = a.random_box_m
    cfg = dict(DEFAULTS, core_m=side, pixel_m=a.gsd, res_az=a.gsd, res_rg=a.gsd, margin=a.margin, h_ref=a.h_ref,
               pulses=64, samples=64)
    sc = Scene(c, cfg)
    info = sc.summary()
    print(f"scene footprint ({how}): lon {W:.5f}..{E_:.5f}, lat {S:.5f}..{N_:.5f} "
          f"({(E_ - W) * mo:.0f} x {(N_ - S) * ml:.0f} m); dwell {info['aperture_full_deg']:.3f} deg, "
          f"a {a.gsd} m chip needs {info['aperture_crop_deg']:.3f} deg")
    reasons = {}
    for i in range(tries):
        lat, lon = rng.uniform(la_a, la_b), rng.uniform(lo_a, lo_b)
        p, why = sc.plan(lat, lon)
        if p is not None:
            dl, do = 0.5 * side / ml, 0.5 * side / mo
            box = [lon - do, lat - dl, lon + do, lat + dl]
            print(f"random {side:.0f} m box (try {i + 1}): "
                  f"--bbox {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            return box
        reasons[why.split(":")[0]] = why
    print(f"no chippable random box in {tries} tries. Reasons seen:")
    for k, v in reasons.items():
        print(f"  {v}")
    hint = {"aperture": "the pass does not sweep enough look angle for this GSD: raise GSD_M (or the input is already "
                        "a chip / crop at this GSD: use the full scene CPHD)",
            "swath": "the box reaches outside the range swath the CPHD stores: use a smaller RANDOM_BOX_M",
            "PRF too low": "the box is too big for the pulse rate at this GSD: use a smaller RANDOM_BOX_M or a larger GSD_M"}
    for k in reasons:
        if k in hint:
            print(f"  -> {hint[k]}")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cphd", default=CPHD_PATH, help="full scene CPHD (FX domain, uncompressed)")
    ap.add_argument("--bbox", default=BBOX, type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="box to chip; leave out (and set BBOX = None) for a random box inside the scene")
    ap.add_argument("--random-box-m", type=float, default=RANDOM_BOX_M, help="side of the random box, metres")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gsd", type=float, default=GSD_M, help="ground resolution in metres, both range and cross-range")
    ap.add_argument("--out", default=OUT_PATH, help="output chip .cphd (a .json is written next to it)")
    ap.add_argument("--margin", type=float, default=MARGIN, help="extra extent around the box, fraction of its side")
    ap.add_argument("--oversample", type=float, default=OVERSAMPLE, help="samples per resolution cell across the extent")
    ap.add_argument("--pulses", type=int, default=PULSES, help="output pulses (0 = from box size)")
    ap.add_argument("--samples", type=int, default=SAMPLES, help="output frequency samples (0 = from box size)")
    ap.add_argument("--h-ref", type=float, default=H_REF, help="ellipsoid height of the chip plane (default: CPHD reference)")
    ap.add_argument("--device", choices=["cpu", "gpu"], default=DEVICE)
    ap.add_argument("--mem-gb", type=float, default=MEM_GB)
    a = ap.parse_args()
    if a.bbox is None:
        a.bbox = random_box(a)
    lo0, la0, lo1, la1 = a.bbox
    lon_min, lon_max, lat_min, lat_max = min(lo0, lo1), max(lo0, lo1), min(la0, la1), max(la0, la1)
    lat, lon = 0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max)
    ml, mo = m_per_deg(lat)
    w_m, h_m = (lon_max - lon_min) * mo, (lat_max - lat_min) * ml
    side = max(w_m, h_m, 16 * a.gsd)
    n_auto = int(math.ceil(side * (1 + a.margin) / a.gsd * a.oversample / 64) * 64)
    cfg = dict(DEFAULTS, core_m=side, pixel_m=a.gsd, res_az=a.gsd, res_rg=a.gsd, margin=a.margin,
               pulses=a.pulses or n_auto, samples=a.samples or n_auto, h_ref=a.h_ref, device=a.device, mem_gb=a.mem_gb)
    t0 = time.time()
    c = CPHD(a.cphd)
    sc = Scene(c, cfg)
    info = sc.summary()
    print(f"box {w_m:.0f} x {h_m:.0f} m at {lat:.6f}, {lon:.6f}; processed as {side * (1 + a.margin):.0f} m square")
    print(f"scene: graze {info['graze_deg']:.1f} deg, dwell {info['aperture_full_deg']:.2f} deg; chip needs aperture "
          f"{info['aperture_crop_deg']:.2f} deg, band {info['band_mhz']:.0f} MHz -> range resolution {info['res_rg_m']:.2f} m")
    if info["res_rg_m"] > a.gsd * 1.01:
        print(f"  note: the CPHD band only allows {info['res_rg_m']:.2f} m in range")
    p, why = sc.plan(lat, lon)
    if p is None:
        max_res = sc.lam / (2 * math.radians(info["aperture_full_deg"]))
        sys.exit(f"cannot cut this chip: {why}" + (f" (finest cross-range here is about {max_res:.2f} m: raise --gsd)"
                                                      if why.startswith("aperture") else ""))
    if not p["alias_free"]:
        print("  warning: Doppler band wide for the pulse count; raise --pulses or --oversample")
    print(f"output {sc.P} pulses x {sc.S} samples ({sc.P * sc.S * 8 / 1e6:.0f} MB); reading pulses "
          f"{p['n0']:,}..{p['n1']:,} of {c.nv:,}")

    xp = get_xp(cfg)
    buf = extract_group(sc, {"chip": p}, workers=cfg["workers"], chunk=cfg["chunk"] if xp is np else 1024, xp=xp)
    Z = resample_slow_time(sc, p, buf.pop("chip"), xp, cfg["resample"])

    with open(a.cphd, "rb") as f, skcphd.Reader(f) as r:
        meta = dict(tree=copy.deepcopy(r.metadata.xmltree))
        try:
            meta["support"] = {s.findtext("{*}Identifier"): r.read_support_array(s.findtext("{*}Identifier"))
                               for s in meta["tree"].findall("{*}Data/{*}SupportArray")}
        except TypeError as e:
            # sarkit 1.12 needs numpy >= 2 to read support arrays (np.asarray(copy=...)); with numpy 1.x, drop them.
            # They are not needed for imaging or for these tools; `pip install -U "numpy>=2"` keeps them.
            print(f"  note: cannot read the CPHD support arrays with numpy {np.__version__} ({e}); "
                  f"writing the chip without them (pip install -U \"numpy>=2\" to keep them)")
            root = meta["tree"].getroot()
            for node in root.findall("{*}Data/{*}SupportArray") + root.findall("{*}SupportArray"):
                node.getparent().remove(node)
            n_sa = root.find("{*}Data/{*}NumSupportArrays")
            if n_sa is not None:
                n_sa.text = "0"
            meta["support"] = {}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_crop(sc, meta, p, Z, a.out, [lon_min, lat_min, lon_max, lat_max])
    js = os.path.splitext(a.out)[0] + ".json"
    with open(js, "w") as f:
        json.dump(dict(source_cphd=os.path.abspath(a.cphd), bbox_geo=[lon_min, lat_min, lon_max, lat_max],
                       centre_lat=lat, centre_lon=lon, h_ref=sc.h_ref, box_m=[round(w_m, 2), round(h_m, 2)],
                       extent_geo=p["ext_bounds"], gsd_m=a.gsd, res_rg_m=sc.res_rg, res_az_m=a.gsd,
                       pulses=sc.P, samples=sc.S, aperture_deg=math.degrees(sc.dtheta), band_hz=sc.B_t,
                       graze_deg=info["graze_deg"], sgn=c.sgn, alias_free=p["alias_free"],
                       # fields crop_to_png.py reads, so the chip can be viewed with it
                       crop_id=os.path.splitext(os.path.basename(a.out))[0],
                       grid=dict(n=int(round(side / a.gsd)), pixel_m=a.gsd), label="chip", n_pos=0, n_ignore=0,
                       boxes=[]), f, indent=1)
    print(f"wrote {a.out} and {js} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
