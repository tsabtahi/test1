#!/usr/bin/env python3
"""
sicd_to_cphd.py — SICD -> CPHD (inverse PFA), with Umbra one-scene download
and a real-vs-synthetic CPHD comparison test.

    pip install "sarpy==2.1.1" finufft numpy scipy lxml shapely matplotlib boto3

THREE WAYS TO RUN
  1. Edit the CONFIG block below, then just:      python sicd_to_cphd.py
  2. Paths on the command line:                   python sicd_to_cphd.py a_SICD.nitf b_SICD.nitf some_folder/
  3. Download one Umbra scene (SICD + real CPHD), convert, and run the test:
         python sicd_to_cphd.py --download auto                  # smallest scene with both products
         python sicd_to_cphd.py --download 2023-11-19-16-12-16_UMBRA-05
         python sicd_to_cphd.py --list-pairs 20                  # see candidate scenes first

  If a real CPHD sits next to a SICD (<scene>_CPHD.cphd), the comparison runs
  automatically. Or pass one explicitly:  --real-cphd path.cphd

OUTPUT  <out_dir>/<scene_id>/
    <scene_id>_SYNTH.cphd          synthesized CPHD (never named like the real one)
    compare/compare.png            test figure
    compare/report.json            numbers

THE TWO TESTS (only when a real CPHD is available)
  A  Geometry go/no-go. Image the REAL CPHD onto the SICD's own pixel grid
     with this script's k-space conventions and compare to the SICD. Wrong
     conventions show up as low correlation, a mirrored match, or an offset.
  B  Phase-history match. Evaluate the SICD's transform at the real CPHD's
     exact samples (re-referenced to the SICD SCP) and compare vector by
     vector. The per-vector phase difference approximates the autofocus
     correction baked into the SICD.

WHAT THE SYNTHETIC CPHD IS NOT
  Not the raw collect: annulus corners PFA discarded are zero (FX1/FX2 mark
  the valid band), autofocus stays baked in, planar wavefront, pulses are a
  Nyquist resampling (not the real PRF), aFRR1/aFRR2 use a nominal chirp rate.
  PFA SICDs only.
"""
from __future__ import annotations

# =================================================================== CONFIG
# Used when you run the script with no arguments. Command-line flags override.
CONFIG = {
    # SICD files or folders (folders are searched for *_SICD.nitf).
    # Windows paths: use r"C:\path\to\file_SICD.nitf"
    "sicd_files": [
        # r"/data/umbra_phase/2023-11-19-16-12-16_UMBRA-05/2023-11-19-16-12-16_UMBRA-05_SICD.nitf",
    ],
    # Optional real CPHD for the test (only used with a single SICD).
    # Leave None to auto-detect <scene>_CPHD.cphd next to the SICD.
    "real_cphd": None,
    # Download one Umbra open-data scene first: None, "auto", or a scene id.
    "download": None,
    "download_dir": "./umbra_data",
    "task_filter": None,           # e.g. "Beet Piler" to restrict --download auto
    "out_dir": "./cphd_out",
    "chip_size": 4096,             # synthetic CPHD chip around the SCP; None = full image
    "compare": True,               # run tests A/B when a real CPHD is available
    "compare_chip": 1024,          # test A image size (pixels, centred on SCP)
    "compare_vectors": 512,        # test B: real vectors sampled across the aperture
    "check": False,                # run sarpy's CPHD consistency checker (needs pytest)
}
# ==========================================================================

import argparse
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
C = 299_792_458.0
TWO_PI = 2.0 * np.pi


# =================================================================== Umbra download
BUCKET, REGION, PREFIX = "umbra-open-data-catalog", "us-west-2", "sar-data/tasks/"
PRODUCTS = {"_SICD.nitf": "SICD", "_CPHD.cphd": "CPHD", "_METADATA.json": "METADATA"}
SCENE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_UMBRA-\d+$")


def _s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", region_name=REGION,
                        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 10}))


def _human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def umbra_scenes(cache: Path, refresh=False) -> dict:
    """Scene index of the open-data bucket (~20 list calls, cached to disk)."""
    if cache.exists() and not refresh:
        objs = json.loads(cache.read_text())
    else:
        print("[index] listing s3://umbra-open-data-catalog (once, then cached) ...")
        objs = []
        for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX):
            objs += [{"key": o["Key"], "size": o["Size"]} for o in page.get("Contents", [])
                     if o["Key"].endswith(tuple(PRODUCTS))]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(objs))
    scenes = {}
    for o in objs:
        base = o["key"].rsplit("/", 1)[-1]
        for suf, prod in PRODUCTS.items():
            if base.endswith(suf) and SCENE_RE.match(base[:-len(suf)]):
                sid = base[:-len(suf)]
                parts = o["key"].split("/")
                task = parts[2] if len(parts) > 3 else "unknown"
                sc = scenes.setdefault(sid, {"scene_id": sid, "task": task, "files": {}})
                if prod not in sc["files"] or o["size"] > sc["files"][prod]["size"]:
                    sc["files"][prod] = o
    return scenes


def umbra_pairs(scenes, task_filter=None):
    """Scenes that have both SICD and CPHD, smallest first."""
    out = [s for s in scenes.values() if "SICD" in s["files"] and "CPHD" in s["files"]
           and (not task_filter or task_filter.lower() in s["task"].lower())]
    return sorted(out, key=lambda s: s["files"]["SICD"]["size"] + s["files"]["CPHD"]["size"])


def umbra_download(scene_arg: str, dest: Path, task_filter=None) -> tuple[Path, Path]:
    from boto3.s3.transfer import TransferConfig
    scenes = umbra_scenes(dest / "_index.json")
    if scene_arg == "auto":
        pairs = umbra_pairs(scenes, task_filter)
        if not pairs:
            sys.exit("no scene with both SICD and CPHD matched")
        sc = pairs[0]
    else:
        sc = scenes.get(scene_arg)
        if sc is None:
            sys.exit(f"scene {scene_arg} not found in the open-data bucket")
        missing = {"SICD", "CPHD"} - set(sc["files"])
        if missing:
            sys.exit(f"scene {scene_arg} has no {', '.join(missing)} in the bucket")
    sid = sc["scene_id"]
    tot = sum(f["size"] for f in sc["files"].values())
    print(f"[download] {sid}  ({sc['task']})  {_human(tot)}")
    s3, cfg = _s3(), TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20,
                                    max_concurrency=8)
    paths = {}
    for prod, f in sc["files"].items():
        p = dest / sid / f["key"].rsplit("/", 1)[-1]
        paths[prod] = p
        if p.exists() and p.stat().st_size == f["size"]:
            print(f"  skip {p.name} (present)")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        print(f"  get  {p.name}  {_human(f['size'])}")
        s3.download_file(BUCKET, f["key"], str(tmp), Config=cfg)
        if tmp.stat().st_size != f["size"]:
            tmp.unlink(missing_ok=True)
            sys.exit(f"size mismatch on {p.name}")
        os.replace(tmp, p)
    return paths["SICD"], paths["CPHD"]


# =================================================================== geometry
@dataclass
class Axis:
    ss: float
    kctr: float
    sgn: int
    irbw: float
    dkcoa: float = 0.0
    wgt: np.ndarray | None = None

    def weight(self, dk):
        dk = np.asarray(dk, float)
        if self.wgt is None or len(self.wgt) < 2:
            return np.ones_like(dk)
        w = np.asarray(self.wgt, float)
        grid = np.linspace(-self.irbw / 2, self.irbw / 2, len(w)) + self.dkcoa
        return np.interp(dk, grid, w, left=w[0], right=w[-1])

    def in_support(self, dk, guard=0.0):
        return np.abs(np.asarray(dk) - self.dkcoa) <= (self.irbw / 2 - guard)


class Geometry:
    """k-space conventions: scatterer phase = SGN * 2 pi k . x (SGN = SICD Sgn);
    k3 = (f/c)(u_tx + u_rcv), u = unit(radar -> SCP); scatterers in the focus
    plane map to the image plane along IPN:  k_img = k3 - fpn (k3.ipn)/(ipn.fpn)."""

    def __init__(self, sicd, allow_non_pfa=False):
        algo = str(sicd.ImageFormation.ImageFormAlgo).upper()
        if algo != "PFA":
            msg = f"ImageFormAlgo={algo}: only PFA SICDs can be inverted exactly."
            if not allow_non_pfa:
                raise ValueError(msg + " Use --allow-non-pfa for an approximation.")
            print(f"[warn] {msg} Proceeding as if PFA.")
        unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)

        def axis(d):
            dk = 0.0
            if d.DeltaKCOAPoly is not None:
                co = np.asarray(d.DeltaKCOAPoly.Coefs, float)
                dk = float(co.flat[0])
                if np.any(np.abs(co.ravel()[1:]) > 1e-12):
                    print("[warn] spatially varying DeltaKCOAPoly: using constant term")
            return Axis(float(d.SS), float(d.KCtr), int(d.Sgn), float(d.ImpRespBW), dk,
                        None if d.WgtFunct is None else np.asarray(d.WgtFunct, float))

        self.sicd = sicd
        self.scp = np.asarray(sicd.GeoData.SCP.ECF.get_array(), float)
        self.u_row = unit(sicd.Grid.Row.UVectECF.get_array())
        self.u_col = unit(sicd.Grid.Col.UVectECF.get_array())
        self.fpn = unit(sicd.PFA.FPN.get_array()) if sicd.PFA is not None else unit(self.scp)
        self.ipn = unit(np.cross(self.u_row, self.u_col))
        self._arp = sicd.Position.ARPPoly
        self.t_start = float(sicd.ImageFormation.TStartProc)
        self.t_end = float(sicd.ImageFormation.TEndProc)
        self.f_min = float(sicd.ImageFormation.TxFrequencyProc.MinProc)
        self.f_max = float(sicd.ImageFormation.TxFrequencyProc.MaxProc)
        self.row, self.col = axis(sicd.Grid.Row), axis(sicd.Grid.Col)
        if self.row.sgn != self.col.sgn:
            raise ValueError("Row.Sgn != Col.Sgn is not supported")
        self.sgn = self.row.sgn
        idata = sicd.ImageData
        self.scp_pixel = (float(idata.SCPPixel.Row) - int(idata.FirstRow or 0),
                          float(idata.SCPPixel.Col) - int(idata.FirstCol or 0))

    def arp_pos(self, t):
        return np.atleast_2d(self._arp(np.atleast_1d(np.asarray(t, float))))

    def arp_vel(self, t):
        return np.atleast_2d(self._arp.derivative_eval(np.atleast_1d(np.asarray(t, float)), der_order=1))

    def project(self, w):
        """(N,3) 3-D spatial-frequency directions -> (row, col) image-plane coefficients."""
        wp = w - np.outer(w @ self.ipn / float(self.ipn @ self.fpn), self.fpn)
        return wp @ self.u_row, wp @ self.u_col

    def axis_coeffs(self, t):
        """k_img = (2 f / c) * (a_row, a_col) for the SICD's own ARP track."""
        d = self.scp[None, :] - self.arp_pos(t)
        return self.project(d / np.linalg.norm(d, axis=1, keepdims=True))

    def coeffs_from_positions(self, tx, rcv):
        """k_img = f * (A_row, A_col) from arbitrary transmit/receive positions."""
        ut = self.scp[None, :] - tx
        ur = self.scp[None, :] - rcv
        w = (ut / np.linalg.norm(ut, axis=1, keepdims=True)
             + ur / np.linalg.norm(ur, axis=1, keepdims=True)) / C
        return self.project(w)

    def k_image(self, f, t):
        ar, ac = self.axis_coeffs(t)
        s = 2.0 * np.atleast_1d(np.asarray(f, float)) / C
        return ar[:, None] * s[None, :], ac[:, None] * s[None, :]

    def max_delay_range(self, corners, t):
        ar, ac = self.axis_coeffs(t)
        return float(np.max(np.abs(ar[:, None] * corners[None, :, 0] + ac[:, None] * corners[None, :, 1])))

    def choose_sampling(self, corners, oversample):
        """df = c / (4 D os) -> CPHD FX_OSR == os;  dt from max phase-rate spread."""
        tt = np.linspace(self.t_start, self.t_end, 257)
        df = C / (4.0 * max(self.max_delay_range(corners, tt), 1e-6) * oversample)
        ar, ac = self.axis_coeffs(tt)
        rate = (np.gradient(ar, tt)[:, None] * corners[None, :, 0]
                + np.gradient(ac, tt)[:, None] * corners[None, :, 1])
        dt = 1.0 / (2.0 * max((2.0 * self.f_max / C) * np.max(np.abs(rate)), 1e-12) * oversample)
        nf = int(np.ceil((self.f_max - self.f_min) / df)) + 1
        nt = int(np.ceil((self.t_end - self.t_start) / dt)) + 1
        return np.linspace(self.t_start, self.t_end, nt), np.linspace(self.f_min, self.f_max, nf)

    def check(self):
        """First-principles k-space vs the SICD's declared support."""
        t = np.linspace(self.t_start, self.t_end, 65)
        f = np.linspace(self.f_min, self.f_max, 65)
        kr, kc = self.k_image(f, t)
        rep = {"row_centre_rel_err": abs(np.median(kr) - self.row.kctr) / max(abs(self.row.kctr), 1e-9),
               "col_centre_err_frac_irbw": abs(np.median(kc) - self.col.kctr - self.col.dkcoa) / max(self.col.irbw, 1e-9),
               "row_span_ratio": np.ptp(kr) / max(self.row.irbw, 1e-9),
               "col_span_ratio": np.ptp(kc) / max(self.col.irbw, 1e-9)}
        rep["ok"] = bool(rep["row_centre_rel_err"] < 0.02 and rep["col_centre_err_frac_irbw"] < 0.1
                         and rep["row_span_ratio"] > 0.9 and rep["col_span_ratio"] > 0.9)
        return {k: (float(v) if not isinstance(v, bool) else v) for k, v in rep.items()}


# =================================================================== transforms
def chip_corners(g, shape, origin):
    nr, nc = shape
    rr = ((origin[0] - g.scp_pixel[0] - 0.5) * g.row.ss, (origin[0] + nr - 1 - g.scp_pixel[0] + 0.5) * g.row.ss)
    cc = ((origin[1] - g.scp_pixel[1] - 0.5) * g.col.ss, (origin[1] + nc - 1 - g.scp_pixel[1] + 0.5) * g.col.ss)
    return np.array([(r, c) for r in rr for c in cc])


def centred_chip(g, n_rows, n_cols, size):
    if size is None:
        return 0, 0, n_rows, n_cols
    nr, nc = min(size, n_rows), min(size, n_cols)
    r0 = max(0, min(int(round(g.scp_pixel[0] - nr / 2)), n_rows - nr))
    c0 = max(0, min(int(round(g.scp_pixel[1] - nc / 2)), n_cols - nc))
    return r0, c0, nr, nc


def support_mask(g, dkr, dkc, shape, guard):
    nr, nc = shape
    return (g.row.in_support(dkr, guard / (nr * g.row.ss)) & g.col.in_support(dkc, guard / (nc * g.col.ss))
            & (np.abs(TWO_PI * dkr * g.row.ss) < np.pi) & (np.abs(TWO_PI * dkc * g.col.ss) < np.pi))


def image_ft(img, g, origin, dkr, dkc, eps=1e-9):
    """Exact Fourier transform of an image chip at spatial-frequency offsets
    (dkr, dkc) (1-D arrays, cycles/m from KCtr):  sum_x I(x) exp(SGN j 2pi dk.x)."""
    import finufft
    nr, nc = img.shape
    pr, pc = TWO_PI * dkr * g.row.ss, TWO_PI * dkc * g.col.ss
    v = finufft.nufft2d2(pr, pc, np.ascontiguousarray(img, np.complex128), isign=g.sgn, eps=eps)
    sr = nr // 2 + origin[0] - g.scp_pixel[0]
    sc = nc // 2 + origin[1] - g.scp_pixel[1]
    return v * np.exp(g.sgn * 1j * (pr * sr + pc * sc))


def image_from_samples(values, g, origin, shape, dkr, dkc, eps=1e-9):
    """Adjoint of image_ft (density-compensated values in): accumulate onto the chip grid."""
    import finufft
    nr, nc = shape
    pr, pc = TWO_PI * dkr * g.row.ss, TWO_PI * dkc * g.col.ss
    sr = nr // 2 + origin[0] - g.scp_pixel[0]
    sc = nc // 2 + origin[1] - g.scp_pixel[1]
    c = values.astype(np.complex128) * np.exp(-g.sgn * 1j * (pr * sr + pc * sc))
    return finufft.nufft2d1(pr, pc, c, n_modes=(nr, nc), isign=-g.sgn, eps=eps) * (g.row.ss * g.col.ss)


def deweight(g, dkr, dkc, floor):
    return np.maximum(g.row.weight(dkr), floor) * np.maximum(g.col.weight(dkc), floor)


def inverse_pfa(img, g, origin, oversample=1.25, do_deweight=True, wgt_floor=0.1, edge_guard=2.0):
    t, f = g.choose_sampling(chip_corners(g, img.shape, origin), oversample)
    kr, kc = g.k_image(f, t)
    dkr, dkc = kr - g.row.kctr, kc - g.col.kctr
    valid = support_mask(g, dkr, dkc, img.shape, edge_guard)
    sig = np.zeros(kr.shape, np.complex64)
    if valid.any():
        vals = image_ft(img, g, origin, dkr[valid], dkc[valid])
        if do_deweight:
            vals /= deweight(g, dkr[valid], dkc[valid], wgt_floor)
        sig[valid] = vals
    has = valid.any(axis=1)
    fg = np.broadcast_to(f[None, :], valid.shape)
    fx1 = np.where(has, np.where(valid, fg, np.inf).min(1), f[0])
    fx2 = np.where(has, np.where(valid, fg, -np.inf).max(1), f[-1])
    return sig, valid, t, f, fx1, fx2, has


# =================================================================== CPHD writing
PVP_LAYOUT = [("TxTime", 1), ("TxPos", 3), ("TxVel", 3), ("RcvTime", 1), ("RcvPos", 3),
              ("RcvVel", 3), ("SRPPos", 3), ("aFDOP", 1), ("aFRR1", 1), ("aFRR2", 1),
              ("FX1", 1), ("FX2", 1), ("TOA1", 1), ("TOA2", 1), ("TDTropoSRP", 1),
              ("SC0", 1), ("SCSS", 1), ("SIGNAL", 1)]


def _llh(ecf):
    from sarpy.geometry.geocoords import ecf_to_geodetic
    return np.asarray(ecf_to_geodetic(np.asarray(ecf, float)), float)


def _tref(v):
    rx, rr = np.linalg.norm(v["TxPos"] - v["SRPPos"]), np.linalg.norm(v["RcvPos"] - v["SRPPos"])
    return float(v["TxTime"] + rx / (rx + rr) * (v["RcvTime"] - v["TxTime"]))


def _xml_no_ns(meta):
    from lxml import etree
    raw = meta.to_xml_bytes() if hasattr(meta, "to_xml_bytes") else meta.to_xml_string().encode()
    root = etree.fromstring(raw)
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def build_pvps(g, t, f, fx1, fx2, has, corners, pulse_length_s):
    from sarpy.io.phase_history.cphd1_elements import PVP
    nt = len(t)
    tx_pos, tx_vel = g.arp_pos(t), g.arp_vel(t)
    srp = np.repeat(g.scp[None, :], nt, 0)
    rcv_t = t + 2.0 * np.linalg.norm(tx_pos - srp, axis=1) / C
    rcv_pos, rcv_vel = g.arp_pos(rcv_t), g.arp_vel(rcv_t)

    def rdot(p, v):
        u = p - srp
        return (v * u / np.linalg.norm(u, axis=1, keepdims=True)).sum(1)

    fx_c = 0.5 * (fx1 + fx2)
    afrr1 = 2.0 * fx_c / (C * (f[-1] - f[0]) / pulse_length_s)
    toa = 2.0 * g.max_delay_range(corners, t) / C
    vals = {"TxTime": t, "TxPos": tx_pos, "TxVel": tx_vel, "RcvTime": rcv_t, "RcvPos": rcv_pos,
            "RcvVel": rcv_vel, "SRPPos": srp,
            "aFDOP": -(rdot(tx_pos, tx_vel) + rdot(rcv_pos, rcv_vel)) / C,
            "aFRR1": afrr1, "aFRR2": afrr1 / fx_c, "FX1": fx1, "FX2": fx2,
            "TOA1": np.full(nt, -toa), "TOA2": np.full(nt, toa), "TDTropoSRP": np.zeros(nt),
            "SC0": np.full(nt, f[0]), "SCSS": np.full(nt, f[1] - f[0]), "SIGNAL": has.astype(np.int64)}
    kw, off = {}, 0
    for name, n in PVP_LAYOUT:
        cls = (PVP.PerVectorParameterXYZ if n == 3 else
               PVP.PerVectorParameterI8 if name == "SIGNAL" else PVP.PerVectorParameterF8)
        kw[name] = cls(Offset=off)
        off += n
    pvp_type = PVP.PVPType(**kw)
    rec = np.zeros(nt, dtype=pvp_type.get_vector_dtype())
    for name in rec.dtype.names:
        rec[name] = vals[name]
    return rec, pvp_type, off


def build_meta(g, rec, pvp_type, n_words, shape, corners, core_name, pol):
    from sarpy.consistency.cphd_consistency import calc_refgeom_parameters
    from sarpy.io.complex.sicd_elements.blocks import Poly2DType
    from sarpy.io.complex.sicd_elements.CollectionInfo import RadarModeType
    from sarpy.io.phase_history.cphd1_elements import (
        CPHD, Channel, CollectionID, Data, Dwell, Global, ReferenceGeometry as RG, SceneCoordinates as SC)
    from sarpy.io.phase_history.cphd1_elements.blocks import AreaType

    nt, nf = shape
    up = g.fpn
    uiax = g.u_row - (g.u_row @ up) * up
    uiax /= np.linalg.norm(uiax)
    uiay = np.cross(up, uiax)
    xy = []
    for xr, xc in corners:
        p = xr * g.u_row + xc * g.u_col
        pg = p - g.ipn * (p @ up) / float(g.ipn @ up)
        xy.append((pg @ uiax, pg @ uiay))
    xy = np.array(xy)
    (x1, y1), (x2, y2) = xy.min(0), xy.max(0)
    corner_pts = []
    for i, (x, y) in enumerate([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], 1):
        lat, lon, _ = _llh(g.scp + x * uiax + y * uiay)
        corner_pts.append(SC.LatLonCornerType(Lat=lat, Lon=lon, index=i))
    tref1, tref2 = _tref(rec[0]), _tref(rec[-1])
    t_cod, t_dwell = 0.5 * (tref1 + tref2), tref2 - tref1
    fx_min, fx_max = float(rec["FX1"].min()), float(rec["FX2"].max())
    toa1, toa2 = float(rec["TOA1"].min()), float(rec["TOA2"].max())
    fx_fixed = bool(np.ptp(rec["FX1"]) == 0 and np.ptp(rec["FX2"]) == 0)
    start = np.datetime64(str(g.sicd.Timeline.CollectStart).replace("Z", ""))
    collector = str(getattr(g.sicd.CollectionInfo, "CollectorName", None) or "UNKNOWN")

    def make(refgeom):
        return CPHD.CPHDType(
            CollectionID=CollectionID.CollectionIDType(
                CollectorName=collector, CoreName=core_name, CollectType="MONOSTATIC",
                RadarMode=RadarModeType(ModeType="SPOTLIGHT"),
                Classification="UNCLASSIFIED", ReleaseInfo="UNRESTRICTED"),
            Global=Global.GlobalType(
                DomainType="FX", SGN=int(g.sgn),
                Timeline=Global.TimelineType(CollectionStart=start, TxTime1=float(rec["TxTime"][0]),
                                             TxTime2=float(rec["TxTime"][-1])),
                FxBand=Global.FxBandType(FxMin=fx_min, FxMax=fx_max),
                TOASwath=Global.TOASwathType(TOAMin=toa1, TOAMax=toa2)),
            SceneCoordinates=SC.SceneCoordinatesType(
                EarthModel="WGS_84", IARP=SC.IARPType(ECF=g.scp, LLH=_llh(g.scp)),
                ReferenceSurface=SC.ReferenceSurfaceType(Planar=SC.ECFPlanarType(uIAX=uiax, uIAY=uiay)),
                ImageArea=AreaType(X1Y1=[x1, y1], X2Y2=[x2, y2]), ImageAreaCornerPoints=corner_pts),
            Data=Data.DataType(
                SignalArrayFormat="CF8", NumBytesPVP=8 * n_words,
                Channels=[Data.ChannelSizeType(Identifier="1", NumVectors=nt, NumSamples=nf,
                                               SignalArrayByteOffset=0, PVPArrayByteOffset=0)]),
            Channel=Channel.ChannelType(
                RefChId="1", FXFixedCPHD=fx_fixed, TOAFixedCPHD=True, SRPFixedCPHD=True,
                Parameters=[Channel.ChannelParametersType(
                    Identifier="1", RefVectorIndex=nt // 2, FXFixed=fx_fixed, TOAFixed=True, SRPFixed=True,
                    Polarization=Channel.PolarizationType(TxPol=pol[0], RcvPol=pol[1]),
                    FxC=0.5 * (fx_min + fx_max), FxBW=fx_max - fx_min, TOASaved=toa2 - toa1,
                    DwellTimes=Channel.DwellTimesType(CODId="cod", DwellId="dwell"))]),
            PVP=pvp_type,
            Dwell=Dwell.DwellType(
                CODTimes=[Dwell.CODTimeType(Identifier="cod", CODTimePoly=Poly2DType(Coefs=[[t_cod]]))],
                DwellTimes=[Dwell.DwellTimeType(Identifier="dwell", DwellTimePoly=Poly2DType(Coefs=[[t_dwell]]))]),
            ReferenceGeometry=refgeom)

    ref = rec[nt // 2]
    placeholder = RG.ReferenceGeometryType(
        SRP=RG.SRPType(ECF=g.scp, IAC=[0.0, 0.0, 0.0]), ReferenceTime=_tref(ref),
        SRPCODTime=t_cod, SRPDwellTime=t_dwell,
        Monostatic=RG.MonostaticType(
            ARPPos=ref["TxPos"], ARPVel=ref["TxVel"], SideOfTrack="L", SlantRange=1.0, GroundRange=1.0,
            DopplerConeAngle=90.0, GrazeAngle=45.0, IncidenceAngle=45.0, AzimuthAngle=0.0,
            TwistAngle=0.0, SlopeAngle=45.0, LayoverAngle=0.0))
    rp = calc_refgeom_parameters(_xml_no_ns(make(placeholder)), {"1": rec})   # CPHD 1.0.1 Sec 6.5
    gg, mm = rp.refgeom, rp.monostat
    return make(RG.ReferenceGeometryType(
        SRP=RG.SRPType(ECF=gg["SRP/ECF"], IAC=gg["SRP/IAC"]),
        ReferenceTime=float(gg["ReferenceTime"]), SRPCODTime=float(gg["SRPCODTime"]),
        SRPDwellTime=float(gg["SRPDwellTime"]),
        Monostatic=RG.MonostaticType(**{k: (v if k in ("ARPPos", "ARPVel", "SideOfTrack") else float(v))
                                        for k, v in mm.items()})))


def check_cphd(path):
    import sarpy.consistency.cphd_consistency as cc
    con = cc.CphdConsistency.from_file(str(path), check_signal_data=True)
    con.check()
    flagged = {n: [x.get("details") for x in d.get("details", []) if not x.get("passed", True)]
               for n, d in con.failures().items()}
    flagged = {k: v for k, v in flagged.items() if v}
    flagged.pop("check_image_grid_exists", None)        # advisory only
    return len(con.all()), flagged


def sicd_pol(sicd):
    tp = getattr(sicd.ImageFormation, "TxRcvPolarizationProc", None)
    return tuple(str(tp).split(":")[:2]) if tp and ":" in str(tp) else ("V", "V")


def convert(sicd_path: Path, out_path: Path, a) -> dict:
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.phase_history.cphd import CPHDWriter1
    t0 = time.time()
    reader = open_complex(str(sicd_path))
    sicd = reader.get_sicds_as_tuple()[0]
    g = Geometry(sicd, allow_non_pfa=a.allow_non_pfa)
    chk = g.check()
    print(f"  geometry check: {'ok' if chk['ok'] else 'FAILED'} (row ctr err {chk['row_centre_rel_err']:.4f}, "
          f"col ctr err {chk['col_centre_err_frac_irbw']:.4f}, spans {chk['row_span_ratio']:.2f}/{chk['col_span_ratio']:.2f})")
    rep = {"geometry_check": chk}
    if not chk["ok"] and not a.force:
        print("  k-space does not match the SICD's declared support — not writing (--force overrides)")
        return {**rep, "status": "geometry_check_failed"}
    n_rows, n_cols = reader.get_data_size_as_tuple()[0]
    r0, c0, nr, nc = centred_chip(g, n_rows, n_cols, a.chip_size)
    print(f"  image {n_rows}x{n_cols}, chip rows {r0}:{r0 + nr} cols {c0}:{c0 + nc}")
    img = np.asarray(reader[r0:r0 + nr, c0:c0 + nc], dtype=np.complex64)
    sig, valid, t, f, fx1, fx2, has = inverse_pfa(img, g, (r0, c0), a.oversample, not a.no_deweight,
                                                  a.wgt_floor, a.edge_guard)
    print(f"  phase history {sig.shape[0]} pulses x {sig.shape[1]} samples, {valid.mean() * 100:.1f}% in support")
    corners = chip_corners(g, (nr, nc), (r0, c0))
    rec, pvp_type, n_words = build_pvps(g, t, f, fx1, fx2, has, corners, a.pulse_length)
    meta = build_meta(g, rec, pvp_type, n_words, sig.shape, corners, out_path.stem, sicd_pol(sicd))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with CPHDWriter1(str(tmp), meta, check_existence=False) as w:
        w.write_file({"1": rec}, {"1": sig})
    tmp.replace(out_path)
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)")
    rep.update(status="ok", synthetic_cphd=str(out_path), chip=[r0, c0, nr, nc], ph_shape=list(sig.shape))
    if a.check:
        n, flagged = check_cphd(out_path)
        print(f"  consistency: {n} checks, " + ("all passed" if not flagged else f"FLAGGED {flagged}"))
        rep["cphd_check"] = {"n_checks": n, "flagged": flagged}
    return rep


# =================================================================== comparison
def _fidelity(a, b):
    a, b = a.ravel().astype(np.complex128), b.ravel().astype(np.complex128)
    ea, eb = np.vdot(a, a).real, np.vdot(b, b).real
    if ea == 0 or eb == 0:
        return 0.0
    return float(abs(np.vdot(a, b)) / np.sqrt(ea * eb))


def _pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _xcorr_offset(a, b):
    """Integer (row, col) shift that best aligns magnitude image b onto a."""
    A = np.fft.fft2(a - a.mean())
    B = np.fft.fft2(b - b.mean())
    xc = np.abs(np.fft.ifft2(A * np.conj(B)))
    i, j = np.unravel_index(np.argmax(xc), xc.shape)
    return [int(i if i <= a.shape[0] // 2 else i - a.shape[0]),
            int(j if j <= a.shape[1] // 2 else j - a.shape[1])]


class RealCPHD:
    """Real CPHD channel, re-referenced to the SICD SCP, in the SICD's sign convention."""

    def __init__(self, path, g, pol):
        from sarpy.io.phase_history.converter import open_phase_history
        self.r = open_phase_history(str(path))
        m = self.r.cphd_meta
        if str(m.Global.DomainType).upper() != "FX":
            raise ValueError(f"real CPHD domain is {m.Global.DomainType}; only FX is supported")
        self.sgn = int(m.Global.SGN)
        self.index = 0
        if len(m.Channel.Parameters) > 1:       # pick the channel matching the SICD polarization
            for i, p in enumerate(m.Channel.Parameters):
                if p.Polarization is not None and (p.Polarization.TxPol, p.Polarization.RcvPol) == tuple(pol):
                    self.index = i
        self.nv = int(m.Data.Channels[self.index].NumVectors)
        self.ns = int(m.Data.Channels[self.index].NumSamples)
        pv = self.r.read_pvp_array(self.index)
        self.g = g
        self.sc0, self.scss = pv["SC0"].astype(float), pv["SCSS"].astype(float)
        self.fx1, self.fx2 = pv["FX1"].astype(float), pv["FX2"].astype(float)
        self.signal_ok = pv["SIGNAL"] != 0 if "SIGNAL" in pv.dtype.names else np.ones(self.nv, bool)
        self.amp = pv["AmpSF"].astype(float) if "AmpSF" in pv.dtype.names else np.ones(self.nv)
        tx, rcv, srp = pv["TxPos"].astype(float), pv["RcvPos"].astype(float), pv["SRPPos"].astype(float)
        n = lambda v: np.linalg.norm(v, axis=1)
        # delay of SCP relative to the file's SRP, per vector
        self.tau_ref = (n(tx - g.scp) + n(rcv - g.scp) - n(tx - srp) - n(rcv - srp)) / C
        self.A_r, self.A_c = g.coeffs_from_positions(tx, rcv)
        self.dA = np.abs(self.A_r * np.gradient(self.A_c) - np.gradient(self.A_r) * self.A_c)
        self.srp_offset_m = float(np.median(n(srp - g.scp)))

    def block(self, v0, v1):
        """Signal, frequencies, (dkr, dkc) and band mask for vectors v0:v1."""
        s = np.asarray(self.r.read_chip(slice(v0, v1), slice(None), index=self.index), np.complex128)
        s = s.reshape(v1 - v0, self.ns) * self.amp[v0:v1, None]
        f = self.sc0[v0:v1, None] + np.arange(self.ns)[None, :] * self.scss[v0:v1, None]
        s *= np.exp(-self.sgn * 1j * TWO_PI * f * self.tau_ref[v0:v1, None])     # SRP -> SCP
        if self.sgn != self.g.sgn:
            s = np.conj(s)
        band = (f >= self.fx1[v0:v1, None]) & (f <= self.fx2[v0:v1, None]) & self.signal_ok[v0:v1, None]
        dkr = f * self.A_r[v0:v1, None] - self.g.row.kctr
        dkc = f * self.A_c[v0:v1, None] - self.g.col.kctr
        return s, f, dkr, dkc, band


def compare(sicd_path: Path, real_path: Path, out_dir: Path, a) -> dict:
    from sarpy.io.complex.converter import open_complex
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = open_complex(str(sicd_path))
    sicd = reader.get_sicds_as_tuple()[0]
    g = Geometry(sicd, allow_non_pfa=a.allow_non_pfa)
    real = RealCPHD(real_path, g, sicd_pol(sicd))
    n_rows, n_cols = reader.get_data_size_as_tuple()[0]
    rep = {"sicd": str(sicd_path), "real_cphd": str(real_path), "real_vectors": real.nv,
           "real_samples": real.ns, "real_sgn": real.sgn, "real_srp_offset_from_scp_m": real.srp_offset_m}
    print(f"  real CPHD: {real.nv} vectors x {real.ns} samples, SGN {real.sgn}, "
          f"SRP {real.srp_offset_m:.1f} m from SICD SCP")

    # ---------------- test A: image the real CPHD onto the SICD grid
    r0, c0, nr, nc = centred_chip(g, n_rows, n_cols, a.compare_chip)
    sicd_chip = np.asarray(reader[r0:r0 + nr, c0:c0 + nc], np.complex64)
    img = np.zeros((nr, nc), np.complex128)
    used = 0
    for v0 in range(0, real.nv, 1024):
        v1 = min(v0 + 1024, real.nv)
        s, f, dkr, dkc, band = real.block(v0, v1)
        m = band & support_mask(g, dkr, dkc, (nr, nc), a.edge_guard)
        if not m.any():
            continue
        jac = real.scss[v0:v1, None] * f * real.dA[v0:v1, None]          # polar density comp.
        vals = s[m] * jac[m] * g.row.weight(dkr[m]) * g.col.weight(dkc[m])   # apply SICD taper
        img += image_from_samples(vals, g, (r0, c0), (nr, nc), dkr[m], dkc[m])
        used += int(m.sum())
        print(f"\r  test A: imaging real CPHD {v1}/{real.nv} vectors", end="", flush=True)
    print()
    ma, mb = np.abs(sicd_chip).astype(float), np.abs(img)
    corr_mag = _pearson(ma, mb)
    mirrors = {"flip_rows": _pearson(ma, mb[::-1, :]), "flip_cols": _pearson(ma, mb[:, ::-1]),
               "flip_both": _pearson(ma, mb[::-1, ::-1])}
    offset = _xcorr_offset(ma, mb)
    mirrored = max(mirrors.values()) > corr_mag
    passed_a = corr_mag > 0.6 and not mirrored and max(abs(o) for o in offset) <= 2
    rep["test_A"] = {"chip": [r0, c0, nr, nc], "samples_used": used,
                     "magnitude_corr": corr_mag, "complex_corr": _fidelity(sicd_chip, img),
                     "mirror_corrs": mirrors, "offset_pixels_row_col": offset,
                     "mirrored": bool(mirrored), "pass": bool(passed_a)}
    print(f"  test A: |corr| magnitude {corr_mag:.3f}, complex {rep['test_A']['complex_corr']:.3f}, "
          f"offset {offset} px, best mirror {max(mirrors.values()):.3f} -> {'PASS' if passed_a else 'FAIL'}")

    # ---------------- test B: synthetic vs real at the real samples
    big = n_rows * n_cols > 64e6
    rb0, cb0, nrb, ncb = centred_chip(g, n_rows, n_cols, 8000 if big else None)
    full = np.asarray(reader[rb0:rb0 + nrb, cb0:cb0 + ncb], np.complex64)
    idx = np.unique(np.linspace(0, real.nv - 1, min(a.compare_vectors, real.nv)).astype(int))
    rho, ph_real, ph_syn = [], [], []
    for i in idx:
        s, f, dkr, dkc, band = real.block(i, i + 1)
        m = band & support_mask(g, dkr, dkc, full.shape, a.edge_guard)
        row_r = np.full(real.ns, np.nan, complex)
        row_s = np.full(real.ns, np.nan, complex)
        if m.sum() > 8:
            syn = image_ft(full, g, (rb0, cb0), dkr[m], dkc[m]) / deweight(g, dkr[m], dkc[m], a.wgt_floor)
            re_ = s[m]
            den = np.sqrt(np.vdot(re_, re_).real * np.vdot(syn, syn).real)
            rho.append(np.vdot(syn, re_) / den if den > 0 else np.nan)
            row_r[m[0]], row_s[m[0]] = re_, syn
        else:
            rho.append(np.nan)
        ph_real.append(row_r)
        ph_syn.append(row_s)
    rho = np.array(rho)
    ok = np.isfinite(rho)
    phase = np.full(len(rho), np.nan)
    phase[ok] = np.unwrap(np.angle(rho[ok]))
    u = np.linspace(-1, 1, len(rho))
    resid = phase.copy()
    if ok.sum() > 3:
        resid[ok] -= np.polyval(np.polyfit(u[ok], phase[ok], 1), u[ok])    # drop constant + linear
    R, S = np.array(ph_real), np.array(ph_syn)
    Sal = S * np.exp(1j * np.nan_to_num(np.angle(rho)))[:, None]
    fin = np.isfinite(R) & np.isfinite(Sal)
    corr_aligned = _fidelity(R[fin], Sal[fin]) if fin.any() else 0.0
    corr_raw = _fidelity(R[fin], S[fin]) if fin.any() else 0.0
    rep["test_B"] = {"vectors_compared": int(ok.sum()), "sicd_region": [rb0, cb0, nrb, ncb],
                     "median_per_vector_coherence": float(np.nanmedian(np.abs(rho))),
                     "global_coherence_raw": corr_raw,
                     "global_coherence_per_vector_phase_removed": corr_aligned,
                     "phase_residual_rms_rad_after_linear": float(np.sqrt(np.nanmean(resid ** 2))),
                     "phase_residual_ptp_rad": float(np.nanmax(resid) - np.nanmin(resid))}
    print(f"  test B: per-vector coherence median {rep['test_B']['median_per_vector_coherence']:.3f}, "
          f"global {corr_raw:.3f} raw / {corr_aligned:.3f} phase-aligned, "
          f"residual phase {rep['test_B']['phase_residual_rms_rad_after_linear']:.2f} rad rms")
    np.savez_compressed(out_dir / "test_B_phase.npz", vector_index=idx, rho=rho, phase=phase, resid=resid)

    # ---------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def db(x, floor=-45):
            v = 20 * np.log10(np.abs(x) + 1e-12)
            return np.clip(v - np.nanmax(v), floor, 0)

        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        ax[0, 0].imshow(db(sicd_chip), cmap="gray", aspect="auto")
        ax[0, 0].set_title("SICD chip")
        ax[0, 1].imshow(db(img), cmap="gray", aspect="auto")
        ax[0, 1].set_title("REAL CPHD imaged on SICD grid (this script's geometry)")
        ax[0, 2].axis("off")
        ta, tb = rep["test_A"], rep["test_B"]
        ax[0, 2].text(0.02, 0.95, "\n".join([
            f"TEST A  {'PASS' if ta['pass'] else 'FAIL'}",
            f"magnitude corr   {ta['magnitude_corr']:.3f}",
            f"complex corr     {ta['complex_corr']:.3f}",
            f"offset (px)      {ta['offset_pixels_row_col']}",
            "mirrors  " + "  ".join(f"{k}={v:.2f}" for k, v in ta["mirror_corrs"].items()),
            "", "TEST B",
            f"per-vector coherence (median) {tb['median_per_vector_coherence']:.3f}",
            f"global coherence raw          {tb['global_coherence_raw']:.3f}",
            f"  per-vector phase removed    {tb['global_coherence_per_vector_phase_removed']:.3f}",
            f"phase residual rms (rad)      {tb['phase_residual_rms_rad_after_linear']:.2f}",
        ]), family="monospace", fontsize=11, va="top")
        ax[1, 0].imshow(db(R), cmap="viridis", aspect="auto")
        ax[1, 0].set_title("real CPHD |S| (sampled vectors, SICD support)")
        ax[1, 0].set_xlabel("sample")
        ax[1, 0].set_ylabel("sampled vector")
        ax[1, 1].imshow(db(S), cmap="viridis", aspect="auto")
        ax[1, 1].set_title("synthetic from SICD, same samples")
        ax[1, 1].set_xlabel("sample")
        ax[1, 2].plot(idx, np.abs(rho), "k.", ms=3)
        ax[1, 2].set_ylim(0, 1.05)
        ax[1, 2].set_xlabel("real vector index")
        ax[1, 2].set_ylabel("per-vector |coherence|")
        a2 = ax[1, 2].twinx()
        a2.plot(idx, resid, "r-", lw=1)
        a2.set_ylabel("phase residual after linear (rad) ~ autofocus", color="r")
        fig.suptitle(f"{sicd_path.name}  vs  {Path(real_path).name}")
        fig.tight_layout()
        fig.savefig(out_dir / "compare.png", dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] figure not written: {e}")

    rep["seconds"] = round(time.time() - t0, 1)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"  compare -> {out_dir}  ({rep['seconds']}s)")
    return rep


# =================================================================== main
def find_sicds(items):
    out = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(p.rglob("*_SICD.nitf")) or sorted(p.rglob("*.nitf")) + sorted(p.rglob("*.ntf"))
        elif p.exists():
            out.append(p)
        else:
            print(f"[warn] not found: {p}")
    return out


def scene_id(p: Path):
    n = p.name
    for suf in ("_SICD.nitf", ".nitf", ".ntf"):
        if n.endswith(suf):
            return n[:-len(suf)]
    return p.stem


def find_real_cphd(sicd: Path):
    sid = scene_id(sicd)
    for cand in (sicd.with_name(f"{sid}_CPHD.cphd"), sicd.with_name(f"{sid}.cphd")):
        if cand.exists():
            return cand
    hits = [h for h in sicd.parent.glob("*.cphd") if h.name.startswith(sid) and "_SYNTH" not in h.name]
    return hits[0] if len(hits) == 1 else None


def main():
    cf = CONFIG
    ap = argparse.ArgumentParser(description="SICD -> CPHD (inverse PFA) + Umbra download + real-vs-synthetic test",
                                 epilog="With no arguments the CONFIG block at the top of the script is used.")
    ap.add_argument("inputs", nargs="*", help="SICD files or folders (default: CONFIG['sicd_files'])")
    ap.add_argument("--real-cphd", type=Path, default=cf["real_cphd"], help="real CPHD for the test (single SICD)")
    ap.add_argument("--download", nargs="?", const="auto", default=cf["download"],
                    help="download one Umbra scene (SICD + CPHD): 'auto' or a scene id")
    ap.add_argument("--list-pairs", type=int, metavar="N", help="list the N smallest scenes with SICD+CPHD and exit")
    ap.add_argument("--download-dir", type=Path, default=Path(cf["download_dir"]))
    ap.add_argument("--task-filter", default=cf["task_filter"])
    ap.add_argument("-o", "--out-dir", type=Path, default=Path(cf["out_dir"]))
    ap.add_argument("--chip-size", type=int, default=cf["chip_size"])
    ap.add_argument("--full", action="store_true", help="convert the whole image")
    ap.add_argument("--no-compare", action="store_true", default=not cf["compare"])
    ap.add_argument("--compare-chip", type=int, default=cf["compare_chip"])
    ap.add_argument("--compare-vectors", type=int, default=cf["compare_vectors"])
    ap.add_argument("--check", action="store_true", default=cf["check"])
    ap.add_argument("--oversample", type=float, default=1.25)
    ap.add_argument("--no-deweight", action="store_true")
    ap.add_argument("--wgt-floor", type=float, default=0.1)
    ap.add_argument("--edge-guard", type=float, default=2.0)
    ap.add_argument("--pulse-length", type=float, default=50e-6)
    ap.add_argument("--allow-non-pfa", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.full:
        a.chip_size = None

    if a.list_pairs:
        pairs = umbra_pairs(umbra_scenes(a.download_dir / "_index.json"), a.task_filter)
        print(f"{len(pairs)} scenes with SICD + CPHD; smallest {a.list_pairs}:")
        for s in pairs[:a.list_pairs]:
            print(f"  {s['scene_id']}  SICD {_human(s['files']['SICD']['size']):>9}  "
                  f"CPHD {_human(s['files']['CPHD']['size']):>9}  {s['task']}")
        return 0

    jobs = []
    if a.download:
        sp, cp = umbra_download(a.download, a.download_dir, a.task_filter)
        jobs.append((sp, cp))
    sicds = find_sicds(a.inputs if a.inputs else cf["sicd_files"])
    for s in sicds:
        real = a.real_cphd if (a.real_cphd and len(sicds) == 1 and not a.download) else find_real_cphd(s)
        jobs.append((s, real))
    if not jobs:
        ap.print_help()
        print("\nNo input. Put SICD paths in CONFIG['sicd_files'], pass them on the command line, "
              "or use --download auto.")
        return 1

    summary = []
    for i, (sp, real) in enumerate(jobs, 1):
        sid = scene_id(sp)
        print(f"\n[{i}/{len(jobs)}] {sid}")
        out = a.out_dir / sid
        try:
            rep = {"scene_id": sid, **convert(sp, out / f"{sid}_SYNTH.cphd", a)}
            if real and not a.no_compare and rep.get("status") == "ok":
                rep["compare"] = compare(sp, Path(real), out / "compare", a)
            elif not real:
                print("  (no real CPHD found — comparison skipped)")
        except Exception as e:
            import traceback
            traceback.print_exc(limit=3)
            rep = {"scene_id": sid, "status": "error", "error": repr(e)}
        summary.append(rep)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n{sum(r.get('status') == 'ok' for r in summary)}/{len(summary)} converted -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



pip install "sarpy==2.1.1" finufft numpy scipy lxml shapely matplotlib boto3

python sicd_to_cphd.py --list-pairs 10    # smallest Umbra scenes that have both SICD and CPHD
python sicd_to_cphd.py --download auto    # download the smallest pair, convert, test
python sicd_to_cphd.py --download 2023-11-19-16-12-16_UMBRA-05
