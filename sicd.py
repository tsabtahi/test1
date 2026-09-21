#!/usr/bin/env python3
"""
sicd_to_cphd.py — synthesize a CPHD 1.x file from a SICD image (inverse PFA).

    pip install "sarpy==2.1.1" finufft numpy scipy lxml shapely

    Easiest: put your SICD paths in the EDIT HERE block below, then
    python sicd_to_cphd.py

    Or on the command line (overrides the block):
    python sicd_to_cphd.py scene_SICD.nitf                    # -> scene.cphd, 4096 chip at SCP
    python sicd_to_cphd.py a_SICD.nitf b_SICD.nitf ./folder --out-dir ./cphd
    python sicd_to_cphd.py scene_SICD.nitf -o out.cphd --chip-size 2048
    python sicd_to_cphd.py scene_SICD.nitf --full             # whole image (memory heavy)
    python sicd_to_cphd.py scene_SICD.nitf --check            # + sarpy consistency check (needs pytest)

How it works
    A PFA-formed SICD image is the inverse Fourier transform of the polar phase
    history, cropped to a rectangle in spatial frequency. This script evaluates
    the image's Fourier transform exactly (type-2 NUFFT) at the polar-annulus
    frequencies the collection traced, removes the SICD taper, and writes the
    result as FX-domain CPHD with per-vector parameters.

What the output is NOT
    * Not the raw collect: annulus corners PFA discarded are zero (FX1/FX2 per
      vector mark the valid band), autofocus stays baked in.
    * Planar wavefront (PFA's own model) — no wavefront curvature.
    * Pulse times are a Nyquist resampling, not the real PRF.
    * aFRR1/aFRR2 use a nominal chirp rate; SICD does not record the waveform.
    * PFA SICDs only (RMA etc. refused unless --allow-non-pfa).

Conventions: scatterer phase = SGN * 2 pi k . x, SGN = SICD Grid Sgn = CPHD SGN.
Scatterers are assumed in the focus plane (PFA's assumption) and map into the
image plane along the image-plane normal.
"""
from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  EDIT HERE — then just run:  python sicd_to_cphd.py                  ║
# ║  (anything given on the command line overrides this block)           ║
# ╚══════════════════════════════════════════════════════════════════════╝
SICD_PATHS = [
    # Any mix of the three forms below. Use r"..." strings on Windows.
    # r"/data/umbra_phase/2023-11-19-16-12-16_UMBRA-05/2023-11-19-16-12-16_UMBRA-05_SICD.nitf",
    # r"/data/umbra_phase",                     # folder: every *_SICD.nitf inside, recursively
    # r"/data/umbra_phase/*/*UMBRA-05_SICD.nitf",   # glob pattern
]
OUTPUT_DIR = None        # None = write each <scene>.cphd next to its SICD; or r"/data/cphd_out"
CHIP_SIZE = 4096         # square chip centred on the scene centre point
FULL_IMAGE = False       # True = whole image (memory heavy on large scenes)
SKIP_EXISTING = True     # don't redo scenes whose .cphd already exists
RUN_CHECK = False        # sarpy CPHD consistency check after each file (needs pytest)
# ════════════════════════════════════════════════════════════════════════

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

C = 299_792_458.0
TWO_PI = 2.0 * np.pi


# =============================================================== geometry
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
    def __init__(self, sicd, allow_non_pfa=False):
        algo = str(sicd.ImageFormation.ImageFormAlgo).upper()
        if algo != "PFA":
            msg = f"ImageFormAlgo={algo}: only PFA SICDs can be inverted exactly."
            if not allow_non_pfa:
                raise SystemExit(msg + " Use --allow-non-pfa for an approximation.")
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
        ipn = np.cross(self.u_row, self.u_col)
        self.ipn = ipn / np.linalg.norm(ipn)
        self._arp = sicd.Position.ARPPoly
        self.t_start = float(sicd.ImageFormation.TStartProc)
        self.t_end = float(sicd.ImageFormation.TEndProc)
        self.f_min = float(sicd.ImageFormation.TxFrequencyProc.MinProc)
        self.f_max = float(sicd.ImageFormation.TxFrequencyProc.MaxProc)
        self.row, self.col = axis(sicd.Grid.Row), axis(sicd.Grid.Col)
        if self.row.sgn != self.col.sgn:
            raise SystemExit("Row.Sgn != Col.Sgn is not supported")
        self.sgn = self.row.sgn
        idata = sicd.ImageData
        self.scp_pixel = (float(idata.SCPPixel.Row) - int(idata.FirstRow or 0),
                          float(idata.SCPPixel.Col) - int(idata.FirstCol or 0))

    def arp_pos(self, t):
        return np.atleast_2d(self._arp(np.atleast_1d(np.asarray(t, float))))

    def arp_vel(self, t):
        return np.atleast_2d(self._arp.derivative_eval(np.atleast_1d(np.asarray(t, float)), der_order=1))

    def axis_coeffs(self, t):
        """k_image = (2 f / c) * (a_row, a_col).  k_img = k3 - fpn (k3.ipn)/(ipn.fpn)."""
        d = self.scp[None, :] - self.arp_pos(t)
        w = d / np.linalg.norm(d, axis=1, keepdims=True)
        wp = w - np.outer(w @ self.ipn / float(self.ipn @ self.fpn), self.fpn)
        return wp @ self.u_row, wp @ self.u_col

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
        b = (2.0 * self.f_max / C) * np.max(np.abs(rate))
        dt = 1.0 / (2.0 * max(b, 1e-12) * oversample)
        nf = int(np.ceil((self.f_max - self.f_min) / df)) + 1
        nt = int(np.ceil((self.t_end - self.t_start) / dt)) + 1
        return np.linspace(self.t_start, self.t_end, nt), np.linspace(self.f_min, self.f_max, nf)

    def check(self):
        """First-principles k-space vs the SICD's declared support. A failure
        means a convention mismatch — output would be wrong."""
        t = np.linspace(self.t_start, self.t_end, 65)
        f = np.linspace(self.f_min, self.f_max, 65)
        kr, kc = self.k_image(f, t)
        rep = {
            "row_centre_rel_err": abs(np.median(kr) - self.row.kctr) / max(abs(self.row.kctr), 1e-9),
            "col_centre_err_frac_irbw": abs(np.median(kc) - self.col.kctr - self.col.dkcoa) / max(self.col.irbw, 1e-9),
            "row_span_ratio": np.ptp(kr) / max(self.row.irbw, 1e-9),
            "col_span_ratio": np.ptp(kc) / max(self.col.irbw, 1e-9),
        }
        rep["ok"] = bool(rep["row_centre_rel_err"] < 0.02 and rep["col_centre_err_frac_irbw"] < 0.1
                         and rep["row_span_ratio"] > 0.9 and rep["col_span_ratio"] > 0.9)
        return rep


# =============================================================== inverse PFA
def chip_corners(g: Geometry, shape, origin):
    nr, nc = shape
    xr0 = (origin[0] - g.scp_pixel[0] - 0.5) * g.row.ss
    xr1 = (origin[0] + nr - 1 - g.scp_pixel[0] + 0.5) * g.row.ss
    xc0 = (origin[1] - g.scp_pixel[1] - 0.5) * g.col.ss
    xc1 = (origin[1] + nc - 1 - g.scp_pixel[1] + 0.5) * g.col.ss
    return np.array([(r, c) for r in (xr0, xr1) for c in (xc0, xc1)])


def inverse_pfa(img, g: Geometry, origin, oversample=1.25, deweight=True,
                wgt_floor=0.1, edge_guard=2.0, eps=1e-9):
    import finufft

    img = np.ascontiguousarray(img, dtype=np.complex128)
    nr, nc = img.shape
    t, f = g.choose_sampling(chip_corners(g, img.shape, origin), oversample)
    kr, kc = g.k_image(f, t)
    dkr, dkc = kr - g.row.kctr, kc - g.col.kctr
    ar, ac = TWO_PI * dkr * g.row.ss, TWO_PI * dkc * g.col.ss
    valid = (g.row.in_support(dkr, edge_guard / (nr * g.row.ss))
             & g.col.in_support(dkc, edge_guard / (nc * g.col.ss))
             & (np.abs(ar) < np.pi) & (np.abs(ac) < np.pi))

    sig = np.zeros(kr.shape, np.complex64)
    if valid.any():
        pr, pc = ar[valid], ac[valid]
        vals = finufft.nufft2d2(pr, pc, img, isign=g.sgn, eps=eps)
        sr = nr // 2 + origin[0] - g.scp_pixel[0]
        sc = nc // 2 + origin[1] - g.scp_pixel[1]
        vals *= np.exp(g.sgn * 1j * (pr * sr + pc * sc))
        if deweight:   # floor each 1-D taper separately (corners!)
            vals /= (np.maximum(g.row.weight(dkr[valid]), wgt_floor)
                     * np.maximum(g.col.weight(dkc[valid]), wgt_floor))
        sig[valid] = vals

    has = valid.any(axis=1)
    fg = np.broadcast_to(f[None, :], valid.shape)
    fx1 = np.where(has, np.where(valid, fg, np.inf).min(1), f[0])
    fx2 = np.where(has, np.where(valid, fg, -np.inf).max(1), f[-1])
    return sig, valid, t, f, fx1, fx2, has


# =============================================================== CPHD
PVP_LAYOUT = [("TxTime", 1), ("TxPos", 3), ("TxVel", 3), ("RcvTime", 1), ("RcvPos", 3),
              ("RcvVel", 3), ("SRPPos", 3), ("aFDOP", 1), ("aFRR1", 1), ("aFRR2", 1),
              ("FX1", 1), ("FX2", 1), ("TOA1", 1), ("TOA2", 1), ("TDTropoSRP", 1),
              ("SC0", 1), ("SCSS", 1), ("SIGNAL", 1)]


def _llh(ecf):
    from sarpy.geometry.geocoords import ecf_to_geodetic
    return np.asarray(ecf_to_geodetic(np.asarray(ecf, float)), float)


def _tref(v):
    rx = np.linalg.norm(v["TxPos"] - v["SRPPos"])
    rr = np.linalg.norm(v["RcvPos"] - v["SRPPos"])
    return float(v["TxTime"] + rx / (rx + rr) * (v["RcvTime"] - v["TxTime"]))


def _xml_no_ns(meta):
    from lxml import etree
    raw = meta.to_xml_bytes() if hasattr(meta, "to_xml_bytes") else meta.to_xml_string().encode()
    root = etree.fromstring(raw)
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def build_pvps(g: Geometry, t, f, fx1, fx2, has, corners, pulse_length_s):
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
    vals = {
        "TxTime": t, "TxPos": tx_pos, "TxVel": tx_vel,
        "RcvTime": rcv_t, "RcvPos": rcv_pos, "RcvVel": rcv_vel, "SRPPos": srp,
        "aFDOP": -2.0 * 0.5 * (rdot(tx_pos, tx_vel) + rdot(rcv_pos, rcv_vel)) / C,
        "aFRR1": afrr1, "aFRR2": afrr1 / fx_c, "FX1": fx1, "FX2": fx2,
        "TOA1": np.full(nt, -toa), "TOA2": np.full(nt, toa), "TDTropoSRP": np.zeros(nt),
        "SC0": np.full(nt, f[0]), "SCSS": np.full(nt, f[1] - f[0]),
        "SIGNAL": has.astype(np.int64),
    }
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


def build_meta(g: Geometry, rec, pvp_type, n_words, shape, corners, core_name, pol):
    from sarpy.consistency.cphd_consistency import calc_refgeom_parameters
    from sarpy.io.complex.sicd_elements.blocks import Poly2DType
    from sarpy.io.complex.sicd_elements.CollectionInfo import RadarModeType
    from sarpy.io.phase_history.cphd1_elements import (
        CPHD, Channel, CollectionID, Data, Dwell, Global,
        ReferenceGeometry as RG, SceneCoordinates as SC)
    from sarpy.io.phase_history.cphd1_elements.blocks import AreaType

    nt, nf = shape
    up = g.fpn
    uiax = g.u_row - (g.u_row @ up) * up
    uiax /= np.linalg.norm(uiax)
    uiay = np.cross(up, uiax)
    xy = []
    for xr, xc in corners:      # image plane -> focus plane along IPN -> IA coords
        p = xr * g.u_row + xc * g.u_col
        pg = p - g.ipn * (p @ up) / float(g.ipn @ up)
        xy.append((pg @ uiax, pg @ uiay))
    xy = np.array(xy)
    (x1, y1), (x2, y2) = xy.min(0), xy.max(0)
    corner_pts = [SC.LatLonCornerType(Lat=_llh(g.scp + x * uiax + y * uiay)[0],
                                      Lon=_llh(g.scp + x * uiax + y * uiay)[1], index=i)
                  for i, (x, y) in enumerate([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], 1)]

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
                ImageArea=AreaType(X1Y1=[x1, y1], X2Y2=[x2, y2]),
                ImageAreaCornerPoints=corner_pts),
            Data=Data.DataType(
                SignalArrayFormat="CF8", NumBytesPVP=8 * n_words,
                Channels=[Data.ChannelSizeType(Identifier="1", NumVectors=nt, NumSamples=nf,
                                               SignalArrayByteOffset=0, PVPArrayByteOffset=0)]),
            Channel=Channel.ChannelType(
                RefChId="1", FXFixedCPHD=fx_fixed, TOAFixedCPHD=True, SRPFixedCPHD=True,
                Parameters=[Channel.ChannelParametersType(
                    Identifier="1", RefVectorIndex=nt // 2, FXFixed=fx_fixed, TOAFixed=True,
                    SRPFixed=True, Polarization=Channel.PolarizationType(TxPol=pol[0], RcvPol=pol[1]),
                    FxC=0.5 * (fx_min + fx_max), FxBW=fx_max - fx_min, TOASaved=toa2 - toa1,
                    DwellTimes=Channel.DwellTimesType(CODId="cod", DwellId="dwell"))]),
            PVP=pvp_type,
            Dwell=Dwell.DwellType(
                CODTimes=[Dwell.CODTimeType(Identifier="cod", CODTimePoly=Poly2DType(Coefs=[[t_cod]]))],
                DwellTimes=[Dwell.DwellTimeType(Identifier="dwell",
                                                DwellTimePoly=Poly2DType(Coefs=[[t_dwell]]))]),
            ReferenceGeometry=refgeom)

    ref = rec[nt // 2]
    placeholder = RG.ReferenceGeometryType(
        SRP=RG.SRPType(ECF=g.scp, IAC=[0.0, 0.0, 0.0]), ReferenceTime=_tref(ref),
        SRPCODTime=t_cod, SRPDwellTime=t_dwell,
        Monostatic=RG.MonostaticType(
            ARPPos=ref["TxPos"], ARPVel=ref["TxVel"], SideOfTrack="L", SlantRange=1.0,
            GroundRange=1.0, DopplerConeAngle=90.0, GrazeAngle=45.0, IncidenceAngle=45.0,
            AzimuthAngle=0.0, TwistAngle=0.0, SlopeAngle=45.0, LayoverAngle=0.0))
    # reference geometry per CPHD 1.0.1 Sec 6.5, computed by sarpy's own routine
    rp = calc_refgeom_parameters(_xml_no_ns(make(placeholder)), {"1": rec})
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
    return len(con.all()), {k: v for k, v in flagged.items() if v}


# =============================================================== main
def expand_paths(items) -> list[Path]:
    """Files, folders (recursive *_SICD.nitf) and glob patterns -> sorted unique files."""
    import glob
    out = []
    for it in items:
        it = str(it).strip()
        if not it:
            continue
        p = Path(it).expanduser()
        if any(ch in it for ch in "*?["):
            out += [Path(x) for x in glob.glob(str(p), recursive=True)]
        elif p.is_dir():
            out += sorted(p.rglob("*_SICD.nitf"))
        elif p.is_file():
            out.append(p)
        else:
            print(f"[warn] not found: {it}")
    seen, uniq = set(), []
    for f in out:
        k = f.resolve()
        if k not in seen and f.is_file():
            seen.add(k)
            uniq.append(f)
    return sorted(uniq)


def scene_stem(p: Path) -> str:
    return p.name[:-len("_SICD.nitf")] if p.name.endswith("_SICD.nitf") else p.stem


def convert(sicd_path: Path, out: Path, a) -> str:
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.phase_history.cphd import CPHDWriter1

    t0 = time.time()
    stem = scene_stem(sicd_path)
    reader = open_complex(str(sicd_path))
    sicd = reader.get_sicds_as_tuple()[0]
    g = Geometry(sicd, allow_non_pfa=a.allow_non_pfa)

    chk = g.check()
    print(f"  geometry check: {'ok' if chk['ok'] else 'FAILED'}  "
          f"(row ctr err {chk['row_centre_rel_err']:.4f}, col ctr err {chk['col_centre_err_frac_irbw']:.4f}, "
          f"span ratios {chk['row_span_ratio']:.2f}/{chk['col_span_ratio']:.2f})")
    if not chk["ok"] and not a.force:
        raise RuntimeError("k-space does not match the SICD's declared support (convention "
                           "mismatch) — not written. Use --force to override.")

    n_rows, n_cols = reader.get_data_size_as_tuple()[0]
    if a.full:
        r0, c0, nr, nc = 0, 0, n_rows, n_cols
    elif a.chip:
        r0, c0, nr, nc = a.chip
    else:
        nr, nc = min(a.chip_size, n_rows), min(a.chip_size, n_cols)
        r0 = int(round(g.scp_pixel[0] - nr / 2))
        c0 = int(round(g.scp_pixel[1] - nc / 2))
    r0, c0 = max(0, min(r0, n_rows - nr)), max(0, min(c0, n_cols - nc))
    print(f"  image {n_rows}x{n_cols}, chip rows {r0}:{r0 + nr} cols {c0}:{c0 + nc}")

    img = np.asarray(reader[r0:r0 + nr, c0:c0 + nc], dtype=np.complex64)
    sig, valid, t, f, fx1, fx2, has = inverse_pfa(
        img, g, (r0, c0), a.oversample, not a.no_deweight, a.wgt_floor, a.edge_guard)
    print(f"  phase history {sig.shape[0]} pulses x {sig.shape[1]} samples, "
          f"{valid.mean() * 100:.1f}% inside image support")

    corners = chip_corners(g, (nr, nc), (r0, c0))
    rec, pvp_type, n_words = build_pvps(g, t, f, fx1, fx2, has, corners, a.pulse_length)
    pol = ("V", "V")
    tp = getattr(sicd.ImageFormation, "TxRcvPolarizationProc", None)
    if tp and ":" in str(tp):
        pol = tuple(str(tp).split(":")[:2])
    meta = build_meta(g, rec, pvp_type, n_words, sig.shape, corners, stem, pol)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with CPHDWriter1(str(tmp), meta, check_existence=False) as w:
        w.write_file({"1": rec}, {"1": sig})
    tmp.replace(out)
    msg = f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)"

    if a.check:
        n, flagged = check_cphd(out)
        flagged.pop("check_image_grid_exists", None)   # advisory only
        msg += f"; consistency {n} checks " + ("all passed" if not flagged else f"FLAGGED {flagged}")
    return msg


def main():
    ap = argparse.ArgumentParser(
        description="SICD -> CPHD (inverse PFA). With no paths, uses SICD_PATHS at the top of this file.")
    ap.add_argument("sicd", nargs="*", help="SICD files, folders, or glob patterns")
    ap.add_argument("-o", "--out", type=Path, help="output .cphd (single input only)")
    ap.add_argument("--out-dir", type=Path, default=Path(OUTPUT_DIR) if OUTPUT_DIR else None,
                    help="write all outputs here (default: next to each SICD)")
    ap.add_argument("--chip-size", type=int, default=CHIP_SIZE)
    ap.add_argument("--chip", type=int, nargs=4, metavar=("ROW0", "COL0", "NROWS", "NCOLS"))
    ap.add_argument("--full", action="store_true", default=FULL_IMAGE)
    ap.add_argument("--overwrite", action="store_true", default=not SKIP_EXISTING)
    ap.add_argument("--check", action="store_true", default=RUN_CHECK)
    ap.add_argument("--oversample", type=float, default=1.25, help="phase-history oversampling (= FX_OSR)")
    ap.add_argument("--no-deweight", action="store_true", help="keep the SICD taper")
    ap.add_argument("--wgt-floor", type=float, default=0.1)
    ap.add_argument("--edge-guard", type=float, default=2.0, help="k-bins masked at support edge")
    ap.add_argument("--pulse-length", type=float, default=50e-6, help="nominal, sets aFRR1/aFRR2")
    ap.add_argument("--allow-non-pfa", action="store_true")
    ap.add_argument("--force", action="store_true", help="write even if the geometry check fails")
    a = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    source = a.sicd if a.sicd else SICD_PATHS
    files = expand_paths(source)
    if not files:
        where = "the command line" if a.sicd else "SICD_PATHS at the top of this script"
        sys.exit(f"No SICD files found from {where}. Add paths there and re-run.")
    if a.out and len(files) > 1:
        sys.exit("-o/--out takes a single input; use --out-dir (or OUTPUT_DIR) for several.")

    print(f"{len(files)} SICD file(s) to convert")
    results = []
    for i, f in enumerate(files, 1):
        out = a.out or ((a.out_dir / f"{scene_stem(f)}.cphd") if a.out_dir else f.with_name(f"{scene_stem(f)}.cphd"))
        print(f"\n[{i}/{len(files)}] {f.name}")
        if out.exists() and not a.overwrite:
            print(f"  skip: {out} exists (set SKIP_EXISTING = False or pass --overwrite)")
            results.append((f.name, "skip"))
            continue
        try:
            print("  " + convert(f, out, a))
            results.append((f.name, "ok"))
        except Exception as e:          # keep the batch going
            print(f"  ERROR: {e}")
            results.append((f.name, f"error: {e}"))

    n_ok = sum(r == "ok" for _, r in results)
    n_skip = sum(r == "skip" for _, r in results)
    n_err = len(results) - n_ok - n_skip
    print(f"\ndone: {n_ok} converted, {n_skip} skipped, {n_err} failed")
    for name, r in results:
        if r.startswith("error"):
            print(f"  {name}: {r}")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
