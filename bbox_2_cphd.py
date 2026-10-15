#!/usr/bin/env python3
"""Cut a CPHD chip for a lat/lon box at a chosen ground resolution.

Set the inputs in the block below and run:  python chip_cphd.py
(Command-line flags still work and override the block, e.g. --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX --gsd 2.)

The chip is a valid CPHD whose reference point is the box centre: pulses re-referenced to that point, range gated to
the box (plus --margin), band cut for --gsd ground-range resolution, Doppler low-passed and resampled to pulses
uniform in look angle spanning lambda / (2 gsd). It is the same processing as build_crops.py (which it imports), for
one box of any size instead of the fixed 512 m training crops. A JSON with the geometry is written beside it
(it also works as the JSON for crop_to_png.py, to look at the chip).

Pulses / samples are chosen from the box size: (box side x (1 + margin) / gsd) x --oversample, rounded up to a multiple
of 64, unless --pulses / --samples are given. The box is processed as a square of its longer side.
Run where build_crops.py is (e.g. inside the cphd-gpu image: /app/chip_cphd.py).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_crops as bc


def scene_footprint(cphd_path, c):
    """Scene footprint [lon_min, lat_min, lon_max, lat_max] from the CPHD header: the image-area corner points if given,
    else the image area rectangle around the reference point, else 500 m around the reference point."""
    import sarkit.cphd as skcphd
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
    E, N, U = bc.enu_basis(lat0, lon0)
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
            ml, mo = bc.m_per_deg(lat0)
            cs = [(x, y) for x in (float(x1), float(x2)) for y in (float(y1), float(y2))]
            de = [x * (ux @ E) + y * (uy @ E) for x, y in cs]
            dn = [x * (ux @ N) + y * (uy @ N) for x, y in cs]
            return [lon0 + min(de) / mo, lat0 + min(dn) / ml, lon0 + max(de) / mo, lat0 + max(dn) / ml], "image area"
    ml, mo = bc.m_per_deg(lat0)
    return [lon0 - 500 / mo, lat0 - 500 / ml, lon0 + 500 / mo, lat0 + 500 / ml], "500 m around the reference point"


def random_box(a, tries=30):
    """A random square box of a.random_box_m inside the scene footprint that the CPHD can actually chip."""
    rng = np.random.default_rng(a.seed)
    c = bc.CPHD(a.cphd)
    (W, S, E_, N_), how = scene_footprint(a.cphd, c)
    lat_c = 0.5 * (S + N_)
    ml, mo = bc.m_per_deg(lat_c)
    half = 0.5 * a.random_box_m * (1 + a.margin)
    lo_a, lo_b, la_a, la_b = W + half / mo, E_ - half / mo, S + half / ml, N_ - half / ml
    if lo_a > lo_b or la_a > la_b:
        raise SystemExit(f"a {a.random_box_m:.0f} m box does not fit in the scene footprint ({how}); lower --random-box-m")
    side = a.random_box_m
    cfg = dict(bc.DEFAULTS, core_m=side, pixel_m=a.gsd, res_az=a.gsd, res_rg=a.gsd, margin=a.margin, h_ref=a.h_ref,
               pulses=64, samples=64)
    sc = bc.Scene(c, cfg)
    for i in range(tries):
        lat, lon = rng.uniform(la_a, la_b), rng.uniform(lo_a, lo_b)
        p, why = sc.plan(lat, lon)
        if p is not None:
            dl, do = 0.5 * side / ml, 0.5 * side / mo
            box = [lon - do, lat - dl, lon + do, lat + dl]
            print(f"random {side:.0f} m box (footprint from the {how}, try {i + 1}): "
                  f"--bbox {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            return box
    raise SystemExit(f"no chippable random box found in {tries} tries (last reason: {why})")


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
    ml, mo = bc.m_per_deg(lat)
    w_m, h_m = (lon_max - lon_min) * mo, (lat_max - lat_min) * ml
    side = max(w_m, h_m, 16 * a.gsd)
    n_auto = int(math.ceil(side * (1 + a.margin) / a.gsd * a.oversample / 64) * 64)
    cfg = dict(bc.DEFAULTS, core_m=side, pixel_m=a.gsd, res_az=a.gsd, res_rg=a.gsd, margin=a.margin,
               pulses=a.pulses or n_auto, samples=a.samples or n_auto, h_ref=a.h_ref, device=a.device, mem_gb=a.mem_gb)
    t0 = time.time()
    c = bc.CPHD(a.cphd)
    sc = bc.Scene(c, cfg)
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

    xp = bc.get_xp(cfg)
    buf = bc.extract_group(sc, {"chip": p}, workers=cfg["workers"], chunk=cfg["chunk"] if xp is np else 1024, xp=xp)
    Z = bc.resample_slow_time(sc, p, buf.pop("chip"), xp, cfg["resample"])

    import copy, sarkit.cphd as skcphd
    with open(a.cphd, "rb") as f, skcphd.Reader(f) as r:
        meta = dict(tree=copy.deepcopy(r.metadata.xmltree))
        meta["support"] = {s.findtext("{*}Identifier"): r.read_support_array(s.findtext("{*}Identifier"))
                           for s in meta["tree"].findall("{*}Data/{*}SupportArray")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bc.write_crop(sc, meta, p, Z, a.out, [lon_min, lat_min, lon_max, lat_max])
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
