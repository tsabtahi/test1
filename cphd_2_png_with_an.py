"""Make a PNG of one CPHD crop with its label boxes, with the boxes moved for the height difference.

All heights here are ellipsoid heights (metres above WGS-84), the same as the GEC's RPC HEIGHT_OFF.
The crop is formed on a plane at h_ref (crop JSON). The GEC labels were made on a plane at the GEC's HEIGHT_OFF.
A point shows up (h_point - h_plane) x tan(graze) towards the radar, so a ship's box on our plane sits
    (HEIGHT_OFF - h_ref) x tan(graze) towards the radar
from where the GEC put it. The ship's own height cancels, so no geoid or water level is needed for this. The image is
formed as before (IMAGE_HEIGHT = "h_ref"); only the box positions change. Orange dashed = boxes as labelled,
green = moved. OUT_JSON saves the moved boxes in lat/lon, ready to write back into the crop JSON.

Optional: IMAGE_HEIGHT = "water" images at the water's ellipsoid height (geoid + water level) to see true positions.

Set the paths below, then:  python crop_to_png.py
Needs: pip install numpy sarkit matplotlib   (pyproj only for IMAGE_HEIGHT = "water")
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sarkit.cphd as skcphd

# ------------------------------------------------------------------ set these
CPHD_PATH = r"C:\Users\you\Desktop\crops\crop.cphd"
JSON_PATH = r"C:\Users\you\Desktop\crops\crop.json"
OUT_PNG = r"C:\Users\you\Desktop\crops\crop.png"
PIXEL_M = 1.0              # 2.0 is 4x faster
DB_RANGE = (-5, 35)        # display range, dB above the crop median
OUT_JSON = None            # e.g. r"...\crop_boxes_moved.json": also save the moved boxes (lat/lon)

GEC_HEIGHT = None          # ellipsoid height the GEC / labels were geocoded at (RPC HEIGHT_OFF), m:
                           #   gdalinfo <scene>_GEC.tif | findstr HEIGHT_OFF
                           # None = assume the labels were made at the crop's h_ref (then nothing moves)
IMAGE_HEIGHT = "h_ref"     # image plane, all heights are ellipsoid heights (HAE, metres above WGS-84):
                           #   "h_ref" = the crop's own height: image formed as before, only the boxes move (default)
                           #   "gec"   = at GEC_HEIGHT, the same plane as the GEC, so the boxes need no move
                           #   "water" = at the water surface (geoid + water level): ships at their true position
                           #   or a number in metres
WATER_LEVEL_MSL = 0.0      # only for "water": water above mean sea level (0 at sea, about 174 for Lake Erie)
GEOID_N = None             # only for "water": geoid N in metres; None = look it up (EGM2008 via pyproj)
# ------------------------------------------------------------------

C = 299792458.0
A, E2 = 6378137.0, 6.69437999014e-3                     # WGS-84


def llh_to_ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


def geoid_undulation(lat, lon):
    """EGM2008 geoid height above the WGS-84 ellipsoid, metres (EGM96 if EGM2008 is not available)."""
    import pyproj
    from pyproj import Transformer
    pyproj.network.set_network_enabled(True)               # lets PROJ fetch the geoid grid on first use
    for vert in ("3855", "5773"):                          # EGM2008 height, EGM96 height
        try:
            t = Transformer.from_crs("EPSG:4979", f"EPSG:4326+{vert}", always_xy=True, only_best=True)
            _, _, H = t.transform(lon, lat, 0.0)            # orthometric height of the ellipsoid surface = -N
            if np.isfinite(H) and abs(H) > 1e-6:
                return -float(H), "EGM2008" if vert == "3855" else "EGM96"
        except Exception as e:
            last = e
    raise RuntimeError("no geoid grid available (offline?): set GEOID_N by hand, e.g. from "
                       "https://geographiclib.sourceforge.io/cgi-bin/GeoidEval")


def metres_per_degree(lat):
    s = math.sin(math.radians(lat))
    m_lat = math.pi / 180 * A * (1 - E2) / (1 - E2 * s * s) ** 1.5
    m_lon = math.pi / 180 * A / math.sqrt(1 - E2 * s * s) * math.cos(math.radians(lat))
    return m_lat, m_lon


# 1. read the crop: signal and per-pulse geometry
with open(CPHD_PATH, "rb") as f, skcphd.Reader(f) as r:
    xml = r.metadata.xmltree
    ch = xml.find("{*}Data/{*}Channel").findtext("{*}Identifier")
    sgn = int(xml.findtext("{*}Global/{*}SGN"))
    pv = r.read_pvps(ch)
    sig = r.read_signal(ch)
X = (sig["real"] + 1j * sig["imag"]) if sig.dtype.names else sig.astype(np.complex128)
srp = np.asarray(pv["SRPPos"][0], float)                 # crop centre (reference point)
tx = np.asarray(pv["TxPos"], float) - srp                # antenna positions relative to the crop centre
rc = np.asarray(pv["RcvPos"], float) - srp
f0 = np.asarray(pv["SC0"], float)                        # first frequency of each pulse
scss = float(pv["SCSS"][0])                              # frequency step
P, S = X.shape
with open(JSON_PATH) as f:
    meta = json.load(f)

# 2. the output grid: north-up, regular in lat/lon, centred on the crop (same as the labels)
size_m = meta["grid"]["n"] * meta["grid"]["pixel_m"]
n = int(round(size_m / PIXEL_M))
lat0, lon0, h_ref = meta["centre_lat"], meta["centre_lon"], float(meta["h_ref"])
h_gec = h_ref if GEC_HEIGHT is None else float(GEC_HEIGHT)
if IMAGE_HEIGHT == "water" or GEOID_N is not None:      # the geoid is only needed for the water surface
    N_geoid, src = (geoid_undulation(lat0, lon0) if GEOID_N is None else (float(GEOID_N), "typed in"))
    h_water = N_geoid + WATER_LEVEL_MSL                  # an ellipsoid height
else:
    N_geoid, src, h_water = float("nan"), "not used", float("nan")
h = {"gec": h_gec, "water": h_water, "h_ref": h_ref}.get(IMAGE_HEIGHT, None)
h = float(IMAGE_HEIGHT) if h is None else h

# viewing geometry at the crop centre, to read the shift: a plane-height error dh moves the scene dh x tan(graze)
la, lo = math.radians(lat0), math.radians(lon0)
east = np.array([-math.sin(lo), math.cos(lo), 0.0])
north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
u = (tx[len(tx) // 2] + rc[len(rc) // 2]) / 2
u /= np.linalg.norm(u)                                   # crop centre -> radar
graze = math.degrees(math.asin(u @ up))
radar_az = math.degrees(math.atan2(u @ east, u @ north)) % 360
tg = math.tan(math.radians(graze))
toward = np.array([u @ east, u @ north])
toward /= np.linalg.norm(toward)                          # ground direction towards the radar (east, north)
print(f"crop centre {lat0:.6f}, {lon0:.6f}; radar at azimuth {radar_az:.0f} deg (0 = north), graze {graze:.1f} deg")
if np.isfinite(h_water):
    print(f"geoid N {N_geoid:.2f} m ({src}) + water level {WATER_LEVEL_MSL:.1f} m = water at {h_water:.2f} m ellipsoid height")
print(f"crop h_ref {h_ref:.2f} m, labels (GEC) at {h_gec:.2f} m{' (assumed = h_ref)' if GEC_HEIGHT is None else ''}, "
      f"image plane at {h:.2f} m")
# a point at height a on a plane at height b appears (a - b) x tan(graze) towards the radar.
# label boxes sit where the GEC plane put the ships; on our plane h they belong (h_gec - h) x tan(graze) towards the radar
move_m = (h_gec - h) * tg
dE, dN = move_m * toward
print(f"boxes moved {move_m:+.1f} m towards the radar (east {dE:+.1f} m, north {dN:+.1f} m)")
m_lat, m_lon = metres_per_degree(lat0)
dlat, dlon = PIXEL_M / m_lat, PIXEL_M / m_lon
lat_n, lon_w = lat0 + n * dlat / 2, lon0 - n * dlon / 2
lats = lat_n - (np.arange(n) + 0.5) * dlat
lons = lon_w + (np.arange(n) + 0.5) * dlon
LON, LAT = np.meshgrid(lons, lats)
pix = llh_to_ecef(LAT, LON, np.full(LAT.shape, h)).reshape(-1, 3) - srp
pp = (pix ** 2).sum(1)

# 3. range compression: taper, one sign convention, inverse FFT along frequency
if sgn > 0:
    X = np.conj(X)
X = X * np.hanning(P)[:, None] * np.hanning(S)[None, :]
K = 4 * S                                                # 4x oversampled range profile
prof = np.fft.fftshift(np.fft.ifft(X, n=K, axis=1), axes=1)


def delay_diff(t):
    """two-way path to each pixel minus path to the crop centre, for antenna position t (stable form)"""
    num = pp - 2 * (pix @ t)
    return num / (np.sqrt(t @ t + num) + np.sqrt(t @ t))


# 4. backprojection: every pulse adds its range profile, read at each pixel's delay, with the carrier removed
img = np.zeros(len(pix), complex)
for p in range(P):
    dtau = (delay_diff(tx[p]) + delay_diff(rc[p])) / C
    k = dtau * K * scss + K // 2
    k0 = np.clip(np.floor(k).astype(int), 0, K - 2)
    w = k - k0
    val = prof[p, k0] * (1 - w) + prof[p, k0 + 1] * w
    img += np.where((k >= 0) & (k < K - 1), val * np.exp(2j * np.pi * f0[p] * dtau), 0)
    if p % 128 == 0:
        print(f"pulse {p} of {P}")
amp = np.abs(img).reshape(n, n)
db = 20 * np.log10(np.maximum(amp, 1e-30) / np.median(amp))

# 5. draw the image, the original boxes (orange, dashed) and the boxes moved for height (green; yellow = ignored)
fig, ax = plt.subplots(figsize=(8, 8.6))
ax.imshow(db, cmap="gray", vmin=DB_RANGE[0], vmax=DB_RANGE[1], extent=[0, size_m, size_m, 0])
moved = []
for b in meta.get("boxes", []):
    lo0, la0, lo1, la1 = b["bbox_geo"]
    x0, x1 = (lo0 - lon_w) * m_lon, (lo1 - lon_w) * m_lon
    y0, y1 = (lat_n - la1) * m_lat, (lat_n - la0) * m_lat
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="orange", linewidth=1, linestyle="--"))
    colour = "yellow" if b.get("ignore") else "lime"
    ax.add_patch(Rectangle((x0 + dE, y0 - dN), x1 - x0, y1 - y0, fill=False, edgecolor=colour, linewidth=1.5))
    ax.text(x0 + dE, y0 - dN - 3, f"{b.get('ann_id')}  {b.get('visible', 1):.0%}", color=colour, fontsize=8)
    moved.append(dict(ann_id=b.get("ann_id"), ignore=b.get("ignore", False), bbox_geo_original=b["bbox_geo"],
                      bbox_geo_moved=[lo0 + dE / m_lon, la0 + dN / m_lat, lo1 + dE / m_lon, la1 + dN / m_lat]))
ax.set_xlim(0, size_m); ax.set_ylim(size_m, 0)
ax.set_xlabel("metres east"); ax.set_ylabel("metres south")
ax.set_title(f"{meta['crop_id']}   image plane h = {h:.1f} m, labels made at {h_gec:.1f} m (ellipsoid heights)\n"
             f"orange dashed = boxes as labelled, green = moved {move_m:+.1f} m towards the radar "
             f"(az {radar_az:.0f} deg, graze {graze:.0f} deg)", fontsize=9)
if OUT_JSON:
    with open(OUT_JSON, "w") as fh:
        json.dump(dict(crop_id=meta["crop_id"], h_water=h_water, geoid_n=N_geoid, h_gec=h_gec, h_ref=h_ref, image_h=h,
                       graze_deg=graze, radar_az_deg=radar_az, move_towards_radar_m=move_m, east_m=dE, north_m=dN,
                       boxes=moved), fh, indent=1)
    print("saved", OUT_JSON)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130)
print("saved", OUT_PNG)
