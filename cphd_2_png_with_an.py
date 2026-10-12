"""Make a PNG of one CPHD crop with its label boxes, to check the boxes sit on the ships.

Set the three paths below, then:  python crop_to_png.py
Needs: pip install numpy sarkit matplotlib
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
H_OVERRIDE = None          # image plane height (m above ellipsoid). None = the crop's h_ref.
                           # To test the height shift, set it to the GEC's HEIGHT_OFF:  gdalinfo <scene>_GEC.tif | findstr HEIGHT_OFF
# ------------------------------------------------------------------

C = 299792458.0
A, E2 = 6378137.0, 6.69437999014e-3                     # WGS-84


def llh_to_ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


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
lat0, lon0, h = meta["centre_lat"], meta["centre_lon"], meta["h_ref"]
if H_OVERRIDE is not None:
    h = float(H_OVERRIDE)

# viewing geometry at the crop centre, to read the shift: a plane-height error dh moves the scene dh / tan(graze)
la, lo = math.radians(lat0), math.radians(lon0)
east = np.array([-math.sin(lo), math.cos(lo), 0.0])
north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
u = (tx[len(tx) // 2] + rc[len(rc) // 2]) / 2
u /= np.linalg.norm(u)                                   # crop centre -> radar
graze = math.degrees(math.asin(u @ up))
radar_az = math.degrees(math.atan2(u @ east, u @ north)) % 360
print(f"crop h_ref {meta['h_ref']:.1f} m, image plane at {h:.1f} m")
print(f"radar is at azimuth {radar_az:.0f} deg (0 = north, 90 = east), graze {graze:.1f} deg")
print(f"each 10 m of height error moves the scene {10 / math.tan(math.radians(graze)):.1f} m along that azimuth")
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

# 5. draw the image and the boxes (from their lat/lon corners)
fig, ax = plt.subplots(figsize=(8, 8.4))
ax.imshow(db, cmap="gray", vmin=DB_RANGE[0], vmax=DB_RANGE[1], extent=[0, size_m, size_m, 0])
for b in meta.get("boxes", []):
    lo0, la0, lo1, la1 = b["bbox_geo"]
    x0, x1 = (lo0 - lon_w) * m_lon, (lo1 - lon_w) * m_lon
    y0, y1 = (lat_n - la1) * m_lat, (lat_n - la0) * m_lat
    colour = "yellow" if b.get("ignore") else "lime"
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour, linewidth=1.5))
    ax.text(x0, y0 - 3, f"{b.get('ann_id')}  {b.get('visible', 1):.0%}", color=colour, fontsize=8)
ax.set_xlim(0, size_m); ax.set_ylim(size_m, 0)
ax.set_xlabel("metres east"); ax.set_ylabel("metres south")
ax.set_title(f"{meta['crop_id']}   plane h = {h:.1f} m\n{meta['label']}: {meta['n_pos']} ship(s), {meta['n_ignore']} ignored "
             f"(green = ship, yellow = ignored); radar at az {radar_az:.0f} deg, graze {graze:.0f} deg", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130)
print("saved", OUT_PNG)
