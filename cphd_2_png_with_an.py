"""Form an image from a CPHD crop or chip and save it as a PNG (to look at) and a GeoTIFF (to open in QGIS).

Set the paths below and run:  python cphd_to_image.py

What it does, step by step:
  1. reads the phase history (pulses x frequencies, complex) and every pulse's antenna position;
  2. lays a north-up lat/lon grid over the chip at PIXEL_M spacing, on a flat surface at height H (ellipsoid);
  3. range compression: inverse FFT along frequency turns each pulse into echo strength versus distance;
  4. backprojection: for every pixel and every pulse, read the echo at that pixel's distance from the antenna and
     add them all up (with the phase that distance implies). Bright pixels = where echoes add up = scatterers.
  5. saves |image| in dB as a PNG (with label boxes if a JSON is given) and as a GeoTIFF:
     band 1 = amplitude, band 2 = dB, EPSG:4326, so it overlays the GEC in QGIS.

Meant for chips and training crops (up to a few thousand pixels a side); a full scene would take hours this way.
Needs: pip install numpy sarkit matplotlib rasterio   (without rasterio the .tif is written with a .tfw world file)
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sarkit.cphd as skcphd

# ------------------------------------------------------------------ set these
CPHD_PATH = r"/home/tabtahi/SATLOCK/dataset/chips/chip1.cphd"
JSON_PATH = None           # optional: the chip/crop .json, to draw its boxes and use its centre/size/h_ref
OUT_BASE = r"/home/tabtahi/SATLOCK/dataset/chips/chip1_image"   # writes OUT_BASE.png and OUT_BASE.tif
PIXEL_M = 1.0              # output pixel size, metres
SIZE_M = None              # side of the square image, metres; None = from the JSON or the CPHD image area
H = None                   # image plane, ellipsoid height in metres; None = the JSON h_ref, else the CPHD SRP height
DB_RANGE = (-5, 35)        # PNG display range, dB above the median
# ------------------------------------------------------------------

C = 299792458.0
A, E2 = 6378137.0, 6.69437999014e-3                     # WGS-84


def llh_to_ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h) * np.sin(la)], -1)


def ecef_to_llh(x, y, z):
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - E2))
    for _ in range(10):                                  # fixed-point iteration, converges to mm in a few steps
        n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - E2 * n / (n + h)))
    return math.degrees(lat), math.degrees(lon), h


def metres_per_degree(lat):
    s = math.sin(math.radians(lat))
    return (math.pi / 180 * A * (1 - E2) / (1 - E2 * s * s) ** 1.5,
            math.pi / 180 * A / math.sqrt(1 - E2 * s * s) * math.cos(math.radians(lat)))


# 1. read the phase history and geometry
with open(CPHD_PATH, "rb") as f, skcphd.Reader(f) as r:
    xml = r.metadata.xmltree
    ch = xml.find("{*}Data/{*}Channel").findtext("{*}Identifier")
    sgn = int(xml.findtext("{*}Global/{*}SGN"))
    pv = r.read_pvps(ch)
    sig = r.read_signal(ch)
    ia = xml.find("{*}SceneCoordinates/{*}ImageArea")
X = (sig["real"] + 1j * sig["imag"]) if sig.dtype.names else sig.astype(np.complex128)
if "AmpSF" in pv.dtype.names:
    X = X * np.nan_to_num(np.asarray(pv["AmpSF"], float))[:, None]
P, S = X.shape
srp = np.asarray(pv["SRPPos"][P // 2], float)           # reference point = chip centre (for our chips and crops)
meta = json.load(open(JSON_PATH)) if JSON_PATH else {}
lat0, lon0, h_srp = ecef_to_llh(*srp)
lat0, lon0 = meta.get("centre_lat", lat0), meta.get("centre_lon", lon0)
h = float(H) if H is not None else float(meta.get("h_ref", h_srp))
if SIZE_M is not None:
    size_m = float(SIZE_M)
elif "grid" in meta:
    size_m = meta["grid"]["n"] * meta["grid"]["pixel_m"]
elif ia is not None and ia.find("{*}X1Y1") is not None:
    size_m = 2 * float(ia.findtext("{*}X2Y2/{*}X"))
else:
    size_m = 512.0
tx = np.asarray(pv["TxPos"], float) - srp
rc = np.asarray(pv["RcvPos"], float) - srp
f0 = np.asarray(pv["SC0"], float)
scss = float(pv["SCSS"][P // 2])
# viewing geometry at the centre: radar direction and grazing angle (a plane-height change dh moves points
# dh x tan(graze) on the ground, along the radar direction)
_la, _lo = math.radians(lat0), math.radians(lon0)
_E = np.array([-math.sin(_lo), math.cos(_lo), 0.0])
_N = np.array([-math.sin(_la) * math.cos(_lo), -math.sin(_la) * math.sin(_lo), math.cos(_la)])
_U = np.array([math.cos(_la) * math.cos(_lo), math.cos(_la) * math.sin(_lo), math.sin(_la)])
_u = (tx[P // 2] + rc[P // 2]) / 2
_u /= np.linalg.norm(_u)
graze = math.degrees(math.asin(_u @ _U))
radar_az = math.degrees(math.atan2(_u @ _E, _u @ _N)) % 360
print(f"{os.path.basename(CPHD_PATH)}: {P} pulses x {S} samples, centre {lat0:.6f}, {lon0:.6f}, "
      f"plane {h:.2f} m (SRP {h_srp:.2f} m), image {size_m:.0f} m at {PIXEL_M} m; radar az {radar_az:.0f} deg, "
      f"graze {graze:.1f} deg (10 m of height = {10 * math.tan(math.radians(graze)):.1f} m on the ground)")

# 2. north-up lat/lon pixel grid on the plane at height h, positions relative to the chip centre
n = int(round(size_m / PIXEL_M))
m_lat, m_lon = metres_per_degree(lat0)
dlat, dlon = PIXEL_M / m_lat, PIXEL_M / m_lon
lat_n, lon_w = lat0 + n * dlat / 2, lon0 - n * dlon / 2
LON, LAT = np.meshgrid(lon_w + (np.arange(n) + 0.5) * dlon, lat_n - (np.arange(n) + 0.5) * dlat)
pix = llh_to_ecef(LAT, LON, np.full(LAT.shape, h)).reshape(-1, 3) - srp
pp = (pix ** 2).sum(1)

# 3. range compression: one sign convention, taper, inverse FFT along frequency (4x oversampled)
if sgn > 0:
    X = np.conj(X)
X = X * np.hanning(P)[:, None] * np.hanning(S)[None, :]
K = 4 * S
prof = np.fft.fftshift(np.fft.ifft(X, n=K, axis=1), axes=1)


def delay_diff(t):
    """two-way path to each pixel minus path to the chip centre, for antenna position t (stable form)"""
    num = pp - 2 * (pix @ t)
    return num / (np.sqrt(t @ t + num) + np.sqrt(t @ t))


# 4. backprojection
img = np.zeros(len(pix), complex)
for p in range(P):
    dtau = (delay_diff(tx[p]) + delay_diff(rc[p])) / C
    k = dtau * K * scss + K // 2
    k0 = np.clip(np.floor(k).astype(int), 0, K - 2)
    w = k - k0
    val = prof[p, k0] * (1 - w) + prof[p, k0 + 1] * w
    img += np.where((k >= 0) & (k < K - 1), val * np.exp(2j * np.pi * f0[p] * dtau), 0)
    if p % 128 == 0:
        print(f"  pulse {p} of {P}")
amp = np.abs(img).reshape(n, n).astype(np.float32)
db = (20 * np.log10(np.maximum(amp, 1e-30) / np.median(amp))).astype(np.float32)

# 5a. PNG
fig, ax = plt.subplots(figsize=(8, 8.4))
ax.imshow(db, cmap="gray", vmin=DB_RANGE[0], vmax=DB_RANGE[1], extent=[0, size_m, size_m, 0])
for b in meta.get("boxes", []):
    lo0, la0, lo1, la1 = b["bbox_geo"]
    x0, x1 = (lo0 - lon_w) * m_lon, (lo1 - lon_w) * m_lon
    y0, y1 = (lat_n - la1) * m_lat, (lat_n - la0) * m_lat
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow" if b.get("ignore") else "lime"))
ax.set_xlabel("metres east"); ax.set_ylabel("metres south")
ax.set_title(f"{os.path.basename(CPHD_PATH)}  centre {lat0:.5f}, {lon0:.5f}  plane {h:.1f} m  ({PIXEL_M} m pixels)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_BASE + ".png", dpi=130)
print("saved", OUT_BASE + ".png")

# 5b. GeoTIFF: band 1 amplitude, band 2 dB, north-up EPSG:4326 (pixel corners: lon_w, lat_n)
try:
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(OUT_BASE + ".tif", "w", driver="GTiff", width=n, height=n, count=2, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(lon_w, lat_n, dlon, dlat)) as dst:
        dst.write(amp, 1)
        dst.write(db, 2)
        dst.set_band_description(1, "amplitude")
        dst.set_band_description(2, "dB above median")
        dst.update_tags(SOURCE_CPHD=os.path.basename(CPHD_PATH), PLANE_HEIGHT_HAE_M=f"{h:.3f}", PIXEL_M=str(PIXEL_M),
                        GRAZE_DEG=f"{graze:.4f}", RADAR_AZIMUTH_DEG=f"{radar_az:.4f}")
    print("saved", OUT_BASE + ".tif (GeoTIFF, EPSG:4326)")
except ImportError:
    import tifffile
    tifffile.imwrite(OUT_BASE + ".tif", np.stack([amp, db]))
    with open(OUT_BASE + ".tfw", "w") as fh:                  # world file: pixel size and centre of the top-left pixel
        fh.write(f"{dlon}\n0\n0\n{-dlat}\n{lon_w + dlon / 2}\n{lat_n - dlat / 2}\n")
    print("saved", OUT_BASE + ".tif with a .tfw world file (rasterio not installed; set the layer CRS to EPSG:4326)")
