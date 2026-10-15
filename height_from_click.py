"""Tkinter GUI: drag the label boxes onto the ships, read the shift and the height it implies, save a corrected JSON.

All boxes move together (one translation for the whole image): drag ONE box onto its ship, drag again to fine-tune.

Set the paths below and run from a terminal / Anaconda Prompt:  python box_shift_tk.py
(Spyder: Run > Run in an external system terminal, or just run it - tkinter opens its own window.)

  drag a box          move ALL boxes (orange dashed = original, green = moved)
  mouse wheel         zoom in / out around the mouse;  right-drag = pan
  Reset / Save        buttons at the bottom (keys: r, s)

Metrics (all heights are ellipsoid heights, metres above WGS-84):
  translation      east / north in metres and degrees
  along the radar  part of the shift towards (+) / away from (-) the radar -> height offset = along / tan(graze);
                   ships sit at  h_plane + offset  (h_plane = height the image was formed at, from the .tif tags)
  across           part of the shift height cannot explain (label alignment, timing, moving ships ...)
Save writes the input JSON with every bbox_geo moved (originals kept as bbox_geo_original) plus a "box_translation"
record, and lists the heights found in the data (CPHD, GEC, sea level) next to the ships' height.

Needs: numpy rasterio pillow (tkinter comes with Python). Inputs: the .tif from cphd_to_image.py and a JSON with
bbox_geo boxes (annotations.json "annotations" or a crop / chip .json "boxes").
"""
import copy, json, math
import tkinter as tk
import numpy as np
if int(np.__version__.split(".")[0]) < 2:              # sarkit needs np.asarray(copy=...): add it on numpy 1.x
    _np_asarray = np.asarray

    def _asarray_compat(a, dtype=None, order=None, *, copy=None, **kw):
        if copy:
            return np.array(a, dtype=dtype, order=order, copy=True, **kw)
        return _np_asarray(a, dtype=dtype, order=order, **kw)

    np.asarray = _asarray_compat
from PIL import Image, ImageTk

# ------------------------------------------------------------------ set these
TIF_PATH = r"C:\data\chip1_image.tif"            # from cphd_to_image.py
BOXES_JSON = r"C:\data\annotations.json"         # annotations.json or a crop / chip .json
OUT_JSON = r"C:\data\annotations_shifted.json"   # corrected boxes are written here
CPHD_PATH = None                                 # optional: compare with the CPHD's reference heights
GEC_PATH = None                                  # optional: compare with the GEC's RPC HEIGHT_OFF
DB_RANGE = (-5, 35)
WINDOW_PX = 850                                  # initial size of the image view
# ------------------------------------------------------------------

A, E2 = 6378137.0, 6.69437999014e-3


def m_per_deg(lat):
    s = math.sin(math.radians(lat))
    return (math.pi / 180 * A * (1 - E2) / (1 - E2 * s * s) ** 1.5,
            math.pi / 180 * A / math.sqrt(1 - E2 * s * s) * math.cos(math.radians(lat)))


def shift_metrics(dE, dN, graze_deg, radar_az_deg, h_plane, lat):
    az = math.radians(radar_az_deg)
    tE, tN = math.sin(az), math.cos(az)                        # ground direction towards the radar
    along, across = dE * tE + dN * tN, -dE * tN + dN * tE
    dh = along / math.tan(math.radians(graze_deg))
    ml, mo = m_per_deg(lat)
    return dict(east_m=dE, north_m=dN, dlon_deg=dE / mo, dlat_deg=dN / ml, total_m=math.hypot(dE, dN),
                along_radar_m=along, across_m=across, height_offset_m=dh, h_plane=h_plane, h_ships=h_plane + dh,
                graze_deg=graze_deg, radar_az_deg=radar_az_deg)


def data_heights(lat, lon, h_plane):
    out = {"image plane used": h_plane}
    if CPHD_PATH:
        try:
            import sarkit.cphd as skcphd
            with open(CPHD_PATH, "rb") as f, skcphd.Reader(f) as r:
                x = r.metadata.xmltree
                hae = x.findtext("{*}SceneCoordinates/{*}IARP/{*}LLH/{*}HAE")
                if hae is not None:
                    out["CPHD IARP HAE"] = float(hae)
                srp = np.median(np.asarray(r.read_pvps(x.findtext("{*}Data/{*}Channel/{*}Identifier"))["SRPPos"],
                                           float), 0)
            p = math.hypot(srp[0], srp[1]); la = math.atan2(srp[2], p * (1 - E2)); hh = 0.0
            for _ in range(10):
                n = A / math.sqrt(1 - E2 * math.sin(la) ** 2); hh = p / math.cos(la) - n
                la = math.atan2(srp[2], p * (1 - E2 * n / (n + hh)))
            out["CPHD SRP height (median)"] = hh
        except Exception as e:
            print(f"  (could not read the CPHD heights: {e})")
    if GEC_PATH:
        try:
            import rasterio
            with rasterio.open(GEC_PATH) as g:
                if g.rpcs is not None:
                    out["GEC RPC HEIGHT_OFF"] = float(g.rpcs.height_off)
        except Exception as e:
            print(f"  (could not read the GEC HEIGHT_OFF: {e})")
    try:
        import pyproj
        from pyproj import Transformer
        pyproj.network.set_network_enabled(True)
        _, _, ho = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True).transform(lon, lat, 0.0)
        if np.isfinite(ho) and abs(ho) > 1e-6:
            out["sea surface (EGM2008 geoid N)"] = -float(ho)
    except Exception:
        pass
    return out


def box_items(d):
    if isinstance(d, list):
        d = d[0]
    for key in ("annotations", "boxes"):
        if isinstance(d.get(key), list):
            return d, [b for b in d[key] if b.get("bbox_geo")]
    return d, []


class App:
    def __init__(self, root):
        import rasterio
        with rasterio.open(TIF_PATH) as src:
            band = src.read(2 if src.count >= 2 else 1).astype(float)
            if src.count < 2:
                band = 20 * np.log10(np.maximum(band, 1e-30) / np.median(band))
            self.tags, self.T, (self.H, self.W) = src.tags(), src.transform, src.shape
        self.h_plane = float(self.tags.get("PLANE_HEIGHT_HAE_M", "nan"))
        self.graze = float(self.tags.get("GRAZE_DEG", "nan"))
        self.az = float(self.tags.get("RADAR_AZIMUTH_DEG", "nan"))
        if not (np.isfinite(self.graze) and np.isfinite(self.az)):
            raise SystemExit("the .tif has no GRAZE_DEG / RADAR_AZIMUTH_DEG tags: remake it with the latest cphd_to_image.py")
        T = self.T
        self.lon_c, self.lat_c = T.c + T.a * self.W / 2, T.f + T.e * self.H / 2
        self.ml, self.mo = m_per_deg(self.lat_c)
        g = np.clip((band - DB_RANGE[0]) / (DB_RANGE[1] - DB_RANGE[0]) * 255, 0, 255).astype(np.uint8)
        self.pil = Image.fromarray(g)

        self.raw = json.load(open(BOXES_JSON))
        _, items = box_items(self.raw)
        self.boxes = []                                        # boxes in image pixel coordinates (col, row)
        for it in items:
            lo0, la0, lo1, la1 = it["bbox_geo"]
            c0, c1 = (lo0 - T.c) / T.a, (lo1 - T.c) / T.a
            r0, r1 = (la1 - T.f) / T.e, (la0 - T.f) / T.e     # T.e < 0: north edge is the smaller row
            if c1 > 0 and c0 < self.W and r1 > 0 and r0 < self.H:
                self.boxes.append((c0, r0, c1, r1))
        print(f"{len(items)} boxes in the JSON, {len(self.boxes)} inside the image; plane {self.h_plane:.2f} m, "
              f"graze {self.graze:.1f} deg, radar az {self.az:.0f} deg")

        self.dc = self.dr = 0.0                                # current shift in pixels (columns, rows)
        self.scale = WINDOW_PX / max(self.W, self.H)           # screen pixels per image pixel
        self.ox = self.oy = 0.0                                # screen position of image pixel (0, 0)
        self.drag = self.pan = None

        root.title("box shift: drag a box onto its ship")
        self.cv = tk.Canvas(root, width=WINDOW_PX, height=WINDOW_PX, bg="black", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        bar = tk.Frame(root); bar.pack(fill="x")
        tk.Button(bar, text="Reset (r)", command=self.reset).pack(side="left")
        tk.Button(bar, text="Save (s)", command=self.save).pack(side="left")
        self.info = tk.Label(bar, text="", anchor="w", justify="left", font=("Consolas", 10), wraplength=WINDOW_PX - 160)
        self.info.pack(side="left", fill="x", expand=True)
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<B1-Motion>", self.on_drag)
        self.cv.bind("<ButtonRelease-1>", self.on_release)
        self.cv.bind("<ButtonPress-3>", lambda e: setattr(self, "pan", (e.x, e.y, self.ox, self.oy)))
        self.cv.bind("<B3-Motion>", self.on_pan)
        self.cv.bind("<MouseWheel>", lambda e: self.zoom(e, 1.25 if e.delta > 0 else 0.8))   # Windows / macOS
        self.cv.bind("<Button-4>", lambda e: self.zoom(e, 1.25))                              # Linux
        self.cv.bind("<Button-5>", lambda e: self.zoom(e, 0.8))
        root.bind("r", lambda e: self.reset())
        root.bind("s", lambda e: self.save())
        self.render()

    # ---- drawing
    def to_screen(self, c, r):
        return self.ox + c * self.scale, self.oy + r * self.scale

    def to_image(self, x, y):
        return (x - self.ox) / self.scale, (y - self.oy) / self.scale

    def render(self):
        cv = self.cv
        cv.delete("all")
        w, h = max(1, int(self.W * self.scale)), max(1, int(self.H * self.scale))
        self.tkimg = ImageTk.PhotoImage(self.pil.resize((w, h), Image.NEAREST if self.scale > 1 else Image.BILINEAR))
        cv.create_image(self.ox, self.oy, image=self.tkimg, anchor="nw")
        for c0, r0, c1, r1 in self.boxes:
            cv.create_rectangle(*self.to_screen(c0, r0), *self.to_screen(c1, r1), outline="orange", dash=(4, 3))
            cv.create_rectangle(*self.to_screen(c0 + self.dc, r0 + self.dr), *self.to_screen(c1 + self.dc, r1 + self.dr),
                                outline="lime", width=2)
        a = math.radians(self.az)                              # arrow towards the radar (east = +x, north = -y)
        x0, y0 = 60, int(self.cv.winfo_height() or WINDOW_PX) - 60
        cv.create_line(x0, y0, x0 + 45 * math.sin(a), y0 - 45 * math.cos(a), fill="yellow", width=3, arrow="last")
        cv.create_text(x0, y0 + 18, text="to radar", fill="yellow")
        m = self.metrics()
        self.info.config(text=f"shift E {m['east_m']:+.1f} m  N {m['north_m']:+.1f} m | along radar "
                              f"{m['along_radar_m']:+.1f} m -> height offset {m['height_offset_m']:+.1f} m "
                              f"(ships at {m['h_ships']:.1f} m) | across {m['across_m']:+.1f} m")

    def metrics(self):
        dE = self.dc * self.T.a * self.mo                     # pixels -> metres east / north
        dN = self.dr * self.T.e * self.ml
        return shift_metrics(dE, dN, self.graze, self.az, self.h_plane, self.lat_c)

    # ---- mouse
    def on_press(self, e):
        c, r = self.to_image(e.x, e.y)
        for c0, r0, c1, r1 in self.boxes:
            if c0 + self.dc <= c <= c1 + self.dc and r0 + self.dr <= r <= r1 + self.dr:
                self.drag = (c, r, self.dc, self.dr)
                return

    def on_drag(self, e):
        if self.drag:
            c, r = self.to_image(e.x, e.y)
            c_start, r_start, dc0, dr0 = self.drag
            self.dc, self.dr = dc0 + c - c_start, dr0 + r - r_start
            self.render()

    def on_release(self, e):
        if self.drag:
            self.drag = None
            m = self.metrics()
            print(f"shift east {m['east_m']:+.2f} m, north {m['north_m']:+.2f} m; along radar {m['along_radar_m']:+.2f} m "
                  f"-> height offset {m['height_offset_m']:+.2f} m; across {m['across_m']:+.2f} m")

    def on_pan(self, e):
        if self.pan:
            x, y, ox, oy = self.pan
            self.ox, self.oy = ox + e.x - x, oy + e.y - y
            self.render()

    def zoom(self, e, f):
        c, r = self.to_image(e.x, e.y)
        self.scale *= f
        self.ox, self.oy = e.x - c * self.scale, e.y - r * self.scale
        self.render()

    # ---- actions
    def reset(self):
        self.dc = self.dr = 0.0
        self.render()

    def save(self):
        m = self.metrics()
        print(f"\ntranslation: east {m['east_m']:+.2f} m, north {m['north_m']:+.2f} m "
              f"(lon {m['dlon_deg']:+.8f} deg, lat {m['dlat_deg']:+.8f} deg)")
        print(f"along radar {m['along_radar_m']:+.2f} m -> height offset {m['height_offset_m']:+.2f} m: ships at "
              f"{m['h_ships']:.2f} m ellipsoid height (image plane {self.h_plane:.2f} m); across {m['across_m']:+.2f} m")
        cands = data_heights(self.lat_c, self.lon_c, self.h_plane)
        print("heights in the data (difference to the ships' height):")
        best = None
        for k, v in cands.items():
            print(f"  {k:32s} {v:9.2f} m   ({v - m['h_ships']:+.2f} m)")
            if best is None or abs(v - m["h_ships"]) < abs(best[1]):
                best = (k, v - m["h_ships"])
        if best:
            print(f"closest: {best[0]} ({best[1]:+.2f} m)" + ("  -> matches" if abs(best[1]) <= 3 else
                                                              "  -> none of these heights explains the shift"))
        out = copy.deepcopy(self.raw)
        o_root, o_items = box_items(out)
        for it in o_items:                                     # same metres for every box, degrees at its own latitude
            lo0, la0, lo1, la1 = it["bbox_geo"]
            bl, bo = m_per_deg(0.5 * (la0 + la1))
            dlo, dla = m["east_m"] / bo, m["north_m"] / bl
            it["bbox_geo_original"] = [lo0, la0, lo1, la1]
            it["bbox_geo"] = [lo0 + dlo, la0 + dla, lo1 + dlo, la1 + dla]
        o_root["box_translation"] = dict(m, source_tif=TIF_PATH, source_json=BOXES_JSON, data_heights=cands,
                                         note="every bbox_geo moved by east_m / north_m; originals in bbox_geo_original")
        with open(OUT_JSON, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"saved {len(o_items)} moved boxes to {OUT_JSON}")
        self.info.config(text=self.info.cget("text") + "   [saved]")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
