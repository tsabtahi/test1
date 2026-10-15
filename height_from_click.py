"""Drag the label boxes onto the ships, read off the shift and the height it implies, and save a corrected JSON.

All boxes move together (one translation for the whole image), so you only need to drag ONE box onto its ship;
drag again to fine-tune (any box). Then press  s  to save.

Set the paths below and run:  python box_shift_gui.py      (in Spyder first run  %matplotlib qt  in the console)

  drag a box (left mouse)   move ALL boxes by the same amount; orange dashed = original, green = moved
  r                         reset to the original positions
  s                         save the corrected JSON (+ print the metrics and the heights found in the data)
  toolbar zoom / pan        work as usual (dragging is off while a toolbar tool is active)

Metrics (all heights are ellipsoid heights, metres above WGS-84):
  translation      east / north in metres, and in degrees (lon / lat)
  along the radar  part of the shift towards (+) / away from (-) the radar  -> height offset = along / tan(graze):
                   ships sit at  h_plane + offset  (h_plane = height the image was formed at)
  across           part of the shift that height cannot explain (label alignment, timing, moving ships ...)
The saved JSON is the input JSON with every bbox_geo moved by the translation (the original kept as
bbox_geo_original), plus a "box_translation" record with all the numbers.

Inputs: the GeoTIFF from cphd_to_image.py (it stores the plane height, graze and radar direction) and the JSON with
boxes (annotations.json "annotations", or a crop / chip .json "boxes"; each with bbox_geo
[lon_min, lat_min, lon_max, lat_max]).
"""
import copy, json, math
import numpy as np
if int(np.__version__.split(".")[0]) < 2:              # sarkit needs np.asarray(copy=...): add it on numpy 1.x
    _np_asarray = np.asarray

    def _asarray_compat(a, dtype=None, order=None, *, copy=None, **kw):
        if copy:
            return np.array(a, dtype=dtype, order=order, copy=True, **kw)
        return _np_asarray(a, dtype=dtype, order=order, **kw)

    np.asarray = _asarray_compat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------ set these
TIF_PATH = r"C:\data\chip1_image.tif"            # from cphd_to_image.py
BOXES_JSON = r"C:\data\annotations.json"         # annotations.json or a crop / chip .json
OUT_JSON = r"C:\data\annotations_shifted.json"   # corrected boxes are written here
CPHD_PATH = None                                 # optional: compare with the CPHD's reference heights
GEC_PATH = None                                  # optional: compare with the GEC's RPC HEIGHT_OFF
DB_RANGE = (-5, 35)
# ------------------------------------------------------------------

A, E2 = 6378137.0, 6.69437999014e-3


def m_per_deg(lat):
    s = math.sin(math.radians(lat))
    return (math.pi / 180 * A * (1 - E2) / (1 - E2 * s * s) ** 1.5,
            math.pi / 180 * A / math.sqrt(1 - E2 * s * s) * math.cos(math.radians(lat)))


def shift_metrics(dE, dN, graze_deg, radar_az_deg, h_plane, lat):
    az = math.radians(radar_az_deg)
    tE, tN = math.sin(az), math.cos(az)                        # ground direction towards the radar
    along = dE * tE + dN * tN
    across = -dE * tN + dN * tE
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
    """The list holding the boxes in this JSON (annotations.json or crop/chip json)."""
    if isinstance(d, list):
        d = d[0]
    for key in ("annotations", "boxes"):
        if isinstance(d.get(key), list):
            return d, [b for b in d[key] if b.get("bbox_geo")]
    return d, []


def main():
    import rasterio
    with rasterio.open(TIF_PATH) as src:
        img = src.read(2 if src.count >= 2 else 1).astype(float)
        if src.count < 2:
            img = 20 * np.log10(np.maximum(img, 1e-30) / np.median(img))
        tags, T, (H_, W_) = src.tags(), src.transform, src.shape
    h_plane = float(tags.get("PLANE_HEIGHT_HAE_M", "nan"))
    graze, az = float(tags.get("GRAZE_DEG", "nan")), float(tags.get("RADAR_AZIMUTH_DEG", "nan"))
    if not (np.isfinite(graze) and np.isfinite(az)):
        raise SystemExit("the GeoTIFF has no GRAZE_DEG / RADAR_AZIMUTH_DEG tags: remake it with the latest cphd_to_image.py")
    lon_w, lat_n = T.c, T.f
    lon_c, lat_c = lon_w + T.a * W_ / 2, lat_n + T.e * H_ / 2
    ml, mo = m_per_deg(lat_c)
    E = lambda lon: (lon - lon_c) * mo
    N = lambda lat: (lat - lat_c) * ml
    ext = [E(lon_w), E(lon_w + T.a * W_), N(lat_n + T.e * H_), N(lat_n)]

    raw = json.load(open(BOXES_JSON))
    root, items = box_items(raw)
    boxes = []
    for it in items:
        lo0, la0, lo1, la1 = it["bbox_geo"]
        boxes.append(dict(e0=E(lo0), e1=E(lo1), n0=N(la0), n1=N(la1)))
    inside = [i for i, b in enumerate(boxes) if b["e1"] > ext[0] and b["e0"] < ext[1] and b["n1"] > ext[2] and b["n0"] < ext[3]]
    print(f"{len(items)} boxes in the JSON, {len(inside)} inside the image; plane {h_plane:.2f} m, "
          f"graze {graze:.1f} deg, radar az {az:.0f} deg")

    fig, ax = plt.subplots(figsize=(9, 9.3))
    ax.imshow(img, cmap="gray", vmin=DB_RANGE[0], vmax=DB_RANGE[1], extent=ext, origin="upper")
    orig, moved = [], []
    for i in inside:
        b = boxes[i]
        w, h = b["e1"] - b["e0"], b["n1"] - b["n0"]
        orig.append(ax.add_patch(Rectangle((b["e0"], b["n0"]), w, h, fill=False, edgecolor="orange", lw=1, ls="--")))
        moved.append(ax.add_patch(Rectangle((b["e0"], b["n0"]), w, h, fill=False, edgecolor="lime", lw=1.5)))
    x0, y0 = ext[0] + 0.12 * (ext[1] - ext[0]), ext[2] + 0.12 * (ext[3] - ext[2])
    L = 0.1 * (ext[1] - ext[0])
    ax.annotate("", xy=(x0 + L * math.sin(math.radians(az)), y0 + L * math.cos(math.radians(az))), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=2))
    ax.text(x0, y0 - 0.05 * (ext[3] - ext[2]), "to radar", color="yellow", ha="center")
    ax.set_xlabel("metres east"); ax.set_ylabel("metres north")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    st = dict(dE=0.0, dN=0.0, drag=None)

    def redraw():
        for k, i in enumerate(inside):
            moved[k].set_xy((boxes[i]["e0"] + st["dE"], boxes[i]["n0"] + st["dN"]))
        m = shift_metrics(st["dE"], st["dN"], graze, az, h_plane, lat_c)
        ax.set_title(f"shift east {m['east_m']:+.1f} m, north {m['north_m']:+.1f} m  |  along radar {m['along_radar_m']:+.1f} m"
                     f" -> height offset {m['height_offset_m']:+.1f} m (ships at {m['h_ships']:.1f} m)  |  across "
                     f"{m['across_m']:+.1f} m\ndrag a box onto its ship;  r = reset,  s = save", fontsize=9)
        fig.canvas.draw_idle()

    def toolbar_busy():
        return fig.canvas.toolbar is not None and bool(fig.canvas.toolbar.mode)

    def on_press(ev):
        if ev.inaxes is not ax or ev.button != 1 or toolbar_busy():
            return
        for i in inside:
            b = boxes[i]
            if b["e0"] + st["dE"] <= ev.xdata <= b["e1"] + st["dE"] and b["n0"] + st["dN"] <= ev.ydata <= b["n1"] + st["dN"]:
                st["drag"] = (ev.xdata, ev.ydata, st["dE"], st["dN"])
                return

    def on_move(ev):
        if st["drag"] is None or ev.inaxes is not ax or ev.xdata is None:
            return
        x, y, e, n = st["drag"]
        st["dE"], st["dN"] = e + ev.xdata - x, n + ev.ydata - y
        redraw()

    def on_release(ev):
        if st["drag"] is not None:
            st["drag"] = None
            m = shift_metrics(st["dE"], st["dN"], graze, az, h_plane, lat_c)
            print(f"shift east {m['east_m']:+.2f} m, north {m['north_m']:+.2f} m (|{m['total_m']:.1f}| m); along radar "
                  f"{m['along_radar_m']:+.2f} m -> height offset {m['height_offset_m']:+.2f} m; across {m['across_m']:+.2f} m")

    def on_key(ev):
        if ev.key == "r":
            st["dE"] = st["dN"] = 0.0
            redraw()
        elif ev.key == "s":
            save()

    def save():
        m = shift_metrics(st["dE"], st["dN"], graze, az, h_plane, lat_c)
        print(f"\ntranslation: east {m['east_m']:+.2f} m, north {m['north_m']:+.2f} m  "
              f"(lon {m['dlon_deg']:+.8f} deg, lat {m['dlat_deg']:+.8f} deg)")
        print(f"along radar {m['along_radar_m']:+.2f} m -> height offset {m['height_offset_m']:+.2f} m: "
              f"ships at {m['h_ships']:.2f} m ellipsoid height (image plane {h_plane:.2f} m)")
        print(f"across radar {m['across_m']:+.2f} m (not explained by height)")
        cands = data_heights(lat_c, lon_c, h_plane)
        print("heights in the data (difference to the ships' height from your drag):")
        best = None
        for k, v in cands.items():
            print(f"  {k:32s} {v:9.2f} m   ({v - m['h_ships']:+.2f} m)")
            if best is None or abs(v - m["h_ships"]) < abs(best[1]):
                best = (k, v - m["h_ships"])
        if best:
            print(f"closest: {best[0]} ({best[1]:+.2f} m)" + ("  -> matches" if abs(best[1]) <= 3 else
                                                              "  -> none of these heights explains the shift"))
        out = copy.deepcopy(raw)
        o_root, o_items = box_items(out)
        for it in o_items:                                     # same metres for every box, degrees at its own latitude
            lo0, la0, lo1, la1 = it["bbox_geo"]
            bl, bo = m_per_deg(0.5 * (la0 + la1))
            dlo, dla = st["dE"] / bo, st["dN"] / bl
            it["bbox_geo_original"] = [lo0, la0, lo1, la1]
            it["bbox_geo"] = [lo0 + dlo, la0 + dla, lo1 + dlo, la1 + dla]
        o_root["box_translation"] = dict(m, source_tif=TIF_PATH, source_json=BOXES_JSON, data_heights=cands,
                                         note="every bbox_geo moved by east_m / north_m; originals in bbox_geo_original")
        with open(OUT_JSON, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"saved {len(o_items)} moved boxes to {OUT_JSON}")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
