"""Small GUI: drag label boxes onto the ships in our CPHD image, and get the height that explains the shift.

Set the paths below and run:  python height_from_clicks.py
(In Spyder, first run  %matplotlib qt  in the console so the window is interactive.)

How to use
  1. LEFT-click inside a box (green) to pick it - it turns red.
  2. LEFT-click on the ship's true centre in the image - the box moves there (cyan) and an arrow shows the shift.
  3. Repeat for a few ships (3-10 spread over the image is plenty).  RIGHT-click = un-pick,  u = undo last move,
     s = save the moves and the result to OUT_JSON.  Use the toolbar zoom/pan freely (clicks are ignored while a
     toolbar tool is active).

What it computes (all heights are ellipsoid heights, metres above WGS-84)
  A ship at height h_true, imaged on a plane at height h_plane, appears (h_true - h_plane) x tan(graze) TOWARDS the
  radar from its true position. So for each moved box:
      shift along the radar direction  d  ->  height offset  dh = d / tan(graze),   h_true = h_plane + dh
      shift across it                       -> NOT a height effect (labels, timing, moving ship, ...)
  It prints the mean and spread over your moves, then compares h_true with the heights found in the data
  (the image plane used, the CPHD reference point / IARP heights, the sea surface = geoid N if pyproj can get it,
  and the GEC HEIGHT_OFF if you give the GEC). A good match tells you which height to image at.

Inputs: the GeoTIFF from cphd_to_image.py (it carries the plane height, graze and radar direction), and a JSON with
boxes: a crop/chip .json ("boxes"), or an annotations.json ("annotations"), each with bbox_geo
[lon_min, lat_min, lon_max, lat_max].
"""
import json, math, os
import numpy as np
if int(np.__version__.split(".")[0]) < 2:              # sarkit needs np.asarray(copy=...): add it on numpy 1.x
    _np_asarray = np.asarray

    def _asarray_compat(a, dtype=None, order=None, *, copy=None, **kw):
        if copy:
            return np.array(a, dtype=dtype, order=order, copy=True, **kw)
        return _np_asarray(a, dtype=dtype, order=order, **kw)

    np.asarray = _asarray_compat
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ------------------------------------------------------------------ set these
TIF_PATH = r"C:\data\chip1_image.tif"          # from cphd_to_image.py
BOXES_JSON = r"C:\data\annotations.json"       # crop/chip .json or annotations.json with bbox_geo boxes
CPHD_PATH = None                               # optional: the CPHD, to compare with its reference heights
GEC_PATH = None                                # optional: the GEC .tif, to compare with its RPC HEIGHT_OFF
OUT_JSON = r"C:\data\height_from_clicks.json"  # where 's' saves the result
DB_RANGE = (-5, 35)
# ------------------------------------------------------------------

A, E2 = 6378137.0, 6.69437999014e-3


def m_per_deg(lat):
    s = math.sin(math.radians(lat))
    return (math.pi / 180 * A * (1 - E2) / (1 - E2 * s * s) ** 1.5,
            math.pi / 180 * A / math.sqrt(1 - E2 * s * s) * math.cos(math.radians(lat)))


def height_from_moves(moves, graze_deg, radar_az_deg, h_plane):
    """moves: list of (dE, dN) in metres, from the labelled box centre to where the ship is in the image."""
    az = math.radians(radar_az_deg)
    towards = np.array([math.sin(az), math.cos(az)])          # ground direction towards the radar (east, north)
    across_dir = np.array([-towards[1], towards[0]])
    d = np.array(moves, float).reshape(-1, 2)
    along, across = d @ towards, d @ across_dir
    dh = along / math.tan(math.radians(graze_deg))
    return dict(n=len(d), along_m=along, across_m=across, dh=dh, h_true=h_plane + dh,
                dh_mean=float(dh.mean()) if len(d) else float("nan"), dh_std=float(dh.std()) if len(d) > 1 else 0.0,
                across_mean=float(across.mean()) if len(d) else float("nan"))


def data_heights(cphd_path, gec_path, lat, lon, h_plane):
    """Heights found in the data, to compare with the clicked estimate."""
    out = {"image plane used": h_plane}
    if cphd_path:
        try:
            import sarkit.cphd as skcphd
            with open(cphd_path, "rb") as f, skcphd.Reader(f) as r:
                x = r.metadata.xmltree
                hae = x.findtext("{*}SceneCoordinates/{*}IARP/{*}LLH/{*}HAE")
                if hae is not None:
                    out["CPHD IARP HAE"] = float(hae)
                ch = x.findtext("{*}Data/{*}Channel/{*}Identifier")
                srp = np.median(np.asarray(r.read_pvps(ch)["SRPPos"], float), 0)
            p = math.hypot(srp[0], srp[1]); la = math.atan2(srp[2], p * (1 - E2))
            for _ in range(10):
                n = A / math.sqrt(1 - E2 * math.sin(la) ** 2); hh = p / math.cos(la) - n
                la = math.atan2(srp[2], p * (1 - E2 * n / (n + hh)))
            out["CPHD SRP height (median)"] = hh
        except Exception as e:
            print(f"  (could not read CPHD heights: {e})")
    if gec_path:
        try:
            import rasterio
            with rasterio.open(gec_path) as g:
                if g.rpcs is not None:
                    out["GEC RPC HEIGHT_OFF"] = float(g.rpcs.height_off)
        except Exception as e:
            print(f"  (could not read GEC HEIGHT_OFF: {e})")
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


def load_boxes(path):
    d = json.load(open(path))
    if isinstance(d, list):
        d = d[0]
    items = d.get("boxes") or d.get("annotations") or []
    return [(b.get("ann_id", b.get("id", i)), b["bbox_geo"]) for i, b in enumerate(items) if b.get("bbox_geo")]


def main():
    import rasterio
    with rasterio.open(TIF_PATH) as src:
        img = src.read(2 if src.count >= 2 else 1).astype(float)
        if src.count < 2:
            img = 20 * np.log10(np.maximum(img, 1e-30) / np.median(img))
        tags, T, (H_, W_) = src.tags(), src.transform, src.shape
    h_plane = float(tags.get("PLANE_HEIGHT_HAE_M", "nan"))
    graze = float(tags.get("GRAZE_DEG", "nan"))
    az = float(tags.get("RADAR_AZIMUTH_DEG", "nan"))
    if not (np.isfinite(graze) and np.isfinite(az)):
        raise SystemExit("the GeoTIFF has no GRAZE_DEG / RADAR_AZIMUTH_DEG tags: make it with the latest cphd_to_image.py")
    lon_w, lat_n = T.c, T.f
    lon_c, lat_c = lon_w + T.a * W_ / 2, lat_n + T.e * H_ / 2
    ml, mo = m_per_deg(lat_c)
    to_e = lambda lon: (lon - lon_c) * mo                     # metres east / north of the image centre
    to_n = lambda lat: (lat - lat_c) * ml
    ext = [to_e(lon_w), to_e(lon_w + T.a * W_), to_n(lat_n + T.e * H_), to_n(lat_n)]

    boxes = []
    for bid, (lo0, la0, lo1, la1) in load_boxes(BOXES_JSON):
        e0, e1, n0, n1 = to_e(lo0), to_e(lo1), to_n(la0), to_n(la1)
        if e1 > ext[0] and e0 < ext[1] and n1 > ext[2] and n0 < ext[3]:
            boxes.append(dict(id=bid, e0=e0, e1=e1, n0=n0, n1=n1, geo=[lo0, la0, lo1, la1]))
    print(f"{len(boxes)} boxes inside the image; plane {h_plane:.2f} m, graze {graze:.1f} deg, radar az {az:.0f} deg")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray", vmin=DB_RANGE[0], vmax=DB_RANGE[1], extent=ext, origin="upper")
    patches = []
    for b in boxes:
        r = Rectangle((b["e0"], b["n0"]), b["e1"] - b["e0"], b["n1"] - b["n0"], fill=False, edgecolor="lime", lw=1.2)
        ax.add_patch(r)
        patches.append(r)
    L = 0.12 * (ext[1] - ext[0])                                # arrow showing the radar direction
    ax.annotate("", xy=(ext[0] + 0.15 * (ext[1] - ext[0]) + L * math.sin(math.radians(az)),
                        ext[2] + 0.15 * (ext[3] - ext[2]) + L * math.cos(math.radians(az))),
                xytext=(ext[0] + 0.15 * (ext[1] - ext[0]), ext[2] + 0.15 * (ext[3] - ext[2])),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=2))
    ax.text(ext[0] + 0.02 * (ext[1] - ext[0]), ext[2] + 0.03 * (ext[3] - ext[2]), "to radar", color="yellow")
    ax.set_xlabel("metres east"); ax.set_ylabel("metres north")
    state = dict(sel=None, moves=[], drawn=[])

    def refresh_title():
        if not state["moves"]:
            ax.set_title("click a box, then click the ship's true centre  (u = undo, s = save)", fontsize=10)
        else:
            r = height_from_moves([m["d"] for m in state["moves"]], graze, az, h_plane)
            ax.set_title(f"{r['n']} moves: height offset {r['dh_mean']:+.1f} +/- {r['dh_std']:.1f} m  ->  ships at "
                         f"{h_plane + r['dh_mean']:.1f} m ellipsoid height;  across-range {r['across_mean']:+.1f} m",
                         fontsize=10)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is not ax or (fig.canvas.toolbar is not None and fig.canvas.toolbar.mode):
            return
        if ev.button == 3:                                        # right click: un-pick
            if state["sel"] is not None:
                patches[state["sel"]].set_edgecolor("lime")
            state["sel"] = None
            fig.canvas.draw_idle()
            return
        if ev.button != 1:
            return
        if state["sel"] is None:                                  # pick the box under the click
            for i, b in enumerate(boxes):
                if b["e0"] <= ev.xdata <= b["e1"] and b["n0"] <= ev.ydata <= b["n1"]:
                    state["sel"] = i
                    patches[i].set_edgecolor("red")
                    fig.canvas.draw_idle()
                    return
            return
        b = boxes[state["sel"]]                                   # second click: the ship's true centre
        ce, cn = 0.5 * (b["e0"] + b["e1"]), 0.5 * (b["n0"] + b["n1"])
        d = (ev.xdata - ce, ev.ydata - cn)
        w, h = b["e1"] - b["e0"], b["n1"] - b["n0"]
        art = [ax.add_patch(Rectangle((ev.xdata - w / 2, ev.ydata - h / 2), w, h, fill=False, edgecolor="cyan", lw=1.5)),
               ax.add_patch(FancyArrowPatch((ce, cn), (ev.xdata, ev.ydata), arrowstyle="->", color="cyan",
                                            mutation_scale=12))]
        state["moves"].append(dict(id=b["id"], d=d, from_en=(ce, cn), to_en=(ev.xdata, ev.ydata)))
        state["drawn"].append(art)
        patches[state["sel"]].set_edgecolor("orange")
        state["sel"] = None
        r = height_from_moves([d], graze, az, h_plane)
        print(f"box {b['id']}: moved east {d[0]:+.1f} m, north {d[1]:+.1f} m -> along radar {r['along_m'][0]:+.1f} m "
              f"(height offset {r['dh'][0]:+.1f} m), across {r['across_m'][0]:+.1f} m")
        refresh_title()

    def on_key(ev):
        if ev.key == "u" and state["moves"]:
            state["moves"].pop()
            for a in state["drawn"].pop():
                a.remove()
            refresh_title()
        elif ev.key == "s":
            report()

    def report():
        if not state["moves"]:
            print("no moves yet")
            return
        r = height_from_moves([m["d"] for m in state["moves"]], graze, az, h_plane)
        h_true = h_plane + r["dh_mean"]
        print(f"\n{r['n']} moves: height offset {r['dh_mean']:+.2f} +/- {r['dh_std']:.2f} m -> ships at {h_true:.2f} m "
              f"(ellipsoid); across-range shift {r['across_mean']:+.1f} m (not explained by height)")
        cands = data_heights(CPHD_PATH, GEC_PATH, lat_c, lon_c, h_plane)
        print("heights in the data (difference to the clicked estimate):")
        best = None
        for k, v in cands.items():
            diff = v - h_true
            print(f"  {k:32s} {v:9.2f} m   ({diff:+.2f} m)")
            if best is None or abs(diff) < abs(best[1]):
                best = (k, diff)
        if best is not None:
            tol = max(3.0, 2 * r["dh_std"])
            print(f"closest: {best[0]} ({best[1]:+.2f} m) -> "
                  + ("consistent with the clicks" if abs(best[1]) <= tol else "none of these heights explains the shift"))
        if OUT_JSON:
            with open(OUT_JSON, "w") as fh:
                json.dump(dict(tif=TIF_PATH, boxes_json=BOXES_JSON, h_plane=h_plane, graze_deg=graze, radar_az_deg=az,
                               moves=[dict(id=m["id"], east_m=m["d"][0], north_m=m["d"][1]) for m in state["moves"]],
                               height_offset_mean_m=r["dh_mean"], height_offset_std_m=r["dh_std"],
                               ships_height_m=h_true, across_range_mean_m=r["across_mean"], data_heights=cands),
                          fh, indent=1, default=float)
            print("saved", OUT_JSON)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh_title()
    plt.show()
    report()                                                      # also print the result when the window closes


if __name__ == "__main__":
    main()
