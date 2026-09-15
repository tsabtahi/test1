


# Python 3.9+
#
# Install:
#   pip install -r requirements.txt
#
# tkinter is NOT a pip package - it comes from your Python install:
#   - python.org Windows/macOS installers: included by default
#   - Anaconda/Miniconda: included
#   - Debian/Ubuntu:  sudo apt install python3-tk
#   - Fedora/RHEL:    sudo dnf install python3-tkinter
#   - Arch:           sudo pacman -S tk
#   - macOS Homebrew: brew install python-tk
# Verify with:  python -c "import tkinter"
 
rasterio>=1.3
numpy>=1.24
Pillow>=9.0

GeoTIFF Object Overlay

Place PNG object templates on a GeoTIFF and export a new GeoTIFF. CRS, transform, nodata and extra bands are preserved.

Install
bash
pip install -r requirements.txt

tkinter comes with python.org / Anaconda installs. On Linux add it separately: sudo apt install python3-tk (Debian/Ubuntu) or sudo dnf install python3-tkinter (Fedora). Verify: python -c "import tkinter"

Run

F5 in Spyder or python geotiff_overlay_gui.py. Under Spyder it opens in a separate process; if no window appears, read gui_launch.log next to the script.

Use
Load GeoTIFF
Add PNG template - draw a polygon around the object (right-click undo, wheel zoom, right-drag pan)
Place it: drag to move, Ctrl+wheel scale, Shift+wheel rotate. Metre dimensions show on the object; Width (m) sets exact size, Measure (m) gives distance + bearing from the imagery.
Check the Native footprint panel - that is what gets written.
Export GeoTIFF (placements saved to a .json sidecar, reloadable).

Tones: leave on match local background; slider ~150 = realistic pop. Output grid auto (template GSD) + chip around objects keeps the template at full resolution without a huge file.











#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geotiff_overlay_gui.py - place PNG object templates on a GeoTIFF, export a
new GeoTIFF (CRS / transform / nodata / extra bands preserved).

INSTALL   pip install rasterio numpy Pillow     (tkinter ships with Anaconda)

RUN       F5 in Spyder or: python geotiff_overlay_gui.py
          Under Spyder it launches as a separate process; if no window
          appears, read gui_launch.log next to this file.
          RUN.GEOTIFF / RUN.TEMPLATE below can pre-load files.

USE       1. Load GeoTIFF
          2. Add PNG template -> draw a polygon around the object
             (right-click undo, wheel zoom, right-drag pan)
          3. Place: drag = move, Ctrl+wheel = scale, Shift+wheel = rotate,
             arrows = nudge, Delete = remove. Metre dimensions show on the
             object; Width (m) sets exact size; Measure (m) reads distance
             and bearing off the imagery.
          4. Native footprint panel = exactly what gets written.
          5. Export GeoTIFF (placements go to a reloadable .json sidecar).

TONES     keep "match local background"; slider ~150 = realistic pop.
          Output grid "auto (template GSD)" + "chip around objects" keeps
          the template at full resolution without a huge file.
"""

import os
import sys
import gc
import json
import math
import subprocess
import time
import traceback

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _HAVE_TK = True
except Exception:                                                # pragma: no cover
    # Lets the file still be imported (e.g. to reuse the export functions from a
    # headless script) on a Python build without tkinter.
    _HAVE_TK = False

    class _NoTk(object):
        Tk = Toplevel = Canvas = Text = Listbox = object

        def __getattr__(self, name):
            return object

    tk = ttk = filedialog = messagebox = _NoTk()

from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps
try:
    from PIL import ImageTk
except Exception:                                                # pragma: no cover
    ImageTk = None

import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling


# --------------------------------------------------------------------------- #
#  Settings -- edit these if you want the tool to pre-load files on launch
# --------------------------------------------------------------------------- #
class RUN:
    GEOTIFF = ""            # optional path pre-loaded at startup
    TEMPLATE = ""           # optional PNG pre-loaded at startup
    PREVIEW_MAX = 1600      # max preview dimension in pixels (display only)
    STRETCH = (2.0, 98.0)   # percentile stretch used for the display preview
    OUT_SUFFIX = "_overlay"  # default suffix for the exported GeoTIFF
    BLOCK = 1024            # window size used when copying the raster
    ZOOM_STEP = 1.25        # wheel zoom factor
    ZOOM_MAX = 32.0         # max canvas px per raster px
    # Tk cannot safely share an IPython kernel's threads (Spyder throws
    # "Tcl_AsyncDelete: async handler deleted by the wrong thread" and kills
    # the kernel).  When True, running under Spyder/IPython relaunches the GUI
    # in its own interpreter instead.  Set False to force in-kernel running.
    # local tone matching: ring of background sampled around the object
    RING_FRAC = 0.6         # ring pad = this fraction of the larger bbox side
    RING_MIN = 8            # ...but at least this many px
    RING_MAX = 512          # ...and at most this many px
    RING_MIN_SAMPLES = 200  # fall back to scene stretch below this many px
    DETACH_FROM_SPYDER = True
    DETACH_NO_CONSOLE = True    # hide the child's console; it logs to a file
    DETACH_WAIT = 2.5           # seconds to watch the child before reporting


COLOR_MODES = ["original", "grayscale", "black & white"]


# =========================================================================== #
#  Raster helpers
# =========================================================================== #
def pick_display_bands(ds):
    """Return the 1-based band indexes used for display (3 for RGB, 1 for gray)."""
    if ds.count >= 3:
        return [1, 2, 3]
    return [1]


def stretch_limits(arr, plo, phi, nodata=None):
    """Per-band (lo, hi) percentile limits, ignoring nodata."""
    lims = []
    for b in range(arr.shape[0]):
        v = arr[b].astype("float64").ravel()
        if nodata is not None:
            v = v[v != nodata]
        v = v[np.isfinite(v)]
        if v.size == 0:
            lims.append((0.0, 255.0))
            continue
        lo, hi = np.percentile(v, [plo, phi])
        if hi <= lo:
            lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1.0
        lims.append((float(lo), float(hi)))
    return lims


def to_uint8(arr, lims):
    """Apply per-band limits and return a HxWxN uint8 array."""
    out = np.empty(arr.shape, dtype="uint8")
    for b in range(arr.shape[0]):
        lo, hi = lims[b % len(lims)]
        x = (arr[b].astype("float32") - lo) / max(hi - lo, 1e-9)
        out[b] = np.clip(x * 255.0, 0, 255).astype("uint8")
    return np.moveaxis(out, 0, -1)


def array_to_rgb(arr, lims):
    """Bands-first array -> PIL RGB image using the given stretch limits."""
    rgb = to_uint8(arr, lims)
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    elif rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    return Image.fromarray(np.ascontiguousarray(rgb), "RGB")


def build_preview(path, max_dim, stretch):
    """
    Read a decimated preview of the raster.

    Returns a dict of preview image + raster metadata.
    """
    with rasterio.open(path) as ds:
        bands = pick_display_bands(ds)
        decim = max(1.0, max(ds.width, ds.height) / float(max_dim))
        ow = max(1, int(round(ds.width / decim)))
        oh = max(1, int(round(ds.height / decim)))
        data = ds.read(
            indexes=bands,
            out_shape=(len(bands), oh, ow),
            resampling=Resampling.bilinear,
        )
        lims = stretch_limits(data, stretch[0], stretch[1], ds.nodata)
        info = dict(
            image=array_to_rgb(data, lims),
            decim=float(ds.width) / ow,
            bands=bands,
            lims=lims,
            width=ds.width,
            height=ds.height,
            transform=ds.transform,
            crs=ds.crs,
            dtype=ds.dtypes[0],
            count=ds.count,
            res=ds.res,
            nodata=ds.nodata,
        )
    return info


def read_window_rgb(path, bands, lims, win, out_w, out_h, nearest=False):
    """Read one window at an explicit output size and return a PIL RGB image."""
    with rasterio.open(path) as ds:
        data = ds.read(
            indexes=bands,
            window=win,
            out_shape=(len(bands), out_h, out_w),
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
            boundless=False,
        )
    return array_to_rgb(data, lims)


def ring_stats(ds_like, bands, x0, y0, x1, y1, pad, nodata=None,
               decimate_to=None, exclude=None):
    """
    Per-band (mean, std, n) of the background ring around bbox (x0..x1, y0..y1),
    excluding the object and nodata.  Returns None when the ring is too small
    to trust.

    exclude = (mask2d, ex0, ey0): the object's actual alpha contour placed at
    (ex0, ey0).  With it, only pixels UNDER the object are excluded, so the
    empty corners of a rotated object's bbox still count as background - the
    ring hugs the contour instead of the rectangle.  Without it, the whole
    bbox is excluded (legacy behaviour).
    """
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1 = min(ds_like.width, x1 + pad)
    ry1 = min(ds_like.height, y1 + pad)
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    win = Window(rx0, ry0, rx1 - rx0, ry1 - ry0)
    kw = {}
    f = 1.0
    if decimate_to and max(rx1 - rx0, ry1 - ry0) > decimate_to:
        f = max(rx1 - rx0, ry1 - ry0) / float(decimate_to)
        kw = dict(out_shape=(len(bands),
                             max(1, int(round((ry1 - ry0) / f))),
                             max(1, int(round((rx1 - rx0) / f)))),
                  resampling=Resampling.nearest)   # nearest keeps the histogram
    data = ds_like.read(indexes=bands, window=win, **kw).astype("float32")
    h, w = data.shape[1:]
    mask = np.ones((h, w), dtype=bool)
    if exclude is not None:
        em, ex0, ey0 = exclude
        if f > 1.0:
            # rasterio's nearest picks ~cell-centre samples; align to centres
            # and dilate by the decimation radius so no object edge slips in
            em = em.copy()
            for _ in range(min(8, int(math.ceil(f / 2.0)) + 1)):
                d2 = em.copy()
                d2[1:, :] |= em[:-1, :]; d2[:-1, :] |= em[1:, :]
                d2[:, 1:] |= em[:, :-1]; d2[:, :-1] |= em[:, 1:]
                em = d2
        rr = ((np.arange(h) + 0.5) * f + ry0).astype(int) - int(ey0)
        cc = ((np.arange(w) + 0.5) * f + rx0).astype(int) - int(ex0)
        RR = np.clip(rr, 0, em.shape[0] - 1)[:, None]
        CC = np.clip(cc, 0, em.shape[1] - 1)[None, :]
        hit = em[RR, CC]
        valid = ((rr >= 0) & (rr < em.shape[0]))[:, None] \
            & ((cc >= 0) & (cc < em.shape[1]))[None, :]
        mask &= ~(hit & valid)
    else:
        fx0 = max(0, int((x0 - rx0) / f)); fy0 = max(0, int((y0 - ry0) / f))
        fx1 = min(w, int(math.ceil((x1 - rx0) / f)))
        fy1 = min(h, int(math.ceil((y1 - ry0) / f)))
        if fx1 > fx0 and fy1 > fy0:
            mask[fy0:fy1, fx0:fx1] = False
    out = []
    for i in range(data.shape[0]):
        v = data[i][mask]
        if nodata is not None:
            v = v[v != nodata]
        v = v[np.isfinite(v)]
        if v.size < RUN.RING_MIN_SAMPLES:
            return None
        out.append((float(v.mean()), float(v.std()), int(v.size)))
    return out


def ring_pad_for(w, h):
    return int(min(RUN.RING_MAX, max(RUN.RING_MIN,
                                     round(RUN.RING_FRAC * max(w, h)))))


def object_channel_stats(chans, alpha_mask):
    """(mean, std>=1) per channel over the opaque part of the object."""
    out = []
    for c in chans:
        v = c[alpha_mask]
        if v.size < 10:
            return None
        out.append((float(v.mean()), max(float(v.std()), 1.0)))
    return out


def apply_linear_rgb(img, maps):
    """Apply per-channel v' = A*v + B to the RGB of an RGBA image (display)."""
    a = np.array(img)
    rgb = a[..., :3].astype("float32")
    for i in range(3):
        A, B = maps[min(i, len(maps) - 1)]
        rgb[..., i] = rgb[..., i] * A + B
    a[..., :3] = np.clip(rgb, 0, 255).astype("uint8")
    return Image.fromarray(a, "RGBA")


def dtype_range(dt):
    dt = np.dtype(dt)
    if dt.kind == "u":
        return 0.0, float(np.iinfo(dt).max)
    if dt.kind == "i":
        return float(np.iinfo(dt).min), float(np.iinfo(dt).max)
    return 0.0, 1.0            # float rasters: assume reflectance-ish 0..1


# =========================================================================== #
#  Template / mask / colour helpers
# =========================================================================== #
def load_template(path):
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def apply_polygon_mask(img, points, invert=False, feather=0.0):
    """Keep only what is inside the polygon (or outside, if invert)."""
    if not points or len(points) < 3:
        return img
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in points], fill=255)
    if invert:
        mask = ImageOps.invert(mask)
    if feather and feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather)))
    out = img.copy()
    out.putalpha(ImageChops.multiply(img.getchannel("A"), mask))
    return out


def remove_bg_color(img, rgb, tol):
    """Knock out pixels whose colour is within `tol` of `rgb` (max-channel dist)."""
    a = np.array(img)
    d = np.max(np.abs(a[..., :3].astype("int16") - np.array(rgb, dtype="int16")),
               axis=-1)
    a[..., 3] = np.where(d <= int(tol), 0, a[..., 3]).astype("uint8")
    return Image.fromarray(a, "RGBA")


def apply_color_mode(img, mode="original", thresh=128, invert=False):
    """
    Recolour the RGB channels of an RGBA image, leaving alpha untouched.

    mode : "original"      leave colours alone
           "grayscale"     luminance (ITU-R 601)
           "black & white"  hard threshold on luminance -> pure 0 / 255
    """
    if mode == "original" and not invert:
        return img
    a = np.array(img)
    rgb = a[..., :3].astype("float32")
    if mode == "original":
        out = rgb
    else:
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if mode == "black & white":
            lum = np.where(lum >= float(thresh), 255.0, 0.0)
        out = np.repeat(lum[..., None], 3, axis=-1)
    if invert:
        out = 255.0 - out
    a[..., :3] = np.clip(out, 0, 255).astype("uint8")
    return Image.fromarray(a, "RGBA")


RESAMPLE = {"lanczos (sharpest)": Image.LANCZOS,
            "area  (sensor-like)": Image.BOX,
            "nearest (hard/aliased)": Image.NEAREST}


def harden_alpha(img, cut=128):
    """Binarise alpha so the silhouette does not blend into the terrain."""
    a = np.array(img)
    a[..., 3] = np.where(a[..., 3] >= int(cut), 255, 0).astype("uint8")
    return Image.fromarray(a, "RGBA")


def unsharp_rgba(img, percent, radius=1.2, threshold=2):
    """Unsharp-mask the RGB channels only; alpha is left hard."""
    if not percent or percent <= 0:
        return img
    rgb = img.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=radius, percent=int(percent),
                                threshold=int(threshold)))
    out = rgb.convert("RGBA")
    out.putalpha(img.getchannel("A"))
    return out


def apply_tone_contrast(img, contrast):
    """Expand the object's RGB contrast about mid-grey (display-domain)."""
    if abs(contrast - 1.0) < 1e-3:
        return img
    a = np.array(img)
    v = 128.0 + (a[..., :3].astype("float32") - 128.0) * float(contrast)
    a[..., :3] = np.clip(v, 0, 255).astype("uint8")
    return Image.fromarray(a, "RGBA")


def paste_rgba(base, img, x, y):
    """alpha_composite that tolerates negative / overflowing offsets."""
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1 = min(base.width, x + img.width)
    y1 = min(base.height, y + img.height)
    if x1 <= x0 or y1 <= y0:
        return
    base.alpha_composite(img.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def trim_to_alpha(img):
    """Crop to the bounding box of non-transparent pixels."""
    bbox = img.getchannel("A").getbbox()
    if bbox is None:
        return img
    return img.crop(bbox)


# =========================================================================== #
#  Overlay object
# =========================================================================== #
class Overlay(object):
    """
    One placed template instance.  Geometry is in raster pixel space.

    `base` is the masked template with original colours; `src` is `base` after
    the colour mode has been applied and is what actually gets composited.
    """

    _n = 0

    def __init__(self, base_rgba, path="", name=None, cx=0.0, cy=0.0,
                 scale=1.0, rot=0.0, opacity=1.0):
        Overlay._n += 1
        self.path = path
        self.name = name or "obj_%02d" % Overlay._n
        self.cx = float(cx)                 # centre, raster pixel coords
        self.cy = float(cy)
        self.scale = float(scale)           # 1.0 => 1 template px = 1 raster px
        self.rot = float(rot)               # degrees, CCW
        self.opacity = float(opacity)       # 0..1
        # mask state, kept so placements can be re-created from JSON
        self.poly = []
        self.invert = False
        self.feather = 0.0
        self.bg = None
        self.bg_tol = 0
        # colour state
        self.color_mode = "original"
        self.thresh = 128
        self.invert_lum = False
        self.sharpen = 0.0                  # unsharp-mask strength, percent
        self.hard_edge = False              # binarise alpha after resampling
        self.filt = "lanczos (sharpest)"    # downsample kernel
        self.set_base(base_rgba)

    # -- colour ------------------------------------------------------------ #
    def set_base(self, base_rgba):
        self.base = base_rgba
        self.apply_color()

    def apply_color(self):
        self.src = apply_color_mode(self.base, self.color_mode, self.thresh,
                                    self.invert_lum)

    def set_color(self, mode=None, thresh=None, invert=None):
        if mode is not None:
            self.color_mode = mode
        if thresh is not None:
            self.thresh = int(thresh)
        if invert is not None:
            self.invert_lum = bool(invert)
        self.apply_color()

    # -- geometry ---------------------------------------------------------- #
    def scaled_size(self, view=1.0):
        w = max(1, int(round(self.src.width * self.scale * view)))
        h = max(1, int(round(self.src.height * self.scale * view)))
        return w, h

    def render(self, view=1.0, contrast=1.0):
        """
        Return the transformed RGBA image at `view` * scale.

        Rotation is done at the template's native resolution and the result is
        resampled to its final size in ONE step.  Resampling twice (resize then
        rotate) costs roughly 12% of the edge energy on a hard-edged template.
        Doing the downsample last also anti-aliases properly.
        """
        img = self.src
        if abs(self.rot) > 1e-6:
            img = img.rotate(self.rot, resample=Image.BICUBIC, expand=True)
        s = self.scale * float(view)
        w = max(1, int(round(img.width * s)))
        h = max(1, int(round(img.height * s)))
        if (w, h) != img.size:
            kern = RESAMPLE.get(self.filt, Image.LANCZOS)
            img = img.resize((w, h), kern if w * h < 4e7 else Image.BILINEAR)
        if self.hard_edge:
            img = harden_alpha(img)
        # scale the unsharp radius to the object: a 1.2 px radius on a 20 px
        # object smears across 10% of it
        img = unsharp_rgba(img, self.sharpen,
                           radius=min(1.5, max(0.5, min(w, h) / 40.0)))
        img = apply_tone_contrast(img, contrast)
        if self.opacity < 0.999:
            a = img.getchannel("A").point(lambda v: int(v * self.opacity))
            img.putalpha(a)
        return img

    def bbox(self, view=1.0):
        """(left, top, right, bottom) of the rendered object, in view pixels."""
        img_w, img_h = self.scaled_size(view)
        if abs(self.rot) > 1e-6:
            r = math.radians(self.rot)
            c, s = abs(math.cos(r)), abs(math.sin(r))
            w = img_w * c + img_h * s
            h = img_w * s + img_h * c
        else:
            w, h = img_w, img_h
        cx, cy = self.cx * view, self.cy * view
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def hit(self, x, y, view=1.0):
        l, t, r, b = self.bbox(view)
        return (l <= x <= r) and (t <= y <= b)

    def to_dict(self):
        return dict(name=self.name, path=self.path, cx=self.cx, cy=self.cy,
                    scale=self.scale, rot=self.rot, opacity=self.opacity,
                    poly=self.poly, invert=self.invert, feather=self.feather,
                    bg=self.bg, bg_tol=self.bg_tol,
                    color_mode=self.color_mode, thresh=self.thresh,
                    invert_lum=self.invert_lum, sharpen=self.sharpen,
                    hard_edge=self.hard_edge, filt=self.filt)


def rebuild_from_dict(d):
    """Re-create an Overlay from a saved placement dict."""
    img = load_template(d["path"])
    if d.get("bg") is not None:
        img = remove_bg_color(img, d["bg"], d.get("bg_tol", 0))
    if d.get("poly"):
        img = apply_polygon_mask(img, d["poly"], d.get("invert", False),
                                 d.get("feather", 0.0))
    img = trim_to_alpha(img)
    ov = Overlay(img, path=d["path"], name=d.get("name"), cx=d["cx"], cy=d["cy"],
                 scale=d["scale"], rot=d["rot"], opacity=d.get("opacity", 1.0))
    ov.poly = d.get("poly", [])
    ov.invert = d.get("invert", False)
    ov.feather = d.get("feather", 0.0)
    ov.bg = d.get("bg")
    ov.bg_tol = d.get("bg_tol", 0)
    ov.sharpen = float(d.get("sharpen", 0.0))
    ov.hard_edge = bool(d.get("hard_edge", False))
    ov.filt = d.get("filt", "lanczos (sharpest)")
    ov.set_color(d.get("color_mode", "original"), d.get("thresh", 128),
                 d.get("invert_lum", False))
    return ov


# =========================================================================== #
#  Export
# =========================================================================== #
def copy_raster(src_path, dst_path, block=1024, factor=1, window=None,
                log=print):
    """
    Write a tiled/LZW copy of the source, preserving georeferencing.

    factor > 1 writes a finer output grid (transform scaled to match) so small
    objects land on more pixels.  The imagery is upsampled NEAREST, so every
    output pixel is still an actual source measurement - no invented DN values,
    just replicated ones.
    """
    factor = int(max(1, factor))
    with rasterio.open(src_path) as src:
        if window is None:
            window = Window(0, 0, src.width, src.height)
        ox, oy = int(window.col_off), int(window.row_off)
        sw, sh = int(window.width), int(window.height)
        profile = src.profile.copy()
        profile.update(driver="GTiff", tiled=True, blockxsize=512, blockysize=512,
                       compress="LZW", BIGTIFF="IF_SAFER")
        profile.pop("photometric", None)
        profile.update(
            width=sw * factor, height=sh * factor,
            transform=src.window_transform(window) * rasterio.Affine.scale(
                1.0 / factor, 1.0 / factor))
        if (sw, sh) != (src.width, src.height):
            log("  chip: source window x %d..%d, y %d..%d (%d x %d px)"
                % (ox, ox + sw, oy, oy + sh, sw, sh))
        if factor > 1:
            log("  output grid %dx finer: %d x %d px, %g units/px (imagery "
                "upsampled NEAREST - replicated, not invented, values)"
                % (factor, sw * factor, sh * factor,
                   abs(src.transform.a) / factor))
        with rasterio.open(dst_path, "w", **profile) as dst:
            if src.colorinterp:
                try:
                    dst.colorinterp = src.colorinterp
                except Exception:
                    pass
            n = 0
            for row in range(0, sh, block):
                h = min(block, sh - row)
                src_win = Window(ox, oy + row, sw, h)
                if factor == 1:
                    dst.write(src.read(window=src_win),
                              window=Window(0, row, sw, h))
                else:
                    data = src.read(window=src_win,
                                    out_shape=(src.count, h * factor,
                                               sw * factor),
                                    resampling=Resampling.nearest)
                    dst.write(data, window=Window(0, row * factor,
                                                  sw * factor, h * factor))
                n += 1
            log("  copied %d row-blocks (%dx%d, %d band(s), %s)"
                % (n, sw * factor, sh * factor, src.count, src.dtypes[0]))


def composite_overlay(dst, ov, bands, lims, match_range=True, contrast=1.0,
                      factor=1, origin=(0, 0), tone_mode=None, log=print):
    """tone_mode: "stretch" (scene percentile window), "local" (ring mean/std,
    contrast = object-to-background sigma ratio), "dtype" (full range).
    None keeps the legacy match_range behaviour."""
    """
    Blend one Overlay into an open, writable rasterio dataset `dst`.

    bands : 1-based band indexes that carry the visible image
    lims  : [(lo, hi), ...] display stretch limits, one per entry in `bands`
    """
    img = ov.render(view=float(factor))
    ow, oh = img.size
    left = int(math.floor((ov.cx - origin[0]) * factor - ow / 2.0))
    top = int(math.floor((ov.cy - origin[1]) * factor - oh / 2.0))

    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(dst.width, left + ow), min(dst.height, top + oh)
    if x1 <= x0 or y1 <= y0:
        log("  %s: fully outside the image, skipped" % ov.name)
        return False

    img = img.crop((x0 - left, y0 - top, x1 - left, y1 - top))
    a = np.array(img)
    ov_rgb = a[..., :3].astype("float32")
    alpha = a[..., 3].astype("float32") / 255.0

    win = Window(x0, y0, x1 - x0, y1 - y0)
    dmin, dmax = dtype_range(dst.dtypes[0])

    if len(bands) == 1:                       # single-band / grayscale target
        chans = [0.299 * ov_rgb[..., 0] + 0.587 * ov_rgb[..., 1]
                 + 0.114 * ov_rgb[..., 2]]
    else:
        chans = [ov_rgb[..., i] for i in range(3)]

    if tone_mode is None:
        tone_mode = "stretch" if match_range else "dtype"

    local_maps = None
    if tone_mode == "local":
        pad = ring_pad_for(x1 - x0, y1 - y0)
        st = ring_stats(dst, bands, x0, y0, x1, y1, pad, dst.nodata,
                        exclude=(alpha >= 0.02, x0, y0))
        ost = object_channel_stats(chans, alpha >= 0.5)
        if st is None or ost is None:
            log("  %s: background ring too small/empty - falling back to "
                "scene-stretch tones" % ov.name)
            tone_mode = "stretch"
        else:
            local_maps = []
            for i in range(len(bands)):
                mu_b, sd_b, n = st[i]
                mu_o, sd_o = ost[i]
                gain = sd_b * float(contrast) / sd_o
                local_maps.append((gain, mu_b - gain * mu_o))
                log("  %s b%d: ring n=%d mu=%.1f sd=%.1f | obj mu=%.1f "
                    "sd=%.1f | gain %.3f offset %.1f"
                    % (ov.name, bands[i], n, mu_b, sd_b, mu_o, sd_o,
                       gain, mu_b - gain * mu_o))

    for i, b in enumerate(bands):
        base = dst.read(b, window=win).astype("float32")
        if tone_mode == "local":
            A, B = local_maps[i]
            val = chans[i] * A + B
        elif tone_mode == "stretch":
            lo, hi = lims[i] if i < len(lims) else (dmin, dmax)
            # widen the mapped range about its midpoint so the object is not
            # crushed into the scene's percentile window; it may exceed lo/hi
            # and is clipped to the dtype range below
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * float(contrast)
            lo, hi = mid - half, mid + half
            val = lo + (chans[i] / 255.0) * (hi - lo)
        else:
            val = dmin + (chans[i] / 255.0) * (dmax - dmin)
        blended = np.clip(base * (1.0 - alpha) + val * alpha, dmin, dmax)
        dst.write(blended.astype(dst.dtypes[b - 1]), b, window=win)

    log("  %s: %dx%d px at (%d, %d), rot %.1f deg, scale %.4f, %s, sharpen %d%%"
        % (ov.name, x1 - x0, y1 - y0, x0, y0, ov.rot, ov.scale, ov.color_mode,
           int(ov.sharpen)))
    if ov.scale > 1.05:
        log("    NOTE: template enlarged %.1fx past its native pixels - the "
            "object cannot be sharper than the source PNG." % ov.scale)
    return True


def export_geotiff(src_path, dst_path, overlays, bands, lims,
                   match_range=True, contrast=1.0, factor=1, window=None,
                   tone_mode=None, block=1024, log=print):
    log("Copying source raster -> %s" % os.path.basename(dst_path))
    copy_raster(src_path, dst_path, block=block, factor=factor, window=window,
                log=log)
    origin = (0, 0) if window is None else (int(window.col_off),
                                            int(window.row_off))
    log("Compositing %d object(s) at full resolution..." % len(overlays))
    ok = 0
    with rasterio.open(dst_path, "r+") as dst:
        for ov in overlays:
            if composite_overlay(dst, ov, bands, lims, match_range, contrast,
                                 factor=factor, origin=origin,
                                 tone_mode=tone_mode, log=log):
                ok += 1
    log("Done. %d/%d object(s) written." % (ok, len(overlays)))
    return ok


# =========================================================================== #
#  Template editor window
# =========================================================================== #
def checkerboard(size, sq=8):
    w, h = size
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            if ((x // sq) + (y // sq)) % 2:
                d.rectangle([x, y, x + sq - 1, y + sq - 1], fill=(205, 205, 205))
    return img


class TemplateEditor(tk.Toplevel):
    """
    Draw a polygon / knock out a background colour / pick the colour mode.

    The view zooms and pans so the polygon can be placed against the object's
    actual edge:
        wheel        zoom about the cursor
        right-DRAG   pan            (a plain right-CLICK still undoes a point)
        middle-drag  pan
    Above 200% the template is shown nearest-neighbour, so you see the real
    pixels you are cutting along - that is what makes a tight footprint
    possible, which in turn gives the tone matcher clean background right up
    to the object.
    """

    CANW, CANH = 840, 540

    def __init__(self, master, path, on_done):
        tk.Toplevel.__init__(self, master)
        self.title("Template editor  -  %s" % os.path.basename(path))
        self.on_done = on_done
        self.path = path
        self.orig = load_template(path)

        self.points = []          # in original template pixel coords
        self.mode = tk.StringVar(value="poly")
        self.invert = tk.BooleanVar(value=False)
        self.feather = tk.DoubleVar(value=0.0)
        self.tol = tk.DoubleVar(value=30)
        self.trim = tk.BooleanVar(value=True)
        self.cmode = tk.StringVar(value="original")
        self.thresh = tk.DoubleVar(value=128)
        self.invlum = tk.BooleanVar(value=False)
        self.bg = None

        # view state: vs = canvas px per template px, (ox, oy) = template
        # coordinate at canvas (0, 0)
        self.vs = 1.0
        self.ox = self.oy = 0.0
        self._pan = None
        self._rpress = None
        self._styled_key = None
        self._styled_img = None

        bar = ttk.Frame(self, padding=6)
        bar.pack(side="top", fill="x")
        ttk.Radiobutton(bar, text="Draw polygon", variable=self.mode,
                        value="poly", command=self._refresh).pack(side="left")
        ttk.Radiobutton(bar, text="Pick background colour", variable=self.mode,
                        value="bg", command=self._refresh).pack(side="left",
                                                                padx=(8, 16))
        ttk.Button(bar, text="Undo point", command=self.undo).pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left",
                                                               padx=4)
        ttk.Checkbutton(bar, text="Invert (keep outside)", variable=self.invert,
                        command=self._refresh).pack(side="left", padx=12)

        bar2 = ttk.Frame(self, padding=(6, 0, 6, 4))
        bar2.pack(side="top", fill="x")
        ttk.Label(bar2, text="Feather px").pack(side="left")
        ttk.Scale(bar2, from_=0, to=12, variable=self.feather, length=110,
                  command=lambda e: self._refresh()).pack(side="left",
                                                          padx=(4, 14))
        ttk.Label(bar2, text="BG tolerance").pack(side="left")
        ttk.Scale(bar2, from_=0, to=180, variable=self.tol, length=140,
                  command=lambda e: self._refresh()).pack(side="left", padx=4)
        ttk.Checkbutton(bar2, text="Trim to content", variable=self.trim
                        ).pack(side="left", padx=12)

        bar3 = ttk.Frame(self, padding=(6, 0, 6, 4))
        bar3.pack(side="top", fill="x")
        ttk.Label(bar3, text="Colour").pack(side="left")
        cb = ttk.Combobox(bar3, values=COLOR_MODES, textvariable=self.cmode,
                          state="readonly", width=14)
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh())
        ttk.Label(bar3, text="B/W threshold").pack(side="left", padx=(12, 2))
        self.sc_thresh = ttk.Scale(bar3, from_=1, to=254, variable=self.thresh,
                                   length=150, command=lambda e: self._refresh())
        self.sc_thresh.pack(side="left")
        self.lbl_thresh = ttk.Label(bar3, text="128", width=4,
                                    foreground="#666")
        self.lbl_thresh.pack(side="left", padx=3)
        ttk.Checkbutton(bar3, text="Invert tone", variable=self.invlum,
                        command=self._refresh).pack(side="left", padx=10)

        bar4 = ttk.Frame(self, padding=(6, 0, 6, 6))
        bar4.pack(side="top", fill="x")
        ttk.Label(bar4, text="View").pack(side="left")
        ttk.Button(bar4, text="-", width=3,
                   command=lambda: self.zoom_by(1 / 1.3)).pack(side="left",
                                                               padx=(6, 0))
        ttk.Button(bar4, text="+", width=3,
                   command=lambda: self.zoom_by(1.3)).pack(side="left", padx=3)
        ttk.Button(bar4, text="Fit", command=self.zoom_fit).pack(side="left")
        ttk.Button(bar4, text="1:1", command=lambda: self.set_zoom(1.0)
                   ).pack(side="left", padx=3)
        self.lbl_zoom = ttk.Label(bar4, text="-", foreground="#666")
        self.lbl_zoom.pack(side="left", padx=8)
        self.lbl_pos = ttk.Label(bar4, text="", foreground="#666")
        self.lbl_pos.pack(side="right")

        self.canvas = tk.Canvas(self, width=self.CANW, height=self.CANH,
                                background="#333", highlightthickness=0)
        self.canvas.pack(side="top", padx=6)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", self.rdown)
        self.canvas.bind("<B3-Motion>", self.rmove)
        self.canvas.bind("<ButtonRelease-3>", self.rup)
        self.canvas.bind("<Button-2>", self.rdown)
        self.canvas.bind("<B2-Motion>", self.rmove)
        self.canvas.bind("<ButtonRelease-2>", lambda e: self.rup(e, undo=False))
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Button-4>", lambda e: self.wheel(e, 120))
        self.canvas.bind("<Button-5>", lambda e: self.wheel(e, -120))
        self.canvas.bind("<Motion>", self.hover)

        self.hint = ttk.Label(self, padding=6, foreground="#555", text="")
        self.hint.pack(side="top", anchor="w")

        btns = ttk.Frame(self, padding=6)
        btns.pack(side="bottom", fill="x")
        ttk.Button(btns, text="Use this object", command=self.accept
                   ).pack(side="right")
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right",
                                                                   padx=6)

        self._tkimg = None
        self.zoom_fit()
        self.transient(master)
        self.grab_set()

    # ---------------------------------------------------------- view maths -- #
    def c2t(self, x, y):
        return self.ox + x / self.vs, self.oy + y / self.vs

    def t2c(self, tx, ty):
        return (tx - self.ox) * self.vs, (ty - self.oy) * self.vs

    def fit_vs(self):
        return min(self.CANW / float(self.orig.width),
                   self.CANH / float(self.orig.height))

    def clamp(self):
        vw, vh = self.CANW / self.vs, self.CANH / self.vs
        W, H = self.orig.width, self.orig.height
        self.ox = (W - vw) / 2.0 if vw >= W else min(max(self.ox, 0.0), W - vw)
        self.oy = (H - vh) / 2.0 if vh >= H else min(max(self.oy, 0.0), H - vh)

    def set_zoom(self, nv, anchor=None):
        nv = max(self.fit_vs() * 0.5, min(32.0, float(nv)))
        ax, ay = anchor if anchor else (self.CANW / 2.0, self.CANH / 2.0)
        tx, ty = self.c2t(ax, ay)
        self.vs = nv
        self.ox, self.oy = tx - ax / nv, ty - ay / nv
        self.clamp()
        self._refresh()

    def zoom_by(self, f, anchor=None):
        self.set_zoom(self.vs * f, anchor)

    def zoom_fit(self):
        self.vs = self.fit_vs()
        self.ox = (self.orig.width - self.CANW / self.vs) / 2.0
        self.oy = (self.orig.height - self.CANH / self.vs) / 2.0
        self.clamp()
        self._refresh()

    def wheel(self, ev, delta=None):
        d = delta if delta is not None else ev.delta
        self.zoom_by(1.3 if d > 0 else 1 / 1.3, anchor=(ev.x, ev.y))

    # ---------------------------------------------------------------- mask -- #
    def masked(self):
        img = self.orig
        if self.bg is not None:
            img = remove_bg_color(img, self.bg, int(self.tol.get()))
        if len(self.points) >= 3:
            img = apply_polygon_mask(img, self.points, self.invert.get(),
                                     self.feather.get())
        return img

    def styled(self):
        """Masked + colour-moded template at native res, cached."""
        key = (tuple(tuple(p) for p in self.points), self.invert.get(),
               round(self.feather.get(), 2), int(self.tol.get()),
               tuple(self.bg) if self.bg else None, self.cmode.get(),
               int(self.thresh.get()), self.invlum.get())
        if key != self._styled_key:
            self._styled_img = apply_color_mode(self.masked(),
                                                self.cmode.get(),
                                                int(self.thresh.get()),
                                                self.invlum.get())
            self._styled_key = key
        return self._styled_img

    # ------------------------------------------------------------- drawing -- #
    def _refresh(self):
        self.lbl_thresh.config(text="%d" % int(self.thresh.get()))
        self.sc_thresh.state(["!disabled"]
                             if self.cmode.get() == "black & white"
                             else ["disabled"])

        img = self.styled()
        base = checkerboard((self.CANW, self.CANH)).convert("RGBA")
        tx0, ty0 = max(0.0, self.ox), max(0.0, self.oy)
        tx1 = min(float(img.width), self.ox + self.CANW / self.vs)
        ty1 = min(float(img.height), self.oy + self.CANH / self.vs)
        if tx1 > tx0 and ty1 > ty0:
            ow = max(1, int(round((tx1 - tx0) * self.vs)))
            oh = max(1, int(round((ty1 - ty0) * self.vs)))
            slab = img.resize((ow, oh),
                              Image.NEAREST if self.vs >= 2.0
                              else Image.BILINEAR,
                              box=(tx0, ty0, tx1, ty1))
            paste_rgba(base, slab, (tx0 - self.ox) * self.vs,
                       (ty0 - self.oy) * self.vs)
        self._tkimg = ImageTk.PhotoImage(base)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tkimg)

        if self.points:
            pts = [self.t2c(p[0], p[1]) for p in self.points]
            if len(pts) >= 2:
                flat = []
                for p in pts + [pts[0]]:
                    flat.extend(p)
                self.canvas.create_line(*flat, fill="#00e0ff", width=2)
            for x, y in pts:
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                        fill="#00e0ff", outline="")
        self.lbl_zoom.config(text="%.0f%%" % (100.0 * self.vs))
        if self.mode.get() == "bg":
            self.hint.config(text="Click a background pixel to knock that "
                                  "colour out; adjust tolerance to taste.")
        else:
            self.hint.config(text="Left-click adds a point  |  right-click "
                                  "undoes, right-DRAG pans  |  wheel zooms  |  "
                                  "%d point(s)" % len(self.points))

    # ---------------------------------------------------------------- mouse -- #
    def hover(self, ev):
        tx, ty = self.c2t(ev.x, ev.y)
        if 0 <= tx < self.orig.width and 0 <= ty < self.orig.height:
            self.lbl_pos.config(text="px %.1f, %.1f" % (tx, ty))
        else:
            self.lbl_pos.config(text="")

    def click(self, ev):
        tx, ty = self.c2t(ev.x, ev.y)
        if not (0 <= tx <= self.orig.width and 0 <= ty <= self.orig.height):
            return
        if self.mode.get() == "bg":
            px = int(min(max(tx, 0), self.orig.width - 1))
            py = int(min(max(ty, 0), self.orig.height - 1))
            self.bg = list(self.orig.getpixel((px, py))[:3])
        else:
            self.points.append([tx, ty])
        self._refresh()

    def rdown(self, ev):
        self._rpress = (ev.x, ev.y)
        self._pan = (ev.x, ev.y, self.ox, self.oy)

    def rmove(self, ev):
        if not self._pan:
            return
        x, y, ox, oy = self._pan
        self.ox = ox - (ev.x - x) / self.vs
        self.oy = oy - (ev.y - y) / self.vs
        self.clamp()
        self._refresh()

    def rup(self, ev, undo=True):
        moved = (self._rpress
                 and abs(ev.x - self._rpress[0]) + abs(ev.y - self._rpress[1])
                 > 4)
        self._pan = None
        self._rpress = None
        if undo and not moved:
            self.undo()

    # ------------------------------------------------------------- actions -- #
    def undo(self):
        if self.points:
            self.points.pop()
        self._refresh()

    def clear(self):
        self.points = []
        self.bg = None
        self._refresh()

    def accept(self):
        img = self.masked()
        if self.trim.get():
            img = trim_to_alpha(img)
        if img.getchannel("A").getbbox() is None:
            messagebox.showwarning("Empty object",
                                   "Everything is transparent - adjust the "
                                   "mask.", parent=self)
            return
        ov = Overlay(img, path=self.path)
        ov.poly = self.points
        ov.invert = self.invert.get()
        ov.feather = float(self.feather.get())
        ov.bg = self.bg
        ov.bg_tol = int(self.tol.get())
        ov.set_color(self.cmode.get(), int(self.thresh.get()),
                     self.invlum.get())
        self.destroy()
        self.on_done(ov)


# =========================================================================== #
#  Scrollable side panel
# =========================================================================== #
class ScrollFrame(ttk.Frame):
    """
    A fixed-width column whose contents scroll vertically.

    The control stack is taller than a 1080p window, so without this the
    bottom panels (Export, View, log) fall off the screen with no way to
    reach them.
    """

    def __init__(self, master, width=310):
        ttk.Frame.__init__(self, master)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0,
                                borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient="vertical",
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._win, width=e.width))
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._wheel(e, 120))
        self.canvas.bind("<Button-5>", lambda e: self._wheel(e, -120))

    def _wheel(self, ev, delta=None):
        d = delta if delta is not None else ev.delta
        self.canvas.yview_scroll(-1 if d > 0 else 1, "units")
        return "break"

    def bind_wheel(self, widget=None, skip=()):
        """Bind the wheel on every child so scrolling works anywhere in the
        column - except widgets that need the wheel themselves."""
        w = self.inner if widget is None else widget
        if not isinstance(w, skip):
            try:
                w.bind("<MouseWheel>", self._wheel)
                w.bind("<Button-4>", lambda e: self._wheel(e, 120))
                w.bind("<Button-5>", lambda e: self._wheel(e, -120))
            except Exception:
                pass
        for child in w.winfo_children():
            self.bind_wheel(child, skip)


# =========================================================================== #
#  Main application
# =========================================================================== #
class App(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        self.title("GeoTIFF Object Overlay")
        # fit the actual display rather than assuming a big monitor
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1380, sw - 60), min(900, sh - 100)
        self.geometry("%dx%d+%d+%d" % (w, h, max(0, (sw - w) // 2),
                                       max(0, (sh - h) // 3)))
        self.minsize(820, 520)

        self.info = None            # preview / raster metadata
        self.tif_path = None
        self.overlays = []
        self.sel = None
        self.view_lims = []

        # --- viewport state: ds = canvas px per raster px, (ox, oy) = raster
        #     coordinate shown at canvas pixel (0, 0)
        self.ds = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self._fitted = False
        self._base_img = None
        self._base_key = None

        self._drag = None
        self._pan = None
        self._redraw_pending = False
        self.aoi = None                 # (x0, y0, x1, y1) in raster px
        self._aoi_mode = False
        self._aoi_start = None
        self.measure_pts = []           # measuring polyline, raster px
        self._measure = False
        self._cursor_r = None
        self._tkimg = None
        self._suspend = False

        self._build_ui()

        if RUN.GEOTIFF and os.path.isfile(RUN.GEOTIFF):
            self.load_tif(RUN.GEOTIFF)
        if RUN.TEMPLATE and os.path.isfile(RUN.TEMPLATE) and self.info:
            self.add_template(RUN.TEMPLATE)

    # ------------------------------------------------------------------ UI -- #
    def _build_ui(self):
        root = ttk.Frame(self, padding=6)
        root.pack(fill="both", expand=True)

        scroll = ScrollFrame(root, width=306)
        scroll.pack(side="left", fill="y", padx=(0, 8))
        panel = scroll.inner
        self._scroll = scroll

        # --- files
        f = ttk.LabelFrame(panel, text="1.  Files", padding=8)
        f.pack(fill="x")
        ttk.Button(f, text="Load GeoTIFF...", command=self.on_load_tif
                   ).pack(fill="x")
        ttk.Button(f, text="Add PNG template...", command=self.on_add_template
                   ).pack(fill="x", pady=(4, 0))
        self.lbl_tif = ttk.Label(f, text="(no raster loaded)", wraplength=268,
                                 foreground="#666")
        self.lbl_tif.pack(fill="x", pady=(6, 0))

        # --- objects
        f = ttk.LabelFrame(panel, text="2.  Objects", padding=8)
        f.pack(fill="x", pady=8)
        self.lst = tk.Listbox(f, height=5, exportselection=False)
        self.lst.pack(fill="x")
        self.lst.bind("<<ListboxSelect>>", self.on_pick)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Duplicate", command=self.on_dup
                   ).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Edit mask", command=self.on_edit_mask
                   ).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(row, text="Delete", command=self.on_del
                   ).pack(side="left", expand=True, fill="x")

        # --- colour
        f = ttk.LabelFrame(panel, text="3.  Object colour", padding=8)
        f.pack(fill="x")
        r = ttk.Frame(f)
        r.pack(fill="x")
        self.v_cmode = tk.StringVar(value="original")
        cb = ttk.Combobox(r, values=COLOR_MODES, textvariable=self.v_cmode,
                          state="readonly", width=14)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_color())
        self.v_invlum = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="Invert tone", variable=self.v_invlum,
                        command=self.on_color).pack(side="left", padx=8)
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="B/W threshold").pack(side="left")
        self.v_thresh = tk.DoubleVar(value=128)
        self.sc_thresh = ttk.Scale(r, from_=1, to=254, variable=self.v_thresh,
                                   length=150, command=lambda e: self.on_color())
        self.sc_thresh.pack(side="left", padx=4)
        self.lbl_thresh = ttk.Label(r, text="128", width=4, foreground="#666")
        self.lbl_thresh.pack(side="left")

        # --- transform
        f = ttk.LabelFrame(panel, text="4.  Transform", padding=8)
        f.pack(fill="x", pady=8)

        self.v_rot = tk.DoubleVar(value=0.0)
        self.v_scale = tk.DoubleVar(value=1.0)
        self.v_op = tk.DoubleVar(value=100.0)

        ttk.Label(f, text="Rotation (deg CCW)").pack(anchor="w")
        r = ttk.Frame(f)
        r.pack(fill="x")
        ttk.Scale(r, from_=-180, to=180, variable=self.v_rot, length=205,
                  command=lambda e: self.on_slider()).pack(side="left")
        self.e_rot = ttk.Entry(r, width=7)
        self.e_rot.pack(side="left", padx=4)
        self.e_rot.bind("<Return>", lambda e: self.on_entry("rot"))

        ttk.Label(f, text="Scale (template px -> raster px)").pack(anchor="w",
                                                                   pady=(6, 0))
        r = ttk.Frame(f)
        r.pack(fill="x")
        ttk.Scale(r, from_=0.01, to=20.0, variable=self.v_scale, length=205,
                  command=lambda e: self.on_slider()).pack(side="left")
        self.e_scale = ttk.Entry(r, width=7)
        self.e_scale.pack(side="left", padx=4)
        self.e_scale.bind("<Return>", lambda e: self.on_entry("scale"))

        ttk.Label(f, text="Opacity %").pack(anchor="w", pady=(6, 0))
        ttk.Scale(f, from_=5, to=100, variable=self.v_op, length=205,
                  command=lambda e: self.on_slider()).pack(anchor="w")

        self.v_sharp = tk.DoubleVar(value=0.0)
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(6, 0))
        ttk.Label(r, text="Sharpen %").pack(side="left")
        ttk.Scale(r, from_=0, to=250, variable=self.v_sharp, length=140,
                  command=lambda e: self.on_slider()).pack(side="left", padx=4)
        self.lbl_sharp = ttk.Label(r, text="0", width=4, foreground="#666")
        self.lbl_sharp.pack(side="left")

        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        self.v_hard = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="Hard edges", variable=self.v_hard,
                        command=self.on_render_opts).pack(side="left")
        self.v_filt = tk.StringVar(value="lanczos (sharpest)")
        cbf = ttk.Combobox(r, values=list(RESAMPLE.keys()),
                           textvariable=self.v_filt, state="readonly", width=19)
        cbf.pack(side="left", padx=4)
        cbf.bind("<<ComboboxSelected>>", lambda e: self.on_render_opts())

        r = ttk.Frame(f)
        r.pack(fill="x", pady=(8, 0))
        ttk.Label(r, text="Width (m)").pack(side="left")
        self.e_m = ttk.Entry(r, width=9)
        self.e_m.pack(side="left", padx=4)
        self.e_m.bind("<Return>", lambda e: self.on_entry("metres"))
        ttk.Button(r, text="Native 1:1", width=10, command=self.native_size
                   ).pack(side="left", padx=4)
        self.lbl_objdims = ttk.Label(f, text="-", foreground="#666")
        self.lbl_objdims.pack(anchor="w", pady=(3, 0))

        r = ttk.Frame(f)
        r.pack(fill="x", pady=(6, 0))
        ttk.Label(r, text="Centre px  X").pack(side="left")
        self.e_cx = ttk.Entry(r, width=8)
        self.e_cx.pack(side="left", padx=3)
        ttk.Label(r, text="Y").pack(side="left")
        self.e_cy = ttk.Entry(r, width=8)
        self.e_cy.pack(side="left", padx=3)
        for e in (self.e_cx, self.e_cy):
            e.bind("<Return>", lambda ev: self.on_entry("centre"))

        # --- true 1:1 footprint
        f = ttk.LabelFrame(panel, text="Native footprint (what gets written)",
                           padding=8)
        f.pack(fill="x")
        self.pv_canvas = tk.Canvas(f, width=272, height=110, background="#111",
                                   highlightthickness=0)
        self.pv_canvas.pack()
        self.v_fp_bg = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="show on imagery (as it will blend)",
                        variable=self.v_fp_bg,
                        command=self.update_native_preview
                        ).pack(anchor="w", pady=(2, 0))
        self.lbl_fp = ttk.Label(f, text="-", foreground="#666",
                                wraplength=268, justify="left")
        self.lbl_fp.pack(anchor="w", pady=(3, 0))
        ttk.Button(f, text="Match template resolution",
                   command=self.match_template_res).pack(fill="x", pady=(4, 0))

        # --- export
        f = ttk.LabelFrame(panel, text="5.  Export", padding=8)
        f.pack(fill="x")
        r = ttk.Frame(f)
        r.pack(fill="x")
        ttk.Label(r, text="Tones").pack(side="left")
        self.v_tone = tk.StringVar(value="match local background")
        cbt = ttk.Combobox(r, values=["match local background",
                                      "match scene stretch",
                                      "full dtype range"],
                           textvariable=self.v_tone, state="readonly", width=21)
        cbt.pack(side="left", padx=4)
        cbt.bind("<<ComboboxSelected>>", lambda e: self.on_tone_mode())
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        self.lbl_conname = ttk.Label(r, text="sigma ratio %")
        self.lbl_conname.pack(side="left")
        self.v_contrast = tk.DoubleVar(value=100.0)
        ttk.Scale(r, from_=20, to=300, variable=self.v_contrast, length=110,
                  command=lambda e: self.on_contrast()).pack(side="left", padx=4)
        self.lbl_contrast = ttk.Label(r, text="100", width=4, foreground="#666")
        self.lbl_contrast.pack(side="left")
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="Output grid").pack(side="left")
        self.v_factor = tk.StringVar(value="1x (native)")
        cbg = ttk.Combobox(r, values=["1x (native)", "2x finer", "4x finer",
                                      "8x finer", "auto (template GSD)"],
                           textvariable=self.v_factor, state="readonly", width=18)
        cbg.pack(side="left", padx=4)
        cbg.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="Extent").pack(side="left")
        self.v_extent = tk.StringVar(value="full scene")
        cbe = ttk.Combobox(r, values=["full scene", "chip around objects",
                                      "custom AOI (drag)"],
                           textvariable=self.v_extent, state="readonly", width=18)
        cbe.pack(side="left", padx=4)
        cbe.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        ttk.Label(r, text="margin m").pack(side="left")
        self.e_margin = ttk.Entry(r, width=6)
        self.e_margin.insert(0, "50")
        self.e_margin.pack(side="left", padx=3)
        self.e_margin.bind("<Return>", lambda e: self.redraw())

        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        self.btn_aoi = ttk.Button(r, text="Draw AOI", command=self.arm_aoi)
        self.btn_aoi.pack(side="left", expand=True, fill="x")
        ttk.Button(r, text="Use view", command=self.aoi_from_view
                   ).pack(side="left", expand=True, fill="x", padx=4)
        ttk.Button(r, text="Clear", command=self.clear_aoi
                   ).pack(side="left", expand=True, fill="x")

        self.lbl_cost = ttk.Label(f, text="-", foreground="#666",
                                  wraplength=268, justify="left")
        self.lbl_cost.pack(anchor="w", pady=(4, 0))
        ttk.Button(f, text="Export GeoTIFF...", command=self.on_export
                   ).pack(fill="x", pady=(6, 0))
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        ttk.Button(r, text="Save placements", command=self.on_save_json
                   ).pack(side="left", expand=True, fill="x")
        ttk.Button(r, text="Load", command=self.on_load_json
                   ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        # --- view
        f = ttk.LabelFrame(panel, text="View", padding=8)
        f.pack(fill="x", pady=8)
        r = ttk.Frame(f)
        r.pack(fill="x")
        ttk.Button(r, text="-", width=3,
                   command=lambda: self.zoom_by(1 / RUN.ZOOM_STEP)).pack(side="left")
        ttk.Button(r, text="+", width=3,
                   command=lambda: self.zoom_by(RUN.ZOOM_STEP)).pack(side="left", padx=3)
        ttk.Button(r, text="Fit", command=self.zoom_fit).pack(side="left", padx=3)
        ttk.Button(r, text="1:1", command=lambda: self.set_zoom(1.0)
                   ).pack(side="left", padx=3)
        ttk.Button(r, text="Object", command=self.zoom_to_object).pack(side="left")
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        self.lbl_zoom = ttk.Label(r, text="-", foreground="#666")
        self.lbl_zoom.pack(side="left")
        ttk.Button(r, text="Restretch to view", command=self.restretch
                   ).pack(side="right")
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(4, 0))
        self.btn_measure = ttk.Button(r, text="Measure (m)",
                                      command=self.arm_measure)
        self.btn_measure.pack(side="left", expand=True, fill="x")
        ttk.Button(r, text="Clear", command=self.clear_measure
                   ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        # --- log
        self.log_box = tk.Text(panel, height=6, wrap="word",
                               font=("Consolas", 8))
        self.log_box.pack(fill="x", pady=(0, 6))

        # --- canvas
        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, background="#1e1e1e", highlightthickness=0)
        self.hs = ttk.Scrollbar(right, orient="horizontal", command=self.xview)
        self.vs = ttk.Scrollbar(right, orient="vertical", command=self.yview)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vs.grid(row=0, column=1, sticky="ns")
        self.hs.grid(row=1, column=0, sticky="ew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas.bind("<Configure>", self.on_configure)
        self.canvas.bind("<Button-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Motion>", self.on_hover)
        for b in ("2", "3"):                       # middle / right drag = pan
            self.canvas.bind("<Button-%s>" % b, self.pan_start)
            self.canvas.bind("<B%s-Motion>" % b, self.pan_move)
            self.canvas.bind("<ButtonRelease-%s>" % b, self.pan_end)
        self.canvas.bind("<MouseWheel>", self.on_wheel)          # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self.on_wheel(e, 120))
        self.canvas.bind("<Button-5>", lambda e: self.on_wheel(e, -120))
        # bound on the CANVAS, not the toplevel, so typing in the entry boxes
        # is not intercepted (Delete used to remove the selected object while
        # editing a number, arrows used to nudge it, "-" zoomed out)
        self.canvas.config(takefocus=1)
        self.canvas.bind("<Left>", lambda e: self.nudge(-1, 0))
        self.canvas.bind("<Right>", lambda e: self.nudge(1, 0))
        self.canvas.bind("<Up>", lambda e: self.nudge(0, -1))
        self.canvas.bind("<Down>", lambda e: self.nudge(0, 1))
        self.canvas.bind("<Delete>", lambda e: self.on_del())
        self.canvas.bind("<plus>", lambda e: self.zoom_by(RUN.ZOOM_STEP))
        self.canvas.bind("<minus>", lambda e: self.zoom_by(1 / RUN.ZOOM_STEP))
        self.bind("<Escape>", lambda e: (self.cancel_aoi(),
                                         self._measure
                                         and self.arm_measure()))

        # the Text log and Listbox keep their own wheel behaviour
        self._scroll.bind_wheel(skip=(tk.Text, tk.Listbox))

        self.status = ttk.Label(self, text="Load a GeoTIFF to begin.  "
                                           "Wheel = zoom, Ctrl+wheel = scale "
                                           "object, Shift+wheel = rotate, "
                                           "right-drag = pan.",
                                relief="sunken", anchor="w", padding=3)
        self.status.pack(side="bottom", fill="x")

    # --------------------------------------------------------------- utils -- #
    def log(self, msg):
        self.log_box.insert("end", str(msg) + "\n")
        self.log_box.see("end")
        self.update_idletasks()

    def cw(self):
        return max(1, self.canvas.winfo_width())

    def ch(self):
        return max(1, self.canvas.winfo_height())

    def mpp(self):
        """
        Metres per raster pixel.

        A GeoTIFF served in a geographic CRS has its resolution in DEGREES, so
        taking transform.a as metres would be wrong by ~1e5.  Convert using the
        scene-centre latitude in that case.
        """
        if not self.info:
            return 1.0
        rx = abs(self.info["res"][0])
        crs = self.info["crs"]
        try:
            geographic = crs is not None and crs.is_geographic
        except Exception:
            geographic = False
        if geographic:
            _, lat = self.px_to_map(self.info["width"] / 2.0,
                                    self.info["height"] / 2.0)
            return rx * 111320.0 * max(0.05, math.cos(math.radians(lat)))
        return rx

    def px_to_map(self, px, py):
        if not self.info:
            return None
        return self.info["transform"] * (px, py)

    def canvas_to_raster(self, x, y):
        return self.ox + x / self.ds, self.oy + y / self.ds

    def raster_to_canvas(self, px, py):
        return (px - self.ox) * self.ds, (py - self.oy) * self.ds

    # ---------------------------------------------------------------- load -- #
    def on_load_tif(self):
        p = filedialog.askopenfilename(
            title="Select GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff *.TIF *.TIFF"),
                       ("All files", "*.*")])
        if p:
            self.load_tif(p)

    def load_tif(self, path):
        try:
            self.log("Reading %s ..." % os.path.basename(path))
            self.info = build_preview(path, RUN.PREVIEW_MAX, RUN.STRETCH)
        except Exception as exc:
            messagebox.showerror("Could not open raster", str(exc))
            self.log(traceback.format_exc())
            return
        self.tif_path = path
        self.view_lims = list(self.info["lims"])
        i = self.info
        self.lbl_tif.config(
            text="%s\n%d x %d px, %d band(s), %s\nres %.3f x %.3f, %s"
                 % (os.path.basename(path), i["width"], i["height"], i["count"],
                    i["dtype"], i["res"][0], i["res"][1],
                    str(i["crs"]) if i["crs"] else "no CRS"))
        self.log("  %d x %d, display bands %s, preview decim %.2fx"
                 % (i["width"], i["height"], i["bands"], i["decim"]))
        if not i["crs"]:
            self.log("  WARNING: source has no CRS - output will inherit that.")
        self._base_key = None
        self.zoom_fit()

    def on_add_template(self):
        if not self.info:
            messagebox.showinfo("Load a raster first",
                                "Load the GeoTIFF before adding templates.")
            return
        p = filedialog.askopenfilename(
            title="Select template image",
            filetypes=[("Images", "*.png *.PNG *.jpg *.jpeg *.tif *.tiff"),
                       ("All files", "*.*")])
        if p:
            self.add_template(p)

    def add_template(self, path):
        TemplateEditor(self, path, self._template_ready)

    def _template_ready(self, ov):
        # drop it in the middle of whatever is currently on screen
        ov.cx, ov.cy = self.canvas_to_raster(self.cw() / 2.0, self.ch() / 2.0)
        target = min(self.info["width"], self.cw() / self.ds) / 6.0
        ov.scale = max(0.01, min(20.0, target / max(1, ov.src.width)))
        self.overlays.append(ov)
        self.sel = ov
        self.sync_list()
        self.push_controls()
        self.log("Added %s (%dx%d template px, %s)"
                 % (ov.name, ov.src.width, ov.src.height, ov.color_mode))
        self.redraw()

    # -------------------------------------------------------------- object -- #
    def sync_list(self):
        self.lst.delete(0, "end")
        for ov in self.overlays:
            self.lst.insert("end", ov.name)
        if self.sel in self.overlays:
            self.lst.selection_clear(0, "end")
            self.lst.selection_set(self.overlays.index(self.sel))

    def on_pick(self, ev=None):
        s = self.lst.curselection()
        if s:
            self.sel = self.overlays[s[0]]
            self.push_controls()
            self.redraw()

    def on_dup(self):
        if not self.sel:
            return
        s = self.sel
        ov = Overlay(s.base.copy(), path=s.path, cx=s.cx + 40 / self.ds,
                     cy=s.cy + 40 / self.ds, scale=s.scale, rot=s.rot,
                     opacity=s.opacity)
        ov.poly, ov.invert = list(s.poly), s.invert
        ov.feather, ov.bg, ov.bg_tol = s.feather, s.bg, s.bg_tol
        ov.set_color(s.color_mode, s.thresh, s.invert_lum)
        self.overlays.append(ov)
        self.sel = ov
        self.sync_list()
        self.push_controls()
        self.redraw()

    def on_edit_mask(self):
        if not self.sel:
            return
        old = self.sel

        def done(new_ov):
            new_ov.cx, new_ov.cy = old.cx, old.cy
            new_ov.rot, new_ov.opacity, new_ov.scale = old.rot, old.opacity, old.scale
            new_ov.name = old.name
            self.overlays[self.overlays.index(old)] = new_ov
            self.sel = new_ov
            self.sync_list()
            self.push_controls()
            self.redraw()

        TemplateEditor(self, old.path, done)

    def on_del(self):
        if self.sel in self.overlays:
            self.log("Removed %s" % self.sel.name)
            self.overlays.remove(self.sel)
            self.sel = self.overlays[-1] if self.overlays else None
            self.sync_list()
            self.push_controls()
            self.redraw()

    # ------------------------------------------------------------ controls -- #
    def push_controls(self):
        """Overlay state -> widgets."""
        self._suspend = True
        ov = self.sel
        if ov:
            self.v_rot.set(ov.rot)
            self.v_scale.set(min(20.0, max(0.01, ov.scale)))
            self.v_op.set(ov.opacity * 100.0)
            self.v_sharp.set(ov.sharpen)
            self.lbl_sharp.config(text="%d" % int(ov.sharpen))
            self.v_hard.set(ov.hard_edge)
            self.v_filt.set(ov.filt)
            self.v_cmode.set(ov.color_mode)
            self.v_thresh.set(ov.thresh)
            self.v_invlum.set(ov.invert_lum)
            self.lbl_thresh.config(text="%d" % ov.thresh)
            self._thresh_on = None          # force a re-evaluation
            self._sync_thresh_state()
            self._set(self.e_rot, "%.1f" % ov.rot)
            self._set(self.e_scale, "%.4f" % ov.scale)
            self._set(self.e_cx, "%.1f" % ov.cx)
            self._set(self.e_cy, "%.1f" % ov.cy)
            if self.info:
                mpp = self.mpp()
                self._set(self.e_m, "%.2f" % (ov.src.width * ov.scale * mpp))
                self.lbl_objdims.config(
                    text="object: %.2f x %.2f m  (%.0f x %.0f raster px)"
                         % (ov.src.width * ov.scale * mpp,
                            ov.src.height * ov.scale * mpp,
                            ov.src.width * ov.scale,
                            ov.src.height * ov.scale))
        self._suspend = False

    @staticmethod
    def _set(entry, text):
        entry.delete(0, "end")
        entry.insert(0, text)

    def sync_readouts(self):
        """
        Refresh the numeric entries and labels ONLY.

        Deliberately does not touch the Scale variables: writing a Scale's own
        variable from inside its command callback fights the drag and makes the
        thumb stick or snap back.
        """
        ov = self.sel
        if not ov:
            return
        self._suspend = True
        try:
            self._set(self.e_rot, "%.1f" % ov.rot)
            self._set(self.e_scale, "%.4f" % ov.scale)
            self._set(self.e_cx, "%.1f" % ov.cx)
            self._set(self.e_cy, "%.1f" % ov.cy)
            self.lbl_sharp.config(text="%d" % int(ov.sharpen))
            self.lbl_thresh.config(text="%d" % int(ov.thresh))
            if self.info:
                mpp = self.mpp()
                self._set(self.e_m,
                          "%.2f" % (ov.src.width * ov.scale * mpp))
                self.lbl_objdims.config(
                    text="object: %.2f x %.2f m  (%.0f x %.0f raster px)"
                         % (ov.src.width * ov.scale * mpp,
                            ov.src.height * ov.scale * mpp,
                            ov.src.width * ov.scale,
                            ov.src.height * ov.scale))
        finally:
            self._suspend = False

    def _sync_thresh_state(self):
        """Enable/disable the B/W slider only when the mode actually changes -
        reconfiguring a ttk widget mid-drag can drop its grab."""
        want = self.v_cmode.get() == "black & white"
        if want != getattr(self, "_thresh_on", None):
            self._thresh_on = want
            self.sc_thresh.state(["!disabled"] if want else ["disabled"])

    def on_color(self):
        if self._suspend or not self.sel:
            return
        self.sel.set_color(self.v_cmode.get(), int(self.v_thresh.get()),
                           self.v_invlum.get())
        self.lbl_thresh.config(text="%d" % int(self.v_thresh.get()))
        self._sync_thresh_state()
        self.request_redraw()

    def on_slider(self):
        if self._suspend or not self.sel:
            return
        self.sel.rot = float(self.v_rot.get())
        self.sel.scale = float(self.v_scale.get())
        self.sel.opacity = float(self.v_op.get()) / 100.0
        self.sel.sharpen = float(self.v_sharp.get())
        self.sync_readouts()
        self.request_redraw()

    def auto_factor(self):
        """Grid refinement that gives each template pixel its own output pixel."""
        if not self.overlays:
            return 1
        smin = min(o.scale for o in self.overlays if o.scale > 0)
        return int(max(1, min(64, math.ceil(1.0 / smin))))

    def factor(self):
        v = self.v_factor.get()
        if v.startswith("auto"):
            return self.auto_factor()
        return {"1x (native)": 1, "2x finer": 2, "4x finer": 4,
                "8x finer": 8}.get(v, 1)

    def _map_xy(self, p):
        return self.info["transform"] * (p[0], p[1])

    def seg_m(self, p0, p1):
        """Metres between two raster-pixel points, honest about the CRS."""
        x0, y0 = self._map_xy(p0)
        x1, y1 = self._map_xy(p1)
        crs = self.info["crs"]
        try:
            if crs is not None and crs.is_geographic:
                # haversine on lon/lat
                R = 6371008.8
                la0, la1 = math.radians(y0), math.radians(y1)
                dla = la1 - la0
                dlo = math.radians(x1 - x0)
                a = (math.sin(dla / 2) ** 2
                     + math.cos(la0) * math.cos(la1) * math.sin(dlo / 2) ** 2)
                return 2 * R * math.asin(min(1.0, math.sqrt(a)))
            fac = 1.0
            try:
                fac = float(crs.linear_units_factor[1])   # e.g. US ft -> m
            except Exception:
                pass
            return math.hypot(x1 - x0, y1 - y0) * fac
        except Exception:
            return math.hypot(x1 - x0, y1 - y0)

    def seg_bearing(self, p0, p1):
        """Azimuth of the segment, degrees clockwise from map north."""
        x0, y0 = self._map_xy(p0)
        x1, y1 = self._map_xy(p1)
        return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0

    def measure_total(self, pts):
        return sum(self.seg_m(pts[i], pts[i + 1])
                   for i in range(len(pts) - 1))

    def arm_measure(self):
        if not self.info:
            return
        self._measure = not self._measure
        if self._measure:
            self._aoi_mode = False
            self.canvas.config(cursor="tcross")
            self.btn_measure.config(text="measuring...")
            self.status.config(text="Click points to measure; right-click "
                                    "removes the last one; Esc or the button "
                                    "ends; Clear wipes the line.")
        else:
            self.canvas.config(cursor="")
            self.btn_measure.config(text="Measure (m)")
        self.request_redraw()

    def clear_measure(self):
        self.measure_pts = []
        self._cursor_r = None
        if self._measure:
            self.arm_measure()          # toggles off + redraws
        else:
            self.request_redraw()

    def arm_aoi(self):
        """Next left-drag on the image defines the crop rectangle."""
        if not self.info:
            return
        self._aoi_mode = True
        self._aoi_start = None
        self._measure = False
        self.btn_measure.config(text="Measure (m)")
        self.canvas.config(cursor="crosshair")
        self.btn_aoi.config(text="drag on image...")
        self.status.config(text="Drag on the image to set the crop area  "
                                "(Esc to cancel).")

    def cancel_aoi(self):
        self._aoi_mode = False
        self._aoi_start = None
        self.canvas.config(cursor="")
        self.btn_aoi.config(text="Draw AOI")
        self.redraw()

    def clear_aoi(self):
        self.aoi = None
        if self.v_extent.get().startswith("custom"):
            self.v_extent.set("full scene")
        self.cancel_aoi()

    def aoi_from_view(self):
        """Use exactly what is on screen as the crop area."""
        if not self.info:
            return
        W, H = self.info["width"], self.info["height"]
        x0 = int(max(0, math.floor(self.ox)))
        y0 = int(max(0, math.floor(self.oy)))
        x1 = int(min(W, math.ceil(self.ox + self.cw() / self.ds)))
        y1 = int(min(H, math.ceil(self.oy + self.ch() / self.ds)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        self.aoi = (x0, y0, x1, y1)
        self.v_extent.set("custom AOI (drag)")
        self.log("AOI from view: x %d..%d, y %d..%d (%d x %d px)"
                 % (x0, x1, y0, y1, x1 - x0, y1 - y0))
        self.redraw()

    def aoi_window(self):
        if not self.aoi or not self.info:
            return None
        x0, y0, x1, y1 = self.aoi
        x0 = int(max(0, min(x0, self.info["width"] - 1)))
        y0 = int(max(0, min(y0, self.info["height"] - 1)))
        x1 = int(max(x0 + 1, min(x1, self.info["width"])))
        y1 = int(max(y0 + 1, min(y1, self.info["height"])))
        return Window(x0, y0, x1 - x0, y1 - y0)

    def export_window(self):
        """Source-pixel window to export: whole scene, or a chip round the objects."""
        if not self.info:
            return None
        mode = self.v_extent.get()
        if mode.startswith("custom"):
            return self.aoi_window()
        if mode == "full scene" or not self.overlays:
            return None
        try:
            margin = float(self.e_margin.get()) / max(self.mpp(), 1e-9)
        except ValueError:
            margin = 0.0
        ls, ts, rs, bs = [], [], [], []
        for ov in self.overlays:
            l, t, r, b = ov.bbox(1.0)
            ls.append(l); ts.append(t); rs.append(r); bs.append(b)
        x0 = int(max(0, math.floor(min(ls) - margin)))
        y0 = int(max(0, math.floor(min(ts) - margin)))
        x1 = int(min(self.info["width"], math.ceil(max(rs) + margin)))
        y1 = int(min(self.info["height"], math.ceil(max(bs) + margin)))
        if x1 <= x0 or y1 <= y0:
            return None
        return Window(x0, y0, x1 - x0, y1 - y0)

    def update_cost(self):
        """Show the output size before the user commits to it."""
        if not self.info:
            return
        fac = self.factor()
        win = self.export_window()
        sw = int(win.width) if win is not None else self.info["width"]
        sh = int(win.height) if win is not None else self.info["height"]
        w, h = sw * fac, sh * fac
        gb = w * h * self.info["count"] * np.dtype(self.info["dtype"]).itemsize / 1e9
        gsd = self.mpp() / fac
        src = ("full scene" if win is None else
               ("AOI" if self.v_extent.get().startswith("custom") else "chip"))
        txt = ("%s: %d x %d px  @ %.4f m/px  ~%.2f GB raw"
               % (src, w, h, gsd, gb))
        if fac > 1:
            txt += "  (%dx upsample)" % fac
        warn = gb > 20
        if warn:
            txt += "   [!] very large - use a chip extent"
        self.lbl_cost.config(text=txt, foreground="#a33" if warn else "#666")

    def match_template_res(self):
        """
        Set the output grid fine enough that one template pixel maps to one
        output pixel, and chip the extent so the file stays tractable.
        """
        if not self.sel:
            return
        fac = self.auto_factor()
        self.v_factor.set("auto (template GSD)")
        if self.v_extent.get() == "full scene" and fac > 2:
            self.v_extent.set("chip around objects")
            self.log("Extent switched to chip: %dx over the full scene would "
                     "be %dx the pixel count." % (fac, fac * fac))
        self.log("Output grid -> %dx (%.4f m/px, template GSD). "
                 "Adjust the margin to trade context for file size."
                 % (fac, self.mpp() / fac))
        self.redraw()

    def on_render_opts(self):
        if self._suspend or not self.sel:
            return
        self.sel.hard_edge = self.v_hard.get()
        self.sel.filt = self.v_filt.get()
        self.redraw()

    def tone_mode(self):
        return {"match local background": "local",
                "match scene stretch": "stretch",
                "full dtype range": "dtype"}.get(self.v_tone.get(), "local")

    def on_tone_mode(self):
        mode = self.tone_mode()
        self.lbl_conname.config(
            text="sigma ratio %" if mode == "local" else "contrast %")
        self.invalidate_tone_maps()
        if mode == "local":
            self.log("Tones: local - object mapped to the mean/std of a ring "
                     "of background around it; slider = object-vs-background "
                     "sigma ratio (150 recommended).")
        self.request_redraw()

    def invalidate_tone_maps(self):
        for ov in self.overlays:
            ov._pv_tone = None

    def on_contrast(self):
        self.lbl_contrast.config(text="%d" % int(self.v_contrast.get()))
        self.invalidate_tone_maps()
        self.request_redraw()

    def contrast(self):
        return float(self.v_contrast.get()) / 100.0 \
            if self.tone_mode() in ("local", "stretch") else 1.0

    def preview_tone_map(self, ov):
        """
        Per-channel (A, B) so the preview shows the SAME tones the export will
        write: object 0-255 -> local-map DN -> display via the view stretch.
        Cached per object; while dragging the last map is reused and refreshed
        on release.
        """
        if self.tone_mode() != "local" or not self.info:
            return None
        key = (int(ov.cx // 4), int(ov.cy // 4), round(ov.scale, 4),
               round(ov.rot, 1), ov.color_mode, int(ov.thresh),
               ov.invert_lum, round(self.contrast(), 3),
               tuple(tuple(l) for l in self.view_lims))
        cached = getattr(ov, "_pv_tone", None)
        if cached and (cached[0] == key or self._drag is not None):
            return cached[1]
        maps = self._compute_preview_map(ov)
        ov._pv_tone = (key, maps)
        return maps

    def _compute_preview_map(self, ov):
        try:
            l, t, r, b = ov.bbox(1.0)
            x0, y0 = int(math.floor(l)), int(math.floor(t))
            x1, y1 = int(math.ceil(r)), int(math.ceil(b))
            pad = ring_pad_for(x1 - x0, y1 - y0)
            exclude = None
            if (x1 - x0) * (y1 - y0) <= 4e6:
                r1 = ov.render(view=1.0)
                exclude = (np.array(r1)[..., 3] >= 5,
                           int(math.floor(ov.cx - r1.width / 2.0)),
                           int(math.floor(ov.cy - r1.height / 2.0)))
            with rasterio.open(self.tif_path) as ds:
                st = ring_stats(ds, self.info["bands"], x0, y0, x1, y1, pad,
                                self.info["nodata"], decimate_to=384,
                                exclude=exclude)
            if st is None:
                return None
            a = np.array(ov.src)
            m = a[..., 3] >= 128
            if len(self.info["bands"]) == 1:
                lum = (0.299 * a[..., 0] + 0.587 * a[..., 1]
                       + 0.114 * a[..., 2]).astype("float32")
                ost = object_channel_stats([lum], m)
            else:
                ost = object_channel_stats(
                    [a[..., i].astype("float32") for i in range(3)], m)
            if ost is None:
                return None
            k = self.contrast()
            maps = []
            for i, (mu_b, sd_b, n) in enumerate(st):
                mu_o, sd_o = ost[min(i, len(ost) - 1)]
                gain = sd_b * k / sd_o
                off = mu_b - gain * mu_o
                lo, hi = self.view_lims[min(i, len(self.view_lims) - 1)]
                span = max(hi - lo, 1e-9)
                maps.append((gain * 255.0 / span, (off - lo) * 255.0 / span))
            return maps
        except Exception:
            self.log(traceback.format_exc().strip().splitlines()[-1])
            return None

    def native_size(self):
        """Set scale to 1.0 - one template pixel per raster pixel, the sharpest
        placement possible for this PNG."""
        if not self.sel:
            return
        self.sel.scale = 1.0
        self.push_controls()
        self.redraw()

    def on_entry(self, what):
        if not self.sel:
            return
        try:
            if what == "rot":
                self.sel.rot = float(self.e_rot.get())
            elif what == "scale":
                self.sel.scale = max(0.001, float(self.e_scale.get()))
            elif what == "centre":
                self.sel.cx = float(self.e_cx.get())
                self.sel.cy = float(self.e_cy.get())
            elif what == "metres":
                mpp = self.mpp()
                want = float(self.e_m.get())
                self.sel.scale = max(0.001,
                                     want / mpp / max(1, self.sel.src.width))
        except ValueError:
            return
        self.push_controls()
        self.redraw()

    def nudge(self, dx, dy):
        if not self.sel:
            return
        self.sel.cx += dx
        self.sel.cy += dy
        self.sync_readouts()
        self.request_redraw()

    # ----------------------------------------------------------- view/zoom -- #
    def fit_ds(self):
        if not self.info:
            return 1.0
        return min(self.cw() / float(self.info["width"]),
                   self.ch() / float(self.info["height"]))

    def clamp_origin(self):
        if not self.info:
            return
        vw, vh = self.cw() / self.ds, self.ch() / self.ds
        W, H = self.info["width"], self.info["height"]
        self.ox = (W - vw) / 2.0 if vw >= W else min(max(self.ox, 0.0), W - vw)
        self.oy = (H - vh) / 2.0 if vh >= H else min(max(self.oy, 0.0), H - vh)

    def set_zoom(self, new_ds, anchor=None):
        if not self.info:
            return
        lo = self.fit_ds() * 0.5
        new_ds = max(min(lo, 1.0), min(RUN.ZOOM_MAX, float(new_ds)))
        ax, ay = anchor if anchor else (self.cw() / 2.0, self.ch() / 2.0)
        rx, ry = self.canvas_to_raster(ax, ay)
        self.ds = new_ds
        self.ox = rx - ax / self.ds
        self.oy = ry - ay / self.ds
        self.clamp_origin()
        self.redraw()

    def zoom_by(self, factor, anchor=None):
        self.set_zoom(self.ds * factor, anchor)

    def zoom_fit(self):
        if not self.info:
            return
        self.ds = self.fit_ds()
        self.ox = (self.info["width"] - self.cw() / self.ds) / 2.0
        self.oy = (self.info["height"] - self.ch() / self.ds) / 2.0
        self.clamp_origin()
        self._fitted = True
        self.redraw()

    def zoom_to_object(self):
        """Centre on the selected object, framed tight to its alpha contour."""
        if not self.sel or not self.info:
            return
        l, t, r, b = self.sel.bbox(1.0)
        if (r - l) * (b - t) <= 4e6:
            img = self.sel.render(view=1.0)
            ab = img.getchannel("A").getbbox()
            if ab:
                bl = self.sel.cx - img.width / 2.0
                bt = self.sel.cy - img.height / 2.0
                l, t = bl + ab[0], bt + ab[1]
                r, b = bl + ab[2], bt + ab[3]
        w, h = max(1.0, r - l), max(1.0, b - t)
        target = min(self.cw() / (w * 1.25), self.ch() / (h * 1.25))
        self.ds = max(min(self.fit_ds(), 1.0), min(RUN.ZOOM_MAX, target))
        self.ox = (l + r) / 2.0 - self.cw() / (2.0 * self.ds)
        self.oy = (t + b) / 2.0 - self.ch() / (2.0 * self.ds)
        self.clamp_origin()
        self.redraw()

    def restretch(self):
        """Recompute display stretch from the pixels currently on screen."""
        if not self.info:
            return
        W, H = self.info["width"], self.info["height"]
        x0 = int(max(0, math.floor(self.ox)))
        y0 = int(max(0, math.floor(self.oy)))
        x1 = int(min(W, math.ceil(self.ox + self.cw() / self.ds)))
        y1 = int(min(H, math.ceil(self.oy + self.ch() / self.ds)))
        if x1 <= x0 or y1 <= y0:
            return
        win = Window(x0, y0, x1 - x0, y1 - y0)
        ow = min(512, x1 - x0)
        oh = min(512, y1 - y0)
        with rasterio.open(self.tif_path) as ds:
            data = ds.read(indexes=self.info["bands"], window=win,
                           out_shape=(len(self.info["bands"]), oh, ow),
                           resampling=Resampling.bilinear)
        self.view_lims = stretch_limits(data, RUN.STRETCH[0], RUN.STRETCH[1],
                                        self.info["nodata"])
        self.invalidate_tone_maps()
        self.log("Display stretch -> %s  (export still uses the full-scene "
                 "stretch)" % [tuple(round(v, 1) for v in l)
                               for l in self.view_lims])
        self._base_key = None
        self.redraw()

    def xview(self, *args):
        self._scroll(args, axis="x")

    def yview(self, *args):
        self._scroll(args, axis="y")

    def _scroll(self, args, axis):
        if not self.info:
            return
        span = (self.cw() if axis == "x" else self.ch()) / self.ds
        total = self.info["width"] if axis == "x" else self.info["height"]
        cur = self.ox if axis == "x" else self.oy
        if args[0] == "moveto":
            new = float(args[1]) * total
        elif args[0] == "scroll":
            n = float(args[1])
            new = cur + n * span * (0.1 if args[2] == "units" else 0.9)
        else:
            return
        if axis == "x":
            self.ox = new
        else:
            self.oy = new
        self.clamp_origin()
        self.redraw()

    def _sync_scrollbars(self):
        if not self.info:
            return
        for bar, off, span, total in (
                (self.hs, self.ox, self.cw() / self.ds, self.info["width"]),
                (self.vs, self.oy, self.ch() / self.ds, self.info["height"])):
            first = max(0.0, min(1.0, off / float(total)))
            last = max(0.0, min(1.0, (off + span) / float(total)))
            bar.set(first, max(last, first + 1e-4))

    def on_configure(self, ev=None):
        if not self.info:
            return
        if not self._fitted:
            self.zoom_fit()
        else:
            self.clamp_origin()
            self.redraw()

    # ------------------------------------------------------------- drawing -- #
    def render_base(self):
        """
        Build the visible slab of imagery at the current zoom.

        Zoomed out to preview resolution or coarser, the cached preview is
        resampled (fast).  Zoomed in past it, the pixels are read straight from
        the source file so the view is genuinely full resolution.
        """
        cw, ch = self.cw(), self.ch()
        key = (round(self.ox, 4), round(self.oy, 4), round(self.ds, 8), cw, ch,
               tuple(tuple(l) for l in self.view_lims))
        if key == self._base_key and self._base_img is not None:
            return self._base_img

        W, H = self.info["width"], self.info["height"]
        canvas_img = Image.new("RGB", (cw, ch), (30, 30, 30))

        rx0 = max(0.0, self.ox)
        ry0 = max(0.0, self.oy)
        rx1 = min(float(W), self.ox + cw / self.ds)
        ry1 = min(float(H), self.oy + ch / self.ds)
        if rx1 > rx0 and ry1 > ry0:
            ix0, iy0 = int(math.floor(rx0)), int(math.floor(ry0))
            ix1, iy1 = int(math.ceil(rx1)), int(math.ceil(ry1))
            out_w = max(1, int(round((ix1 - ix0) * self.ds)))
            out_h = max(1, int(round((iy1 - iy0) * self.ds)))
            decim = self.info["decim"]
            try:
                if self.ds <= (1.0 / decim) * 1.001:
                    # preview already has at least this much detail
                    pv = self.info["image"]
                    box = (ix0 / decim, iy0 / decim, ix1 / decim, iy1 / decim)
                    box = (max(0, box[0]), max(0, box[1]),
                           min(pv.width, box[2]), min(pv.height, box[3]))
                    slab = pv.resize((out_w, out_h), Image.BILINEAR, box=box)
                else:
                    slab = read_window_rgb(
                        self.tif_path, self.info["bands"], self.view_lims,
                        Window(ix0, iy0, ix1 - ix0, iy1 - iy0),
                        out_w, out_h, nearest=(self.ds >= 2.0))
            except Exception:
                self.log(traceback.format_exc().strip().splitlines()[-1])
                slab = Image.new("RGB", (out_w, out_h), (60, 20, 20))
            canvas_img.paste(slab,
                             (int(round((ix0 - self.ox) * self.ds)),
                              int(round((iy0 - self.oy) * self.ds))))

        self._base_img = canvas_img
        self._base_key = key
        return canvas_img

    def request_redraw(self):
        """Coalesce rapid updates (slider drags, object drags) into one redraw
        per idle cycle instead of one per motion event."""
        if self._redraw_pending:
            return
        self._redraw_pending = True
        self.after_idle(self._flush_redraw)

    def _flush_redraw(self):
        self._redraw_pending = False
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.info:
            return
        comp = self.render_base().convert("RGBA")
        con = self.contrast()
        local = self.tone_mode() == "local"
        for ov in self.overlays:
            maps = self.preview_tone_map(ov) if local else None
            img = ov.render(view=self.ds,
                            contrast=1.0 if maps is not None else
                            (1.0 if local else con))
            if maps is not None:
                img = apply_linear_rgb(img, maps)
            # place from the ACTUAL rendered size, exactly as the exporter does,
            # so preview and output agree to the pixel
            l = math.floor(ov.cx * self.ds - img.width / 2.0)
            t = math.floor(ov.cy * self.ds - img.height / 2.0)
            paste_rgba(comp, img, l - self.ox * self.ds, t - self.oy * self.ds)
        self._tkimg = ImageTk.PhotoImage(comp)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tkimg)
        if self.sel:
            ov = self.sel
            cx, cy = self.raster_to_canvas(ov.cx, ov.cy)
            # oriented rectangle following the object's own edges
            w2 = ov.src.width * ov.scale * self.ds / 2.0
            h2 = ov.src.height * ov.scale * self.ds / 2.0
            th = math.radians(ov.rot)
            co, si = math.cos(th), math.sin(th)

            def rp(dx_, dy_):
                # CCW-visual rotation in screen coords (y down)
                return (cx + dx_ * co + dy_ * si, cy - dx_ * si + dy_ * co)

            p0, p1 = rp(-w2, -h2), rp(w2, -h2)
            p2, p3 = rp(w2, h2), rp(-w2, h2)
            self.canvas.create_polygon(*(p0 + p1 + p2 + p3), fill="",
                                       outline="#00e0ff", dash=(4, 3), width=1)
            self.canvas.create_line(cx - 6, cy, cx + 6, cy, fill="#00e0ff")
            self.canvas.create_line(cx, cy - 6, cx, cy + 6, fill="#00e0ff")

            # live dimensions in metres, pinned to the object's edges
            mpp = self.mpp()
            Lm = ov.src.width * ov.scale * mpp    # along the template's width
            Wm = ov.src.height * ov.scale * mpp   # along its height
            for (a, b2), dim in (((p0, p1), Lm), ((p1, p2), Wm)):
                mx, my = (a[0] + b2[0]) / 2.0, (a[1] + b2[1]) / 2.0
                vx, vy = mx - cx, my - cy
                n = math.hypot(vx, vy) or 1.0
                self._mtext(mx + vx / n * 14, my + vy / n * 14,
                            "%.1f m" % dim, colour="#00e0ff",
                            anchor="center")
        if self.measure_pts:
            self._draw_measure()
        if self.aoi:
            x0, y0, x1, y1 = self.aoi
            cx0, cy0 = self.raster_to_canvas(x0, y0)
            cx1, cy1 = self.raster_to_canvas(x1, y1)
            active = self.v_extent.get().startswith("custom")
            col = "#ffd400" if active else "#8a7a2a"
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline=col,
                                         width=2, dash=(6, 4))
            self.canvas.create_text(
                cx0 + 4, cy0 - 9, anchor="w", fill=col,
                text="AOI %d x %d px  (%.0f x %.0f m)%s"
                     % (x1 - x0, y1 - y0, (x1 - x0) * self.mpp(),
                        (y1 - y0) * self.mpp(),
                        "" if active else "  - extent not set to AOI"))
        self._sync_scrollbars()
        if self._drag is None:          # skip the re-render mid-drag
            self.update_native_preview()
            self.update_cost()
        native = 100.0 * self.ds
        self.lbl_zoom.config(text="zoom %.1f%% of native%s"
                                  % (native, "  (full-res reads)"
                                     if self.ds > 1.0 / self.info["decim"] else ""))
        self.update_status()

    def update_native_preview(self):
        """
        Show the object at exactly 1 template-render pixel per RASTER pixel,
        magnified with nearest-neighbour.  The main canvas can be zoomed to any
        level, which makes a 20-pixel object look like a 120-pixel one; this
        panel always shows the truth.
        """
        c = self.pv_canvas
        c.delete("all")
        self._pv_tk = None
        if not self.sel:
            self.lbl_fp.config(text="-")
            return
        l, t, r, b = self.sel.bbox(1.0)
        if (r - l) * (b - t) > 4e6:
            self.lbl_fp.config(text="%d x %d raster px (too large to preview)"
                                    % (int(r - l), int(b - t)))
            return
        fac = self.factor()
        maps = self.preview_tone_map(self.sel) \
            if self.tone_mode() == "local" else None
        img = self.sel.render(view=float(fac),
                              contrast=1.0 if maps is not None
                              else self.contrast())
        if maps is not None:
            img = apply_linear_rgb(img, maps)
        w, h = img.size

        # composite onto the real imagery at the exact output-grid position,
        # so this panel shows the blend the export will produce
        disp = img
        if self.v_fp_bg.get() and self.tif_path:
            try:
                ol = int(math.floor(self.sel.cx * fac - w / 2.0))
                ot = int(math.floor(self.sel.cy * fac - h / 2.0))
                mpad = max(3, int(round(0.15 * max(w, h) / fac)))
                sx0 = max(0, int(math.floor(self.sel.cx - (w / fac) / 2.0))
                          - mpad)
                sy0 = max(0, int(math.floor(self.sel.cy - (h / fac) / 2.0))
                          - mpad)
                sx1 = min(self.info["width"],
                          int(math.ceil(self.sel.cx + (w / fac) / 2.0)) + mpad)
                sy1 = min(self.info["height"],
                          int(math.ceil(self.sel.cy + (h / fac) / 2.0)) + mpad)
                if sx1 > sx0 and sy1 > sy0:
                    ow2, oh2 = (sx1 - sx0) * fac, (sy1 - sy0) * fac
                    dfp = min(1.0, 1200.0 / max(ow2, oh2))
                    bg = read_window_rgb(
                        self.tif_path, self.info["bands"], self.view_lims,
                        Window(sx0, sy0, sx1 - sx0, sy1 - sy0),
                        max(1, int(round(ow2 * dfp))),
                        max(1, int(round(oh2 * dfp))),
                        nearest=(fac > 1)).convert("RGBA")
                    obj = img if dfp >= 0.999 else img.resize(
                        (max(1, int(round(w * dfp))),
                         max(1, int(round(h * dfp)))), Image.BILINEAR)
                    paste_rgba(bg, obj, (ol - sx0 * fac) * dfp,
                               (ot - sy0 * fac) * dfp)
                    disp = bg
            except Exception:
                self.log(traceback.format_exc().strip().splitlines()[-1])
                disp = img
        w2, h2 = disp.size
        bw, bh = 272, 110
        if w2 <= bw and h2 <= bh:
            k = max(1, int(min(bw / float(w2), bh / float(h2))))
            disp = disp.resize((w2 * k, h2 * k), Image.NEAREST) if k > 1 \
                else disp
            note = "  (shown %dx)" % k if k > 1 else ""
        else:
            k = min(bw / float(w2), bh / float(h2))
            disp = disp.resize((max(1, int(w2 * k)), max(1, int(h2 * k))),
                               Image.LANCZOS)
            note = "  (shown %.2fx)" % k
        back = Image.new("RGBA", disp.size, (64, 64, 64, 255))
        back.alpha_composite(disp.convert("RGBA"))
        self._pv_tk = ImageTk.PhotoImage(back)
        c.create_image(bw // 2, bh // 2, anchor="center", image=self._pv_tk)
        eff = self.sel.scale * fac              # template px -> output px
        txt = ("%d x %d output px  =  %.1f x %.1f m%s"
               % (w, h, w * self.mpp() / fac, h * self.mpp() / fac, note))
        if eff < 0.98:
            txt += ("\n[!] template downsampled %.1fx - %d of its %d px "
                    "survive per row" % (1.0 / eff, w,
                                         int(round(w / max(eff, 1e-6)))))
        elif eff >= 0.98:
            txt += "\nfull template resolution (1 template px = 1 output px)"
        self.lbl_fp.config(text=txt,
                           foreground="#a33" if eff < 0.98 else "#2a7")

    def _mtext(self, x, y, text, colour="#ff9f40", anchor="w"):
        self.canvas.create_text(x + 1, y + 1, anchor=anchor, fill="#000000",
                                text=text, font=("TkDefaultFont", 8))
        self.canvas.create_text(x, y, anchor=anchor, fill=colour,
                                text=text, font=("TkDefaultFont", 8))

    def _draw_measure(self):
        pts = [self.raster_to_canvas(*p) for p in self.measure_pts]
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            self.canvas.create_line(x0, y0, x1, y1, fill="#ff9f40", width=2)
            d = self.seg_m(self.measure_pts[i], self.measure_pts[i + 1])
            self._mtext((x0 + x1) / 2 + 5, (y0 + y1) / 2 - 5, "%.1f m" % d)
        for x, y in pts:
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                    fill="#ff9f40", outline="#000")
        if self._measure and self._cursor_r is not None:
            lx, ly = pts[-1]
            cx, cy = self.raster_to_canvas(*self._cursor_r)
            self.canvas.create_line(lx, ly, cx, cy, fill="#ff9f40",
                                    width=1, dash=(4, 3))
            d = self.seg_m(self.measure_pts[-1], self._cursor_r)
            self._mtext(cx + 8, cy - 8, "%.1f m" % d)
        if len(self.measure_pts) >= 2:
            ex, ey = pts[-1]
            self._mtext(ex + 8, ey + 10, "total %.1f m"
                        % self.measure_total(self.measure_pts))

    def update_status(self):
        if not self.info:
            return
        if not self.sel:
            self.status.config(text="%d object(s).  Wheel = zoom, right-drag = pan."
                                    % len(self.overlays))
            return
        ov = self.sel
        mpp = self.mpp()
        m = self.px_to_map(ov.cx, ov.cy)
        txt = "%s [%s]  |  centre px (%.1f, %.1f)" % (ov.name, ov.color_mode,
                                                      ov.cx, ov.cy)
        if m:
            txt += "  |  map (%.3f, %.3f)" % m
        txt += ("  |  %.1f x %.1f m  |  rot %.1f deg"
                % (ov.src.width * ov.scale * mpp,
                   ov.src.height * ov.scale * mpp, ov.rot))
        l, t, r, b = ov.bbox(1.0)
        fw, fh = int(round(r - l)), int(round(b - t))
        txt += "  |  footprint %d x %d raster px" % (fw, fh)
        if max(fw, fh) < 40:
            txt += "  [!] too few pixels to carry template detail"
        if ov.scale > 1.05:
            txt += ("   [!] enlarged %.1fx past native - max sharp width is "
                    "%.1f m" % (ov.scale, ov.src.width * mpp))
        self.status.config(text=txt)

    # ---------------------------------------------------------------- mouse -- #
    def on_hover(self, ev):
        if self._measure and self.info:
            self._cursor_r = self.canvas_to_raster(ev.x, ev.y)
            if self.measure_pts:
                seg = self.seg_m(self.measure_pts[-1], self._cursor_r)
                tot = self.measure_total(self.measure_pts) + seg
                brg = self.seg_bearing(self.measure_pts[-1], self._cursor_r)
                self.status.config(
                    text="segment %.1f m  |  bearing %.1f deg  |  total "
                         "%.1f m  |  %d point(s)"
                         % (seg, brg, tot, len(self.measure_pts)))
            self.request_redraw()
            return
        if not self.info or self.sel:
            return
        px, py = self.canvas_to_raster(ev.x, ev.y)
        if 0 <= px < self.info["width"] and 0 <= py < self.info["height"]:
            m = self.px_to_map(px, py)
            self.status.config(text="px (%.1f, %.1f)   map (%.3f, %.3f)"
                                    % (px, py, m[0], m[1]))

    def on_down(self, ev):
        self.canvas.focus_set()
        if not self.info:
            return
        px, py = self.canvas_to_raster(ev.x, ev.y)
        if self._measure:
            self.measure_pts.append((px, py))
            self.request_redraw()
            return
        if self._aoi_mode:
            self._aoi_start = (px, py)
            self.aoi = (int(px), int(py), int(px) + 1, int(py) + 1)
            return
        for ov in reversed(self.overlays):
            if ov.hit(px, py, 1.0):
                self.sel = ov
                self._drag = (px - ov.cx, py - ov.cy)
                self.sync_list()
                self.push_controls()
                self.redraw()
                return
        self._drag = None

    def on_move(self, ev):
        if self._aoi_mode and self._aoi_start:
            px, py = self.canvas_to_raster(ev.x, ev.y)
            sx, sy = self._aoi_start
            W, H = self.info["width"], self.info["height"]
            x0 = int(max(0, min(sx, px))); x1 = int(min(W, max(sx, px)))
            y0 = int(max(0, min(sy, py))); y1 = int(min(H, max(sy, py)))
            self.aoi = (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))
            self.redraw()
            return
        if not self._drag or not self.sel:
            return
        px, py = self.canvas_to_raster(ev.x, ev.y)
        self.sel.cx = px - self._drag[0]
        self.sel.cy = py - self._drag[1]
        self.sync_readouts()
        self.request_redraw()

    def on_up(self, ev):
        if self._drag is not None and self.sel is not None:
            self.sel._pv_tone = None        # re-sample the ring where it landed
        if self._aoi_mode:
            self._aoi_mode = False
            self._aoi_start = None
            self.canvas.config(cursor="")
            self.btn_aoi.config(text="Draw AOI")
            if self.aoi and (self.aoi[2] - self.aoi[0]) >= 4 \
                    and (self.aoi[3] - self.aoi[1]) >= 4:
                x0, y0, x1, y1 = self.aoi
                self.v_extent.set("custom AOI (drag)")
                self.log("AOI set: x %d..%d, y %d..%d  (%d x %d px, "
                         "%.1f x %.1f m)"
                         % (x0, x1, y0, y1, x1 - x0, y1 - y0,
                            (x1 - x0) * self.mpp(), (y1 - y0) * self.mpp()))
            else:
                self.aoi = None
                self.log("AOI too small - discarded.")
            self.redraw()
            return
        self._drag = None
        self.update_native_preview()

    def pan_start(self, ev):
        self.canvas.focus_set()
        self._pan = (ev.x, ev.y, self.ox, self.oy)
        self.canvas.config(cursor="fleur")

    def pan_move(self, ev):
        if not self._pan or not self.info:
            return
        x, y, ox, oy = self._pan
        self.ox = ox - (ev.x - x) / self.ds
        self.oy = oy - (ev.y - y) / self.ds
        self.clamp_origin()
        self.redraw()

    def pan_end(self, ev):
        moved = self._pan and (abs(ev.x - self._pan[0])
                               + abs(ev.y - self._pan[1]) > 4)
        self._pan = None
        self.canvas.config(cursor="tcross" if self._measure else "")
        if self._measure and not moved and self.measure_pts:
            self.measure_pts.pop()
            self.request_redraw()

    def on_wheel(self, ev, delta=None):
        d = delta if delta is not None else ev.delta
        ctrl = bool(ev.state & 0x0004)
        shift = bool(ev.state & 0x0001)
        if ctrl and self.sel:                       # scale the object
            self.sel.scale = max(0.001, self.sel.scale * (1.06 if d > 0 else 1 / 1.06))
        elif shift and self.sel:                    # rotate the object
            self.sel.rot = (self.sel.rot + (5 if d > 0 else -5) + 180) % 360 - 180
        else:                                       # zoom the view
            self.zoom_by(RUN.ZOOM_STEP if d > 0 else 1 / RUN.ZOOM_STEP,
                         anchor=(ev.x, ev.y))
            return
        self.push_controls()
        self.redraw()

    # --------------------------------------------------------------- export -- #
    def on_export(self):
        if not self.info:
            return
        if not self.overlays:
            messagebox.showinfo("Nothing to do", "Add at least one object first.")
            return
        stem, _ = os.path.splitext(self.tif_path)
        out = filedialog.asksaveasfilename(
            title="Save composited GeoTIFF",
            initialfile=os.path.basename(stem) + RUN.OUT_SUFFIX + ".tif",
            defaultextension=".tif", filetypes=[("GeoTIFF", "*.tif")])
        if not out:
            return
        try:
            self.config(cursor="watch")
            self.update()
            export_geotiff(self.tif_path, out, self.overlays,
                           self.info["bands"], self.info["lims"],
                           contrast=self.contrast(), factor=self.factor(),
                           window=self.export_window(),
                           tone_mode=self.tone_mode(), block=RUN.BLOCK,
                           log=self.log)
            side = os.path.splitext(out)[0] + ".json"
            self._write_json(side)
            self.log("Sidecar: %s" % os.path.basename(side))
            messagebox.showinfo("Export complete", "Written:\n%s" % out)
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Export failed", str(exc))
        finally:
            self.config(cursor="")

    def _write_json(self, path):
        d = dict(source=self.tif_path, bands=self.info["bands"],
                 objects=[o.to_dict() for o in self.overlays])
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)

    def on_close(self):
        """Quit the event loop before destroying, so Tcl tears down in order."""
        try:
            self.quit()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def on_save_json(self):
        if not self.overlays:
            return
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         filetypes=[("JSON", "*.json")])
        if p:
            self._write_json(p)
            self.log("Saved placements -> %s" % p)

    def on_load_json(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            with open(p) as fh:
                d = json.load(fh)
            if not self.info and d.get("source") and os.path.isfile(d["source"]):
                self.load_tif(d["source"])
            for od in d.get("objects", []):
                self.overlays.append(rebuild_from_dict(od))
            self.sel = self.overlays[-1] if self.overlays else None
            self.sync_list()
            self.push_controls()
            self.redraw()
            self.log("Loaded %d placement(s)" % len(d.get("objects", [])))
        except Exception as exc:
            self.log(traceback.format_exc())
            messagebox.showerror("Load failed", str(exc))


# =========================================================================== #
def _under_ipython():
    """True inside Spyder / Jupyter / any IPython kernel."""
    if "IPython" not in sys.modules:
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        return False
    if ip is None:
        return False
    return type(ip).__name__ != "TerminalInteractiveShell"


def _script_path():
    p = globals().get("__file__")
    return os.path.abspath(p) if p else None


def _relaunch_detached():
    """
    Start the GUI in a separate interpreter.

    Tk installs async handlers on the thread that creates the root window.  An
    IPython kernel services comms on other threads, so when the kernel later
    garbage-collects the Tk object the teardown happens on the wrong thread and
    Tcl aborts the whole process - which is the Tcl_AsyncDelete crash and kernel
    restart.  A child process sidesteps it entirely.
    """
    script = _script_path()
    if not script:
        print("Cannot detach: __file__ is not defined (pasted into the "
              "console?).  Running in-kernel instead.")
        return False
    here = os.path.dirname(script) or "."
    log_path = os.path.join(here, "gui_launch.log")

    try:
        fh = open(log_path, "w")
        kw = {"cwd": here, "stdout": fh, "stderr": subprocess.STDOUT}
        if os.name == "nt" and RUN.DETACH_NO_CONSOLE:
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW",
                                          0x08000000)
        elif os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        proc = subprocess.Popen([sys.executable, script, "--gui"], **kw)
    except Exception:
        traceback.print_exc()
        return False

    # a child that dies on startup would otherwise vanish silently
    time.sleep(max(0.0, float(RUN.DETACH_WAIT)))
    if proc.poll() is not None:
        try:
            fh.close()
            with open(log_path) as f:
                tail = f.read()[-4000:]
        except Exception:
            tail = "(could not read %s)" % log_path
        print("The detached GUI process exited immediately (code %s)."
              % proc.returncode)
        print("--- %s ---" % log_path)
        print(tail.strip() or "(no output)")
        print("--- end ---")
        print("Falling back to running in this kernel.")
        return False

    print("GUI launched in a separate process (PID %d) - the window may take "
          "a moment to appear." % proc.pid)
    print("Child output -> %s" % log_path)
    print("Set RUN.DETACH_FROM_SPYDER = False to run it inside the kernel "
          "instead.")
    return True


def main(force_inline=False):
    if not _HAVE_TK:
        print("tkinter is not available in this Python environment.")
        return
    detach = (RUN.DETACH_FROM_SPYDER and not force_inline
              and "--gui" not in sys.argv and _under_ipython())
    if detach and _relaunch_detached():
        return
    if _under_ipython():
        print("NOTE: running Tk inside the kernel. If it crashes with "
              "Tcl_AsyncDelete, run this file in an external terminal or set "
              "the IPython graphics backend to 'Tkinter'.")
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        app.mainloop()
    finally:
        try:
            app.destroy()
        except Exception:
            pass
        del app
        gc.collect()


if __name__ == "__main__":
    main()
