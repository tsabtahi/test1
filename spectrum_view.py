#!/usr/bin/env python3
"""
spectrum_view.py — look at a SICD's frequency response (k-space), and compare
the spectrum of labelled targets against background clutter.

    pip install "sarpy==2.1.1" numpy matplotlib

    python spectrum_view.py scene_SICD.nitf --chip-size 512
    python spectrum_view.py scene_SICD.nitf --chip-size 512 --boxes boxes.json
    python spectrum_view.py scene_SICD.nitf --looks 3 --boxes boxes.json   # sub-aperture too

Why look at k-space at all
    The image and its 2-D spectrum are a Fourier pair. The spectrum shows the
    aperture the sensor actually collected: a rectangle (PFA keeps a rectangle)
    whose width is the bandwidth and whose height is the azimuth aperture,
    shaped by the taper the processor applied. Everything the phase-domain work
    depends on lives here:
      * range axis   = frequency  -> how a scatterer responds across the band
      * azimuth axis = look angle -> how it responds across aspect
    A man-made target (corner, edge, plate) is bright and stable across the
    whole rectangle. Distributed clutter decorrelates across it. That contrast
    is the thing a phase-side detector would exploit, and this tool lets you
    see whether it exists in your data before building anything.

boxes.json  (pixel coordinates in the FULL SICD image)
    [{"name": "ship1", "box": [row0, col0, row1, col1]}, ...]
    or simply  [[row0, col0, row1, col1], ...]
"""
from __future__ import annotations

# =========================================================== CONFIG (edit me)
CONFIG = {
    "input": None,              # SICD path; or pass on the command line
    "boxes": None,              # path to boxes.json, or None
    "chip_size": 512,           # square chip centred on the scene centre pixel
    "chip": None,               # or exactly (row0, col0, n_rows, n_cols)
    "looks": 0,                 # >1 = also show sub-aperture images + coherence
    "out": "spectrum.png",
}
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np

C = 299_792_458.0


def load_sicd_chip(path, chip, chip_size):
    from sarpy.io.complex.converter import open_complex
    rd = open_complex(str(path))
    s = rd.get_sicds_as_tuple()[0]
    n_rows, n_cols = rd.get_data_size_as_tuple()[0]
    if chip:
        r0, c0, nr, nc = chip
    else:
        nr, nc = min(chip_size, n_rows), min(chip_size, n_cols)
        r0 = max(0, min(int(s.ImageData.SCPPixel.Row) - nr // 2, n_rows - nr))
        c0 = max(0, min(int(s.ImageData.SCPPixel.Col) - nc // 2, n_cols - nc))
    img = np.asarray(rd[r0:r0 + nr, c0:c0 + nc], np.complex64)
    meta = {
        "row_ss": float(s.Grid.Row.SS), "col_ss": float(s.Grid.Col.SS),
        "row_irbw": float(s.Grid.Row.ImpRespBW), "col_irbw": float(s.Grid.Col.ImpRespBW),
        "row_kctr": float(s.Grid.Row.KCtr), "col_kctr": float(s.Grid.Col.KCtr),
        "row_wgt": None if s.Grid.Row.WgtFunct is None else np.asarray(s.Grid.Row.WgtFunct, float),
        "col_wgt": None if s.Grid.Col.WgtFunct is None else np.asarray(s.Grid.Col.WgtFunct, float),
        "row_wgt_type": str(getattr(s.Grid.Row.WgtType, "WindowName", None)),
        "col_wgt_type": str(getattr(s.Grid.Col.WgtType, "WindowName", None)),
        "chip": (r0, c0, nr, nc), "full": (n_rows, n_cols),
    }
    return img, meta


def kspace(img):
    """2-D spectrum of the chip, zero frequency centred. Axis units are offsets
    from KCtr in cycles/m once scaled by 1/(N*SS)."""
    return np.fft.fftshift(np.fft.fft2(img))


def db(x, floor=-45):
    v = 20 * np.log10(np.abs(x) + 1e-12)
    return np.clip(v - np.nanmax(v), floor, 0)


def load_boxes(path, chip):
    """Boxes in full-image pixels -> chip-relative, keeping only those inside."""
    if not path:
        return []
    raw = json.loads(Path(path).read_text())
    r0, c0, nr, nc = chip
    out = []
    for i, b in enumerate(raw):
        name, box = (b.get("name", f"box{i}"), b["box"]) if isinstance(b, dict) else (f"box{i}", b)
        br0, bc0, br1, bc1 = [int(v) for v in box]
        br0, br1 = min(br0, br1) - r0, max(br0, br1) - r0
        bc0, bc1 = min(bc0, bc1) - c0, max(bc0, bc1) - c0
        if br1 <= 0 or bc1 <= 0 or br0 >= nr or bc0 >= nc:
            continue
        out.append((name, max(0, br0), max(0, bc0), min(nr, br1), min(nc, bc1)))
    return out


def sub_aperture(img, n_looks, axis=1):
    """Split the azimuth spectrum into n looks. Each look keeps the FULL image
    size (the other sub-bands are zeroed, not cropped) so pixel coordinates —
    and therefore box coordinates — stay valid; the looks are simply lower
    resolution in azimuth."""
    S = np.fft.fftshift(np.fft.fft(img, axis=axis), axes=axis)
    w = S.shape[axis] // n_looks
    looks = []
    for i in range(n_looks):
        band = np.zeros_like(S)
        sl = [slice(None)] * 2
        sl[axis] = slice(i * w, (i + 1) * w)
        band[tuple(sl)] = S[tuple(sl)]
        looks.append(np.fft.ifft(np.fft.ifftshift(band, axes=axis), axis=axis))
    return looks


def coherence(a, b, win=7):
    from scipy.ndimage import uniform_filter
    num_r = uniform_filter((a * np.conj(b)).real, win)
    num_i = uniform_filter((a * np.conj(b)).imag, win)
    den = np.sqrt(uniform_filter(np.abs(a) ** 2, win) * uniform_filter(np.abs(b) ** 2, win))
    return np.abs(num_r + 1j * num_i) / (den + 1e-12)


def main():
    cf = CONFIG
    ap = argparse.ArgumentParser(description="SICD frequency-response (k-space) viewer")
    ap.add_argument("input", nargs="?", default=cf["input"])
    ap.add_argument("--boxes", default=cf["boxes"], help="boxes.json in full-image pixel coords")
    ap.add_argument("--chip-size", type=int, default=cf["chip_size"])
    ap.add_argument("--chip", type=int, nargs=4, default=cf["chip"],
                    metavar=("ROW0", "COL0", "NROWS", "NCOLS"))
    ap.add_argument("--looks", type=int, default=cf["looks"],
                    help="sub-aperture looks (>=2 adds a sub-aperture coherence panel)")
    ap.add_argument("-o", "--out", type=Path, default=Path(cf["out"]))
    a = ap.parse_args()
    if not a.input:
        ap.print_help()
        print("\nNo input. Set CONFIG['input'] or pass a SICD path.")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img, m = load_sicd_chip(a.input, a.chip, a.chip_size)
    nr, nc = img.shape
    boxes = load_boxes(a.boxes, m["chip"])
    S = kspace(img)

    # spatial-frequency offset axes, cycles/m from KCtr
    dkr = (np.arange(nr) - nr // 2) / (nr * m["row_ss"])
    dkc = (np.arange(nc) - nc // 2) / (nc * m["col_ss"])
    print(f"chip {m['chip']} of {m['full']}")
    print(f"  range:   SS {m['row_ss']:.3f} m, IRBW {m['row_irbw']:.2f} cyc/m "
          f"-> resolution {0.886 / m['row_irbw']:.2f} m, taper {m['row_wgt_type']}")
    print(f"  azimuth: SS {m['col_ss']:.3f} m, IRBW {m['col_irbw']:.2f} cyc/m "
          f"-> resolution {0.886 / m['col_irbw']:.2f} m, taper {m['col_wgt_type']}")
    print(f"  Nyquist span {dkr[0]:.2f}..{dkr[-1]:.2f} (range), {dkc[0]:.2f}..{dkc[-1]:.2f} (azimuth) cyc/m")
    print(f"  support fills {m['row_irbw'] / (dkr[-1] - dkr[0]) * 100:.0f}% of range, "
          f"{m['col_irbw'] / (dkc[-1] - dkc[0]) * 100:.0f}% of azimuth Nyquist (the rest is oversampling)")
    if boxes:
        print(f"  {len(boxes)} box(es) inside the chip")

    n_panels = 4 + (1 if boxes else 0) + (1 if a.looks >= 2 else 0)
    ncols = 3
    nrows_fig = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows_fig, ncols, figsize=(6.2 * ncols, 5.2 * nrows_fig))
    axes = np.atleast_1d(axes).ravel()
    k = 0

    # 1. image + boxes
    axes[k].imshow(db(img), cmap="gray", aspect="auto")
    axes[k].set_title("SICD chip |I| (dB)")
    for name, r0, c0, r1, c1 in boxes:
        axes[k].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="lime", lw=1.5))
        axes[k].text(c0, r0 - 4, name, color="lime", fontsize=8)
    k += 1

    # 2. 2-D k-space with the declared support rectangle
    axes[k].imshow(db(S, -60), cmap="magma", aspect="auto",
                   extent=[dkc[0], dkc[-1], dkr[-1], dkr[0]])
    axes[k].add_patch(Rectangle((-m["col_irbw"] / 2, -m["row_irbw"] / 2), m["col_irbw"], m["row_irbw"],
                                fill=False, ec="cyan", lw=1.2, ls="--"))
    axes[k].set_title("2-D spectrum (k-space), dashed = declared support")
    axes[k].set_xlabel("azimuth spatial freq offset (cyc/m)")
    axes[k].set_ylabel("range spatial freq offset (cyc/m)")
    k += 1

    # 3. range cut: the transmit band and its taper
    prof_r = np.abs(S).mean(axis=1)
    axes[k].plot(dkr, 20 * np.log10(prof_r / prof_r.max() + 1e-12), lw=1)
    for x in (-m["row_irbw"] / 2, m["row_irbw"] / 2):
        axes[k].axvline(x, color="cyan", ls="--", lw=1)
    axes[k].set_title(f"range frequency response  (taper: {m['row_wgt_type']})")
    axes[k].set_xlabel("range spatial freq offset (cyc/m)")
    axes[k].set_ylabel("dB")
    axes[k].set_ylim(-40, 2)
    k += 1

    # 4. azimuth cut: the aperture and its taper
    prof_c = np.abs(S).mean(axis=0)
    axes[k].plot(dkc, 20 * np.log10(prof_c / prof_c.max() + 1e-12), lw=1)
    for x in (-m["col_irbw"] / 2, m["col_irbw"] / 2):
        axes[k].axvline(x, color="cyan", ls="--", lw=1)
    axes[k].set_title(f"azimuth (aspect) frequency response  (taper: {m['col_wgt_type']})")
    axes[k].set_xlabel("azimuth spatial freq offset (cyc/m)")
    axes[k].set_ylabel("dB")
    axes[k].set_ylim(-40, 2)
    k += 1

    # 5. per-box spectrum vs clutter — the phase-side detection question
    if boxes:
        clutter = np.ones((nr, nc), bool)
        for _, r0, c0, r1, c1 in boxes:
            clutter[r0:r1, c0:c1] = False
        for name, r0, c0, r1, c1 in boxes[:4]:
            patch = np.zeros_like(img)
            patch[r0:r1, c0:c1] = img[r0:r1, c0:c1]
            Sb = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
            p = Sb.mean(axis=0)
            axes[k].plot(dkc, 20 * np.log10(p / p.max() + 1e-12), lw=1, label=name)
        bg = img * clutter
        Sc = np.abs(np.fft.fftshift(np.fft.fft2(bg)))
        p = Sc.mean(axis=0)
        axes[k].plot(dkc, 20 * np.log10(p / p.max() + 1e-12), "k--", lw=1, label="clutter")
        axes[k].set_title("aspect response: targets vs clutter\n(flat = stable across look angle)")
        axes[k].set_xlabel("azimuth spatial freq offset (cyc/m)")
        axes[k].set_ylabel("dB")
        axes[k].set_ylim(-25, 2)
        axes[k].legend(fontsize=8)
        k += 1

    # 6. sub-aperture coherence
    if a.looks >= 2:
        looks = sub_aperture(img, a.looks)
        gam = np.mean([coherence(looks[i], looks[i + 1]) for i in range(len(looks) - 1)], axis=0)
        im = axes[k].imshow(gam, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axes[k].set_title(f"sub-aperture coherence, {a.looks} looks\n"
                          f"median {np.median(gam):.2f} (high = stable scatterer)")
        for name, r0, c0, r1, c1 in boxes:
            axes[k].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="red", lw=1.5))
        plt.colorbar(im, ax=axes[k], fraction=0.046)
        if boxes:
            inb = np.concatenate([gam[r0:r1, c0:c1].ravel() for _, r0, c0, r1, c1 in boxes])
            mask = np.ones_like(gam, bool)
            for _, r0, c0, r1, c1 in boxes:
                mask[r0:r1, c0:c1] = False
            print(f"  sub-aperture coherence: median {np.median(inb):.3f} inside boxes vs "
                  f"{np.median(gam[mask]):.3f} in clutter")
        k += 1

    for ax in axes[k:]:
        ax.axis("off")
    fig.suptitle(f"{Path(a.input).name}  chip {m['chip']}")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
