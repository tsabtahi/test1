#!/usr/bin/env python3
#pip install numpy sarpy==2.1.1 matplotlib
"""
inverse_omega_k.py — generate SAR raw (unfocused) data from a focused complex
image, by inverting the omega-K (range migration) algorithm.

Reference: A. S. Khwaja, L. Ferro-Famil, E. Pottier, "SAR raw data generation
using inverse SAR image formation algorithms", IGARSS/EURAD 2006; and
"Efficient SAR raw data generation ... based on inverse processing",
IEEE GRSL 6(4):757-761, 2009.

    FORWARD omega-K  (raw data -> image)
        F1  2-D FFT
        F2  reference function multiply        exp(+j theta_ref)
        F3  Stolt interpolation                f_tau  ->  f_tau'
        F4  2-D IFFT

    INVERSE omega-K  (image -> raw data)   = the same blocks, reversed
        I1  2-D FFT
        I2  inverse Stolt interpolation        f_tau' ->  f_tau
        I3  reference function multiply, conjugated   exp(-j theta_ref)
        I4  2-D IFFT

    theta_ref(f_tau, f_eta) = (4 pi R_ref / c) * D  +  pi f_tau^2 / Kr
    D = sqrt( (f0 + f_tau)^2 - (c f_eta / (2 V))^2 )      (range migration term)
    Stolt map:  f0 + f_tau' = D                           (decouples f_tau, f_eta)

Usage
    edit the CONFIG block below, then:   python inverse_omega_k.py
    or pass everything on the command line:
    python inverse_omega_k.py scene_SICD.nitf --chip-size 512 -o raw.npz
    python inverse_omega_k.py image.npy --f0 9.65e9 --fs 1.2e9 --bandwidth 600e6 \
        --velocity 7600 --r-ref 700e3 --azimuth-spacing 0.2 -o raw.npz
    ... add --roundtrip to refocus the raw data and score it against the input
    ... add --quicklook for a PNG

Model assumptions (omega-K is a stripmap, broadside, straight-track algorithm)
    * one effective velocity V, one reference range R_ref, no squint
    * linear-FM transmit pulse of rate Kr = bandwidth / pulse_length
    * a spotlight SICD formed by PFA does not satisfy these; the output is then
      a plausible stripmap-model raw set, not that sensor's true echo data
"""
from __future__ import annotations

# =========================================================== CONFIG (edit me)
# Used when the script is run with no command-line arguments.
# Anything you pass on the command line overrides the matching entry here.
# None means "work it out from the SICD metadata" (or, for .npy input, error).
CONFIG = {
    # SICD (.nitf) or complex image (.npy). Windows: r"C:\data\scene_SICD.nitf"
    "input": None,
    "out": "raw.npz",

    # which part of the image to convert
    "chip_size": 512,            # square chip centred on the scene centre pixel
    "chip": None,                # or exactly: (row0, col0, n_rows, n_cols)

    # radar parameters — None = read from the SICD, a number = use that value.
    # All are REQUIRED for .npy input, since a bare array carries no metadata.
    "f0": None,                  # carrier frequency, Hz          e.g. 9.65e9
    "fs": None,                  # range sample rate, Hz          e.g. 1.2e9
    "bandwidth": None,           # chirp bandwidth, Hz            e.g. 600e6
    "velocity": None,            # effective velocity, m/s        e.g. 100.0
    "r_ref": None,               # reference slant range, m       e.g. 1000.0
    "azimuth_spacing": None,     # along-track sample spacing, m  e.g. 0.2
    "pulse_length": 50e-6,       # sets the chirp rate Kr = bandwidth / pulse_length

    # options
    "absolute_delay": False,     # True = range window starts at zero delay (echo wraps)
    "roundtrip": True,           # refocus with the forward blocks and score
    "quicklook": True,           # write a PNG next to the output
}
# =============================================================================

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
C = 299_792_458.0


# ----------------------------------------------------------------- geometry
class Params:
    """Everything the blocks need. Axis 0 = range, axis 1 = azimuth."""

    def __init__(self, n_rg, n_az, f0, fs, bandwidth, pulse_length, velocity,
                 r_ref, azimuth_spacing, window_at_ref=True):
        self.n_rg, self.n_az = int(n_rg), int(n_az)
        self.f0 = float(f0)                      # carrier, Hz
        self.fs = float(fs)                      # range sample rate, Hz
        self.bandwidth = float(bandwidth)
        self.Kr = float(bandwidth) / float(pulse_length)     # chirp rate, Hz/s
        self.V = float(velocity)                 # effective velocity, m/s
        self.r_ref = float(r_ref)                # reference slant range, m
        self.d_az = float(azimuth_spacing)       # along-track sample spacing, m
        self.prf = self.V / self.d_az
        self.window_at_ref = bool(window_at_ref)   # raw range time measured from 2 R_ref / c

    @property
    def f_tau(self):
        """Baseband range frequency axis (Hz), FFT order."""
        return np.fft.fftfreq(self.n_rg, d=1.0 / self.fs)

    @property
    def f_eta(self):
        """Azimuth (Doppler) frequency axis (Hz), FFT order."""
        return np.fft.fftfreq(self.n_az, d=self.d_az / self.V)

    def migration_factor(self, f_tau):
        """D = sqrt((f0+f_tau)^2 - (c f_eta / 2V)^2), shape (n_rg, n_az).
        Values where the square root goes negative are outside the Doppler
        support of that frequency and are flagged invalid."""
        a = (self.f0 + np.asarray(f_tau).reshape(-1, 1)) ** 2
        b = (C * self.f_eta.reshape(1, -1) / (2.0 * self.V)) ** 2
        inside = a > b
        return np.sqrt(np.where(inside, a - b, 0.0)), inside

    def summary(self):
        return (f"f0={self.f0 / 1e9:.3f} GHz  B={self.bandwidth / 1e6:.0f} MHz  "
                f"Kr={self.Kr / 1e12:.3f} THz/s  fs={self.fs / 1e6:.0f} MHz\n"
                f"  V={self.V:.1f} m/s  R_ref={self.r_ref / 1e3:.2f} km  "
                f"PRF={self.prf:.1f} Hz  grid {self.n_rg} rg x {self.n_az} az")


def _interp_complex(x_new, x_grid, y, taps=8):
    """The Stolt resampler: windowed-sinc interpolation of a complex sequence
    onto arbitrary points. Linear interpolation also "works" but loses about
    0.2% correlation per pass; sinc costs a few lines and is what the
    published algorithm assumes. Out-of-range samples become 0."""
    dx = x_grid[1] - x_grid[0]
    pos = (x_new - x_grid[0]) / dx
    valid = np.isfinite(pos) & (pos >= 0) & (pos <= len(x_grid) - 1)
    pos = np.where(valid, pos, 0.0)
    base = np.floor(pos).astype(int)
    out = np.zeros(x_new.shape, np.complex128)
    for k in range(-taps // 2 + 1, taps // 2 + 1):
        idx = base + k
        ok = valid & (idx >= 0) & (idx < len(x_grid))
        d = pos - idx
        w = np.sinc(d) * (0.54 + 0.46 * np.cos(np.pi * np.clip(d / (taps / 2), -1, 1)))
        out[ok] += y[np.clip(idx, 0, len(x_grid) - 1)][ok] * w[ok]
    return out


# ----------------------------------------------------------- INVERSE blocks
def I1_2d_fft(image):
    """Block I1 — 2-D FFT: focused image -> 2-D frequency domain."""
    return np.fft.fft2(image)


def I2_inverse_stolt_interpolation(S_stolt, p: Params):
    """Block I2 — inverse Stolt interpolation.

    The forward algorithm resampled f_tau -> f_tau' with f0 + f_tau' = D.
    Here we go back: for each point of the uniform *raw* grid f_tau, evaluate
    the Stolt-domain spectrum at f_tau' = D(f_tau, f_eta) - f0. This is the
    block that re-introduces range cell migration."""
    f_tau = np.fft.fftshift(p.f_tau)                  # monotonic for interp
    D, inside = p.migration_factor(f_tau)
    f_tau_prime = D - p.f0                            # (n_rg, n_az)
    S_sorted = np.fft.fftshift(S_stolt, axes=0)
    out = np.zeros_like(S_sorted)
    for j in range(p.n_az):                           # one azimuth line at a time
        col = np.where(inside[:, j], f_tau_prime[:, j], np.inf)
        out[:, j] = _interp_complex(col, f_tau, S_sorted[:, j])
    return np.fft.ifftshift(out, axes=0)


def I3_reference_function_multiply_conjugate(S, p: Params):
    """Block I3 — reference function multiply, conjugated.

    Undoes the bulk compression the forward algorithm applied at R_ref:
    multiply by exp(-j theta_ref). This puts back the range chirp and the
    bulk range migration.

    The extra exp(+j 2 pi f_tau tau_ref) term starts the raw range window at
    the reference range instead of at zero delay. Without it the echo sits at
    the absolute delay 2 R_ref / c and wraps around the (much shorter) range
    window, which is correct but unusable."""
    D, inside = p.migration_factor(p.f_tau)
    theta = (4.0 * np.pi * p.r_ref / C) * D + np.pi * (p.f_tau.reshape(-1, 1) ** 2) / p.Kr
    S = S * np.exp(-1j * theta) * inside
    if p.window_at_ref:
        S = S * np.exp(1j * 2 * np.pi * p.f_tau.reshape(-1, 1) * (2.0 * p.r_ref / C))
    return S


def I4_2d_ifft(S):
    """Block I4 — 2-D IFFT: back to (range time, azimuth time) = raw data."""
    return np.fft.ifft2(S)


def inverse_omega_k(image, p: Params, verbose=True):
    say = (lambda m: print(f"  {m}")) if verbose else (lambda m: None)
    say("[I1] 2-D FFT")
    S = I1_2d_fft(image)
    say("[I2] inverse Stolt interpolation  (f_tau' -> f_tau, re-adds migration)")
    S = I2_inverse_stolt_interpolation(S, p)
    say("[I3] reference function multiply, conjugated  (re-adds chirp + bulk RCM)")
    S = I3_reference_function_multiply_conjugate(S, p)
    say("[I4] 2-D IFFT")
    return I4_2d_ifft(S)


# ----------------------------------------------------------- FORWARD blocks
def F1_2d_fft(raw):
    """Block F1 — 2-D FFT: raw data -> 2-D frequency domain."""
    return np.fft.fft2(raw)


def F2_reference_function_multiply(S, p: Params):
    """Block F2 — reference function multiply: exp(+j theta_ref).
    Range-compresses and bulk-corrects migration for targets at R_ref."""
    D, inside = p.migration_factor(p.f_tau)
    theta = (4.0 * np.pi * p.r_ref / C) * D + np.pi * (p.f_tau.reshape(-1, 1) ** 2) / p.Kr
    if p.window_at_ref:      # undo the range-window shift applied in I3
        S = S * np.exp(-1j * 2 * np.pi * p.f_tau.reshape(-1, 1) * (2.0 * p.r_ref / C))
    return S * np.exp(1j * theta) * inside


def F3_stolt_interpolation(S, p: Params):
    """Block F3 — Stolt interpolation: f_tau -> f_tau', the change of variable
    f0 + f_tau' = D that decouples range and azimuth (differential RCMC)."""
    f_tau = np.fft.fftshift(p.f_tau)
    f_tau_prime = f_tau                                   # uniform target grid
    # invert the map: which f_tau produced this f_tau'?
    a = (p.f0 + f_tau_prime.reshape(-1, 1)) ** 2
    b = (C * p.f_eta.reshape(1, -1) / (2.0 * p.V)) ** 2
    f_tau_src = np.sqrt(a + b) - p.f0
    S_sorted = np.fft.fftshift(S, axes=0)
    out = np.zeros_like(S_sorted)
    for j in range(p.n_az):
        out[:, j] = _interp_complex(f_tau_src[:, j], f_tau, S_sorted[:, j])
    return np.fft.ifftshift(out, axes=0)


def F4_2d_ifft(S):
    """Block F4 — 2-D IFFT: the focused image."""
    return np.fft.ifft2(S)


def forward_omega_k(raw, p: Params, verbose=True):
    say = (lambda m: print(f"  {m}")) if verbose else (lambda m: None)
    say("[F1] 2-D FFT")
    S = F1_2d_fft(raw)
    say("[F2] reference function multiply")
    S = F2_reference_function_multiply(S, p)
    say("[F3] Stolt interpolation")
    S = F3_stolt_interpolation(S, p)
    say("[F4] 2-D IFFT")
    return F4_2d_ifft(S)


# ----------------------------------------------------------------- input
def load_image_and_params(a):
    """Returns (image, Params). SICD metadata fills what it can; CLI overrides."""
    path = Path(a.input)
    if path.suffix.lower() == ".npy":
        img = np.load(path).astype(np.complex128)
        need = [a.f0, a.fs, a.bandwidth, a.velocity, a.r_ref, a.azimuth_spacing]
        if any(v is None for v in need):
            sys.exit("for .npy input give --f0 --fs --bandwidth --velocity --r-ref "
                     "--azimuth-spacing")
        f0, fs, bw, v, rref, daz = need
    else:
        from sarpy.io.complex.converter import open_complex
        rd = open_complex(str(path))
        s = rd.get_sicds_as_tuple()[0]
        n_rows, n_cols = rd.get_data_size_as_tuple()[0]
        if a.chip:
            r0, c0, nr, nc = a.chip
        else:
            nr, nc = min(a.chip_size, n_rows), min(a.chip_size, n_cols)
            r0 = max(0, int(s.ImageData.SCPPixel.Row) - nr // 2)
            c0 = max(0, int(s.ImageData.SCPPixel.Col) - nc // 2)
            r0, c0 = min(r0, n_rows - nr), min(c0, n_cols - nc)
        print(f"  chip rows {r0}:{r0 + nr} cols {c0}:{c0 + nc} of {n_rows}x{n_cols}")
        img = np.asarray(rd[r0:r0 + nr, c0:c0 + nc], np.complex128)
        ifp = s.ImageFormation
        f0 = a.f0 or 0.5 * (float(ifp.TxFrequencyProc.MinProc) + float(ifp.TxFrequencyProc.MaxProc))
        bw = a.bandwidth or (float(ifp.TxFrequencyProc.MaxProc) - float(ifp.TxFrequencyProc.MinProc))
        fs = a.fs or C / (2.0 * float(s.Grid.Row.SS))        # from range pixel spacing
        daz = a.azimuth_spacing or float(s.Grid.Col.SS)
        if a.velocity:
            v = a.velocity
        else:
            t = 0.5 * (float(ifp.TStartProc) + float(ifp.TEndProc))
            v = float(np.linalg.norm(s.Position.ARPPoly.derivative_eval(t, der_order=1)))
        if a.r_ref:
            rref = a.r_ref
        elif s.SCPCOA is not None and s.SCPCOA.SlantRange:
            rref = float(s.SCPCOA.SlantRange)
        else:
            t = 0.5 * (float(ifp.TStartProc) + float(ifp.TEndProc))
            rref = float(np.linalg.norm(np.asarray(s.Position.ARPPoly(t))
                                        - np.asarray(s.GeoData.SCP.ECF.get_array())))
    p = Params(img.shape[0], img.shape[1], f0, fs, bw, a.pulse_length, v, rref, daz,
               window_at_ref=not a.absolute_delay)
    return img, p


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="inverse omega-K: focused image -> raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    cf = CONFIG
    ap.add_argument("input", nargs="?", default=cf["input"],
                    help="SICD (.nitf) or complex image (.npy); default: CONFIG['input']")
    ap.add_argument("-o", "--out", type=Path, default=Path(cf["out"]))
    ap.add_argument("--chip-size", type=int, default=cf["chip_size"], help="square chip at the SCP")
    ap.add_argument("--chip", type=int, nargs=4, default=cf["chip"],
                    metavar=("ROW0", "COL0", "NROWS", "NCOLS"))
    ap.add_argument("--pulse-length", type=float, default=cf["pulse_length"], help="sets Kr = B / Tp")
    ap.add_argument("--f0", type=float, default=cf["f0"], help="carrier frequency, Hz")
    ap.add_argument("--fs", type=float, default=cf["fs"], help="range sample rate, Hz")
    ap.add_argument("--bandwidth", type=float, default=cf["bandwidth"], help="chirp bandwidth, Hz")
    ap.add_argument("--velocity", type=float, default=cf["velocity"], help="effective velocity, m/s")
    ap.add_argument("--r-ref", type=float, default=cf["r_ref"], help="reference slant range, m")
    ap.add_argument("--azimuth-spacing", type=float, default=cf["azimuth_spacing"],
                    help="along-track sample spacing, m")
    ap.add_argument("--absolute-delay", action="store_true", default=cf["absolute_delay"],
                    help="range window starts at zero delay instead of at R_ref (echo will wrap)")
    ap.add_argument("--roundtrip", action="store_true", default=cf["roundtrip"],
                    help="refocus with forward omega-K and score")
    ap.add_argument("--quicklook", action="store_true", default=cf["quicklook"])
    a = ap.parse_args()
    if not a.input:
        ap.print_help()
        print("\nNo input. Set CONFIG['input'] at the top of this file, "
              "or pass a path on the command line.")
        return

    t0 = time.time()
    print(f"input: {a.input}")
    img, p = load_image_and_params(a)
    print("  " + p.summary())

    print("inverse omega-K:")
    raw = inverse_omega_k(img, p)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, raw=raw.astype(np.complex64), f0=p.f0, fs=p.fs,
                        bandwidth=p.bandwidth, Kr=p.Kr, velocity=p.V, r_ref=p.r_ref,
                        azimuth_spacing=p.d_az, prf=p.prf)
    print(f"  raw data {raw.shape} -> {a.out}  ({a.out.stat().st_size / 1e6:.1f} MB)")

    refocused = None
    if a.roundtrip:
        print("forward omega-K (check):")
        refocused = forward_omega_k(raw, p)
        a_, b_ = img.ravel(), refocused.ravel()
        corr = abs(np.vdot(a_, b_)) / np.sqrt(np.vdot(a_, a_).real * np.vdot(b_, b_).real)
        print(f"  refocused vs input image: |corr| = {corr:.4f}")

    if a.quicklook:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def db(x):
            v = 20 * np.log10(np.abs(x) + 1e-12)
            return np.clip(v - v.max(), -45, 0)

        n = 3 if refocused is not None else 2
        fig, ax = plt.subplots(1, n, figsize=(6 * n, 5.5))
        ax[0].imshow(db(img), cmap="gray", aspect="auto")
        ax[0].set_title("input image")
        ax[1].imshow(db(raw), cmap="viridis", aspect="auto")
        ax[1].set_title("raw data (unfocused) — |s|")
        ax[1].set_xlabel("azimuth sample")
        ax[1].set_ylabel("range sample")
        if refocused is not None:
            ax[2].imshow(db(refocused), cmap="gray", aspect="auto")
            ax[2].set_title("refocused by forward omega-K")
        fig.tight_layout()
        png = a.out.with_suffix(".png")
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print(f"  quicklook -> {png}")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
