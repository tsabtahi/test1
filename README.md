


```
"""Print what is inside a CPHD: header, per-pulse parameters (PVP) and I/Q samples.

Edit the paths/numbers below and run:  python cphd_peek.py
Needs sarpy (pip install sarpy).
"""
import numpy as np
from sarpy.io.phase_history.converter import open_phase_history

# ------------------------------------------------------------------ edit these
CPHD_PATH = r"C:\data\2023-06-20-12-06-09_UMBRA-05_CPHD.cphd"
PULSE = None        # pulse (vector) index to show; None = middle pulse
N_SAMPLES = 8       # how many I/Q samples of that pulse to print
SAVE_CSV = None     # e.g. r"C:\data\pulse_iq.csv" to save all I/Q of that pulse
# ------------------------------------------------------------------

reader = open_phase_history(CPHD_PATH)
meta = reader.cphd_meta
ch = meta.Data.Channels[0]
n_pulses, n_samples = ch.NumVectors, ch.NumSamples
p = n_pulses // 2 if PULSE is None else PULSE

def get(obj, path):                                   # safe nested lookup: fields differ by CPHD version
    for k in path.split("."):
        obj = getattr(obj, k, None)
        if obj is None:
            return "n/a"
    return obj

print("=== header ===")
print("collector     :", get(meta, "CollectionID.CollectorName"), "|", get(meta, "CollectionID.CoreName"))
print("mode          :", get(meta, "CollectionID.RadarMode.ModeType"))
print("domain        :", meta.Global.DomainType, "(FX = each sample is a frequency)")
print("SGN           :", meta.Global.SGN)
print("sample format :", meta.Data.SignalArrayFormat, "(CF8 = float32 I + float32 Q; CI2 = int8 I + int8 Q; CI4 = int16 I + int16 Q)")
print("pulses        :", n_pulses, "  samples per pulse:", n_samples)

print(f"\n=== PVP of pulse {p} ===")
pvp = reader.read_pvp_array(index=0)                 # one structured row per pulse
for name in pvp.dtype.names:
    print(f"{name:12s}: {pvp[name][p]}")
sc0, scss = float(pvp["SC0"][p]), float(pvp["SCSS"][p])
print(f"\nband of the whole collection: {pvp['FX1'].min() / 1e9:.4f} to {pvp['FX2'].max() / 1e9:.4f} GHz, "
      f"sample spacing {scss / 1e3:.1f} kHz, collection time {pvp['TxTime'][-1] - pvp['TxTime'][0]:.2f} s")

print(f"\n=== I/Q of pulse {p}, first {N_SAMPLES} samples ===")
raw = reader.read_raw(slice(p, p + 1), slice(0, n_samples), index=0)   # as stored in the file
row = np.asarray(reader[p:p + 1, :]).reshape(-1)                       # converted to complex64
print("stored as:", raw.dtype, raw.shape, "-> last axis is (I, Q)")
print("first stored pairs (I, Q):", np.asarray(raw).reshape(-1, 2)[:3].tolist())
print(f"{'n':>4} {'freq (GHz)':>12} {'I (real)':>12} {'Q (imag)':>12} {'|amp|':>10} {'phase deg':>10}")
for n in range(min(N_SAMPLES, n_samples)):
    z = row[n]
    print(f"{n:4d} {(sc0 + n * scss) / 1e9:12.6f} {z.real:12.5g} {z.imag:12.5g} {abs(z):10.4g} {np.degrees(np.angle(z)):10.1f}")
print("pulse power (mean |z|^2):", float(np.mean(np.abs(row) ** 2)))

if SAVE_CSV:
    f = sc0 + np.arange(n_samples) * scss
    np.savetxt(SAVE_CSV, np.c_[np.arange(n_samples), f, row.real, row.imag],
               delimiter=",", header="n,freq_hz,I,Q", comments="")
    print("saved", SAVE_CSV)
```
