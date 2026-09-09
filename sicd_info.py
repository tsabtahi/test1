#!/usr/bin/env python3
"""
sicd_info.py — verify a downloaded phase test set actually opens and carries phase.

Run inside the container:
    docker run --rm -v $PWD/umbra_phase:/data \
        --entrypoint python umbra-phase:latest /app/sicd_info.py /data

For each scene it prints image size, pixel type, geometry, and the mean/std of
the phase angle of a small centre chip. Phase that is uniform on (-pi, pi] with
std near pi/sqrt(3) ~= 1.81 is the expected signature of complex speckle; a tiny
std means you are looking at a detected product, not complex data.
"""

import sys
from pathlib import Path

import numpy as np
from sarpy.io.complex.converter import open_complex


def chip_phase(reader, n=512):
    rows, cols = reader.data_size
    r0, c0 = max(0, rows // 2 - n // 2), max(0, cols // 2 - n // 2)
    data = reader[r0:r0 + n, c0:c0 + n]
    if not np.iscomplexobj(data):
        return None
    ph = np.angle(data[np.abs(data) > 0])
    return float(ph.mean()), float(ph.std()), int(ph.size)


def main(root: Path) -> int:
    files = sorted(root.rglob("*_SICD.nitf"))
    if not files:
        print(f"no *_SICD.nitf under {root}", file=sys.stderr)
        return 1

    bad = 0
    for f in files:
        try:
            reader = open_complex(str(f))
            sicd = reader.get_sicds_as_tuple()[0]
            rows, cols = reader.data_size
            scpcoa = sicd.SCPCOA
            stats = chip_phase(reader)
            tag = "complex" if stats else "NOT COMPLEX"
            print(f"{f.parent.name}")
            print(f"  size      {rows} x {cols}   {reader.get_data_type()}   [{tag}]")
            print(f"  geometry  inc={scpcoa.IncidenceAng:.2f} graze={scpcoa.GrazeAng:.2f} "
                  f"az={scpcoa.AzimAng:.2f} side={scpcoa.SideOfTrack} "
                  f"slant={scpcoa.SlantRange / 1000:.1f} km")
            print(f"  res       rg={sicd.Grid.Row.ImpRespWid:.3f} m  "
                  f"az={sicd.Grid.Col.ImpRespWid:.3f} m  ifa={sicd.ImageFormation.ImageFormAlgo}")
            if stats:
                mean, std, npix = stats
                print(f"  phase     mean={mean:+.3f} rad  std={std:.3f} rad  n={npix}")
            else:
                bad += 1
        except Exception as e:
            bad += 1
            print(f"{f.parent.name}\n  FAILED: {e}", file=sys.stderr)

    cphd = sorted(root.rglob("*_CPHD.cphd"))
    print(f"\n{len(files)} SICD checked, {bad} problem(s); {len(cphd)} CPHD present")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "/data")))
