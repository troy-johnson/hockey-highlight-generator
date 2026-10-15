"""Download a real elevation grid of the Wasatch front for the Black Ice opening.

    .venv/bin/python v3/blender/wasatch_dem.py --out ~/hockey/cache/wasatch_dem.npz

Source: AWS Open Data "Terrain Tiles" (Terrarium PNG encoding, built from
USGS 3DEP / SRTM and other public sources):
https://registry.opendata.aws/terrain-tiles/
Elevation in metres = R * 256 + G + B / 256 - 32768.

The saved .npz holds `elev` (float32, rows north -> south, columns west -> east)
and the grid bounds `north`, `south`, `west`, `east` in degrees.
"""
import argparse
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
# Salt Lake valley floor to beyond the Wasatch crest, Mount Olympus to Lone Peak.
DEFAULT_BOX = (40.78, 40.50, -112.02, -111.62)     # north, south, west, east


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile_lat(y, z):
    n = math.pi - 2 * math.pi * y / 2 ** z
    return math.degrees(math.atan(math.sinh(n)))


def fetch(z, x, y):
    with urllib.request.urlopen(URL.format(z=z, x=x, y=y), timeout=30) as r:
        data = np.frombuffer(r.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[dem] tile {z}/{x}/{y} did not decode")
    b, g, r = (img[..., i].astype(np.float32) for i in range(3))
    return r * 256 + g + b / 256 - 32768


def main(argv=None):
    p = argparse.ArgumentParser(prog="wasatch_dem")
    p.add_argument("--out", required=True)
    p.add_argument("--zoom", type=int, default=13)
    a = p.parse_args(argv)
    north, south, west, east = DEFAULT_BOX
    x0, y0 = (int(v) for v in tile_xy(north, west, a.zoom))
    x1, y1 = (int(v) for v in tile_xy(south, east, a.zoom))
    rows = []
    for ty in range(y0, y1 + 1):
        rows.append(np.hstack([fetch(a.zoom, tx, ty) for tx in range(x0, x1 + 1)]))
        print(f"[dem] row {ty - y0 + 1}/{y1 - y0 + 1}", flush=True)
    elev = np.vstack(rows).astype(np.float32)
    n = 2 ** a.zoom
    bounds = dict(north=tile_lat(y0, a.zoom), south=tile_lat(y1 + 1, a.zoom),
                  west=x0 / n * 360 - 180, east=(x1 + 1) / n * 360 - 180)
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.out))), exist_ok=True)
    np.savez_compressed(os.path.expanduser(a.out), elev=elev, **bounds)
    print(f"[dem] {elev.shape[1]}x{elev.shape[0]} samples, {elev.min():.0f}-{elev.max():.0f} m, {bounds}")


if __name__ == "__main__":
    sys.exit(main())
