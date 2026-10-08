#!/usr/bin/env python3
"""Render a prediction map GeoTIFF as a PNG figure (shareable artifact).

Soil-moisture values only (RdBu colormap, data-driven scale, 5x4 inches, dpi 150).
Explicit --vmin/--vmax pin a fixed scale for cross-run comparability;
nodata renders as white background.

Exit codes: 0 = figure written; 1 = failure named on stderr.
"""

import argparse
import math
import os
import sys

import numpy as np
from osgeo import gdal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gdal.UseExceptions()


def get_parser():
    p = argparse.ArgumentParser(
        description="Render a prediction map as a PNG figure.")
    p.add_argument("--input", required=True,
                   help="prediction map GeoTIFF, e.g. predictions_TN_2010_01_map.tif")
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--cmap", default="RdBu")
    p.add_argument("--vmin", type=float, default=None,
                   help="fixed scale low (default: data minimum, "
                        "original behavior)")
    p.add_argument("--vmax", type=float, default=None,
                   help="fixed scale high (default: data maximum, "
                        "original behavior)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max-pixels", type=int, default=4000,
                   help="longest preview axis in pixels (decimated read)")
    p.add_argument("--title", default="",
                   help="figure title (generator passes state + train label)")
    return p


def main(argv=None):
    args = get_parser().parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1
    if (args.vmin is not None and args.vmax is not None
            and not args.vmax > args.vmin):
        print("error: --vmax must exceed --vmin", file=sys.stderr)
        return 2
    if args.max_pixels < 1:
        print("error: --max-pixels must be positive", file=sys.stderr)
        return 2
    parent = os.path.dirname(os.path.abspath(args.output))
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        ds = gdal.Open(args.input, 0)
        if ds.RasterCount != 1:
            print("error: expected a single-band prediction map",
                  file=sys.stderr)
            return 1
        scale = max(1, math.ceil(max(ds.RasterXSize, ds.RasterYSize)
                                / args.max_pixels))
        width = max(1, ds.RasterXSize // scale)
        height = max(1, ds.RasterYSize // scale)
        arr = ds.GetRasterBand(1).ReadAsArray(
            buf_xsize=width, buf_ysize=height,
            resample_alg=gdal.GRIORA_Average).astype(float)
        gt = ds.GetGeoTransform()
        wkt = ds.GetProjection()
        full_w, full_h = ds.RasterXSize, ds.RasterYSize
        ds = None
        data = np.ma.masked_invalid(arr)
        if data.count() == 0:
            print("error: no valid pixels to render", file=sys.stderr)
            return 1
        vmin = args.vmin if args.vmin is not None else float(np.nanmin(data))
        vmax = args.vmax if args.vmax is not None else float(np.nanmax(data))
        if not vmax > vmin:
            print("error: degenerate data range", file=sys.stderr)
            return 1

        xmin, ymax = gt[0], gt[3]
        xmax = xmin + full_w * gt[1]
        ymin = ymax + full_h * gt[5]
        geographic = "4326" in wkt
        if not geographic:
            print("warning: input is not EPSG:4326; axes labeled in "
                  "file units", file=sys.stderr)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(data, cmap=args.cmap, vmin=vmin, vmax=vmax,
                       extent=[xmin, xmax, ymin, ymax],
                       origin="upper", interpolation="nearest")
        ax.set_xlabel("lon" if geographic else "X")
        ax.set_ylabel("lat" if geographic else "Y")
        if args.title:
            ax.set_title(args.title)
        h, w = data.shape
        fig.colorbar(im, ax=ax, label="soil moisture (m³/m³)",
                     fraction=0.046 * h / max(w, 1), pad=0.04)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                    pad_inches=0)
        plt.close(fig)
    except (RuntimeError, IOError, OSError, ValueError) as e:
        print(f"error: visualization failed: {e}", file=sys.stderr)
        return 1

    size_kb = os.path.getsize(args.output) / 1024
    print(f"rendered {args.output} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())