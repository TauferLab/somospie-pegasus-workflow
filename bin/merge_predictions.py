#!/usr/bin/env python3
"""Mosaic per-tile prediction rasters into one state map GeoTIFF.

Inputs must share grid/CRS (they do: one eval grid, one model).
Unpredicted areas stay nodata — so a subset run yields a partial map
and eval_tiles: all yields the gap-free state map. An optional
overview pyramid is built for fast visualization.

Exit codes: 0 = map written; 1 = failure named on stderr.
"""

import argparse
import os
import sys
import tempfile

from osgeo import gdal

gdal.UseExceptions()


def get_parser():
    p = argparse.ArgumentParser(
        description="Mosaic prediction tiles into one state map.")
    p.add_argument("-i", "--infiles", nargs="+", required=True,
                   help="prediction tiles in mosaic order")
    p.add_argument("-o", "--outfile", required=True,
                   help="e.g. predictions_TN_2010_01_map.tif")
    p.add_argument("--no-overviews", action="store_true",
                   help="skip the visualization overview pyramid")
    return p


def main(argv=None):
    args = get_parser().parse_args(argv)

    if not args.infiles:
        print("error: at least one input tile required", file=sys.stderr)
        return 2
    for path in args.infiles:
        if not os.path.isfile(path):
            print(f"error: input not found: {path}", file=sys.stderr)
            return 1
    parent = os.path.dirname(os.path.abspath(args.outfile))
    if parent:
        os.makedirs(parent, exist_ok=True)

    fd, vrt_file = tempfile.mkstemp(prefix="mosaic_", suffix=".vrt",
                                    dir=parent)
    os.close(fd)
    try:
        vrt = gdal.BuildVRT(vrt_file, list(args.infiles))
        vrt = None  # closes file
        translate_options = gdal.TranslateOptions(
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
            callback=gdal.TermProgress_nocb)
        gdal.Translate(args.outfile, vrt_file, options=translate_options)
    except RuntimeError as e:
        print(f"error: mosaic failed: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            os.remove(vrt_file)
        except OSError:
            pass

    if not args.no_overviews:
        ds = gdal.Open(args.outfile, 1)
        try:
            gdal.SetConfigOption("COMPRESS_OVERVIEW", "LZW")
            ds.BuildOverviews("NEAREST", [2, 4, 8, 16],
                              callback=gdal.TermProgress_nocb)
        finally:
            ds = None

    size_mb = os.path.getsize(args.outfile) / 1024 / 1024
    print(f"mosaicked {len(args.infiles)} tiles -> {args.outfile} "
          f"({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())