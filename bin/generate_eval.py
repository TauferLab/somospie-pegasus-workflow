#!/usr/bin/env python3
"""Build evaluation GeoTIFF tiles: crop terrain predictors to the ROI,
stack them, cut the requested grid tiles, name the bands.

Tiles contain terrain only (no 'z' band — that is what the model
predicts). Output names follow request order: {prefix}_0000.tif, ...,
so the generator maps config order to job outputs deterministically.

Exit codes: 0 = tiles written; 1 = failure named on stderr.
"""

import argparse
import math
import os
import sys
import tempfile

from osgeo import gdal

gdal.UseExceptions()

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate GeoTIFF evaluation tiles for the model.")
    parser.add_argument("-i", "--infiles", nargs="+",
                        help="Terrain parameter GeoTIF files.")
    parser.add_argument("-p", "--params", nargs="+",
                        help="Terrain parameter identifiers (band names).")
    parser.add_argument("-n", "--ntiles", type=int, required=True,
                        help="Eval grid is ntiles x ntiles.")
    parser.add_argument("--tile", nargs=2, action="append", metavar=("X", "Y"),
                        help="tile to emit, zero-based (repeatable); "
                             "output order follows request order")
    parser.add_argument("-s", "--shpfile",
                        help="ROI cutline zip (OGR opens zipped "
                             "shapefiles transparently).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out-prefix", default="eval")
    return parser


def build_stack(input_files, workdir):
    """Stack input_files into one multiband VRT (separate bands)."""
    fd, vrt_file = tempfile.mkstemp(prefix="stack_", suffix=".vrt",
                                    dir=workdir)
    os.close(fd)
    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(vrt_file, input_files, options=vrt_options)
    vrt = None  # closes file
    return vrt_file


def crop_tile(raster, out_file, n_tiles, idx_x, idx_y):
    """Cut tile (idx_x, idx_y) of the n_tiles x n_tiles grid.

    Tiles partition the raster with non-overlapping windows, so the
    merge step reassembles them exactly (no overlap averaging needed).
    """
    ds = gdal.Open(raster, 0)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ds = None
    x_win_size = int(math.ceil(cols / n_tiles))
    y_win_size = int(math.ceil(rows / n_tiles))

    idx_x = range(0, cols, x_win_size)[idx_x]
    idx_y = range(0, rows, y_win_size)[idx_y]

    if idx_y + y_win_size < rows:
        nrows = y_win_size
    else:
        nrows = rows - idx_y

    if idx_x + x_win_size < cols:
        ncols = x_win_size
    else:
        ncols = cols - idx_x

    translate_options = gdal.TranslateOptions(
        srcWin=[idx_x, idx_y, ncols, nrows],
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
        callback=gdal.TermProgress_nocb)
    gdal.Translate(out_file, raster, options=translate_options)


def crop_region(input_file, shp_file):
    """In-place clip of one raster to the ROI cutline.

    A .zip holding a single shapefile needs OGR's explicit /vsizip/
    prefix (plain .zip paths are rejected as unrecognized format).
    """
    if shp_file.lower().endswith(".zip"):
        shp_file = f"/vsizip/{os.path.abspath(shp_file)}"
    warp_options = gdal.WarpOptions(
        cutlineDSName=shp_file, cropToCutline=True,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
        callback=gdal.TermProgress_nocb)
    warp = gdal.Warp(input_file, input_file, options=warp_options)
    warp = None


def set_band_names(raster, band_names):
    ds = gdal.Open(raster, 1)
    for i, name in enumerate(band_names):
        ds.GetRasterBand(i + 1).SetDescription(name)
    ds = None
    # Keep band names inside the TIFF: drop any PAM sidecar that would
    # otherwise carry them (unstaged sidecars don't travel with the file).
    aux = raster + ".aux.xml"
    if os.path.isfile(aux):
        os.remove(aux)


def main(argv=None):
    args = get_parser().parse_args(argv)

    if not args.infiles or not args.params:
        print("error: -i/-p require at least one predictor",
              file=sys.stderr)
        return 2
    if len(args.infiles) != len(args.params):
        print(f"error: {len(args.infiles)} infiles vs "
              f"{len(args.params)} params", file=sys.stderr)
        return 2
    if args.ntiles < 1:
        print(f"error: bad -n {args.ntiles}, expected >= 1",
              file=sys.stderr)
        return 2
    if not args.tile:
        print("error: at least one --tile X Y is required",
              file=sys.stderr)
        return 2
    tiles = []
    for x, y in args.tile:
        try:
            ix, iy = int(x), int(y)
        except ValueError:
            print(f"error: bad --tile {x} {y}, expected integers",
                  file=sys.stderr)
            return 2
        if not (0 <= ix < args.ntiles and 0 <= iy < args.ntiles):
            print(f"error: tile ({ix}, {iy}) outside "
                  f"0..{args.ntiles - 1} grid", file=sys.stderr)
            return 2
        tiles.append((ix, iy))
    for path in list(args.infiles) + [args.shpfile]:
        if path is None or not os.path.isfile(path):
            print(f"error: input not found: {path}", file=sys.stderr)
            return 1

    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gen_eval_") as workdir:
        try:
            # Stage private copies: the ROI clip rewrites inputs in place.
            staged = []
            for filename in args.infiles:
                dst = os.path.join(workdir, os.path.basename(filename))
                gdal.Translate(dst, filename,
                               options=gdal.TranslateOptions(
                                   creationOptions=["COMPRESS=LZW",
                                                    "TILED=YES",
                                                    "BIGTIFF=YES"]))
                crop_region(dst, args.shpfile)
                staged.append(dst)
            vrt_file = build_stack(staged, workdir)
            outputs = []
            for idx, (ix, iy) in enumerate(tiles):
                out_file = os.path.join(
                    args.output_dir, f"{args.out_prefix}_{idx:04d}.tif")
                crop_tile(vrt_file, out_file, args.ntiles, ix, iy)
                set_band_names(out_file, list(args.params))
                outputs.append(out_file)
        except (RuntimeError, IOError, OSError) as e:
            print(f"error: eval tile build failed: {e}", file=sys.stderr)
            return 1

    print("\n".join(outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())