#!/usr/bin/env python3
"""Build the training GeoTIFF: stack satellite soil-moisture (band 'z')
with terrain predictors, clip to the ROI shapefile, name the bands.

Stacking order (band order downstream) is the -f/-p order, with the
satellite file first. Training features are every band except 'z'.

Exit codes: 0 = output written; 1 = failure named on stderr.
"""

import argparse
import glob
import os
import sys
import tempfile
import zipfile

from osgeo import gdal

gdal.UseExceptions()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate GeoTIFF training file for the model.")
    parser.add_argument("-i", "--infile",
                        help="Satellite soil moisture GeoTIF file.")
    parser.add_argument("-f", "--paramfiles", nargs="+",
                        help="Terrain parameter GeoTIF files.")
    parser.add_argument("-p", "--params", nargs="+",
                        help="Terrain parameter names (band names).")
    parser.add_argument("-s", "--shpfile",
                        help="Shp file to crop into region, in zip.")
    parser.add_argument("-o", "--outfile", help="Training file in tif format.")
    return parser


def build_stack(input_files, output_file):
    """Stack input_files into one multiband GeoTIFF (separate bands).

    Resolution follows the first (satellite) file.
    """
    ds = gdal.Open(input_files[0], 0)
    xmin, xres, _, ymax, _, yres = ds.GetGeoTransform()
    ds = None

    fd, vrt_file = tempfile.mkstemp(prefix="stack_", suffix=".vrt")
    os.close(fd)
    try:
        vrt_options = gdal.BuildVRTOptions(separate=True)
        vrt = gdal.BuildVRT(vrt_file, input_files, options=vrt_options)
        vrt = None  # closes file
        translate_options = gdal.TranslateOptions(
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
            xRes=xres, yRes=yres, callback=gdal.TermProgress_nocb)
        gdal.Translate(output_file, vrt_file, options=translate_options)
    finally:
        try:
            os.remove(vrt_file)
        except OSError:
            pass


def set_band_names(raster, band_names):
    ds = gdal.Open(raster, 1)
    for i, name in enumerate(band_names):
        ds.GetRasterBand(i + 1).SetDescription(name)
    ds = None


def get_band_names(raster):
    ds = gdal.Open(raster, 0)
    names = [ds.GetRasterBand(band + 1).GetDescription()
             for band in range(ds.RasterCount)]
    ds = None
    return names


def get_shp(zip_file, workdir):
    """Extract the ROI zip; return its single .shp path."""
    shp_dir = os.path.join(workdir, "shp_file")
    os.makedirs(shp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(shp_dir)

    paths = sorted(glob.glob(os.path.join(shp_dir, "**", "*.shp"),
                             recursive=True))
    if len(paths) != 1:
        raise IOError(f"expected exactly one .shp in {zip_file}, "
                      f"found {len(paths)}: {paths}")
    return paths[0]


def crop_region(input_file, shp_file, output_file):
    warp_options = gdal.WarpOptions(
        cutlineDSName=shp_file, cropToCutline=True,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
        callback=gdal.TermProgress_nocb)
    warp = gdal.Warp(output_file, input_file, options=warp_options)
    warp = None


def main(argv=None):
    args = get_parser().parse_args(argv)

    for path in [args.infile, args.shpfile, args.outfile]:
        if path is None:
            print("error: -i/-s/-o are all required", file=sys.stderr)
            return 2
    if not args.paramfiles or not args.params:
        print("error: -f/-p require at least one predictor", file=sys.stderr)
        return 2
    if len(args.paramfiles) != len(args.params):
        print(f"error: {len(args.paramfiles)} param files vs "
              f"{len(args.params)} param names", file=sys.stderr)
        return 2
    for path in [args.infile] + list(args.paramfiles) + [args.shpfile]:
        if not os.path.isfile(path):
            print(f"error: input not found: {path}", file=sys.stderr)
            return 1

    parameter_files = [args.infile] + list(args.paramfiles)
    parameter_names = ["z"] + list(args.params)

    with tempfile.TemporaryDirectory(prefix="gen_train_") as workdir:
        try:
            build_stack(parameter_files, args.outfile)
            shp_file = get_shp(args.shpfile, workdir)
            crop_region(args.outfile, shp_file, args.outfile)
            set_band_names(args.outfile, parameter_names)
        except (RuntimeError, IOError, OSError) as e:
            print(f"error: training raster build failed: {e}",
                  file=sys.stderr)
            return 1

    print(get_band_names(args.outfile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
