#!/usr/bin/env python3
"""Prepare terrain inputs: reproject predictor rasters to EPSG:4326 and
build a validated polygon cutline from the ROI shapefile zip.

The cutline must be polygons (a lines-only zip such as channel_network
fails fast here, not deep in the workflow). Invalid geometries are
repaired; a missing .prj is written (EPSG:4326). --out-prefix namespaces
outputs for Pegasus, where raw and processed files share a directory
but must be distinct logical files.

Exit codes: 0 = outputs written (+ manifest); 1 = failure named on
stderr; 2 = argument errors.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile

from osgeo import gdal, ogr, osr

gdal.UseExceptions()

POLYGON_TYPES = {ogr.wkbPolygon, ogr.wkbMultiPolygon,
                 ogr.wkbPolygon25D, ogr.wkbMultiPolygon25D}
SHAPE_EXTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")

def get_parser():
    p = argparse.ArgumentParser(
        description="Reproject terrain predictors; build ROI cutline.")
    p.add_argument("--input-dir", required=True,
                   help="dir with downloaded predictor tifs + roi zip")
    p.add_argument("--param-names", nargs="+", required=True,
                   help="band names, in order (e.g. aspect elevation ...)")
    p.add_argument("--param-files", nargs="+", required=True,
                   help="predictor filenames in input-dir, same order")
    p.add_argument("--roi-zip", required=True,
                   help="ROI filename in input-dir, e.g. drainage_basins.zip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--manifest", required=True, help="output manifest path")
    p.add_argument("--out-prefix", default="",
                   help="prepended to every output filename, e.g. proc_")
    return p


def reproject_tif(src, dst):
    """Warp one raster to EPSG:4326 (source nodata preserved)."""
    warp_options = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES",
                         "NUM_THREADS=ALL_CPUS"],
        multithread=True, warpOptions=["NUM_THREADS=ALL_CPUS"],
        callback=gdal.TermProgress_nocb)
    warp = gdal.Warp(dst, src, options=warp_options)
    warp = None  # closes files


def find_shapefile(shp_dir):
    """The single .shp in the extracted ROI tree, at any depth
    (bucket zips nest files under a per-state folder, e.g. TN/)."""
    paths = sorted(glob.glob(os.path.join(shp_dir, "**", "*.shp"),
                             recursive=True))
    if len(paths) != 1:
        raise IOError(f"expected exactly one .shp in roi zip, "
                      f"found {len(paths)}: {paths}")
    return paths[0]


def check_polygons(shp_path):
    """Fail fast unless the shapefile holds polygons."""
    ds = ogr.Open(shp_path)
    if ds is None or ds.GetLayerCount() == 0:
        raise IOError(f"cannot open shapefile layers: {shp_path}")
    geom_type = ds.GetLayer(0).GetGeomType()
    ds = None
    if geom_type not in POLYGON_TYPES:
        raise IOError(f"roi_shape must be polygons, got "
                      f"{ogr.GeometryTypeToName(geom_type)} "
                      f"(a lines-only zip cannot clip rasters)")
    return geom_type


def repair_shapefile(shp_path, workdir):
    """ogr2ogr -makevalid into workdir; return the repaired .shp path."""
    out_path = os.path.join(
        workdir, os.path.basename(shp_path))
    proc = subprocess.run(
        ["ogr2ogr", "-makevalid", out_path, shp_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not os.path.isfile(out_path):
        err = (proc.stderr.decode().strip().splitlines() or ["failed"])[-1]
        raise IOError(f"ogr2ogr -makevalid failed: {err}")
    return out_path


def ensure_prj(shp_path):
    """Write a .prj (EPSG:4326, ESRI WKT) if the shapefile lacks one."""
    prj_path = os.path.splitext(shp_path)[0] + ".prj"
    if os.path.isfile(prj_path):
        return False
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.MorphToESRI()
    with open(prj_path, "w") as f:
        f.write(srs.ExportToWkt())
    return True


def zip_cutline(shp_path, out_zip):
    """Zip the .shp + companions as the cutline archive (atomic)."""
    stem = os.path.splitext(shp_path)[0]
    tmp = out_zip + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in SHAPE_EXTS:
            part = stem + ext
            if os.path.isfile(part):
                zf.write(part, arcname=os.path.basename(part))
    os.replace(tmp, out_zip)


def main(argv=None):
    args = get_parser().parse_args(argv)

    if len(args.param_names) != len(args.param_files):
        print(f"error: {len(args.param_names)} param-names vs "
              f"{len(args.param_files)} param-files", file=sys.stderr)
        return 2
    if not args.param_files:
        print("error: at least one predictor required", file=sys.stderr)
        return 2

    for filename in list(args.param_files) + [args.roi_zip]:
        if not os.path.isfile(os.path.join(args.input_dir, filename)):
            print(f"error: input not found: {filename}", file=sys.stderr)
            return 1

    os.makedirs(args.output_dir, exist_ok=True)

    pre = args.out_prefix
    out_names = {filename: pre + os.path.basename(filename)
                 for filename in args.param_files}

    # 1. Reproject predictors (order preserved = band order downstream).
    try:
        for name, filename in zip(args.param_names, args.param_files):
            reproject_tif(os.path.join(args.input_dir, filename),
                          os.path.join(args.output_dir, out_names[filename]))
    except RuntimeError as e:
        print(f"error: reproject failed: {e}", file=sys.stderr)
        return 1

    # 2. Unzip ROI, validate single polygon shapefile, repair, ensure .prj.
    roi_base = os.path.splitext(os.path.basename(args.roi_zip))[0]
    cutline_zip = os.path.join(args.output_dir, pre + roi_base + "_cutline.zip")
    with tempfile.TemporaryDirectory(prefix="proc_terrain_") as workdir:
        roi_src = os.path.join(args.input_dir, args.roi_zip)
        try:
            with zipfile.ZipFile(roi_src) as zf:
                zf.extractall(os.path.join(workdir, "roi"))
        except zipfile.BadZipFile as e:
            print(f"error: bad roi zip: {e}", file=sys.stderr)
            return 1
        try:
            shp_path = find_shapefile(os.path.join(workdir, "roi"))
            check_polygons(shp_path)
            fixed = os.path.join(workdir, "fixed")
            os.makedirs(fixed, exist_ok=True)
            repaired = repair_shapefile(shp_path, fixed)
            wrote_prj = ensure_prj(repaired)
            zip_cutline(repaired, cutline_zip)
        except (IOError, OSError) as e:
            print(f"error: roi processing failed: {e}", file=sys.stderr)
            return 1

    manifest = {
        "epsg": 4326,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictors": [{"name": n, "file": out_names[f]}
                       for n, f in zip(args.param_names, args.param_files)],
        "roi_cutline": os.path.basename(cutline_zip),
        "roi_prj_written": wrote_prj,
    }
    tmp = args.manifest + ".part"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, args.manifest)

    print(f"processed {len(args.param_files)} predictors -> "
          f"{args.output_dir}; cutline {os.path.basename(cutline_zip)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())