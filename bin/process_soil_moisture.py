#!/usr/bin/env python3
"""Average one month of ESA CCI daily NetCDFs into a single monthly
soil-moisture GeoTIFF in EPSG:4326.

Input: days-zip + month manifest from acquire_soil_moisture.py
(days actually present; 404 gaps simply average over fewer days).
Averaging is nodata-aware (VRT Python 'average' pixel function).

Exit codes: 0 = output written; 1 = failure named on stderr.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile

from osgeo import gdal


PIXEL_CODE = """band="1" subClass="VRTDerivedRasterBand">
  <PixelFunctionType>average</PixelFunctionType>
  <PixelFunctionLanguage>Python</PixelFunctionLanguage>
  <PixelFunctionCode><![CDATA[
import numpy as np

def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    data = np.ma.array(in_ar, mask=np.equal(in_ar, {nodata}))
    np.ma.mean(data, axis=0, out=out_ar, dtype="float32")
    mask = np.all(data.mask,axis = 0)
    out_ar[mask] = {nodata}
  ]]>
  </PixelFunctionCode>"""

def get_parser():
    p = argparse.ArgumentParser(
        description="Average daily soil-moisture NetCDFs into a monthly GeoTIFF."
    )
    p.add_argument("--days-zip", required=True,
                   help="e.g. sm_2010_01_days.zip from acquire_soil_moisture.py")
    p.add_argument("--month-manifest", required=True,
                   help="month manifest listing the days actually present")
    p.add_argument("--output", required=True,
                   help="e.g. sm_2010_01.tif (EPSG:4326)")
    p.add_argument("--variable", default="sm",
                   help="NetCDF subdataset averaged (ESA CCI soil moisture)")
    return p


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise IOError(f"{' '.join(cmd)} failed: "
                      f"{proc.stderr.decode().strip().splitlines()[-1] if proc.stderr else ''}")
    return proc


def merge_avg(input_files, output_file, workdir):
    """Nodata-aware mean across inputs via a VRT Python pixel function."""
    vrt_path = os.path.join(workdir, "merged.vrt")
    vrt = gdal.BuildVRT(vrt_path, input_files)
    vrt = None  # closes file

    with open(vrt_path) as f:
        contents = f.read()

    if "<NoDataValue>" in contents:
        start = contents.index("<NoDataValue>") + len("<NoDataValue>")
        nodata_value = contents[start:contents.index("</NoDataValue>")]
    else:
        nodata_value = 0

    code = PIXEL_CODE.format(nodata=nodata_value)
    sub1, sub2 = contents.split('band="1">', 1)
    with open(vrt_path, "w") as f:
        f.write(sub1 + code + sub2)

    run(["gdal_translate", "-co", "COMPRESS=LZW", "-co", "TILED=YES",
         "-co", "BIGTIFF=YES", "--config", "GDAL_VRT_ENABLE_PYTHON", "YES",
         vrt_path, output_file])
    os.remove(vrt_path)


def reproject(input_file, projection="EPSG:4326"):
    """In-place warp to the target SRS (NaN nodata, multithreaded)."""
    import numpy as np
    warp_options = gdal.WarpOptions(
        dstSRS=projection,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES",
                         "NUM_THREADS=ALL_CPUS"],
        multithread=True, warpOptions=["NUM_THREADS=ALL_CPUS"],
        dstNodata=np.nan, callback=gdal.TermProgress_nocb)
    warp = gdal.Warp(input_file, input_file, options=warp_options)
    warp = None  # closes files


def main(argv=None):
    args = get_parser().parse_args(argv)

    for path in (args.days_zip, args.month_manifest):
        if not os.path.isfile(path):
            print(f"error: input not found: {path}", file=sys.stderr)
            return 1
    try:
        with open(args.month_manifest) as f:
            manifest = json.load(f)
        days = manifest["days"]
    except (OSError, ValueError, KeyError) as e:
        print(f"error: bad month manifest: {e}", file=sys.stderr)
        return 1
    if not days:
        print("error: month manifest lists zero days", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="proc_sm_", dir=".") as workdir:
        try:
            with zipfile.ZipFile(args.days_zip) as zf:
                zf.extractall(workdir)
        except zipfile.BadZipFile as e:
            print(f"error: bad days-zip: {e}", file=sys.stderr)
            return 1

        subdatasets = []
        for entry in days:
            local = os.path.join(workdir, entry["local"])
            if not os.path.isfile(local):
                print(f"error: manifest lists {entry['local']} "
                      f"but it is missing from the zip", file=sys.stderr)
                return 1
            subdatasets.append(f'NETCDF:"{local}":{args.variable}')

        tmp_tif = os.path.join(workdir, "monthly.tif")
        try:
            merge_avg(subdatasets, tmp_tif, workdir)
            reproject(tmp_tif)
        except (IOError, RuntimeError) as e:
            print(f"error: processing failed: {e}", file=sys.stderr)
            return 1

        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        os.replace(tmp_tif, args.output)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"averaged {len(subdatasets)} days -> {args.output} "
          f"({size_mb:.1f} MB, EPSG:4326)")
    return 0


if __name__ == "__main__":
    sys.exit(main())