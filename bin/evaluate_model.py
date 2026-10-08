#!/usr/bin/env python3
"""Predict soil moisture for one evaluation tile and rasterize it.

Reads an eval raster (terrain bands + pixel x/y, no 'z'), scales rows
with the training scaler, predicts with the pickled RandomForest, and
writes a GeoTIFF reusing the eval raster's geotransform. Predictions
pass through a per-job-unique temp CSV (never a shared fixed path, so
parallel tiles cannot collide).

Exit codes: 0 = output written; 1 = failure named on stderr.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

import pickle

gdal.UseExceptions()


def get_parser():
    p = argparse.ArgumentParser(
        description="Predict soil moisture for one evaluation tile.")
    p.add_argument("-i", "--infile", required=True,
                   help="evaluation GeoTIFF (terrain bands, no 'z')")
    p.add_argument("-o", "--outfile", required=True,
                   help="output predictions GeoTIFF")
    p.add_argument("-s", "--scaler", required=True,
                   help="pickled StandardScaler from training")
    p.add_argument("-m", "--model", required=True,
                   help="pickled RandomForest from training")
    return p


def get_band_names(raster):
    ds = gdal.Open(raster, 0)
    names = [ds.GetRasterBand(band + 1).GetDescription()
             for band in range(ds.RasterCount)]
    ds = None
    return names


def tif2arr(raster_file):
    """Eval raster (+ pixel x/y) into a float32 array, NaN rows dropped.

    Returns (array, kept_mask, geotransform, projection, raster_size).
    kept_mask maps predicted rows back to their pixels for rasterizing.
    """
    ds = gdal.Open(raster_file, 0)
    xmin, res, _, ymax, _, _ = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    xstart = xmin + res / 2
    ystart = ymax - res / 2

    x = np.arange(xstart, xstart + xsize * res, res)
    y = np.arange(ystart, ystart - ysize * res, -res)
    x = np.tile(x[:xsize], ysize)
    y = np.repeat(y[:ysize], xsize)

    band_names = get_band_names(raster_file)

    n_bands = ds.RasterCount
    bands = np.zeros((x.shape[0], n_bands), dtype=np.float32)
    for k in range(1, n_bands + 1):
        band = ds.GetRasterBand(k)
        data = band.ReadAsArray()
        data = np.ma.array(data, mask=np.equal(data, band.GetNoDataValue()))
        data = data.filled(np.nan)
        bands[:, k - 1] = data.flatten().astype(np.float32)
    ds = None

    column_names = ["x", "y"] + band_names
    stack = np.column_stack((x, y, bands)).astype(np.float32)
    df = pd.DataFrame(stack, columns=column_names)
    kept = df.notna().all(axis=1).to_numpy()
    return df.loc[kept].to_numpy(dtype=np.float32), kept, \
        geotransform, projection, (xsize, ysize)


def load_ds(arr, scaler_file, feature_names):
    """Scale feature columns (x, y + bands) with the training scaler."""
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
    frame = pd.DataFrame(arr, columns=["x", "y"] + list(feature_names))
    return scaler.transform(frame)


def predict(scaled, coords, model_file, csv_path):
    """Predict with the pickled model; write x,y,z rows to csv_path.

    coords are the UNSCALED pixel coordinates (geometry must stay in
    map units; only model features get scaled).
    """
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    z = model.predict(scaled)
    out = np.column_stack((coords[:, 0], coords[:, 1], z))
    pd.DataFrame(out, columns=["x", "y", "z"]).to_csv(csv_path, index=False)
    return out


def rasterize(csv_path, output_file, geotransform, projection, raster_size):
    """Point CSV back onto the eval raster's grid (Float32, NaN nodata)."""
    xsize, ysize = raster_size
    xmin = geotransform[0]
    ymax = geotransform[3]
    xres = geotransform[1]
    yres = abs(geotransform[5])
    xmax = xmin + xsize * xres
    ymin = ymax - ysize * yres

    layer = os.path.splitext(os.path.basename(csv_path))[0]
    vrt = (f'<OGRVRTDataSource><OGRVRTLayer name="predictions">'
           f'<SrcDataSource>{csv_path}</SrcDataSource>'
           f'<SrcLayer>{layer}</SrcLayer>'
           f'<GeometryType>wkbPoint</GeometryType>'
           f'<GeometryField encoding="PointFromColumns" x="x" y="y" z="z"/>'
           f'</OGRVRTLayer></OGRVRTDataSource>')
    fd, vrt_path = tempfile.mkstemp(prefix="pred_", suffix=".vrt",
                                    dir=os.path.dirname(
                                        os.path.abspath(output_file)))
    os.close(fd)
    try:
        with open(vrt_path, "w") as f:
            f.write(vrt)
        options = gdal.RasterizeOptions(
            outputBounds=[xmin, ymin, xmax, ymax],
            xRes=xres, yRes=yres, attribute="z",
            outputSRS=projection, noData=float("nan"),
            outputType=gdal.GDT_Float32,
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"])
        gdal.Rasterize(output_file, vrt_path, options=options)
    finally:
        try:
            os.remove(vrt_path)
        except OSError:
            pass


def write_empty(output_file, geotransform, projection, raster_size):
    """All-NaN tile on the eval grid. Empty ROI overlap is legitimate
    (grid cells outside the basins); the merge step treats it as nodata."""
    xsize, ysize = raster_size
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(output_file, xsize, ysize, 1, gdal.GDT_Float32,
                       ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"])
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.Fill(float("nan"))
    band.SetNoDataValue(float("nan"))
    ds = None


def main(argv=None):
    args = get_parser().parse_args(argv)

    for path in (args.infile, args.scaler, args.model):
        if not os.path.isfile(path):
            print(f"error: input not found: {path}", file=sys.stderr)
            return 1
    parent = os.path.dirname(os.path.abspath(args.outfile))
    if parent:
        os.makedirs(parent, exist_ok=True)

    fd, csv_path = tempfile.mkstemp(prefix="pred_", suffix=".csv", dir=parent)
    os.close(fd)
    try:
        try:
            arr, kept, geotransform, projection, raster_size = tif2arr(
                args.infile)
            if arr.shape[0] == 0:
                print(f"note: no valid pixels in {args.infile}; "
                      f"writing all-nodata tile", file=sys.stderr)
                write_empty(args.outfile, geotransform, projection,
                            raster_size)
                return 0
            feature_names = get_band_names(args.infile)
            scaled = load_ds(arr, args.scaler, feature_names)
            predict(scaled, arr[:, 0:2], args.model, csv_path)
            rasterize(csv_path, args.outfile, geotransform, projection,
                      raster_size)
        except (RuntimeError, IOError, OSError, ValueError) as e:
            print(f"error: inference failed: {e}", file=sys.stderr)
            return 1
    finally:
        try:
            os.remove(csv_path)
        except OSError:
            pass

    print(f"predicted {arr.shape[0]} pixels -> {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())