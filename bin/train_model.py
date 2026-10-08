#!/usr/bin/env python3
"""Train a RandomForest regressor on a training GeoTIFF.

Features are every band except 'z' (plus pixel x/y); target is 'z'.
Hyperparameters come from a small RandomizedSearchCV. Outputs the
pickled model, the pickled scaler, and a metrics JSON (validation
RMSE + best params + feature list) for comparing predictor sets.

Exit codes: 0 = all three outputs written; 1 = failure named on stderr.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

gdal.UseExceptions()


def get_parser():
    p = argparse.ArgumentParser(
        description="Train a RandomForest soil-moisture model.")
    p.add_argument("-i", "--infile", required=True,
                   help="training GeoTIFF (band 'z' = soil moisture)")
    p.add_argument("-o", "--outfile", required=True,
                   help="output model pickle, e.g. model.pkl")
    p.add_argument("-s", "--scaler", required=True,
                   help="output scaler pickle, e.g. scaler.pkl")
    p.add_argument("-m", "--metrics", required=True,
                   help="output metrics JSON, e.g. metrics.json")
    p.add_argument("--max-trees", type=int, default=500,
                   help="upper bound on trees tried")
    p.add_argument("--n-iter", type=int, default=10,
                   help="RandomizedSearchCV iterations")
    p.add_argument("--cv", type=int, default=5,
                   help="cross-validation folds")
    p.add_argument("--seed", type=int, default=1024,
                   help="random seed for reproducibility")
    return p


def get_band_names(raster):
    ds = gdal.Open(raster, 0)
    names = [ds.GetRasterBand(band + 1).GetDescription()
             for band in range(ds.RasterCount)]
    ds = None
    return names


def tif2df(raster_file):
    """GeoTIFF bands (+ pixel x/y) into a DataFrame, NaN rows dropped."""
    ds = gdal.Open(raster_file, 0)
    xmin, res, _, ymax, _, _ = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    xstart = xmin + res / 2
    ystart = ymax - res / 2

    x = np.arange(xstart, xstart + xsize * res, res)
    y = np.arange(ystart, ystart - ysize * res, -res)
    x = np.tile(x[:xsize], ysize)
    y = np.repeat(y[:ysize], xsize)

    band_names = get_band_names(raster_file)

    n_bands = ds.RasterCount
    bands = np.zeros((x.shape[0], n_bands))
    for k in range(1, n_bands + 1):
        band = ds.GetRasterBand(k)
        data = band.ReadAsArray()
        data = np.ma.array(data, mask=np.equal(data, band.GetNoDataValue()))
        data = data.filled(np.nan)
        bands[:, k - 1] = data.flatten()
    ds = None

    column_names = ["x", "y"] + band_names
    df = pd.DataFrame(np.column_stack((x, y, bands)),
                      columns=column_names)
    df.dropna(inplace=True)
    return df


def load_ds(train_file, scaler_file):
    """90/10 split on ('z' vs rest); fit + pickle the StandardScaler."""
    train_data = tif2df(train_file)
    print(train_data.shape)

    x_train, x_val, y_train, y_val = train_test_split(
        train_data.loc[:, train_data.columns != "z"],
        train_data.loc[:, "z"], test_size=0.1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    return x_train, x_val, y_train, y_val


def train_rf(x_train, y_train, max_trees, n_iter, cv, seed):
    """RandomForestRegressor under a small randomized search."""
    param_dist = {
        "n_estimators": list(range(300, max_trees + 1, 100)),
        "max_features": ["sqrt"],
        "max_depth": [20, 50, 70],
        "bootstrap": [True],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(n_jobs=-1, random_state=seed),
        param_distributions=param_dist, n_iter=n_iter, cv=cv,
        random_state=seed, n_jobs=-1)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def validate_model(model, x_val, y_val):
    """Validation RMSE (printed and returned)."""
    rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(x_val))))
    print(f"validation RMSE: {rmse:.6f}")
    return rmse


def main(argv=None):
    args = get_parser().parse_args(argv)

    if args.max_trees < 300:
        print(f"error: --max-trees {args.max_trees} below search floor 300",
              file=sys.stderr)
        return 2
    if args.n_iter < 1 or args.cv < 2:
        print("error: --n-iter >= 1 and --cv >= 2 required",
              file=sys.stderr)
        return 2
    if not os.path.isfile(args.infile):
        print(f"error: input not found: {args.infile}", file=sys.stderr)
        return 1
    for path in (args.outfile, args.scaler, args.metrics):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    try:
        feature_names = [c for c in tif2df(args.infile).columns
                         if c not in ("x", "y", "z")]
        x_train, x_val, y_train, y_val = load_ds(args.infile, args.scaler)
        model, best_params = train_rf(x_train, y_train, args.max_trees,
                                      args.n_iter, args.cv, args.seed)
        rmse = validate_model(model, x_val, y_val)
        with open(args.outfile, "wb") as f:
            pickle.dump(model, f)
    except (RuntimeError, IOError, OSError, ValueError) as e:
        print(f"error: training failed: {e}", file=sys.stderr)
        return 1

    metrics = {
        "val_rmse": rmse,
        "best_params": best_params,
        "feature_names": feature_names,
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "max_trees": args.max_trees,
        "n_iter": args.n_iter,
        "cv": args.cv,
        "seed": args.seed,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = args.metrics + ".part"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    os.replace(tmp, args.metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())