#!/usr/bin/env python3
"""Fetch a US state boundary from GADM and package it as a cutline zip.

GADM level-1 USA GeoJSON holds all states; the configured state is
selected by NAME_1 (abbreviation or full name accepted). Output is a
shapefile zip in EPSG:4326, consumable by process_terrain.

The GADM source file persists under --cache-dir (URL-validated), so
state switches reuse it instead of re-downloading; per-state outputs
reuse via the manifest (Tier-1) when nothing changed.

Exit codes: 0 = zip valid on disk (+ manifest); 1 = failure named on
stderr; 2 = argument errors.
"""


import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil 
import time
import urllib.parse
import zipfile

import requests

TRANSFER_TIMEOUT = 120

# Abbreviation -> GADM NAME_1. Full names also accepted verbatim
# (matched case-insensitively) via _NAMES below.
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida",
    "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "PR": "Puerto Rico", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

# Supported region: contiguous US only. STATES above still carries
# AK/HI/DC/PR so those resolve to "outside region" rather than
# "unknown" in the error below.
VALID_STATES: frozenset[str] = frozenset({
    "AL","AR","AZ","CA","CO","CT","DE","FL","GA","IA","ID","IL","IN",
    "KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND",
    "NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD",
    "TN","TX","UT","VA","VT","WA","WI","WV","WY"
})

_NAMES = {v.upper(): k for k, v in STATES.items()}


def get_parser():
    p = argparse.ArgumentParser(
        description="Fetch a US state boundary (GADM) as a cutline zip.")
    p.add_argument("--gadm-url", required=True,
                   help="GADM USA level-1 file, e.g. "
                        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/"
                        "gadm41_USA_1.json.zip")
    p.add_argument("--state", required=True,
                   help="state abbreviation (TN) or full name (Tennessee); "
                        "contiguous US only")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-zip", required=True,
                   help="e.g. state_TN.zip")
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache-dir", default="",
                   help="persist the GADM source file here for reuse "
                        "across states (empty = no caching)")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--force", action="store_true",
                   help="ignore previous outputs, fetch everything again")
    return p


def resolve_state(raw):
    """Abbreviation or full name -> (abbrev, GADM NAME_1).

    Only VALID_STATES (contiguous US) are supported.
    """
    token = raw.strip()
    if len(token) == 2 and token.upper() in STATES:
        abbrev = token.upper()
    else:
        abbrev = _NAMES.get(token.upper(), "")
        if not abbrev:
            raise ValueError(
                f"unknown state {raw!r}; valid abbreviations: "
                f"{', '.join(sorted(VALID_STATES))}")
    if abbrev not in VALID_STATES:
        raise ValueError(
            f"state {raw!r} is outside the supported region "
            f"(contiguous US only)")
    return abbrev, STATES[abbrev]


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = (proc.stderr.decode().strip().splitlines() or ["failed"])[-1]
        raise IOError(f"{' '.join(cmd[:2])} failed: {err}")


def download(url, dest, retries):
    """Small-file download (restart per attempt; resume unnecessary)."""
    last = None
    for attempt in range(1, retries + 1):
        tmp = dest + ".part"
        try:
            with requests.get(url, stream=True,
                              timeout=TRANSFER_TIMEOUT) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except (requests.RequestException, OSError) as e:
            last = e
            try:
                os.remove(tmp)
            except OSError:
                pass
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise IOError(f"download failed after {retries} attempts: {last}")


def cached_source(gadm_url, cache_dir, workdir, retries):
    """Return a local path to the GADM zip, downloading once per URL.

    With --cache-dir, the source persists across runs and states
    (validated by recorded URL); without it, download to tempdir.
    """
    name = os.path.basename(urllib.parse.urlparse(gadm_url).path)
    if not name:
        name = "gadm_usa.json.zip"
    if not cache_dir:
        dest = os.path.join(workdir, name)
        download(gadm_url, dest, retries)
        return dest
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, name)
    sidecar = dest + ".source.json"
    try:
        with open(sidecar) as f:
            cached_url = json.load(f).get("url")
    except (OSError, ValueError):
        cached_url = None
    if cached_url != gadm_url or not os.path.isfile(dest):
        download(gadm_url, dest, retries)
        tmp = sidecar + ".part"
        with open(tmp, "w") as f:
            json.dump({"url": gadm_url}, f)
        os.replace(tmp, sidecar)
    else:
        print(f"reusing cached GADM source ({name})")
    return dest


def load_manifest(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def main(argv=None):
    args = get_parser().parse_args(argv)
    try:
        abbrev, state_name = resolve_state(args.state)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    # Tier 1: reuse a previously built cutline (no network at all).
    if not args.force:
        manifest = load_manifest(args.manifest)
        if (isinstance(manifest, dict)
                and manifest.get("state_name") == state_name
                and manifest.get("gadm_url") == args.gadm_url
                and os.path.isfile(args.output_zip)
                and os.path.getsize(args.output_zip)
                == manifest.get("zip_bytes")):
            print(f"reusing state cutline for {state_name}; "
                  f"no download needed")
            return 0

    with tempfile.TemporaryDirectory(prefix="acq_roi_") as workdir:
        try:
            gadm_zip = cached_source(args.gadm_url, args.cache_dir,
                                     workdir, args.retries)
        except IOError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        try:
            with zipfile.ZipFile(gadm_zip) as zf:
                names = [n for n in zf.namelist()
                         if n.endswith(".json") or n.endswith(".geojson")]
                if len(names) != 1:
                    print(f"error: expected one GeoJSON in GADM zip, "
                          f"found {names}", file=sys.stderr)
                    return 1
                zf.extract(names[0], workdir)
            geojson = os.path.join(workdir, names[0])
        except zipfile.BadZipFile as e:
            print(f"error: bad GADM zip: {e}", file=sys.stderr)
            return 1

        layer = f"state_{abbrev}"
        where = "NAME_1 = '" + state_name.replace("'", "''") + "'"
        shape_dir = os.path.join(workdir, "shp")
        os.makedirs(shape_dir, exist_ok=True)
        if shutil.which("ogr2ogr") is None:                               
            print(                                                        
                "error: software environment is missing GDAL (ogr2ogr) — " 
                "the container was not used; press Build Image, then Run again",  
                file=sys.stderr)                                         
            return 1  
        try:
            run(["ogr2ogr", "-f", "ESRI Shapefile", "-t_srs", "EPSG:4326",
                 "-where", where, "-nln", layer,
                 os.path.join(shape_dir, layer + ".shp"), geojson])
        except IOError as e:
            print(f"error: state extraction failed: {e}", file=sys.stderr)
            return 1

        shp_path = os.path.join(shape_dir, layer + ".shp")
        if not os.path.isfile(shp_path):
            print(f"error: no GADM feature matched {state_name!r}",
                  file=sys.stderr)
            return 1
        parts = [p for p in os.listdir(shape_dir)
                 if p.startswith(layer + ".")]
        if not any(p.endswith(".shp") for p in parts):
            print(f"error: shapefile incomplete for {state_name!r}",
                  file=sys.stderr)
            return 1

        tmp_zip = args.output_zip + ".part"
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for part in sorted(parts):
                zf.write(os.path.join(shape_dir, part), arcname=part)
        os.replace(tmp_zip, args.output_zip)

    manifest = {
        "gadm_url": args.gadm_url,
        "state": abbrev,
        "state_name": state_name,
        "epsg": 4326,
        "zip_bytes": os.path.getsize(args.output_zip),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = args.manifest + ".part"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, args.manifest)

    print(f"state cutline for {state_name} -> {args.output_zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())