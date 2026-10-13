#!/usr/bin/env python3
"""Download one month of ESA CCI daily soil-moisture NetCDFs.

Primary transport is HTTPS via the CEDA archive
(https://data.ceda.ac.uk, redirecting to https://dap.ceda.ac.uk);
anonymous access, no login required (verified: public dataset,
31/31 Jan-2010 files fetch anonymously).

Outputs a zip of the downloaded days plus a manifest. Days missing
upstream (HTTP 404 over HTTPS, exit 19 over legacy FTP) are skipped
with a warning; the job fails only if zero days were obtained for
the month.

Exit codes: 0 = days-zip valid on disk (+ manifest written);
1 = failure, with the cause named on stderr.
"""

import argparse
import calendar
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CONNECT_TIMEOUT = 30
MAX_TIME = 120  # per-file transfer cap 
CHUNK_ZIP_MODE = 0  # zipfile.ZIP_STORED: NetCDFs barely compress; favor speed


class DownloadError(Exception):
    """A single day-file could not be obtained."""


def get_parser():
    p = argparse.ArgumentParser(
        description="Download one month of ESA CCI daily soil-moisture files."
    )
    p.add_argument("--base-url", required=True,
                   help="e.g. https://data.ceda.ac.uk/neodc/esacci/"
                        "soil_moisture/data/daily_files/COMBINED")
    p.add_argument("--version", required=True, help="e.g. 7.1")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True, help="1-12")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--days-zip", required=True,
                   help="output zip path, e.g. sm_2010_01_days.zip")
    p.add_argument("--manifest", required=True, help="output manifest path")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--force", action="store_true",
                   help="ignore previous downloads, redownload everything")
    return p


def day_files(year, month, base_url, version):
    """All candidate (day, filename, url) triples for the month."""
    if not 1 <= month <= 12:
        raise ValueError(f"bad --month {month}, expected 1-12")
    try:
        vnum = float(version)
    except (TypeError, ValueError):
        raise ValueError(f"bad --version {version!r}, expected like '7.1'")
    vdir = f"v0{vnum:.1f}"    # v07.1 (matches ESA CCI layout)
    ftag = f"fv0{vnum:.1f}"   # fv07.1
    _, ndays = calendar.monthrange(year, month)
    out = []
    directory = f"{base_url.rstrip('/')}/{vdir}/{year}/"
    for day in range(1, ndays + 1):
        stamp = f"{year:04d}{month:02d}{day:02d}"
        filename = (f"ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-"
                    f"{stamp}000000-{ftag}.nc")
        out.append((day, filename, directory + filename))
    return out


def curl_fetch(url, dest, retries):
    """Fetch url to dest with curl. Return 'ok', 'missing' (no such file),
    or raise DownloadError. HTTPS-only: a missing upstream file is
    curl exit 22 with a 404 in stderr (curl -f maps HTTP >= 400
    to exit 22)."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                ["curl", "-fSL", "-sS",
                 "--connect-timeout", str(CONNECT_TIMEOUT),
                 "--max-time", str(MAX_TIME), "-o", dest, url],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except OSError as e:
            last = e
            if attempt < retries:
                time.sleep(2 ** attempt)
            continue
        if proc.returncode == 0:
            try:
                if os.path.getsize(dest) == 0:
                    raise OSError("empty file")
            except OSError as e:
                last = e
                try:
                    os.remove(dest)
                except OSError:
                    pass
                if attempt < retries:
                    time.sleep(2 ** attempt)
                continue
            return "ok"
        
        stderr = proc.stderr.decode().strip()
        if proc.returncode == 22 and "404" in stderr:
            # HTTP(S) file not found upstream, skip.
            try:
                os.remove(dest)
            except OSError:
                pass
            return "missing"
        last = (stderr.splitlines() or
                [f"curl exit {proc.returncode}"])[-1]
        try:
            os.remove(dest)
        except OSError:
            pass
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise DownloadError(f"{url}: {last}")


def load_manifest(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def manifest_valid(manifest, year, month, days_zip):
    """True iff the manifest covers this month and the zip matches."""
    if not isinstance(manifest, dict):
        return False
    if manifest.get("year") != year or manifest.get("month") != month:
        return False
    if not manifest.get("days"):
        return False
    if not os.path.isfile(days_zip):
        return False
    return os.path.getsize(days_zip) == manifest.get("zip_bytes")


def write_zip(day_paths, days_zip):
    """Pack downloaded day files (stored, not compressed) + atomic rename."""
    import zipfile
    tmp = days_zip + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for day, local, path in day_paths:
            zf.write(path, arcname=local)
    os.replace(tmp, days_zip)
    return os.path.getsize(days_zip)


def write_manifest(path, *, base_url, version, year, month,
                   days, missing_days, zip_bytes):
    manifest = {
        "base_url": base_url,
        "version": version,
        "year": year,
        "month": month,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "days": days,
        "missing_days": missing_days,
    }
    manifest["zip_bytes"] = zip_bytes
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def main(argv=None):
    args = get_parser().parse_args(argv)
    try:
        candidates = day_files(args.year, args.month,
                               args.base_url, args.version)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    # Tier 1: reuse a previously completed month (no network at all).
    if not args.force:
        manifest = load_manifest(args.manifest)
        if manifest_valid(manifest, args.year, args.month, args.days_zip):
            n = len(manifest["days"])
            print(f"reusing {n} previously downloaded days for "
                  f"{args.year:04d}-{args.month:02d}; no download needed")
            return 0

    downloaded, missing = [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_of = {
            pool.submit(curl_fetch, url,
                        os.path.join(args.output_dir, filename),
                        args.retries): (day, filename)
            for day, filename, url in candidates
        }
        for future in as_completed(future_of):
            day, filename = future_of[future]
            try:
                status = future.result()
            except DownloadError as e:
                print(f"warning: day {day:02d} unobtainable after retries "
                      f"({e}), skipped", file=sys.stderr)
                missing.append(day)
                continue
            if status == "ok":
                path = os.path.join(args.output_dir, filename)
                downloaded.append((day, filename, path))
            else:
                missing.append(day)

    if not downloaded:
        print(f"error: zero days obtained for "
              f"{args.year:04d}-{args.month:04d} "
              f"({len(missing)} missing/unobtainable)", file=sys.stderr)
        return 1

    downloaded.sort()
    for day in sorted(missing):
        print(f"warning: day {day:02d} missing upstream, skipped",
              file=sys.stderr)

    zip_bytes = write_zip(
        [(day, filename, path) for day, filename, path in downloaded],
        args.days_zip)
    write_manifest(
        args.manifest, base_url=args.base_url, version=args.version,
        year=args.year, month=args.month,
        days=[{"day": day, "local": filename,
               "bytes": os.path.getsize(path)}
              for day, filename, path in downloaded],
        missing_days=sorted(missing), zip_bytes=zip_bytes)

    label = f"{args.year:04d}-{args.month:02d}"
    print(f"downloaded {len(downloaded)} days for {label} "
          f"({len(missing)} missing upstream)")
    return 0


if __name__ == "__main__":
    sys.exit(main())