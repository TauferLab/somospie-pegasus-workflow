#!/usr/bin/env python3
"""Download terrain inputs for a state/resolution via a presigned-URL
service, with per-file Dataverse fallback and skip-if-already-downloaded.

The URL service returns short-lived download URLs; 
Dataverse links are plain public HTTPS.

Exit codes: 0 = all wanted files valid on disk;
1 = failure, with the file(s) and cause named on stderr.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time

import requests

SERVICE_TIMEOUT = 60
TRANSFER_TIMEOUT = 300
CHUNK = 1024 * 1024
PLACEHOLDER = "REPLACE_ME" # dataverse sentinel


class ServiceError(Exception):
    """The gen-url call failed (after retries)."""


class DownloadError(Exception):
    """A single file download failed (after retries)."""


class FileFailed(Exception):
    """A wanted file could be obtained from no source."""

    def __init__(self, name, tried):
        super().__init__(f"{name}: " + "; ".join(tried))
        self.name = name
        self.tried = tried


def get_parser():
    p = argparse.ArgumentParser(
        description="Download terrain inputs via presigned-URL service "
                    "(Dataverse fallback, skip-if-downloaded)."
    )
    p.add_argument("--service-url", required=True,
                   help="e.g. https://intersect.nationalsciencedatafabric.org/somospie")
    p.add_argument("--state", required=True, help="e.g. TN")
    p.add_argument("--resolution", required=True, help="e.g. 30m")
    p.add_argument("--want", action="append", required=True, metavar="NAME=FILE",
                   help="wanted file as NAME=filename (repeatable); "
                        "e.g. --want aspect=aspect.tif")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--manifest", required=True, help="output manifest.json path")
    p.add_argument("--fallback-map", default="{}",
                   help="JSON object name->Dataverse URL (embedded by generator "
                        "from dataverse.json[resolution][state])")
    p.add_argument("--no-fallback", action="store_true",
                   help="disable the Dataverse path entirely")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--force", action="store_true",
                   help="ignore previous downloads, redownload everything")
    return p


def parse_wants(raw):
    """Parse [--want NAME=FILE ...] into an ordered [(name, filename)] list."""
    wants = []
    seen = set()
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"bad --want {item!r}, expected NAME=FILE")
        name, filename = item.split("=", 1)
        name, filename = name.strip(), filename.strip()
        if not name or not filename:
            raise ValueError(f"bad --want {item!r}, expected NAME=FILE")
        if name in seen:
            raise ValueError(f"duplicate --want name {name!r}")
        seen.add(name)
        wants.append((name, filename))
    if not wants:
        raise ValueError("at least one --want NAME=FILE is required")
    return wants


def load_manifest(path):
    """Return the previous manifest dict, or None if missing/unreadable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def manifest_valid(manifest, wants, output_dir):
    """True iff the manifest covers exactly this request and every file
    exists on disk with its recorded byte size."""
    if not isinstance(manifest, dict):
        return False
    try:
        recorded = {f["name"]: f["bytes"] for f in manifest.get("files", [])}
    except (TypeError, KeyError, AttributeError):
        return False
    if set(recorded) != {name for name, _ in wants}:
        return False
    for name, filename in wants:
        local = os.path.join(output_dir, os.path.basename(filename))
        if not os.path.isfile(local):
            return False
        if os.path.getsize(local) != recorded[name]:
            return False
    return True


def service_urls(service_url, state, resolution, retries):
    """GET gen-url?state=&resolution=; return [{key, url}]."""
    endpoint = service_url.rstrip("/") + "/api/v1/gen-url"
    params = {"state": state, "resolution": resolution}
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(endpoint, params=params, timeout=SERVICE_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            urls = data.get("urls")
            if not isinstance(urls, list):
                raise ServiceError(f"unexpected response shape: {str(data)[:200]}")
            out = []
            for item in urls:
                if not isinstance(item, dict) or "key" not in item or "url" not in item:
                    raise ServiceError(f"unexpected response shape: {str(data)[:200]}")
                out.append({"key": item["key"], "url": item["url"]})
            return out
        except (requests.RequestException, ValueError, ServiceError) as e:
            last = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise ServiceError(f"gen-url failed after {retries} attempts: {last}")


def _transfer(resp, tmp, have, mode):
    """Write the response body to tmp (append when resuming).

    Returns the total byte count. Content-Length, when present, counts
    only the bytes in this response (remaining bytes under 206).
    """
    expected = resp.headers.get("Content-Length")
    expected = int(expected) if expected is not None else None
    nbytes = 0
    with open(tmp, mode) as f:
        for chunk in resp.iter_content(chunk_size=CHUNK):
            if chunk:
                f.write(chunk)
                nbytes += len(chunk)
    if expected is not None and nbytes != expected:
        raise IOError(f"size mismatch: got {nbytes} new bytes, "
                      f"expected {expected}")
    return have + nbytes


def download_one(url, dest, retries):
    """Stream url to dest, resuming dest.part via Range when present.

    Returns (total bytes, 1-based attempt that succeeded). A stalled
    transfer aborts on the inactivity timeout and resumes on retry.
    """
    last = None
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
            headers = {"Range": f"bytes={have}-"} if have > 0 else {}
            with requests.get(url, stream=True, timeout=TRANSFER_TIMEOUT,
                              headers=headers) as resp:
                if resp.status_code == 416:
                    # Remote changed (or part invalid): restart cleanly.
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    raise DownloadError("stale partial file (HTTP 416)")
                resp.raise_for_status()
                if have > 0 and resp.status_code == 206:
                    mode, base = "ab", have
                else:
                    mode, base = "wb", 0  # fresh or Range ignored: restart
                total = _transfer(resp, tmp, base, mode)
            os.replace(tmp, dest)
            return total, attempt
        except (requests.RequestException, OSError, DownloadError) as e:
            last = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise DownloadError(f"failed after {retries} attempts: {last}")


def valid_fallback_url(raw):
    """A usable fallback entry: non-empty http(s) URL, not a placeholder."""
    return (isinstance(raw, str) and raw.strip()
            and raw.strip() != PLACEHOLDER
            and raw.strip().startswith("http"))


def fetch_file(name, filename, primary, fallback_url, fallback_note,
               output_dir, retries):
    """Obtain one file: primary URL, then fallback URL. Return its record."""
    dest = os.path.join(output_dir, os.path.basename(filename))
    tried = []
    if primary is not None:
        try:
            nbytes, attempts = download_one(primary["url"], dest, retries)
            return {"name": name, "key": primary["key"],
                    "local": os.path.basename(filename), "bytes": nbytes,
                    "source": "url-service", "attempts": attempts}
        except DownloadError as e:
            tried.append(f"url-service: {e}")
    if fallback_url is not None:
        try:
            nbytes, attempts = download_one(fallback_url, dest, retries)
            return {"name": name, "key": filename,
                    "local": os.path.basename(filename), "bytes": nbytes,
                    "source": "dataverse", "attempts": attempts}
        except DownloadError as e:
            tried.append(f"dataverse: {e}")
    else:
        tried.append(f"dataverse: unavailable ({fallback_note})")
    raise FileFailed(name, tried)


def write_manifest(path, *, service_url, state, resolution, fallback_enabled,
                   records, skipped):
    manifest = {
        "service_url": service_url,
        "state": state,
        "resolution": resolution,
        "fallback_enabled": fallback_enabled,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": records,
        "skipped": sorted(skipped),
    }
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def main(argv=None):
    args = get_parser().parse_args(argv)

    try:
        wants = parse_wants(args.want)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        fallback_map = json.loads(args.fallback_map or "{}")
        if not isinstance(fallback_map, dict):
            raise ValueError("must be a JSON object")
    except ValueError as e:
        print(f"error: bad --fallback-map: {e}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    # Tier 1: reuse previously downloaded data (no HTTP at all).
    if not args.force:
        manifest = load_manifest(args.manifest)
        if manifest_valid(manifest, wants, args.output_dir):
            names = ", ".join(name for name, _ in wants)
            print(f"reusing {len(wants)} previously downloaded files "
                  f"({names}); no download needed")
            return 0

    # Primary source: presigned URLs for the whole state/resolution.
    primary = {}
    service_keys = []
    try:
        urls = service_urls(args.service_url, args.state, args.resolution,
                            args.retries)
        for item in urls:
            service_keys.append(os.path.basename(item["key"]))
            primary.setdefault(os.path.basename(item["key"]), item)
    except ServiceError as e:
        print(f"warning: URL service failed ({e}); "
              f"using Dataverse fallback", file=sys.stderr)

    # Per-file plan: primary URL if listed, fallback URL if configured.
    tasks = []
    for name, filename in wants:
        base = os.path.basename(filename)
        item = primary.get(base)
        fallback_url, fallback_note = None, "disabled" \
            if args.no_fallback else "no entry"
        if not args.no_fallback:
            raw = fallback_map.get(name)
            if valid_fallback_url(raw):
                fallback_url = raw.strip()
            elif isinstance(raw, str) and raw.strip():
                fallback_note = "placeholder entry"
        tasks.append((name, filename, item, fallback_url, fallback_note))

    results = {}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers)) as pool:
        future_of = {
            pool.submit(fetch_file, name, filename, item, furl, fnote,
                        args.output_dir, args.retries): name
            for name, filename, item, furl, fnote in tasks
        }
        for future in concurrent.futures.as_completed(future_of):
            try:
                record = future.result()
            except FileFailed as e:
                failures.append(str(e))
            else:
                results[record["name"]] = record

    if failures:
        for message in sorted(failures):
            print(f"error: {message}", file=sys.stderr)
        return 1

    records = [results[name] for name, _ in wants]
    wanted_basenames = {os.path.basename(fn) for _, fn in wants}
    skipped = [k for k in service_keys if k not in wanted_basenames]
    write_manifest(args.manifest, service_url=args.service_url,
                   state=args.state, resolution=args.resolution,
                   fallback_enabled=not args.no_fallback,
                   records=records, skipped=skipped)

    counts = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    breakdown = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"downloaded {len(records)} files ({breakdown}); "
          f"skipped {len(skipped)} unrequested")
    return 0


if __name__ == "__main__":
    sys.exit(main())