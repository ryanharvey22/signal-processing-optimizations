#!/usr/bin/env python3
"""Download only the public RadChar Tiny file; never redistribute it in Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/abcxyzi/radchar-icassp-2023/RadChar-Tiny.h5"
MAX_BYTES = 512 * 1024 * 1024

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/radchar/RadChar-Tiny.h5"))
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise SystemExit(f"Already exists: {out}; refusing to overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".download")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "signal-processing-optimizations/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response, temporary.open("wb") as dst:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("Tiny download exceeded 512 MiB bound")
                dst.write(chunk)
                if total % (32 * 1024 * 1024) == 0:
                    print(f"Downloaded {total // (1024 * 1024)} MiB", flush=True)
        if zipfile.is_zipfile(temporary):
            with zipfile.ZipFile(temporary) as archive:
                entries = [x for x in archive.infolist() if Path(x.filename).name == out.name]
                if len(entries) != 1 or entries[0].file_size > MAX_BYTES:
                    raise ValueError("Archive does not contain exactly one bounded Tiny HDF5")
                # Copy the selected entry; never trust archive paths for filesystem writes.
                with archive.open(entries[0]) as src, out.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
        else:
            with temporary.open("rb") as stream:
                if stream.read(8) != b"\x89HDF\r\n\x1a\n":
                    raise ValueError("Server returned neither HDF5 nor ZIP; authentication may be required")
            os.replace(temporary, out)
        with out.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {"source_url": URL, "sha256": digest, "bytes": out.stat().st_size,
                    "redistribution": "not included; consult the upstream dataset terms"}
        out.with_suffix(".h5.source.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)
    finally:
        temporary.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
