"""Fetch only the tabular CGMacros files from the public PhysioNet ZIP.

The source archive is about 627 MB because it contains meal photographs.  The
PhysioNet endpoint supports HTTP byte ranges, so this script exposes a small
seekable range reader to :mod:`zipfile` and extracts only ``bio.csv`` and the
per-participant ``CGMacros-*.csv`` files.  Raw source files are written under
``output/`` (git-ignored); no repository data or source archive is modified.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import urllib.request
import zlib
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "cgmacros_subset"
URL = "https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
EXPECTED_SUBJECT_IDS = set(range(1, 24)) | set(range(26, 37)) | {38, 39} | set(range(41, 50))


class HTTPRangeReader(io.RawIOBase):
    """Minimal seekable reader backed by exact HTTP Range requests."""

    def __init__(self, url: str):
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.length = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
            self.last_modified = response.headers.get("Last-Modified")
            accepts_ranges = response.headers.get("Accept-Ranges", "").lower()
        if "bytes" not in accepts_ranges:
            raise RuntimeError("CGMacros endpoint does not advertise byte ranges")
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.length + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self.position = min(int(target), self.length)
        return self.position

    def _fetch(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "GlucoBench/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError(f"range request returned HTTP {response.status}")
            return response.read()

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        size = min(int(size), self.length - self.position)
        if size == 0:
            return b""

        start = self.position
        end = start + size
        # Exact ranges are intentional: any read-ahead can cross into an
        # adjacent photograph entry even though only CSV members are extracted.
        result = self._fetch(start, end - 1)
        if len(result) != size:
            raise IOError(f"short range read: expected {size}, received {len(result)}")
        self.position += size
        return result


def safe_target(base: Path, relative: Path) -> Path:
    target = (base / relative).resolve()
    base_resolved = base.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError(f"unsafe archive path: {relative}")
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xFFFFFFFF


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects_dir = OUTPUT_DIR / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)

    reader = HTTPRangeReader(URL)
    extracted: list[dict[str, object]] = []
    with ZipFile(reader) as archive:
        selected = []
        for info in archive.infolist():
            basename = Path(info.filename).name
            lower = basename.lower()
            if lower == "bio.csv" or re.fullmatch(r"cgmacros-\d+\.csv", lower):
                selected.append(info)
        if not any(Path(info.filename).name.lower() == "bio.csv" for info in selected):
            raise RuntimeError("bio.csv was not found in the remote archive")
        subject_infos = [
            info for info in selected
            if re.fullmatch(r"cgmacros-\d+\.csv", Path(info.filename).name.lower())
        ]
        subject_ids = {
            int(re.search(r"(\d+)", Path(info.filename).name).group(1))
            for info in subject_infos
        }
        if len(subject_infos) != 45 or subject_ids != EXPECTED_SUBJECT_IDS:
            raise RuntimeError(
                f"participant CSV inventory mismatch: count={len(subject_infos)}, ids={sorted(subject_ids)}"
            )

        for info in sorted(selected, key=lambda item: item.filename.lower()):
            basename = Path(info.filename).name
            relative = Path("bio.csv") if basename.lower() == "bio.csv" else Path("subjects") / basename
            target = safe_target(OUTPUT_DIR, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            valid_existing = (
                target.exists()
                and target.stat().st_size == info.file_size
                and crc32(target) == info.CRC
            )
            if not valid_existing:
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            extracted.append({
                "archive_member": info.filename,
                "path": str(target.relative_to(ROOT)),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            })

    manifest = {
        "source_url": URL,
        "source_content_length": reader.length,
        "source_etag": reader.etag,
        "source_last_modified": reader.last_modified,
        "range_mode": "exact member and central-directory byte ranges; no read-ahead",
        "extracted_count": len(extracted),
        "files": extracted,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "extracted_count": len(extracted),
        "total_extracted_bytes": sum(int(row["bytes"]) for row in extracted),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
