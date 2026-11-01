#!/usr/bin/env python3
"""Download the IPFS node (Kubo) into tools/ipfs and check it against a pinned hash.

Runs during setup. If it fails, setup carries on: the demo then stores files in a local folder instead,
and the dashboard says so.
"""
from __future__ import annotations

import hashlib
import io
import platform
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "v0.32.1"
BASE = f"https://github.com/ipfs/kubo/releases/download/{VERSION}"
SHA512 = {
    "windows-amd64.zip": "0a78a139fcbd1d04b4d1e5bba05271228add9d5fc74bc311c3551bccf11d76af05112cd8dbf53c18b36d817328eec397f609a525cf33d339a3c685dd5f58652d",
    "darwin-arm64.tar.gz": "fb14edf24a6f42a664af180ed07659d47866ed7a4d023e5be40322526daed8ba792bb337cb51502f2cf51fa9247ca53b0dc07b57b94ca3002fdf3adb90a3b0de",
    "darwin-amd64.tar.gz": "4ee48be2b9ae678eb772cc276845b5972372595ffc87185d75ea30c08366434347277a1093b28383cf4c42b57c6224b7a2f7173108c45d6398105192e145b0ee",
    "linux-amd64.tar.gz": "36b2951de759710ad81f63f3ec3a75c72323be4c6599e737d7f27b437dce02ed0a1662d676137e98f2e88f0b96e41c4b46b2e1667d874f90e3a7ee437912fda4",
    "linux-arm64.tar.gz": "2f7b8589f5a1d715ed853766b0eed206be5bbdc9fed6df9e6b350546336cf9d8395e8f5b167ae05d073ab032ca9bfaa441d022b95b1ebb5bc52413276a22c6c2",
}
TOOLS = Path(__file__).resolve().parent.parent / "tools" / "ipfs"


def package() -> str:
    system = {"Windows": "windows", "Darwin": "darwin", "Linux": "linux"}.get(platform.system(), "")
    machine = {"amd64": "amd64", "x86_64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine().lower(), "")
    return f"{system}-{machine}.{'zip' if system == 'windows' else 'tar.gz'}"


def fetch(url: str) -> bytes:
    context = ssl.create_default_context()
    try:  # python.org builds on macOS ship without system certificates.
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(url, timeout=120, context=context) as response:
        return response.read()


def extract(name: str, blob: bytes) -> bytes:
    """Read only the node's binary, so no path in the archive is ever written to disk."""
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            return archive.read("kubo/ipfs.exe")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        return archive.extractfile("kubo/ipfs").read()


def main() -> int:
    target = TOOLS / ("ipfs.exe" if platform.system() == "Windows" else "ipfs")
    if target.exists():
        print(f"IPFS node already installed: {target}")
        return 0
    name = package()
    if name not in SHA512:
        print(f"No IPFS build for this computer ({platform.system()} {platform.machine()}). The demo will use a local folder.")
        return 0
    try:
        print(f"Downloading IPFS {VERSION} ({name})")
        blob = fetch(f"{BASE}/kubo_{VERSION}_{name}")
        if hashlib.sha512(blob).hexdigest() != SHA512[name]:
            print("The download does not match the pinned hash, so it was not installed. The demo will use a local folder.")
            return 0
        TOOLS.mkdir(parents=True, exist_ok=True)
        target.write_bytes(extract(name, blob))
        target.chmod(0o755)
        print(f"IPFS node installed: {target}")
    except Exception as exc:  # A missing node must not stop the rest of setup.
        print(f"Could not install IPFS ({exc}). The demo will use a local folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
