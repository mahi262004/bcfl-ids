"""Content-addressed store for model updates and global models.

Files go to a private IPFS node when one is running, and to a local folder otherwise. Both use the
same address. Each file is under IPFS's 256 KiB block size and is added as one raw block, so its IPFS
address is just the SHA-256 already on the chain, re-encoded. That is why the contract does not change.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_IPFS_API = "http://127.0.0.1:5901"
_CID_PREFIX = bytes([0x01, 0x55, 0x12, 0x20])  # CIDv1, raw codec, sha2-256, 32 bytes
_RETRY_AFTER = 15.0  # seconds to wait before trying a node that has just failed


class StoreError(Exception):
    pass


class FileMissing(StoreError):
    pass


class FileMismatch(StoreError):
    pass


def cid_from_digest(digest: bytes) -> str:
    return "b" + base64.b32encode(_CID_PREFIX + digest).decode().rstrip("=").lower()


def cid_from_hash(hex_hash: str) -> str:
    """The IPFS address of the file whose SHA-256 is recorded on the chain."""
    return cid_from_digest(bytes.fromhex(hex_hash.removeprefix("0x")))


def cid_for(data: bytes) -> str:
    return cid_from_digest(hashlib.sha256(data).digest())


def digest_of(cid: str) -> bytes:
    raw = base64.b32decode(cid[1:].upper() + "=" * (-len(cid[1:]) % 8))
    if not cid.startswith("b") or raw[:4] != _CID_PREFIX or len(raw) != 36:
        raise StoreError(f"Not a supported address: {cid}")
    return raw[4:]


class IpfsClient:
    """The few calls of Kubo's HTTP API that the store needs."""

    def __init__(self, api_url: str = DEFAULT_IPFS_API) -> None:
        self.api_url = api_url.rstrip("/")

    def _post(self, path: str, data: bytes = b"", headers: dict | None = None, timeout: float = 5.0) -> bytes:
        request = urllib.request.Request(f"{self.api_url}/api/v0/{path}", data=data, headers=headers or {}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def version(self, timeout: float = 1.0) -> str:
        return json.loads(self._post("version", timeout=timeout))["Version"]

    def add(self, data: bytes) -> str:
        boundary = "bcfl-ids-update-file"
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="update"\r\nContent-Type: application/octet-stream\r\n\r\n').encode() + data + f"\r\n--{boundary}--\r\n".encode()
        reply = self._post("add?cid-version=1&raw-leaves=true&pin=false&quieter=true", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return json.loads(reply)["Hash"]

    def cat(self, cid: str) -> bytes:
        return self._post(f"cat?arg={cid}")


class UpdateStore:
    """Puts files on IPFS, or in a local folder if the node is down. Every read checks the hash."""

    def __init__(self, local_dir: Path, ipfs: IpfsClient | None = None) -> None:
        self.local_dir = Path(local_dir)
        self.ipfs = ipfs
        self.ipfs_files = 0
        self.local_files = 0
        self.swapped: dict[str, bytes] = {}
        self._skip_ipfs_until = 0.0

    def _ipfs_usable(self) -> bool:
        return self.ipfs is not None and time.time() >= self._skip_ipfs_until

    def _ipfs_failed(self) -> None:
        self._skip_ipfs_until = time.time() + _RETRY_AFTER

    def put(self, data: bytes) -> dict:
        cid = cid_for(data)
        if self._ipfs_usable():
            try:
                if self.ipfs.add(data) != cid:
                    raise StoreError("the IPFS node returned an unexpected address")
                self.ipfs_files += 1
                return {"cid": cid, "where": "ipfs", "size": len(data)}
            except (OSError, ValueError, KeyError, StoreError):
                self._ipfs_failed()
        self.local_dir.mkdir(parents=True, exist_ok=True)
        path = self.local_dir / cid
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        self.local_files += 1
        return {"cid": cid, "where": "local", "size": len(data)}

    def _read(self, cid: str) -> tuple[bytes, str]:
        if cid in self.swapped:
            return self.swapped[cid], "swapped"
        path = self.local_dir / cid
        if path.exists():
            return path.read_bytes(), "local"
        if self._ipfs_usable():
            try:
                return self.ipfs.cat(cid), "ipfs"
            except urllib.error.HTTPError as exc:
                raise FileMissing(f"The IPFS node does not hold {cid}") from exc
            except OSError:
                self._ipfs_failed()
        raise FileMissing(f"No stored file for {cid}")

    def get(self, cid: str) -> bytes:
        return self.read(cid)[0]

    def read(self, cid: str) -> tuple[bytes, str]:
        data, where = self._read(cid)
        if hashlib.sha256(data).digest() != digest_of(cid):
            raise FileMismatch(f"The file served for {cid} does not match its on-chain hash")
        return data, where

    def status(self) -> dict:
        version = None
        if self.ipfs is not None:
            try:
                version = self.ipfs.version()
            except (OSError, ValueError, KeyError):
                pass
        return {"connected": version is not None, "version": version, "api_url": self.ipfs.api_url if self.ipfs else None, "ipfs_files": self.ipfs_files, "local_files": self.local_files}
