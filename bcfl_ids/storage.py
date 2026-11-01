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
