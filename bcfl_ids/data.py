"""N-BaIoT pipeline: unpack the archive, convert CSVs to per-device arrays, and load client data.

Run as `python -m bcfl_ids.data` once; only the demo bundle in artifacts/demo is needed afterwards.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from .config import CLASSES, DEVICE_IDS, DEVICES, NUM_FEATURES, PROCESSED_DIR, RAW_DIR, SPLITS
from .simulator import ClientData

# Published per-device row counts, checked after conversion: (benign, attack).
EXPECTED_ROWS = {
    1: (49548, 968750), 2: (13113, 822763), 3: (39100, 316400),
    4: (175240, 923437), 5: (62154, 766106), 6: (98514, 738377),
    7: (52150, 323072), 8: (46585, 816471), 9: (19528, 831298),
}


def signed_log(x: np.ndarray) -> np.ndarray:
    """x' = sign(x) * ln(1 + |x|). The features are heavy-tailed and some can be negative."""
    return np.sign(x) * np.log1p(np.abs(x))


def _find_tar() -> str:
    """A bsdtar that can read RAR: built into Windows 10+ and macOS. GNU tar cannot."""
    candidates = [r"C:\Windows\System32\tar.exe", "bsdtar", "tar"]
    for tool in candidates:
        if shutil.which(tool) or Path(tool).exists():
            out = subprocess.run([tool, "--version"], capture_output=True, text=True).stdout
            if "bsdtar" in out or "libarchive" in out:
                return tool
    raise RuntimeError("No bsdtar found, needed to unpack the RAR files in N-BaIoT.")


def unpack(zip_path: Path, dest: Path) -> None:
    """Extract the archive and every per-device RAR. Member paths are validated by the libraries."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    tar = _find_tar()
    for rar in sorted(dest.rglob("*.rar")):
        out = rar.parent / rar.stem
        out.mkdir(exist_ok=True)
        print("unpacking", rar.relative_to(dest), flush=True)
        subprocess.run([tar, "-xf", str(rar), "-C", str(out)], check=True)


def _class_of(path: Path) -> int | None:
    """Map a CSV path such as Danmini/gafgyt_attacks/tcp.csv to its class index."""
    parts = [p.lower() for p in path.parts]
    stem = path.stem.lower().split(".")[-1]
    if stem == "benign_traffic" or "benign" in stem:
        return CLASSES.index("benign")
    for family in ("gafgyt", "mirai"):
        if any(family in p for p in parts):
            label = f"{family}_{stem}"
            if label in CLASSES:
                return CLASSES.index(label)
    return None


def _device_dir(root: Path, folder: str) -> Path:
    matches = [p for p in root.rglob(folder) if p.is_dir()]
    if not matches:
        # Some mirrors keep all files flat and prefix each with the device number.
        raise FileNotFoundError(f"Device folder {folder} not found under {root}")
    return matches[0]


def convert(root: Path, out_dir: Path = PROCESSED_DIR) -> None:
    """Write X_<split>.npy and y_<split>.npy for each device, reading one file at a time."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] | None = None
    report = {}
    for device in DEVICE_IDS:
        folder = _device_dir(root, DEVICES[device][1])
        parts = {s: ([], []) for s in SPLITS}
        counts = {}
        for csv in sorted(folder.rglob("*.csv")):
            cls = _class_of(csv.relative_to(folder))
            if cls is None:
                print("skipping unknown file", csv, flush=True)
                continue
            frame = pd.read_csv(csv, dtype=np.float32)
            if frame.shape[1] != NUM_FEATURES:
                raise ValueError(f"{csv} has {frame.shape[1]} columns")
            names = names or list(frame.columns)
            if list(frame.columns) != names:
                raise ValueError(f"{csv} has a different column order")
            x = signed_log(frame.to_numpy(dtype=np.float32))
            del frame
            n = len(x)
            counts[CLASSES[cls]] = n
            for split, (a, b) in SPLITS.items():
                piece = x[int(n * a) : int(n * b)]
                parts[split][0].append(piece)
                parts[split][1].append(np.full(len(piece), cls, dtype=np.uint8))
            print(f"device {device} {CLASSES[cls]:16s} {n}", flush=True)
        dev_dir = out_dir / f"device_{device}"
        dev_dir.mkdir(exist_ok=True)
        for split, (xs, ys) in parts.items():
            np.save(dev_dir / f"X_{split}.npy", np.concatenate(xs))
            np.save(dev_dir / f"y_{split}.npy", np.concatenate(ys))
        report[str(device)] = counts
        benign, attack = EXPECTED_ROWS[device]
        got_b, got_a = counts.get("benign", 0), sum(v for k, v in counts.items() if k != "benign")
        if (got_b, got_a) != (benign, attack):
            raise ValueError(f"device {device}: expected {(benign, attack)}, got {(got_b, got_a)}")
    (out_dir / "features.json").write_text(json.dumps(names))
    (out_dir / "counts.json").write_text(json.dumps(report, indent=1))
    print("row counts match the published table", flush=True)


def feature_names(base: Path = PROCESSED_DIR) -> list[str]:
    return json.loads((base / "features.json").read_text())


def _stratified(y: np.ndarray, frac: float, cap: int | None, rng: np.random.Generator) -> np.ndarray:
    """Pick rows per class, keeping chronological order within the sample."""
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        k = max(1, int(round(len(idx) * frac)))
        if cap:
            k = min(k, cap)
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def load_clients(frac: float = 0.10, cap: int | None = None, seed: int = 0, base: Path = PROCESSED_DIR, split: str = "test", test_frac: float = 1.0) -> list[ClientData]:
    """Training rows are sampled into RAM; evaluation rows stay memory-mapped unless test_frac < 1."""
    clients = []
    for device in DEVICE_IDS:
        d = base / f"device_{device}"
        rng = np.random.default_rng(seed + device)
        y_tr = np.load(d / "y_train.npy", mmap_mode="r")
        idx = _stratified(np.asarray(y_tr), frac, cap, rng)
        x_tr = np.asarray(np.load(d / "X_train.npy", mmap_mode="r")[idx])
        x_te = np.load(d / f"X_{split}.npy", mmap_mode="r")
        y_te = np.load(d / f"y_{split}.npy", mmap_mode="r")
        if test_frac < 1.0:
            sel = _stratified(np.asarray(y_te), test_frac, None, rng)
            x_te, y_te = np.asarray(x_te[sel]), np.asarray(y_te[sel])
        clients.append(ClientData(device, x_tr, np.asarray(y_tr)[idx], x_te, y_te))
    return clients


if __name__ == "__main__":
    archive = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DIR / "nbaiot.zip"
    target = RAW_DIR / "unpacked"
    if not any(target.rglob("*.csv")) if target.exists() else True:
        unpack(archive, target)
    convert(target)
