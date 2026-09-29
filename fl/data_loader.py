"""
Loads the real N-BaIoT dataset (Meidan et al., 2018), respecting its
actual per-device file structure -- this replaces a generic "one big
CSV" assumption with the dataset's real layout.

Download the dataset from:
    https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot
Extract it into data/raw/ so each device sits in its own folder, e.g.:
    data/raw/Danmini_Doorbell/benign_traffic.csv
    data/raw/Danmini_Doorbell/gafgyt_attacks.rar
    data/raw/Danmini_Doorbell/mirai_attacks.rar
    ...
This matches what the UCI download actually gives you -- per-device
folders, with the attack CSVs bundled inside .rar archives rather
than sitting as individual files. This module extracts those
archives itself (via the OS's tar, which reads .rar) the first time
a device is loaded, caching the result in a sibling folder next to
each archive so later runs don't re-extract.

Dataset facts (confirmed from Meidan et al. and follow-up literature):
    - 9 real IoT devices, one folder each under data/raw/
    - Every device has benign_traffic.csv + gafgyt_attacks.rar
      (5 attack CSVs inside: combo, junk, scan, tcp, udp)
    - 7 of the 9 devices additionally have mirai_attacks.rar
      (5 attack CSVs inside: ack, scan, syn, udp, udpplain) --
      2 devices were never infected with Mirai, which is correct/
      expected, not a bug
    - Each row has 115 numerical traffic-statistics features (no
      need to know their exact names -- the code treats "every
      column except the label" as a feature)

Because each device's data lives in its own folder, this loader
naturally supports the federated setup: load_all_devices() returns
one DataFrame per device, which is exactly one FL client's data --
no artificial random splitting required.
"""

import subprocess
import warnings
from pathlib import Path

import pandas as pd

# Harmless: pandas warns about DataFrame fragmentation when adding a
# couple of label columns to a 115-feature DataFrame. It doesn't
# affect correctness, just a minor performance note -- silenced here
# so it doesn't look like something is broken on every run.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Device folders as extracted from the UCI download, in a fixed order
# so device_id is stable across runs. The UCI archive itself does not
# ship a device_info.csv here, so IDs 1-9 are assigned by this order
# (not necessarily the numbering used in the original paper's table)
# -- what matters for FL is that each ID consistently maps to the
# same real device across the whole project, which this guarantees.
DEVICE_FOLDERS = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]

GAFGYT_ATTACKS = ["combo", "junk", "scan", "tcp", "udp"]
MIRAI_ATTACKS = ["ack", "scan", "syn", "udp", "udpplain"]

DEVICE_IDS = list(range(1, len(DEVICE_FOLDERS) + 1))

# Every column produced by this loader that is NOT one of the 115
# real traffic features -- anything training/explaining on the data
# should exclude these before treating columns as model inputs.
NON_FEATURE_COLUMNS = ["binary_label", "multiclass_label", "device_id", "device_name"]


def _ensure_extracted(archive_path: Path) -> Path | None:
    """Extracts a .rar archive into a sibling folder the first time
    it's needed, and reuses that folder on later calls. Returns None
    if the archive doesn't exist (e.g. mirai_attacks.rar for the 2
    devices that were never infected with Mirai).

    Uses the OS "tar" command rather than a Python rar library: the
    .rar format needs a native decoder, and Windows/most Linux
    distros already ship a tar (bsdtar/libarchive) that reads .rar
    directly, so no extra dependency is required.
    """
    extract_dir = archive_path.with_suffix("")  # e.g. gafgyt_attacks.rar -> gafgyt_attacks/
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return extract_dir  # already extracted -- reused even if the source .rar is gone

    if not archive_path.exists():
        return None

    extract_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["tar", "-xf", str(archive_path), "-C", str(extract_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract {archive_path} (need a tar that reads .rar, "
            f"e.g. Windows' built-in tar.exe or bsdtar/libarchive on Linux): "
            f"{result.stderr}"
        )
    return extract_dir


def _read_csv_if_exists(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.assign(
        binary_label=0 if label == "benign" else 1,
        multiclass_label=label,
    ).copy()
    return df


def load_device_data(device_id: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Loads and labels every available file for one device.

    Returns a single DataFrame combining that device's benign traffic
    and every attack file that actually exists for it (gafgyt always;
    mirai only if present).
    """
    device_name = DEVICE_FOLDERS[device_id - 1]
    device_dir = raw_dir / device_name
    frames = []

    benign = _read_csv_if_exists(device_dir / "benign_traffic.csv", "benign")
    if benign is not None:
        frames.append(benign)

    gafgyt_dir = _ensure_extracted(device_dir / "gafgyt_attacks.rar")
    for attack in GAFGYT_ATTACKS:
        path = gafgyt_dir / f"{attack}.csv" if gafgyt_dir else None
        df = _read_csv_if_exists(path, f"gafgyt_{attack}")
        if df is not None:
            frames.append(df)

    mirai_dir = _ensure_extracted(device_dir / "mirai_attacks.rar")
    for attack in MIRAI_ATTACKS:
        path = mirai_dir / f"{attack}.csv" if mirai_dir else None
        df = _read_csv_if_exists(path, f"mirai_{attack}")
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No files found for device {device_id} ({device_name}) in {device_dir}. "
            "Check that the dataset was extracted there correctly."
        )

    device_df = (
        pd.concat(frames, ignore_index=True)
        .assign(device_id=device_id, device_name=device_name)
        .copy()
    )
    return device_df


def load_all_devices(raw_dir: Path = RAW_DIR) -> dict[int, pd.DataFrame]:
    """Returns {device_id: DataFrame}. Each entry is exactly one
    federated learning client's local dataset -- this is the real,
    non-synthetic non-IID partition your project design calls for."""
    devices = {}
    for device_id in DEVICE_IDS:
        try:
            devices[device_id] = load_device_data(device_id, raw_dir)
        except FileNotFoundError as e:
            print(f"Skipping device {device_id}: {e}")
    return devices


def build_centralized_dataset(raw_dir: Path = RAW_DIR, sample_frac: float | None = None) -> pd.DataFrame:
    """Combines every device into one dataset -- used ONLY for the
    Centralized IDS baseline stage (the PERT-chart stage that comes
    before Federated IDS). The federated pipeline should use
    load_all_devices() instead, to keep devices separate.

    sample_frac: optionally subsample (e.g. 0.05) for a fast first
    run -- N-BaIoT has 7M+ rows in total, more than you need to get
    a working baseline before scaling up.
    """
    devices = load_all_devices(raw_dir)
    combined = pd.concat(devices.values(), ignore_index=True)
    if sample_frac is not None:
        combined = combined.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    devices = load_all_devices()
    for device_id, df in devices.items():
        print(f"Device {device_id} ({df['device_name'].iloc[0]}): {len(df)} rows, "
              f"{df['multiclass_label'].nunique()} classes present")
    print(f"\nLoaded {len(devices)} of {len(DEVICE_IDS)} devices from {RAW_DIR}")
