"""Project-wide constants: paths, devices, class labels and default thresholds."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
DEMO_DIR = ARTIFACTS_DIR / "demo"
RESULTS_DIR = ARTIFACTS_DIR / "results"
RUNTIME_DIR = ROOT / "runtime"
BLOCKCHAIN_DIR = ROOT / "blockchain"

# Client id -> (display name, folder in the N-BaIoT archive). Ids follow the report (Table 9.1).
DEVICES = {
    1: ("Danmini Doorbell", "Danmini_Doorbell"),
    2: ("Ecobee Thermostat", "Ecobee_Thermostat"),
    3: ("Ennio Doorbell", "Ennio_Doorbell"),
    4: ("Philips Baby Monitor", "Philips_B120N10_Baby_Monitor"),
    5: ("Provision PT-737E Camera", "Provision_PT_737E_Security_Camera"),
    6: ("Provision PT-838 Camera", "Provision_PT_838_Security_Camera"),
    7: ("Samsung SNH-1011-N Webcam", "Samsung_SNH_1011_N_Webcam"),
    8: ("SimpleHome XCS7-1002 Camera", "SimpleHome_XCS7_1002_WHT_Security_Camera"),
    9: ("SimpleHome XCS7-1003 Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera"),
}
DEVICE_IDS = tuple(DEVICES)
NO_MIRAI_DEVICES = (3, 7)

# Class 0 is benign; everything else is one attack family.
CLASSES = (
    "benign",
    "gafgyt_combo",
    "gafgyt_junk",
    "gafgyt_scan",
    "gafgyt_tcp",
    "gafgyt_udp",
    "mirai_ack",
    "mirai_scan",
    "mirai_syn",
    "mirai_udp",
    "mirai_udpplain",
)
NUM_CLASSES = len(CLASSES)
BENIGN = 0
NUM_FEATURES = 115

# Family used by the targeted attack.
TARGET_FAMILY = CLASSES.index("mirai_syn")

# Chronological split of every source file, with purge gaps so that the sliding-window
# statistics of a test record cannot overlap the records used for training.
SPLITS = {"train": (0.00, 0.68), "val": (0.70, 0.78), "test": (0.80, 1.00)}


@dataclass(frozen=True)
class Defence:
    """A defence combines independent layers, so each one can be switched off and measured."""

    name: str
    agg: str = "mean"  # mean | median | trimmed | multikrum
    gate: bool = False  # reject updates that fail the cosine / norm-ratio checks
    reputation: bool = False  # keep clients on probation out of the aggregate
    robust_scaler: bool = False  # robust aggregation of the shared feature statistics


DEFENCES = {
    "fedavg": Defence("fedavg"),
    "median": Defence("median", agg="median"),
    "trimmed": Defence("trimmed", agg="trimmed"),
    "multikrum": Defence("multikrum", agg="multikrum"),
    "gate": Defence("gate", gate=True),
    "gate_rep": Defence("gate_rep", gate=True, reputation=True),
    "gate_trim": Defence("gate_trim", agg="trimmed", gate=True),
    "proposed": Defence("proposed", agg="trimmed", gate=True, reputation=True, robust_scaler=True),
}

DEFENCE_LABELS = {
    "fedavg": "No defence (FedAvg)",
    "median": "Coordinate median",
    "trimmed": "Trimmed mean only",
    "multikrum": "Multi-Krum",
    "gate": "Trust gate only",
    "gate_rep": "Trust gate + reputation",
    "gate_trim": "Trust gate + trimmed mean",
    "proposed": "Proposed (full stack)",
    "fog": "Fog nodes + full stack",
}

ATTACK_INFO = {
    "none": "No attacker. Every client trains honestly.",
    "label_flip": "Attackers relabel attack traffic as benign and benign traffic as attack, then train normally.",
    "random_weights": "Attackers skip training and upload random weights.",
    "sign_flip": "Attackers train honestly, then upload the reverse of their update, amplified 3 times.",
    "scaling": "Model replacement: attackers train on flipped labels and amplify the update 10 times.",
    "targeted_evasion": "Attackers teach the model that one family (mirai_syn) is benign, leaving the rest intact.",
    "lie": "A Little Is Enough: a small coordinated shift hidden inside the honest clients' natural spread.",
    "minmax": "Min-Max: the largest reversed shift that stays as close to honest updates as they are to each other.",
    "fang_trim": "Fang attack aimed at trimmed mean: each coordinate is pushed past the far edge of the honest range.",
    "adaptive_gate": "Adaptive: attackers know the gate thresholds and blend in poison until the update only just passes.",
    "scaler_poison": "Attackers lie about their feature statistics, corrupting the shared scale before training starts.",
}
ATTACKS = tuple(ATTACK_INFO)


@dataclass
class SimConfig:
    attack: str = "none"
    n_attackers: int = 0
    defence: str = "proposed"
    rounds: int = 20
    seed: int = 1
    local_epochs: int = 1
    lr: float = 3e-3  # chosen on validation data, see tuning.json
    batch_size: int = 256
    trim: int = 3  # values dropped from each end, per coordinate
    krum_f: int = 3
    cos_min: float = 0.1  # calibrated on validation data, see calibration.json
    ratio_max: float = 2.0  # update size / median update size; honest updates stay below 1.35 on validation data
    attacker_ids: list[int] = field(default_factory=list)  # chosen from the seed when empty

    def to_dict(self) -> dict:
        return asdict(self)


# Reputation rules. blockchain/contracts/TrustLayer.sol uses the same values (a test checks it).
REP_START = 70
REP_GAIN = 2
REP_LOSS = 10
REP_FLOOR = 1
REP_CAP = 100
REP_PROBATION = 50
SCORE_PASS = 50
