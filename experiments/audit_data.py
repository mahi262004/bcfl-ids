"""How easy is N-BaIoT really? One-feature and shallow baselines, then split and device hold-out stress tests.

Runs on the validation split only for model choices; test-split numbers are final measurements.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from bcfl_ids.config import BENIGN, CLASSES, DEVICE_IDS, NUM_CLASSES, PROCESSED_DIR
from bcfl_ids.data import _stratified, feature_names

from .common import save_json


def load_pool(split: str, frac: float, seed: int, cap: int | None = None):
    xs, ys, ds = [], [], []
    for device in DEVICE_IDS:
        d = PROCESSED_DIR / f"device_{device}"
        y = np.asarray(np.load(d / f"y_{split}.npy", mmap_mode="r"))
        idx = _stratified(y, frac, cap, np.random.default_rng(seed + device))
        xs.append(np.asarray(np.load(d / f"X_{split}.npy", mmap_mode="r")[idx]))
        ys.append(y[idx].astype(np.int64))
        ds.append(np.full(len(idx), device))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)


def score(y, pred) -> dict:
    yb, pb = (y != BENIGN).astype(int), (pred != BENIGN).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "binary_accuracy": float(accuracy_score(yb, pb)),
        "binary_mcc": float(matthews_corrcoef(yb, pb)),
        "attack_recall": float(recall_score(yb, pb, zero_division=0)),
        "benign_recall": float(recall_score(1 - yb, 1 - pb, zero_division=0)),
    }


def per_family_recall(y, pred) -> dict:
    return {CLASSES[c]: float((pred[y == c] == c).mean()) for c in range(NUM_CLASSES) if (y == c).any()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.05)
    args = p.parse_args()
    x_tr, y_tr, d_tr = load_pool("train", args.frac, 0)
    x_te, y_te, d_te = load_pool("test", args.frac, 1)
    names = feature_names()
    out: dict = {"train_rows": len(x_tr), "test_rows": len(x_te), "benign_share_test": float((y_te == BENIGN).mean())}

    yb_tr, yb_te = (y_tr != BENIGN).astype(int), (y_te != BENIGN).astype(int)
    out["always_attack"] = {"binary_accuracy": float(yb_te.mean())}
    out["always_most_common_family"] = {"accuracy": float((y_te == np.bincount(y_tr).argmax()).mean())}

    # Best single feature: a depth-1 tree on each feature, chosen by training MCC.
    best = (-1.0, None, None)
    for j in range(x_tr.shape[1]):
        t = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x_tr[:, [j]], yb_tr)
        mcc = matthews_corrcoef(yb_tr, t.predict(x_tr[:, [j]]))
        if mcc > best[0]:
            best = (mcc, j, t)
    pred = best[2].predict(x_te[:, [best[1]]])
    out["best_single_feature"] = {
        "feature": names[best[1]],
        "binary_accuracy": float(accuracy_score(yb_te, pred)), "binary_mcc": float(matthews_corrcoef(yb_te, pred)),
        "attack_recall": float(recall_score(yb_te, pred)), "benign_recall": float(recall_score(1 - yb_te, 1 - pred)),
    }
    print("single feature", out["best_single_feature"], flush=True)

    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_tr, y_tr)
    out["tree_depth3"] = score(y_te, tree.predict(x_te))
    print("tree3", out["tree_depth3"], flush=True)

    scaler = StandardScaler().fit(x_tr)
    lr = LogisticRegression(max_iter=300, n_jobs=None).fit(scaler.transform(x_tr), y_tr)
    out["logistic_regression"] = score(y_te, lr.predict(scaler.transform(x_te)))
    print("lr", out["logistic_regression"], flush=True)

    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=6, random_state=0).fit(x_tr, y_tr)
    pred = rf.predict(x_te)
    out["random_forest"] = score(y_te, pred)
    out["random_forest"]["family_recall"] = per_family_recall(y_te, pred)
    print("rf", out["random_forest"], flush=True)

    # Does a random split flatter the model compared with the chronological split?
    pooled_x = np.concatenate([x_tr, x_te])
    pooled_y = np.concatenate([y_tr, y_te])
    order = np.random.default_rng(0).permutation(len(pooled_x))
    cut = int(len(order) * len(x_tr) / len(pooled_x))
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=6, random_state=0).fit(pooled_x[order[:cut]], pooled_y[order[:cut]])
    out["random_forest_random_split"] = score(pooled_y[order[cut:]], rf2.predict(pooled_x[order[cut:]]))
    print("rf random split", out["random_forest_random_split"], flush=True)

    # Generalisation to a device never seen in training.
    holdout = {}
    for device in DEVICE_IDS:
        m = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=6, random_state=0).fit(x_tr[d_tr != device], y_tr[d_tr != device])
        mask = d_te == device
        holdout[str(device)] = score(y_te[mask], m.predict(x_te[mask]))
        print("device holdout", device, holdout[str(device)], flush=True)
    out["device_holdout"] = holdout

    # A signal that would show capture-session leakage: can traffic reveal which device it is from?
    device_clf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=6, random_state=0).fit(x_tr, d_tr)
    out["device_identifiable_accuracy"] = float((device_clf.predict(x_te) == d_te).mean())
    save_json("data_audit.json", out)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
