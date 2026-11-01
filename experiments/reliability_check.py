"""Does the detector or its explanation state things the evidence does not support?

Three questions:
  model:        is its confidence honest, on real traffic, on nonsense input and on unseen traffic?
  explanation:  does it reflect what the model uses, and does it say anything specific to the verdict?
  retrained:    what happens with attack families or devices that training never saw?

Run `python -m experiments.reliability_check` (needs data/processed). Results go to reliability.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from bcfl_ids.config import BENIGN, CLASSES, DEMO_DIR, DEVICE_IDS, NO_MIRAI_DEVICES, NUM_CLASSES, RESULTS_DIR, SimConfig
from bcfl_ids.data import feature_names, load_clients, signed_log
from bcfl_ids.explain import attack_score, completeness_error, deletion_curve, integrated_gradients, spearman
from bcfl_ids.model import Detector, load_flat
from bcfl_ids.scaler import Scale
from bcfl_ids.simulator import ClientData, run_simulation

from .common import save_json

OUT = RESULTS_DIR / "reliability.json"
# Look-alike pairs, as in metrics.merge_lookalikes.
MERGE = {CLASSES.index("gafgyt_junk"): CLASSES.index("gafgyt_combo"), CLASSES.index("gafgyt_tcp"): CLASSES.index("gafgyt_udp")}
ATTACK_CLASSES = list(range(1, NUM_CLASSES))
OOD_ROWS = 20_000
TOP = 10


def merged(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    for src, dst in MERGE.items():
        out[out == src] = dst
    return out


def build(w) -> Detector:
    m = Detector()
    load_flat(m, torch.as_tensor(w, dtype=torch.float32))
    return m.eval()


@torch.no_grad()
def logits_of(model, x: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    return np.concatenate([model(torch.from_numpy(x[i : i + chunk])).numpy() for i in range(0, len(x), chunk)])


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits) / temperature, dim=1).numpy()


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: the average gap between stated confidence and actual accuracy."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (conf > a) & (conf <= b)
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One number that rescales confidence, fitted on validation data only."""
    x, t = torch.from_numpy(logits), torch.from_numpy(y.astype(np.int64))
    grid = np.exp(np.linspace(np.log(0.25), np.log(50), 80))
    losses = [float(torch.nn.functional.cross_entropy(x / float(g), t)) for g in grid]
    return float(grid[int(np.argmin(losses))])


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Chance that a random positive scores above a random negative, ties counted as half."""
    joined = np.concatenate([pos, neg])
    _, inv, counts = np.unique(joined, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_test(clients, scale, stride: int = 1):
    xs, ys, ds = [], [], []
    for c in clients:
        x = scale.apply(np.asarray(c.x_test[::stride]))
        y = np.asarray(c.y_test[::stride]).astype(np.int64)
        xs.append(x)
        ys.append(y)
        ds.append(np.full(len(y), c.device))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)


def check_calibration(model, clients, val_clients, scale) -> dict:
    x, y, device = load_test(clients, scale)
    logits = logits_of(model, x)
    probs = softmax(logits)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == y
    binary_true, binary_pred = y != BENIGN, pred != BENIGN
    p_benign = probs[:, BENIGN]
    binary_conf = np.maximum(p_benign, 1 - p_benign)
    family_correct = merged(pred) == merged(y)

    wrong = ~correct
    out = {
        "records": len(y),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "ece_11_class": ece(conf, correct),
        "ece_attack_vs_benign": ece(binary_conf, binary_pred == binary_true),
        "ece_merged_families": ece(conf, family_correct),
        "bins": [],
        "wrong_records": int(wrong.sum()),
        "wrong_at_90": float((conf[wrong] >= 0.9).mean()),
        "wrong_at_99": float((conf[wrong] >= 0.99).mean()),
        "wrong_only_within_lookalike_pairs": float((family_correct & wrong).sum() / max(1, wrong.sum())),
        "false_alarms": int((~binary_true & binary_pred).sum()),
        "false_alarms_at_90": float((binary_conf[~binary_true & binary_pred] >= 0.9).mean()),
        "missed_attacks": int((binary_true & ~binary_pred).sum()),
        "missed_attacks_at_90": float((binary_conf[binary_true & ~binary_pred] >= 0.9).mean()),
    }
    for low, high in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)):
        mask = (conf >= low) & (conf < high)
        out["bins"].append({
            "range": f"{low}-{min(high, 1.0)}", "share": float(mask.mean()),
            "stated": float(conf[mask].mean()) if mask.any() else None,
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
            "accuracy_merged": float(family_correct[mask].mean()) if mask.any() else None,
            "attack_vs_benign_accuracy": float((binary_pred == binary_true)[mask].mean()) if mask.any() else None,
        })
    out["per_class"] = {}
    for i, name in enumerate(CLASSES):
        m = y == i
        counts = np.bincount(pred[m], minlength=NUM_CLASSES)
        top = int(counts.argmax())
        out["per_class"][name] = {
            "records": int(m.sum()), "recall": float(correct[m].mean()), "mean_confidence": float(conf[m].mean()),
            "most_often_called": CLASSES[top], "share_called_that": float(counts[top] / m.sum()),
        }
    # Mirai classes on devices that have no Mirai traffic at all: a family that cannot be there.
    no_mirai = np.isin(device, NO_MIRAI_DEVICES)
    mirai_pred = np.isin(pred, [i for i, n in enumerate(CLASSES) if n.startswith("mirai")])
    out["phantom_mirai_on_no_mirai_devices"] = {
        "records": int(no_mirai.sum()), "called_mirai": int((no_mirai & mirai_pred).sum()),
        "share": float((no_mirai & mirai_pred).sum() / max(1, no_mirai.sum())),
        "at_90": float((conf[no_mirai & mirai_pred] >= 0.9).mean()) if (no_mirai & mirai_pred).any() else None,
    }
    # Would rescaling fix it? Fitted on validation data, measured on test data.
    vx, vy, _ = load_test(val_clients, scale, stride=5)
    vlogits = logits_of(model, vx)
    t = fit_temperature(vlogits, vy)
    cooled = softmax(logits, t)
    cc = cooled.max(1)
    out["temperature_scaling"] = {"temperature": t, "ece_after": ece(cc, cooled.argmax(1) == y), "mean_confidence_after": float(cc.mean())}
    # Abstaining: refuse a family name unless confident. Threshold picked on validation data.
    out["coverage"] = []
    for th in (0.5, 0.7, 0.9, 0.99):
        m = conf >= th
        out["coverage"].append({"threshold": th, "answered": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def nonsense_inputs(x_real: np.ndarray, raw_mean: np.ndarray, scale: Scale, rng: np.random.Generator) -> dict:
    """Inputs that are not traffic. Each goes through the same scaling as the prediction API."""
    n, d = OOD_ROWS, x_real.shape[1]
    sample = x_real[rng.integers(0, len(x_real), n)]

    def from_raw(raw: np.ndarray) -> np.ndarray:
        return scale.apply(signed_log(raw.astype(np.float64)))

    return {
        "raw all zeros": from_raw(np.zeros((1, d))),
        "raw all ones": from_raw(np.ones((1, d))),
        "raw all -1000": from_raw(np.full((1, d), -1000.0)),
        "raw all 1e12": from_raw(np.full((1, d), 1e12)),
        "scaled all zeros": np.zeros((1, d), dtype=np.float32),
        "gaussian noise, sd 1": rng.normal(0, 1, (n, d)).astype(np.float32),
        "gaussian noise, sd 5": np.clip(rng.normal(0, 5, (n, d)), -10, 10).astype(np.float32),
        "uniform noise, -10 to 10": rng.uniform(-10, 10, (n, d)).astype(np.float32),
        "real rows, columns shuffled": rng.permuted(sample, axis=1),
        "real columns, rows mixed": np.stack([rng.choice(x_real[:, j], n) for j in range(d)], axis=1),
        "real rows, sign reversed": -sample,
    }


def check_nonsense(model, clients, val_clients, scale, train_x) -> dict:
    rng = np.random.default_rng(7)
    x_test, _, _ = load_test(clients, scale, stride=5)
    x_val, _, _ = load_test(val_clients, scale, stride=5)
    conf_test = softmax(logits_of(model, x_test)).max(1)
    conf_val = softmax(logits_of(model, x_val)).max(1)

    # Three possible detectors for "this is not traffic", all with thresholds set on validation data.
    low, high = train_x.min(0), train_x.max(0)
    mean = train_x.mean(0)
    cov = np.cov(train_x[rng.integers(0, len(train_x), 100_000)], rowvar=False) + 1e-3 * np.eye(train_x.shape[1])
    prec = np.linalg.inv(cov)

    def range_score(x):
        return ((x < low) | (x > high)).mean(1)

    def maha_score(x, chunk=100_000):
        return np.concatenate([np.einsum("ij,jk,ik->i", (x[i : i + chunk] - mean), prec, (x[i : i + chunk] - mean)) for i in range(0, len(x), chunk)])

    limits = {"softmax": np.quantile(conf_val, 0.01), "range": np.quantile(range_score(x_val), 0.99), "distance": np.quantile(maha_score(x_val), 0.99)}
    out = {
        "real_test_reference": {"mean_confidence": float(conf_test.mean()), "at_90": float((conf_test >= 0.9).mean()), "at_99": float((conf_test >= 0.99).mean())},
        "false_refusals_on_real_test": {
            "softmax": float((conf_test < limits["softmax"]).mean()),
            "range": float((range_score(x_test) > limits["range"]).mean()),
            "distance": float((maha_score(x_test) > limits["distance"]).mean()),
        },
        "inputs": {},
    }
    for name, x in nonsense_inputs(x_test, mean, scale, rng).items():
        probs = softmax(logits_of(model, x))
        conf, pred = probs.max(1), probs.argmax(1)
        counts = np.bincount(pred, minlength=NUM_CLASSES)
        out["inputs"][name] = {
            "rows": len(x), "most_often_called": CLASSES[int(counts.argmax())], "share_called_that": float(counts.max() / len(x)),
            "called_attack": float((pred != BENIGN).mean()), "mean_confidence": float(conf.mean()),
            "at_90": float((conf >= 0.9).mean()), "at_99": float((conf >= 0.99).mean()),
            "caught_by_softmax": float((conf < limits["softmax"]).mean()),
            "caught_by_range": float((range_score(x) > limits["range"]).mean()),
            "caught_by_distance": float((maha_score(x) > limits["distance"]).mean()),
            "softmax_auroc_real_vs_this": auroc(conf_test[rng.integers(0, len(conf_test), 20_000)], conf) if len(x) > 1 else None,
        }
    return out


def per_record_agreement(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Do two explanations of the same records name the same features?"""
    sp, ov = [], []
    for u, v in zip(a.numpy(), b.numpy()):
        sp.append(spearman(np.abs(u), np.abs(v)))
        ov.append(len(set(np.argsort(-np.abs(u))[:TOP]) & set(np.argsort(-np.abs(v))[:TOP])) / TOP)
    return {"spearman": float(np.nanmean(sp)), "top10_overlap": float(np.mean(ov))}


def sufficiency(model, x, baseline, target, attr, k: int = TOP) -> dict:
    """Keep only the top k features and reset the rest: how much of the verdict survives?"""
    with torch.no_grad():
        order = attr.argsort(dim=1, descending=True)[:, :k]
        kept = baseline.expand_as(x).clone()
        kept.scatter_(1, order, x.gather(1, order))
        rand = torch.rand(x.shape, generator=torch.Generator().manual_seed(0)).argsort(1)[:, :k]
        kept_r = baseline.expand_as(x).clone()
        kept_r.scatter_(1, rand, x.gather(1, rand))
        fx, fb = attack_score(model(x), target), attack_score(model(baseline.expand_as(x)), target)
        gap = (fx - fb).clamp_min(1e-6)
        return {
            "evidence_kept_top": float(((attack_score(model(kept), target) - fb) / gap).median()),
            "evidence_kept_random": float(((attack_score(model(kept_r), target) - fb) / gap).median()),
            "same_verdict_top": float((model(kept).argmax(1) == target).float().mean()),
            "same_verdict_random": float((model(kept_r).argmax(1) == target).float().mean()),
        }


def sample_records(clients, scale, per_device: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in clients:
        y = np.asarray(c.y_test)
        idx = np.sort(rng.choice(len(y), per_device, replace=False))
        xs.append(scale.apply(np.asarray(c.x_test[idx])))
        ys.append(y[idx].astype(np.int64))
    return torch.from_numpy(np.concatenate(xs)), np.concatenate(ys)


def extra_wrong_records(model, clients, scale, per_device: int = 150, seed: int = 1):
    """Records the model gets wrong, so explanations of wrong answers are measured directly."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in clients:
        y = np.asarray(c.y_test).astype(np.int64)
        idx = np.sort(rng.choice(len(y), min(len(y), 20_000), replace=False))
        x = scale.apply(np.asarray(c.x_test[idx]))
        pred = logits_of(model, x).argmax(1)
        bad = np.flatnonzero((pred != y[idx]) & (pred != BENIGN))
        bad = rng.choice(bad, min(per_device, len(bad)), replace=False)
        xs.append(x[bad])
        ys.append(y[idx][bad])
    return torch.from_numpy(np.concatenate(xs)), np.concatenate(ys)


def check_explanations(model, clients, scale, baseline: torch.Tensor, names: list[str]) -> dict:
    out: dict = {}
    with torch.no_grad():
        b_logits = model(baseline.unsqueeze(0))[0]
    out["baseline"] = {"predicted": CLASSES[int(b_logits.argmax())], "benign_probability": float(torch.softmax(b_logits, 0)[BENIGN])}

    x, y = sample_records(clients, scale)
    with torch.no_grad():
        pred = model(x).argmax(1)
    mask = pred != BENIGN
    xf, yf, target = x[mask], torch.from_numpy(y)[mask.numpy()], pred[mask]
    attr = integrated_gradients(model, xf, baseline, target)
    out["records"] = len(xf)

    def summary(xs, tgt, a) -> dict:
        err = completeness_error(model, xs, baseline, tgt, a)
        rows = deletion_curve(model, xs, baseline, tgt, a, [TOP])
        return {
            "records": len(xs), "integration_error_median": float(np.median(err)), "integration_error_p99": float(np.percentile(err, 99)),
            **{k: rows[0][k] for k in ("evidence_removed_top", "evidence_removed_random", "flipped_top", "flipped_random")},
            **sufficiency(model, xs, baseline, tgt, a),
        }

    out["all_flagged"] = summary(xf, target, attr)
    correct = target == yf
    out["correct_verdicts"] = summary(xf[correct], target[correct], attr[correct])
    wx, wy = extra_wrong_records(model, clients, scale)
    with torch.no_grad():
        wt = model(wx).argmax(1)
    wa = integrated_gradients(model, wx, baseline, wt)
    out["wrong_verdicts"] = summary(wx, wt, wa)
    out["per_family"] = {CLASSES[c]: summary(xf[target == c], target[target == c], attr[target == c]) for c in sorted(set(target.tolist())) if (target == c).sum() >= 20}

    # Is the top bar in the dashboard a push towards the attack, as "strongest signal" implies?
    strongest = attr.gather(1, attr.abs().argmax(1, keepdim=True))[:, 0]
    out["strongest_signal_points_to_normal"] = float((strongest < 0).float().mean())
    out["top10_negative_share"] = float((attr.gather(1, attr.abs().argsort(1, descending=True)[:, :TOP]) < 0).float().mean())

    # Does the explanation say anything about the family, or only "attack, not benign"?
    probs = torch.softmax(model(xf), 1).detach()
    runner = probs.clone()
    runner[:, BENIGN] = 0
    runner[torch.arange(len(xf)), target] = 0
    second = runner.argmax(1)
    rng = torch.Generator().manual_seed(3)
    other = (target - 1 + torch.randint(1, len(ATTACK_CLASSES), target.shape, generator=rng)) % len(ATTACK_CLASSES) + 1
    out["family_specificity"] = {
        "against_runner_up_family": per_record_agreement(attr, integrated_gradients(model, xf, baseline, second)),
        "against_random_family": per_record_agreement(attr, integrated_gradients(model, xf, baseline, other)),
    }
    # After deleting the top 10, where does the verdict go?
    with torch.no_grad():
        order = attr.argsort(dim=1, descending=True)[:, :TOP]
        cut = xf.clone()
        cut.scatter_(1, order, baseline[order])
        after = model(cut).argmax(1)
    out["after_deleting_top10"] = {"same_family": float((after == target).float().mean()), "becomes_benign": float((after == BENIGN).float().mean()), "other_family": float(((after != target) & (after != BENIGN)).float().mean())}

    # Small changes that leave the verdict alone should leave the explanation alone.
    out["stability_to_noise"] = {}
    for sd in (0.01, 0.05, 0.1):
        noisy = xf + torch.normal(0, sd, xf.shape, generator=torch.Generator().manual_seed(11))
        with torch.no_grad():
            same = model(noisy).argmax(1) == target
        out["stability_to_noise"][str(sd)] = {"same_verdict": float(same.float().mean()), **per_record_agreement(attr[same], integrated_gradients(model, noisy[same], baseline, target[same]))}

    # The reference point is a choice. Does a different reasonable choice change the explanation?
    benign_rows = []
    per_device = {}
    rng_np = np.random.default_rng(5)
    for c in clients:
        y_c = np.asarray(c.y_test)
        idx = np.flatnonzero(y_c == BENIGN)
        rows = scale.apply(np.asarray(c.x_test[np.sort(rng_np.choice(idx, min(500, len(idx)), replace=False))]))
        benign_rows.append(rows)
        per_device[c.device] = torch.from_numpy(rows.mean(0))
    alternative = torch.from_numpy(np.median(np.concatenate(benign_rows), axis=0))
    single = per_device[DEVICE_IDS[0]]
    out["baseline_sensitivity"] = {
        "median_benign": per_record_agreement(attr, integrated_gradients(model, xf, alternative, target)),
        "one_device_benign": per_record_agreement(attr, integrated_gradients(model, xf, single, target)),
    }

    # Model randomisation (Adebayo et al., 2018): reset layers from the output down.
    layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    out["cascade_randomisation"] = {}
    randomised = build(torch.nn.utils.parameters_to_vector(model.parameters()).detach())
    rand_layers = [m for m in randomised.net if isinstance(m, torch.nn.Linear)]
    torch.manual_seed(999)
    for depth, layer in enumerate(reversed(rand_layers), start=1):
        layer.reset_parameters()
        with torch.no_grad():
            _ = randomised(xf)
        ra = integrated_gradients(randomised, xf, baseline, target)
        out["cascade_randomisation"][f"last_{depth}_of_{len(layers)}_layers_reset"] = per_record_agreement(attr, ra)
    # How much of the agreement comes from the input alone, with no model at all?
    gap = (xf - baseline)
    out["input_only_explanation"] = per_record_agreement(attr, gap)

    # Non-traffic: does the explanation path notice?
    noise = torch.from_numpy(np.random.default_rng(13).uniform(-10, 10, (500, xf.shape[1])).astype(np.float32))
    with torch.no_grad():
        np_probs = torch.softmax(model(noise), 1)
    np_pred = np_probs.argmax(1)
    flagged = np_pred != BENIGN
    na = integrated_gradients(model, noise[flagged], baseline, np_pred[flagged])
    out["explaining_noise"] = {
        "flagged": float(flagged.float().mean()), "mean_confidence": float(np_probs.max(1).values.mean()),
        "integration_error_median": float(np.median(completeness_error(model, noise[flagged], baseline, np_pred[flagged], na))),
        **{k: deletion_curve(model, noise[flagged], baseline, np_pred[flagged], na, [TOP])[0][k] for k in ("evidence_removed_top", "evidence_removed_random")},
        "refused": False,
    }

    # Correlated features: many top features have a near-copy, so "this feature" is not unique.
    rows = np.concatenate([scale.apply(np.asarray(c.x_test[:: max(1, len(c.x_test) // 5000)])) for c in clients])
    corr = np.nan_to_num(np.corrcoef(rows, rowvar=False))
    np.fill_diagonal(corr, 0)
    top_global = np.argsort(-attr.abs().mean(0).numpy())[:TOP]
    out["correlated_features"] = {
        "features_with_copy_above_95": float((np.abs(corr).max(1) > 0.95).mean()),
        "top10_with_copy_above_95": float((np.abs(corr[top_global]).max(1) > 0.95).mean()),
        "top10": [names[i] for i in top_global],
    }
    return out


def planted_truth(seed: int = 0) -> dict:
    """Train the same network where the truth is known: does IG find the features that decide the label?"""
    rng = np.random.default_rng(seed)
    n, d, signal = 40_000, 115, 5
    x = rng.normal(0, 1, (n, d)).astype(np.float32)
    true_idx = rng.choice(d, signal, replace=False)
    copies = np.setdiff1d(rng.permutation(d), true_idx)[:signal]
    y_score = x[:, true_idx].sum(1)
    y = (y_score > 0).astype(np.int64) * 3  # class 0 is benign, class 3 stands for "attack"
    results = {}
    for name, with_copies in (("independent features", False), ("each signal feature has a near-copy", True)):
        data = x.copy()
        if with_copies:
            data[:, copies] = data[:, true_idx] + rng.normal(0, 0.05, (n, signal)).astype(np.float32)
        torch.manual_seed(seed)
        model = Detector()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        tx, ty = torch.from_numpy(data[:30_000]), torch.from_numpy(y[:30_000])
        for _ in range(8):
            order = torch.randperm(len(tx))
            for s in range(0, len(tx), 256):
                i = order[s : s + 256]
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(tx[i]), ty[i]).backward()
                opt.step()
        model.eval()
        ex = torch.from_numpy(data[30_000:30_500])
        with torch.no_grad():
            pred = model(ex).argmax(1)
        flagged = pred == 3
        target = torch.full((int(flagged.sum()),), 3)
        a = integrated_gradients(model, ex[flagged], torch.zeros(d), target).abs().numpy()
        found = np.argsort(-a, axis=1)[:, :signal]
        truth = set(true_idx.tolist())
        both = truth | set(copies.tolist())
        results[name] = {
            "accuracy": float((model(torch.from_numpy(data[30_000:])).argmax(1).numpy() == y[30_000:]).mean()),
            "top5_are_signal": float(np.mean([len(set(r) & truth) / signal for r in found])),
            "top5_are_signal_or_copy": float(np.mean([len(set(r) & both) / signal for r in found])),
            "chance": signal / d,
        }
    return results


def holdout_run(clients, seed: int, keep_family=None, drop_device=None, shuffle_labels=False) -> dict:
    train = []
    rng = np.random.default_rng(seed + 100)
    for c in clients:
        if c.device == drop_device:
            continue
        x, y = c.x_train, np.asarray(c.y_train)
        if keep_family is not None:
            m = y != keep_family
            x, y = x[m], y[m]
        if shuffle_labels:
            y = rng.permutation(y)
        train.append(ClientData(c.device, x, y, c.x_test, c.y_test))
    return run_simulation(train, SimConfig(defence="proposed", rounds=20, seed=seed), eval_every=20)


def check_retrained(clients, baseline_seed: int = 1) -> dict:
    out: dict = {"unseen_family": {}, "unseen_device": {}}
    full = holdout_run(clients, baseline_seed)
    model, scale = build(full["weights"]), full["scale"]
    x_all, y_all, d_all = load_test(clients, scale)
    probs = softmax(logits_of(model, x_all))
    conf, pred = probs.max(1), probs.argmax(1)
    out["reference_full_training"] = {
        "accuracy": float((pred == y_all).mean()), "attack_recall": float((pred[y_all != BENIGN] != BENIGN).mean()),
        "benign_recall": float((pred[y_all == BENIGN] == BENIGN).mean()), "mean_confidence": float(conf.mean()),
    }
    out["reference_full_training"]["per_device_benign_recall"] = {str(d): float((pred[(y_all == BENIGN) & (d_all == d)] == BENIGN).mean()) for d in DEVICE_IDS}
    out["reference_full_training"]["per_device_attack_recall"] = {str(d): float((pred[(y_all != BENIGN) & (d_all == d)] != BENIGN).mean()) for d in DEVICE_IDS}
    save_partial("retrained", out)

    for family in ATTACK_CLASSES:
        r = holdout_run(clients, baseline_seed, keep_family=family)
        m = build(r["weights"])
        x, y, _ = load_test(clients, r["scale"])
        rows = y == family
        p = softmax(logits_of(m, x[rows]))
        c, q = p.max(1), p.argmax(1)
        counts = np.bincount(q, minlength=NUM_CLASSES)
        top = int(counts.argmax())
        out["unseen_family"][CLASSES[family]] = {
            "records": int(rows.sum()), "flagged_as_attack": float((q != BENIGN).mean()), "called_benign": float((q == BENIGN).mean()),
            "called_benign_at_90": float((c[q == BENIGN] >= 0.9).mean()) if (q == BENIGN).any() else None,
            "named_a_family_it_never_saw": float((q == family).mean()),
            "named_a_known_family": float(((q != BENIGN) & (q != family)).mean()),
            "most_often_called": CLASSES[top], "share_called_that": float(counts[top] / len(q)),
            "mean_confidence": float(c.mean()), "at_90": float((c >= 0.9).mean()), "at_99": float((c >= 0.99).mean()),
        }
        print("family", CLASSES[family], out["unseen_family"][CLASSES[family]], flush=True)
        save_partial("retrained", out)

    for device in DEVICE_IDS:
        r = holdout_run(clients, baseline_seed, drop_device=device)
        m = build(r["weights"])
        c_dev = next(c for c in clients if c.device == device)
        x, y, _ = load_test([c_dev], r["scale"])
        p = softmax(logits_of(m, x))
        c, q = p.max(1), p.argmax(1)
        benign, attack = y == BENIGN, y != BENIGN
        false_alarm = benign & (q != BENIGN)
        missed = attack & (q == BENIGN)
        out["unseen_device"][str(device)] = {
            "benign_records": int(benign.sum()), "benign_recall": float((q[benign] == BENIGN).mean()), "attack_recall": float((q[attack] != BENIGN).mean()),
            "accuracy": float((q == y).mean()), "mean_confidence": float(c.mean()),
            "false_alarms_at_90": float((c[false_alarm] >= 0.9).mean()) if false_alarm.any() else None,
            "missed_attacks_at_90": float((c[missed] >= 0.9).mean()) if missed.any() else None,
        }
        print("device", device, out["unseen_device"][str(device)], flush=True)
        save_partial("retrained", out)

    # Explanations from the same recipe with labels that mean nothing, and from other seeds.
    x, _ = sample_records(clients, scale)
    with torch.no_grad():
        target = model(x).argmax(1)
    mask = target != BENIGN
    xf, tf = x[mask], target[mask]
    base = torch.from_numpy(np.concatenate([scale.apply(np.asarray(c.x_test[np.flatnonzero(np.asarray(c.y_test) == BENIGN)[:500]])) for c in clients]).mean(0))
    attr = integrated_gradients(model, xf, base, tf)

    r = holdout_run(clients, baseline_seed, shuffle_labels=True)
    shuffled = build(r["weights"])
    out["label_randomised_model"] = {"accuracy_on_test": r["final"]["accuracy"], **per_record_agreement(attr, integrated_gradients(shuffled, xf, base, tf))}
    print("label random", out["label_randomised_model"], flush=True)
    save_partial("retrained", out)

    out["other_seeds"] = {}
    for seed in (2, 3):
        r = holdout_run(clients, seed)
        other = build(r["weights"])
        with torch.no_grad():
            same = other(xf).argmax(1) == tf
        out["other_seeds"][str(seed)] = {"same_verdict": float(same.float().mean()), **per_record_agreement(attr[same], integrated_gradients(other, xf[same], base, tf[same]))}
        print("seed", seed, out["other_seeds"][str(seed)], flush=True)
        save_partial("retrained", out)
    return out


def save_partial(key: str, value) -> None:
    current = json.loads(OUT.read_text()) if OUT.exists() else {}
    current[key] = value
    save_json("reliability.json", current)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("deployed", "retrained", "all"), default="all")
    args = p.parse_args()
    started = time.time()

    if args.part in ("deployed", "all"):
        saved = np.load(DEMO_DIR / "model.npz")
        scale = Scale(saved["mean"], saved["std"])
        model = build(saved["weights"])
        baseline = torch.from_numpy(saved["benign_baseline"].astype(np.float32))
        names = json.loads((DEMO_DIR / "features.json").read_text())
        clients = load_clients(frac=0.05, split="test")
        val_clients = load_clients(frac=0.001, split="val")
        train_x = scale.apply(np.concatenate([c.x_train for c in clients]))

        save_partial("calibration", check_calibration(model, clients, val_clients, scale))
        print("calibration done", round(time.time() - started), flush=True)
        save_partial("nonsense", check_nonsense(model, clients, val_clients, scale, train_x))
        print("nonsense done", round(time.time() - started), flush=True)
        save_partial("explanations", check_explanations(model, clients, scale, baseline, names))
        print("explanations done", round(time.time() - started), flush=True)
        save_partial("planted_truth", planted_truth())
        print("planted truth done", round(time.time() - started), flush=True)

    if args.part in ("retrained", "all"):
        clients = load_clients(frac=0.05, split="test")
        save_partial("retrained", check_retrained(clients))
        print("retrained done", round(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
