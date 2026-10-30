"""Evaluate the explanations: completeness, deletion, sanity against random weights, and stability."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from bcfl_ids.config import BENIGN, CLASSES, SimConfig
from bcfl_ids.data import feature_names, load_clients
from bcfl_ids.explain import attack_score, completeness_error, deletion_curve, group_attribution, integrated_gradients, spearman
from bcfl_ids.model import Detector, load_flat
from bcfl_ids.simulator import run_simulation

from .common import save_json


def sample_test(clients, scale, per_client=300, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, benign = [], [], []
    for c in clients:
        y = np.asarray(c.y_test)
        idx = np.sort(rng.choice(len(y), per_client, replace=False))
        xs.append(scale.apply(np.asarray(c.x_test[idx])))
        ys.append(y[idx])
        bi = np.flatnonzero(y == BENIGN)
        bi = np.sort(rng.choice(bi, min(500, len(bi)), replace=False))
        benign.append(scale.apply(np.asarray(c.x_test[bi])))
    return torch.from_numpy(np.concatenate(xs)), np.concatenate(ys), torch.from_numpy(np.concatenate(benign)).mean(0)


def build(w) -> Detector:
    m = Detector()
    load_flat(m, w)
    return m


def attributions(model, x, baseline):
    with torch.no_grad():
        pred = model(x).argmax(1)
    mask = pred != BENIGN
    target = pred[mask]
    return integrated_gradients(model, x[mask], baseline, target, steps=128), x[mask], target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=20)
    args = p.parse_args()
    names = feature_names()
    clients = load_clients(frac=args.frac, split="test")
    out: dict = {}

    runs = {s: run_simulation(clients, SimConfig(seed=s, rounds=args.rounds, defence="proposed"), eval_every=args.rounds) for s in (1, 2, 3)}
    print("trained", {s: round(r["final"]["macro_f1"], 4) for s, r in runs.items()}, flush=True)
    main_run = runs[1]
    scale = main_run["scale"]
    x, y, benign_baseline = sample_test(clients, scale)
    model = build(main_run["weights"])
    global_mean = torch.zeros(x.shape[1])  # the mean of all traffic is zero after scaling

    # Why the choice of baseline matters: what does the model call each reference point?
    with torch.no_grad():
        logits = model(torch.stack([global_mean, benign_baseline]))
    out["baseline_check"] = {
        "global_mean_predicted_as": CLASSES[int(logits[0].argmax())],
        "benign_mean_predicted_as": CLASSES[int(logits[1].argmax())],
    }
    print(out["baseline_check"], flush=True)

    attr, xs, target = attributions(model, x, benign_baseline)
    err = completeness_error(model, xs, benign_baseline, target, attr)
    out["explained_records"] = len(xs)
    out["completeness"] = {"median_relative_error": float(np.median(err)), "p99_relative_error": float(np.percentile(err, 99))}
    out["deletion"] = deletion_curve(model, xs, benign_baseline, target, attr, [1, 5, 10, 20, 30, 50])
    print(out["completeness"], out["deletion"], flush=True)

    mean_abs = attr.abs().mean(0).numpy()
    order = np.argsort(-mean_abs)
    out["top_features"] = [{"feature": names[i], "mean_abs_attribution": float(mean_abs[i])} for i in order[:20]]
    out["groups"] = group_attribution(mean_abs, names)
    out["per_family_top"] = {}
    for cls in sorted(set(target.tolist())):
        fam = attr[target == cls]
        fam_abs = fam.abs().mean(0).numpy()
        idx = np.argsort(-fam_abs)[:5]
        out["per_family_top"][CLASSES[cls]] = {"records": len(fam), "top": [names[i] for i in idx]}

    # Sanity check (Adebayo et al., 2018): explanations of an untrained model must look different.
    torch.manual_seed(999)
    random_model = Detector()
    rand_attr, _, _ = attributions(random_model, x, benign_baseline)
    if len(rand_attr):
        out["sanity_random_model_spearman"] = spearman(mean_abs, rand_attr.abs().mean(0).numpy())
    print("sanity", out.get("sanity_random_model_spearman"), flush=True)

    # Stability across seeds: do differently initialised federations give the same explanation?
    importance = {1: mean_abs}
    for s in (2, 3):
        a, _, _ = attributions(build(runs[s]["weights"]), x, benign_baseline)
        importance[s] = a.abs().mean(0).numpy()
    out["stability_across_seeds_spearman"] = {f"{i}-{j}": spearman(importance[i], importance[j]) for i, j in ((1, 2), (1, 3), (2, 3))}
    out["stability_top20_overlap"] = {f"{i}-{j}": len(set(np.argsort(-importance[i])[:20]) & set(np.argsort(-importance[j])[:20])) / 20 for i, j in ((1, 2), (1, 3), (2, 3))}
    print("stability", out["stability_across_seeds_spearman"], out["stability_top20_overlap"], flush=True)

    # Stability under attack: does the defended model still explain itself the same way?
    out["under_attack"] = {}
    for attack in ("sign_flip", "lie", "fang_trim", "adaptive_gate"):
        r = run_simulation(clients, SimConfig(attack=attack, n_attackers=3, defence="proposed", seed=1, rounds=args.rounds), eval_every=args.rounds)
        m = build(r["weights"])
        a, _, _ = attributions(m, x, benign_baseline)
        imp = a.abs().mean(0).numpy()
        out["under_attack"][attack] = {
            "macro_f1": r["final"]["macro_f1"],
            "spearman_to_clean": spearman(mean_abs, imp),
            "top20_overlap_to_clean": len(set(order[:20]) & set(np.argsort(-imp)[:20])) / 20,
        }
        print(attack, out["under_attack"][attack], flush=True)
    save_json("explain_eval.json", out)


if __name__ == "__main__":
    main()
