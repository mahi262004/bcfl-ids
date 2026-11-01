from __future__ import annotations

import re

import numpy as np
import pytest
import torch

from bcfl_ids import attacks as atk
from bcfl_ids import config as C
from bcfl_ids.chain import RECORD_FIELDS, audit
from bcfl_ids.data import EXPECTED_ROWS, _class_of, signed_log
from bcfl_ids.defences import multi_krum, quality_score, score_updates, trimmed_mean, weighted_mean
from bcfl_ids.explain import attack_score, completeness_error, deletion_curve, integrated_gradients, parse_feature
from bcfl_ids.ledger import InMemoryLedger, canonical_hash
from bcfl_ids.metrics import summarise, wilson
from bcfl_ids.model import Detector, initial_weights, load_flat
from bcfl_ids.scaler import aggregate_pooled, aggregate_robust, client_stats, poison_stats
from bcfl_ids.simulator import choose_attackers, class_weights, run_simulation

from pathlib import Path


# --- data ---------------------------------------------------------------------------------------
def test_published_row_total():
    assert sum(b + a for b, a in EXPECTED_ROWS.values()) == 7_062_606
    assert sum(b for b, _ in EXPECTED_ROWS.values()) == 555_932


def test_signed_log_handles_negatives():
    out = signed_log(np.array([-(np.e - 1), 0.0, np.e - 1]))
    assert out == pytest.approx([-1.0, 0.0, 1.0])


@pytest.mark.parametrize("path,label", [
    ("benign_traffic.csv", "benign"),
    ("gafgyt_attacks/tcp.csv", "gafgyt_tcp"),
    ("mirai_attacks/udpplain.csv", "mirai_udpplain"),
    ("gafgyt_attacks/combo.csv", "gafgyt_combo"),
    ("mirai_attacks/scan.csv", "mirai_scan"),
])
def test_file_to_class(path, label):
    assert C.CLASSES[_class_of(Path(path))] == label


def test_splits_keep_purge_gaps():
    assert C.SPLITS["train"][1] < C.SPLITS["val"][0] and C.SPLITS["val"][1] < C.SPLITS["test"][0]


# --- scaler -------------------------------------------------------------------------------------
def _parts():
    rng = np.random.default_rng(0)
    return [rng.normal(i, 1 + i, size=(100 + 20 * i, 5)).astype(np.float32) for i in range(9)]


def test_pooled_scale_equals_standard_scaler():
    from sklearn.preprocessing import StandardScaler

    parts = _parts()
    s = aggregate_pooled([client_stats(p) for p in parts])
    ref = StandardScaler().fit(np.concatenate(parts).astype(np.float64))
    assert s.mean == pytest.approx(ref.mean_, rel=1e-6)
    assert s.std == pytest.approx(ref.scale_, rel=1e-6)


def test_one_liar_breaks_pooled_but_not_robust_scale():
    stats = [client_stats(p) for p in _parts()]
    honest_means = np.stack([c.s / c.n for c in stats])
    stats[0], stats[1], stats[2] = poison_stats(stats[0]), poison_stats(stats[1]), poison_stats(stats[2])
    pooled, robust = aggregate_pooled(stats), aggregate_robust(stats, 3)
    assert np.abs(pooled.mean).max() > 100 * np.abs(honest_means).max()
    assert (robust.mean <= honest_means.max(0) + 1e-6).all() and (robust.mean >= honest_means.min(0) - 1e-6).all()


def test_constant_feature_left_unscaled():
    s = aggregate_pooled([client_stats(np.ones((10, 3), dtype=np.float32))])
    assert (s.std == 1.0).all()


# --- trust gate ---------------------------------------------------------------------------------
@pytest.mark.parametrize("cos,ratio", [(1.0, 0.0), (0.5, 1.0), (-0.1, 3.0), (-0.1, 0.5), (0.9, 3.0)])
def test_pass_is_at_or_above_contract_mark(cos, ratio):
    score, passed, _ = quality_score(cos, ratio, -0.1, 3.0)
    assert passed and C.SCORE_PASS <= score <= 100


@pytest.mark.parametrize("cos,ratio", [(-0.11, 1.0), (0.9, 3.01), (-1.0, 50.0), (-0.5, 4.0)])
def test_fail_is_below_contract_mark(cos, ratio):
    score, passed, reason = quality_score(cos, ratio, -0.1, 3.0)
    assert not passed and 0 <= score < C.SCORE_PASS and reason


def test_gate_catches_sign_flip_and_scaling():
    honest = torch.randn(1, 200) + 0.05 * torch.randn(9, 200)
    honest[0] = -3 * honest[0]
    honest[1] = 10 * honest[1]
    verdicts = score_updates(honest, -0.1, 3.0)
    assert not verdicts[0].passed and not verdicts[1].passed
    assert all(v.passed for v in verdicts[2:])


def test_size_check_uses_typical_update_size_not_the_norm_of_the_median():
    """Regression: when updates disagree, the median vector is short, so honest updates looked too large."""
    torch.manual_seed(0)
    updates = 0.1 * torch.randn(500) + torch.randn(9, 500)
    old_ratio = updates.norm(dim=1) / updates.median(dim=0).values.norm()
    assert (old_ratio > 2.0).all()  # the old formula would have rejected every one of these
    verdicts = score_updates(updates, 0.1, 2.0)
    assert all(v.norm_ratio < 1.2 for v in verdicts)
    assert all(v.passed for v in verdicts)


# --- aggregators --------------------------------------------------------------------------------
def test_trimmed_mean_stays_in_honest_range():
    honest = torch.randn(6, 50)
    bad = torch.full((3, 50), 1e6)
    out = trimmed_mean(torch.cat([honest, bad]), 3)
    assert (out <= honest.max(0).values + 1e-5).all() and (out >= honest.min(0).values - 1e-5).all()


def test_trimmed_mean_shrinks_trim_when_few_accepted():
    x = torch.tensor([[1.0], [2.0], [3.0]])
    assert trimmed_mean(x, 3).item() == 2.0
    assert trimmed_mean(x[:1], 3).item() == 1.0


def test_weighted_mean():
    x = torch.tensor([[0.0], [10.0]])
    assert weighted_mean(x, torch.tensor([1.0, 3.0])).item() == 7.5


def test_multi_krum_drops_outliers():
    honest = 0.01 * torch.randn(7, 20)
    bad = torch.full((2, 20), 100.0)
    assert multi_krum(torch.cat([honest, bad]), 2).abs().max() < 1


# --- attacks ------------------------------------------------------------------------------------
def test_label_flip_inverts_benign_and_attack():
    y = torch.tensor([0, 1, 5, 10])
    out = atk.poison_labels(y, "label_flip")
    assert out.tolist() == [C.TARGET_FAMILY, 0, 0, 0]


def test_targeted_attack_only_touches_target():
    y = torch.tensor([0, C.TARGET_FAMILY, 3])
    assert atk.poison_labels(y, "targeted_evasion").tolist() == [0, 0, 3]


def test_lie_stays_near_honest_mean():
    honest = torch.randn(6, 100)
    crafted = atk.craft_lie(honest, 9, 3)
    assert (crafted - honest.mean(0)).abs().max() <= 3 * honest.std(0, unbiased=False).max()


def test_minmax_respects_honest_spread():
    honest = 0.5 + 0.2 * torch.randn(6, 100)
    crafted = atk.craft_minmax(honest)
    limit = torch.cdist(honest, honest).max()
    assert torch.cdist(crafted.unsqueeze(0), honest).max() <= limit + 1e-3
    assert torch.dot(crafted - honest.mean(0), honest.mean(0)) < 0


def test_fang_goes_beyond_honest_extremes():
    honest = 1 + torch.randn(6, 100)
    crafted = atk.craft_fang(honest, torch.Generator().manual_seed(0))
    outside = (crafted <= honest.min(0).values) | (crafted >= honest.max(0).values)
    assert outside.all()


def test_adaptive_update_passes_the_gate():
    base = torch.randn(200)
    honest = base + 0.1 * torch.randn(6, 200)
    crafted = atk.craft_adaptive(honest, torch.randn(200), -0.1, 3.0, 3)
    trial = torch.cat([honest, crafted.unsqueeze(0).repeat(3, 1)])
    assert score_updates(trial, -0.1, 3.0)[-1].passed
    assert not torch.allclose(crafted, honest.mean(0))


def test_attacker_choice_is_seeded_and_varies():
    picks = {tuple(choose_attackers(C.SimConfig(attack="sign_flip", n_attackers=3, seed=s))) for s in range(10)}
    assert len(picks) > 3
    assert choose_attackers(C.SimConfig(attack="sign_flip", n_attackers=3, seed=2)) == choose_attackers(C.SimConfig(attack="sign_flip", n_attackers=3, seed=2))


# --- ledger -------------------------------------------------------------------------------------
def test_ledger_rules_match_report_arithmetic():
    led = InMemoryLedger()
    led.register(1)
    for _ in range(3):
        led.submit(1, 1, 0, "0x0", "0x0")
    assert led.reputation(1) == 40 and led.on_probation(1)
    for _ in range(5):
        led.submit(2, 1, 90, "0x0", "0x0")
    assert led.reputation(1) == 50 and not led.on_probation(1)
    for _ in range(10):
        led.submit(3, 1, 0, "0x0", "0x0")
    assert led.reputation(1) == C.REP_FLOOR


def test_ledger_refuses_bad_input():
    led = InMemoryLedger()
    led.register(1)
    with pytest.raises(ValueError):
        led.register(1)
    with pytest.raises(ValueError):
        led.submit(1, 2, 50, "0x0", "0x0")
    with pytest.raises(ValueError):
        led.submit(1, 1, 101, "0x0", "0x0")


def test_python_constants_match_solidity():
    source = (C.BLOCKCHAIN_DIR / "contracts" / "TrustLayer.sol").read_text()
    for name in ("REP_START", "REP_GAIN", "REP_LOSS", "REP_FLOOR", "REP_CAP", "REP_PROBATION", "SCORE_PASS"):
        assert int(re.search(rf"{name} = (\d+);", source).group(1)) == getattr(C, name), name


def test_canonical_hash_ignores_key_order():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


# --- metrics ------------------------------------------------------------------------------------
def test_metrics_on_known_confusion():
    cube = np.zeros((9, C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
    cube[0, 0, 0] = 90   # benign right
    cube[0, 0, 1] = 10   # false alarms
    cube[0, 1, 1] = 80   # family right
    cube[0, 1, 2] = 15   # right as attack, wrong family
    cube[0, 1, 0] = 5    # missed
    out = summarise(cube)
    assert out["benign_recall"] == pytest.approx(0.9)
    assert out["attack_recall"] == pytest.approx(0.95)
    assert out["accuracy"] == pytest.approx(170 / 200)


def test_merging_lookalike_families_forgives_only_those_mixups():
    cube = np.zeros((9, C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
    tcp, udp, syn = C.CLASSES.index("gafgyt_tcp"), C.CLASSES.index("gafgyt_udp"), C.TARGET_FAMILY
    cube[0, tcp, udp] = 50
    cube[0, udp, udp] = 50
    assert summarise(cube)["macro_f1"] < 0.5
    assert summarise(cube)["macro_f1_merged"] == pytest.approx(1.0)
    cube[0, syn, 0] = 50
    assert summarise(cube)["macro_f1_merged"] < 1.0


def test_attacker_devices_leave_headline_metrics():
    cube = np.zeros((9, C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
    cube[0, 0, 1] = 100
    cube[1, 0, 0] = 100
    assert summarise(cube, [1])["benign_recall"] == 1.0
    assert summarise(cube, [1])["devices"]["1"]["malicious"] is True


def test_wilson_interval_widens_for_small_samples():
    small, large = wilson(26, 26), wilson(2600, 2600)
    assert small[0] < large[0] < 1.0


# --- explainability -----------------------------------------------------------------------------
def _trained_model():
    torch.manual_seed(0)
    m = Detector()
    x = torch.randn(400, C.NUM_FEATURES)
    y = (x[:, 0] > 0).long() * C.TARGET_FAMILY
    opt = torch.optim.Adam(m.parameters(), 0.01)
    for _ in range(80):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x), y).backward()
        opt.step()
    return m, x


def test_integrated_gradients_completeness():
    m, x = _trained_model()
    base, sample = torch.zeros(C.NUM_FEATURES), x[:32]
    target = torch.full((32,), C.TARGET_FAMILY)
    attr = integrated_gradients(m, sample, base, target, steps=256)
    assert np.median(completeness_error(m, sample, base, target, attr)) < 0.02


def test_deletion_beats_random_on_a_model_with_one_real_feature():
    m, x = _trained_model()
    sample = x[x[:, 0] > 1][:32]
    base, target = torch.zeros(C.NUM_FEATURES), torch.full((len(sample),), C.TARGET_FAMILY)
    attr = integrated_gradients(m, sample, base, target)
    row = deletion_curve(m, sample, base, target, attr, [5])[0]
    assert row["evidence_removed_top"] > row["evidence_removed_random"]
    assert attr.abs().mean(0).argmax().item() == 0


def test_attack_score_sign():
    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    assert attack_score(logits, torch.tensor([1, 1])).tolist() == [-2.0, 2.0]


def test_parse_feature_names():
    assert parse_feature("HH_jit_L0.01_mean") == ("HH_jit", "L0.01", "mean")
    assert parse_feature("MI_dir_L5_weight") == ("MI_dir", "L5", "weight")
    assert parse_feature("HpHp_L1_pcc") == ("HpHp", "L1", "pcc")


# --- simulation ---------------------------------------------------------------------------------
def cfg(**kw):
    return C.SimConfig(rounds=kw.pop("rounds", 6), batch_size=32, **kw)


def strip(result):
    return result["final"]["accuracy"], result["reputation"], result["weights"].tolist()


def test_same_seed_gives_identical_results(clients):
    a = run_simulation(clients, cfg(rounds=3, seed=5))
    b = run_simulation(clients, cfg(rounds=3, seed=5))
    assert strip(a) == strip(b)
    assert [r["record_hash"] for r in a["records"]] == [r["record_hash"] for r in b["records"]]


def test_honest_federation_learns(clients):
    res = run_simulation(clients, cfg(defence="fedavg"))
    assert res["final"]["macro_f1"] > 0.95
    assert res["false_reject_rate"] == 0


def test_absent_classes_get_zero_weight():
    y = torch.tensor([0, 0, 0, 1])
    w = class_weights(y)
    assert w[2:].sum() == 0 and w[1] > w[0]


def test_sign_flip_hurts_fedavg_but_not_proposed(clients):
    bad = run_simulation(clients, cfg(attack="sign_flip", n_attackers=3, defence="fedavg"))
    good = run_simulation(clients, cfg(attack="sign_flip", n_attackers=3, defence="proposed"))
    assert good["final"]["macro_f1"] > 0.9
    assert good["final"]["macro_f1"] > bad["final"]["macro_f1"] + 0.1
    assert good["attacker_reject_rate"] > 0.9


def test_attackers_fall_into_probation(clients):
    res = run_simulation(clients, cfg(attack="sign_flip", n_attackers=2, defence="proposed", rounds=5))
    for a in res["config"]["attacker_ids"]:
        assert res["reputation"][a] < C.REP_PROBATION


def test_scaler_poison_handled_by_robust_scale(clients):
    res = run_simulation(clients, cfg(attack="scaler_poison", n_attackers=3, defence="proposed"))
    assert res["final"]["macro_f1"] > 0.9


@pytest.mark.parametrize("attack", [a for a in C.ATTACKS if a != "none"])
@pytest.mark.parametrize("defence", list(C.DEFENCES))
def test_every_attack_and_defence_runs_end_to_end(clients, attack, defence):
    res = run_simulation(clients, cfg(attack=attack, n_attackers=3, defence=defence, rounds=2))
    assert np.isfinite(res["final"]["accuracy"])
    assert len(res["records"]) == 18


def test_audit_detects_tampering(clients):
    led = InMemoryLedger()
    res = run_simulation(clients, cfg(rounds=2), ledger=led)
    records = [dict(r) for r in res["records"]]
    assert audit(records, led.entries)["valid"]
    records[3]["cosine"] = 0.999
    outcome = audit(records, led.entries)
    assert not outcome["valid"] and outcome["mismatches"][0]["client"] == records[3]["client"]
    assert set(RECORD_FIELDS) <= set(records[0])
    assert audit(records[:-1], led.entries)["valid"] is False


def test_all_failed_round_keeps_model(clients):
    """Thresholds nobody can meet: the global model must simply stay where it is."""
    res = run_simulation(clients, C.SimConfig(rounds=1, batch_size=32, defence="gate", cos_min=1.1, ratio_max=0.0))
    assert res["history"][0]["accepted"] == 0
    assert torch.equal(res["weights"], initial_weights(1))


def test_load_flat_roundtrip():
    m = Detector()
    w = initial_weights(3)
    load_flat(m, w)
    assert torch.equal(torch.nn.utils.parameters_to_vector(m.parameters()).detach(), w)
