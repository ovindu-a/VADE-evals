"""Offline checks for the Tier-3 code changes (FOLLOWUP_PLAN.md section 2).

C1  vrh_ablation --n_random_movers     block-matched mover nulls, Park draws untouched
C4  text_entity_contrast --mover_subsets  single / pair / leave-one-out mover sets
C5  head_swap_vade --tuples_dir / entity attributes / --alphas (dose install)
C11 head_trace's tag carries a non-test split
C13 head_decode_trace --heads             an explicit head set + its random control

Every default must be unchanged, so each change is tested against a re-statement
of the OLD logic as well as for its new behaviour. Tiny random Qwen2.5-VL from
test_head_followups where a model is needed; no network, no weights.
"""
import json
import random
from collections import Counter

import pytest
import torch

from test_head_followups import tiny                                    # noqa: F401
from methods import head_decode_trace as hdt
from methods import head_swap_vade as hsv
from methods import text_entity_contrast as tec
from methods import vrh_ablation as vrh
from methods.head_trace import trace_tag

MOVERS = hsv.parse_heads(hsv.COMMON10)
PUBLISHED = {"park_vrh": {"heads": [(14, 1), (14, 7), (15, 2), (16, 3), (16, 9), (19, 0)]},
             "park_vrh@1based": {"heads": [(13, 0), (13, 6), (14, 1), (15, 2)]}}


# ---------------------------------------------------------------------------
# C1: vrh_ablation --n_random_movers
# ---------------------------------------------------------------------------

def _old_arms(names, published, movers, n_random, n_heads, seed):
    """vrh_ablation.main()'s arm loop before C1, verbatim."""
    rng = random.Random(seed)
    arms = [("clean", [])]
    for n in names:
        arms.append((f"zero:{n}", published[n]["heads"]))
        for i in range(n_random):
            arms.append((f"zero:{n}:random{i}", vrh.block_matched_random(published[n]["heads"], n_heads, rng,
                                                                         exclude=movers)))
    arms.append(("zero:common10", movers))
    return arms


@pytest.mark.parametrize("n_random", [0, 1, 3])
def test_vrh_arms_without_mover_nulls_equal_the_old_list(n_random):
    names = list(PUBLISHED)
    old = _old_arms(names, PUBLISHED, MOVERS, n_random, 28, seed=0)
    assert vrh.build_arms(names, PUBLISHED, MOVERS, n_random, 28, 0) == old
    assert vrh.build_arms(names, PUBLISHED, MOVERS, n_random, 28, 0, n_random_movers=0) == old


def test_vrh_mover_nulls_are_block_matched_disjoint_deterministic_and_leave_park_draws_alone():
    names = list(PUBLISHED)
    old = _old_arms(names, PUBLISHED, MOVERS, 3, 28, seed=5)
    new = vrh.build_arms(names, PUBLISHED, MOVERS, 3, 28, 5, n_random_movers=5)
    assert new[:len(old)] == old, "existing arms (incl. the Park random draws) must be unchanged and first"
    extra = new[len(old):]
    assert [n for n, _ in extra] == [f"zero:common10:random{i}" for i in range(5)]
    want = Counter(b for b, _ in MOVERS)
    assert want == {21: 2, 22: 4, 23: 4}
    for _, hs in extra:
        assert Counter(b for b, _ in hs) == want
        assert not set(hs) & set(MOVERS), "a mover null must never contain a mover"
        assert len(set(hs)) == len(hs) and all(0 <= h < 28 for _, h in hs)
    assert len({tuple(hs) for _, hs in extra}) == 5, "five independent draws expected"
    assert vrh.build_arms(names, PUBLISHED, MOVERS, 3, 28, 5, n_random_movers=5) == new
    # The movers' RNG is seed + 1, so the first mover draw does not depend on how many
    # Park draws came before it.
    alone = vrh.build_arms(["park_vrh"], PUBLISHED, MOVERS, 0, 28, 5, n_random_movers=1)
    assert alone[-1] == extra[0]


def test_vrh_mover_rank_is_only_skipped_for_the_unrandomized_mover_arm():
    """main() nulls mover_name_rank by EXACT name; the random mover arms keep the
    movers intact and must get a rank, so their names must not collide with it."""
    src = open(vrh.__file__).read()
    assert 'arm == "zero:common10"' in src
    names = [n for n, _ in vrh.build_arms(["park_vrh"], PUBLISHED, MOVERS, 0, 28, 0, n_random_movers=2)]
    assert names.count("zero:common10") == 1


# ---------------------------------------------------------------------------
# C4: text_entity_contrast --mover_subsets
# ---------------------------------------------------------------------------

def test_mover_subsets_of_the_ten_movers():
    sets = tec.mover_subsets(MOVERS)
    singles = {k: v for k, v in sets.items() if k.startswith("single:")}
    pairs = {k: v for k, v in sets.items() if k.startswith("pair:")}
    loos = {k: v for k, v in sets.items() if k.startswith("loo:")}
    assert (len(singles), len(pairs), len(loos), len(sets)) == (10, 45, 10, 65)
    assert singles["single:21.1"] == [(21, 1)]
    assert pairs["pair:21.1+23.17"] == [(21, 1), (23, 17)]
    assert "pair:23.17+21.1" not in pairs, "pairs are unordered, named in mover order"
    assert {frozenset(v) for v in pairs.values()} == {frozenset(p) for p in
                                                       __import__("itertools").combinations(MOVERS, 2)}
    assert loos["loo:-22.13"] == [h for h in MOVERS if h != (22, 13)]
    assert all(len(v) == 9 and len(set(v)) == 9 for v in loos.values())
    assert set(loos) == {f"loo:-{b}.{h}" for b, h in MOVERS}


def test_mover_subsets_degenerate_sizes():
    assert tec.mover_subsets([]) == {}
    one = tec.mover_subsets([(21, 1)])
    assert one == {"single:21.1": [(21, 1)], "loo:-21.1": []}    # empty loo is skipped by `if not heads`


def test_many_set_names_aggregate_and_print():
    """summarize_arms / print_table have no fixed arm list; 65 extra arms must just work."""
    from methods.circuit_common import print_table, summarize_arms
    recs = []
    for name in ["clean", "source_clean"] + [f"install:{n}" for n in tec.mover_subsets(MOVERS)]:
        for i in range(3):
            recs.append({"row_index": i, "attribute": "capital", "cond": "image", "arm": name,
                         "src_hit": name != "clean", "base_hit": name == "clean", "ld": float(i)})
    summ = summarize_arms(recs, {0, 1, 2})
    assert len(summ) == 67 and summ["install:loo:-23.17"]["n"] == 3
    print_table(summ, "smoke")


# ---------------------------------------------------------------------------
# C5: head_swap_vade -- entity attributes, --tuples_dir, --alphas
# ---------------------------------------------------------------------------

def _write_trace(path, heads):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"phase1_ranked": [list(h) for h in heads]}))


def test_attribute_expansion_uses_the_given_list_and_defaults_to_flags(tmp_path):
    animals = ("native_continent", "primary_habitat", "taxonomic_class", "reproduction", "body_covering")
    for i, a in enumerate(animals + hsv.ATTRIBUTES):
        _write_trace(tmp_path / a / "t.json", [(10, i), (11, i)])
    spec = [f"p={tmp_path}/{{attribute}}/t.json#2"]
    assert list(hsv.resolve_head_sets(spec, "flags", 2)["p"]) == list(hsv.ATTRIBUTES)
    got = hsv.resolve_head_sets(spec, "animals", 2, attributes=animals)["p"]
    assert list(got) == list(animals) and got["reproduction"] == [(10, 3), (11, 3)]


def test_flags_entity_attributes_are_ATTRIBUTES_in_order():
    import os
    gt = os.path.join(hsv.DEFAULT_VADE_ROOT, "data", "flags", "ground_truth.json")
    if not os.path.exists(gt):
        pytest.skip("sibling VADE checkout not present")
    assert hsv.entity_attributes(json.load(open(gt))) == hsv.ATTRIBUTES


def _write_tuples(root, attrs, rows_per_attr, offset=0):
    for a in attrs:
        d = root / a
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "test.jsonl", "w") as f:
            for i in range(rows_per_attr):
                f.write(json.dumps({"target_attribute": a, "queried": attrs[i % len(attrs)],
                                    "base": f"B{i}", "source": f"S{i}", "template_id": "t_v1",
                                    "row_index": offset + i}) + "\n")


def test_load_jobs_reads_tuples_dir_when_given_and_vade_data_otherwise(tmp_path):
    attrs = ("x", "y")
    _write_tuples(tmp_path / "vade" / "data" / "ent" / "tuples", attrs, 4)
    pruned = tmp_path / "pruned"
    _write_tuples(pruned, attrs, 2, offset=100)
    jobs, n = hsv.load_jobs(str(tmp_path / "vade"), "ent", "test", attrs, "queried", 0, 0)
    assert n == 8 and sorted(ri for j in jobs for _, ri in j["rows"]) == [0, 0, 1, 1, 2, 2, 3, 3]
    jobs, n = hsv.load_jobs(str(tmp_path / "vade"), "ent", "test", attrs, "queried", 0, 0, tuples_dir=str(pruned))
    assert n == 4 and sorted(ri for j in jobs for _, ri in j["rows"]) == [100, 100, 101, 101]


def test_alpha_tag_names():
    assert [hsv.alpha_tag(a) for a in (0, 0.25, 0.5, 0.75, 1, 1.0)] == ["a0", "a0.25", "a0.5", "a0.75", "a1", "a1"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_alpha_transform_is_exact_at_both_ends_and_linear_between(dtype):
    torch.manual_seed(0)
    have = (torch.randn(3, 1, 16) * torch.logspace(-3, 3, 16)).to(dtype)
    want = (torch.randn(3, 1, 16) * torch.logspace(3, -3, 16)).to(dtype)
    tr = hsv.alpha_transform([22, 21, 22], 0.0)
    assert sorted(tr) == [21, 22]
    assert torch.equal(tr[21](have, want).to(dtype), have)
    assert torch.equal(hsv.alpha_transform([21], 1.0)[21](have, want).to(dtype), want)
    mid = hsv.alpha_transform([21], 0.25)[21](have, want)
    assert torch.allclose(mid, have.float() + 0.25 * (want.float() - have.float()), rtol=1e-5, atol=1e-6)


def _scrambled(tiny, heads, max_new):
    runner, batch = tiny
    adapter, model = runner.adapter, runner.model
    ids, mask, extra = batch["base_input_ids"], batch["attention_mask"], batch["base_extra"]
    z = hsv.capture_donor(adapter, model, sorted({b for b, _ in heads}), ids, mask, extra, max_new, pad_id=0)
    return adapter, model, ids, mask, extra, {b: v.flip(-1) * 3.0 + 1.0 for b, v in z.items()}


def test_alpha_one_is_the_plain_heads_arm_and_alpha_zero_is_clean(tiny):
    heads, max_new, head_dim = [(1, 0), (2, 3)], 4, 8
    adapter, model, ids, mask, extra, z = _scrambled(tiny, heads, max_new)
    blocks = [b for b, _ in heads]
    plain = hsv.patched_generate(adapter, model, heads, z, ids, mask, extra, max_new, 0, head_dim)
    clean = hsv.plain_generate(model, ids, mask, extra, max_new, pad_id=0)
    assert not torch.equal(plain, clean), "the donor must move the answer, or the identities prove nothing"
    one = hsv.patched_generate(adapter, model, heads, z, ids, mask, extra, max_new, 0, head_dim,
                               transform=hsv.alpha_transform(blocks, 1.0))
    zero = hsv.patched_generate(adapter, model, heads, z, ids, mask, extra, max_new, 0, head_dim,
                                transform=hsv.alpha_transform(blocks, 0.0))
    assert torch.equal(one, plain)
    assert torch.equal(zero, clean)


def test_alpha_half_installs_the_midpoint_at_the_prefill(tiny):
    """Read back what the model received at step 0 (nothing upstream of block 1's
    o_proj differs at the prefill, so `have` is the clean value)."""
    block, head, head_dim, max_new = 1, 0, 8, 2
    adapter, model, ids, mask, extra, z = _scrambled(tiny, [(block, head)], max_new)
    clean_in = []
    h = adapter.get_attn_head_output_module(model, block).register_forward_pre_hook(
        lambda _m, a: clean_in.append(a[0][:, -1].clone()))
    try:
        hsv.plain_generate(model, ids, mask, extra, max_new, pad_id=0)
    finally:
        h.remove()
    seen = []
    hsv.patched_generate(adapter, model, [(block, head)], z, ids, mask, extra, max_new, 0, head_dim,
                         transform=hsv.alpha_transform([block], 0.5),
                         observers=[(block, lambda t: seen.append(t[:, -1].clone()))])
    sl = slice(head * head_dim, (head + 1) * head_dim)
    assert torch.allclose(seen[0][:, sl], 0.5 * (clean_in[0][:, sl] + z[block][:, 0, sl]), atol=1e-5)
    assert torch.equal(seen[0][:, head_dim:], clean_in[0][:, head_dim:]), "other heads must be untouched"


def test_head_swap_dry_run_arm_names_with_alphas(tmp_path, capsys, monkeypatch):
    import os
    import sys
    if not os.path.exists(os.path.join(hsv.DEFAULT_VADE_ROOT, "data", "flags", "tuples")):
        pytest.skip("sibling VADE checkout not present")
    base = ["head_swap_vade.py", "--dry_run", "--limit_pairs", "2", "--out_dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", base)
    hsv.main()
    default = capsys.readouterr().out
    assert "['clean', 'heads_common5', 'heads_common10', 'random_heads_common5', 'random_heads_common10', " \
           "'full_image']" in default
    monkeypatch.setattr(sys, "argv", base + ["--head_sets", f"common10={hsv.COMMON10}", "--arms", "heads",
                                             "random_heads", "--alphas", "0", "0.5", "1"])
    hsv.main()
    out = capsys.readouterr().out
    assert "['heads_common10_a0', 'heads_common10_a0.5', 'heads_common10_a1', 'random_heads_common10']" in out
    monkeypatch.setattr(sys, "argv", base + ["--alphas", "1", "1.0"])
    with pytest.raises(AssertionError, match="repeat an arm name"):
        hsv.main()


# ---------------------------------------------------------------------------
# C11: head_trace tag
# ---------------------------------------------------------------------------

def test_trace_tag_unchanged_on_test_split_and_suffixed_otherwise():
    assert trace_tag(21, [21, 22, 23]) == "head_trace_patch21_blocks21-23"
    assert trace_tag(21, [21, 22, 23], split="test") == "head_trace_patch21_blocks21-23"
    assert trace_tag(21, [22]) == "head_trace_patch21_blocks22"
    assert trace_tag(21, [21, 23]) == "head_trace_patch21_blocks21.23"
    assert trace_tag(21, None) == "head_trace_patch21_blocksdefault"
    assert trace_tag(21, [21, 22, 23], text_match=True) == "head_trace_patch21_blocks21-23_text"
    assert trace_tag(21, [21, 22, 23], split="train") == "head_trace_patch21_blocks21-23_train"
    assert trace_tag(21, [21, 22, 23], True, "train") == "head_trace_patch21_blocks21-23_text_train"


# ---------------------------------------------------------------------------
# C13: head_decode_trace --heads
# ---------------------------------------------------------------------------

RANKED = [(b, h) for b in (21, 22, 23) for h in (3, 1, 0, 2, 9, 4, 5, 6)]


def _old_selections(ranked, head_ks, random_repeats, seed):
    """head_decode_trace.main()'s selection block before C13, verbatim."""
    ks = sorted({min(k, len(ranked)) for k in head_ks})
    selections = [(f'top{k}', ranked[:k]) for k in ks]
    for k in ks:
        for repeat in range(random_repeats):
            rng = random.Random(seed + 1000 + 100 * k + repeat)
            selections.append((f'random{k}_r{repeat}', rng.sample(ranked, k)))
    return selections


@pytest.mark.parametrize("ks,reps", [([8, 16], 1), ([10], 2), ([100], 1), ([4, 8], 0)])
def test_selections_without_heads_are_unchanged(ks, reps):
    assert hdt.build_selections(RANKED, ks, reps, 7) == _old_selections(RANKED, ks, reps, 7)


def test_explicit_heads_add_custom_and_a_same_size_random_control():
    custom = hsv.parse_heads("23.4,21.1,22.3,21.3,22.1,23.0,23.2,22.9,21.9,23.5")
    old = _old_selections(RANKED, [8, 16], 1, 7)
    sel = hdt.build_selections(RANKED, [8, 16], 1, 7, custom=custom, patch_layer=21)
    labels = [n for n, _ in sel]
    assert labels == ["top8", "top16", "custom10", "random8_r0", "random16_r0", "random10_r0"]
    assert dict(sel)["custom10"] == sorted(custom)
    for n, hs in old:
        assert dict(sel)[n] == hs, f"{n} changed"
    want = random.Random(7 + 1000 + 100 * 10).sample(RANKED, 10)
    assert dict(sel)["random10_r0"] == want, "the control must be drawn exactly like top-k's"
    # k already a --head_ks size: its existing control is reused, not duplicated.
    sel = hdt.build_selections(RANKED, [10], 2, 7, custom=custom, patch_layer=21)
    assert [n for n, _ in sel] == ["top10", "custom10", "random10_r0", "random10_r1"]
    assert len({n for n, _ in sel}) == len(sel)


def test_explicit_heads_are_validated():
    with pytest.raises(ValueError, match="patch layer"):
        hdt.build_selections(RANKED, [8], 1, 0, custom=[(20, 1), (21, 1)], patch_layer=21)
    with pytest.raises(ValueError, match="ranking"):
        hdt.build_selections(RANKED[:4], [2], 1, 0, custom=hsv.parse_heads("21.1,21.2,21.3,21.4,21.5"))
    with pytest.raises(AssertionError):
        hsv.parse_heads("21.1,21.1")


def _fake_trace_and_vade(tmp_path):
    vade = tmp_path / "VADE"
    ent = vade / "data" / "flags"
    ent.mkdir(parents=True)
    (ent / "a.png").write_bytes(b"x")
    (ent / "b.png").write_bytes(b"x")
    (ent / "ground_truth.json").write_text(json.dumps(
        {"countries": {"A": {"image": "a.png"}, "B": {"image": "b.png"}}}))
    tup = vade / "models" / "Qwen2.5-VL-7B-Instruct" / "flags" / "tuples" / "calling_code"
    tup.mkdir(parents=True)
    (tup / "test.jsonl").write_text("".join(json.dumps(
        {"rule": "match_source", "base": "A", "source": "B", "row_index": i, "queried": "calling_code",
         "template_id": "t_v1"}) + "\n" for i in range(4)))
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "entity": "flags", "attribute": "calling_code", "patch_layer": 21, "positions": "full_image",
        "blocks": [21, 22, 23], "n_rows": 4, "seed": 0,
        "phase1": [{"block": b, "head": h, "delta_resid": float(len(RANKED) - i)}
                   for i, (b, h) in enumerate(RANKED)]}))
    return vade, trace


def test_heads_is_part_of_the_config_and_the_output_directory(tmp_path):
    from methods.head_followup_common import Results, prepare
    vade, trace = _fake_trace_and_vade(tmp_path)
    common = ["--trace", str(trace), "--vade_root", str(vade)]
    a = hdt.build_parser().parse_args(common)
    b = hdt.build_parser().parse_args(common + ["--heads", hsv.COMMON10])
    _, ranked_a, _, cfg_a, out_a = prepare(a, "decode")
    _, ranked_b, _, cfg_b, out_b = prepare(b, "decode")
    assert ranked_a == ranked_b == RANKED
    assert cfg_a["heads"] is None and cfg_b["heads"] == hsv.COMMON10
    assert out_a != out_b, "default out_dir must differ, or a --heads run would resume into a plain one"
    Results(tmp_path / "out", cfg_a)
    with pytest.raises(ValueError, match="different configuration"):
        Results(tmp_path / "out", cfg_b)


def test_decode_dry_run_with_heads(tmp_path, capsys, monkeypatch):
    import sys
    vade, trace = _fake_trace_and_vade(tmp_path)
    monkeypatch.setattr(sys, "argv", ["head_decode_trace.py", "--trace", str(trace), "--vade_root", str(vade),
                                      "--heads", hsv.COMMON10, "--head_ks", "10", "--dry_run"])
    hdt.main()
    out = capsys.readouterr().out
    assert "Explicit head set custom10" in out and "Dry run passed" in out
