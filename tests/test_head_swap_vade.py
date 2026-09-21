"""The continuous head swap must address the right column at the right step.

Every bug this file exists to catch -- an off-by-one in the step index, writing
the first column instead of the last, slicing the wrong head, patching only the
prefill -- produces a plausible generation rather than an error, and would land
in VADE's scorer looking like a result.
"""
import json

import pytest
import torch

from test_head_followups import tiny                                    # noqa: F401
from methods.head_swap_vade import (ALIGN_MODES, all_blocks, batches_by_donor_attribute,
                                    capture_donor, donor_index, head_columns, heads_for,
                                    heads_from_trace, parse_heads, patched_generate, plain_generate,
                                    resolve_head_sets, verify_readback)

HEADS = [(1, 0), (2, 3)]          # tiny model: 4 blocks, 4 heads, head_dim 8, hidden 32
HEAD_DIM = 8
MAX_NEW = 4


def pieces(tiny):
    runner, batch = tiny
    return (runner.adapter, runner.model, batch["base_input_ids"], batch["attention_mask"],
            batch["base_extra"])


def test_parse_heads_round_trips_and_rejects_junk():
    assert parse_heads("21.1,22.19, 23.3") == [(21, 1), (22, 19), (23, 3)]
    assert parse_heads("23.4 21.1") == [(21, 1), (23, 4)]
    for bad in ("21", "21.1.2", "21.1,21.1", ""):
        with pytest.raises(AssertionError):
            parse_heads(bad)


def test_head_columns_selects_only_that_blocks_heads():
    cols = head_columns([(1, 0), (2, 3), (1, 2)], block=1, head_dim=HEAD_DIM)
    assert cols.tolist() == list(range(0, 8)) + list(range(16, 24))
    assert head_columns([(1, 0)], block=9, head_dim=HEAD_DIM).numel() == 0


def test_replaying_a_runs_own_capture_is_an_exact_no_op(tiny):
    """THE identity test. Capturing from a run and replaying it into the SAME run
    writes back exactly what was already there, so the generation must be
    bit-identical. An off-by-one step index, a first-instead-of-last column, or a
    mis-sliced head all break this while still producing fluent text."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    clean = plain_generate(model, ids, mask, extra, MAX_NEW, pad_id=0)
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    assert all(v.shape[1] == MAX_NEW for v in z.values()), "one capture per forward expected"
    replayed = patched_generate(adapter, model, HEADS, z, ids, mask, extra, MAX_NEW, 0, HEAD_DIM)
    assert torch.equal(clean, replayed)


def test_a_different_donor_actually_changes_the_answer(tiny):
    """The mirror of the identity: if nothing changes when the donor genuinely
    differs, the hook is not connected and the no-op above proves nothing."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    scrambled = {b: v.flip(0).roll(1, dims=1) + 5.0 for b, v in z.items()}
    out = patched_generate(adapter, model, HEADS, scrambled, ids, mask, extra, MAX_NEW, 0, HEAD_DIM)
    assert not torch.equal(out, plain_generate(model, ids, mask, extra, MAX_NEW, pad_id=0))


def test_step_t_of_the_donor_lands_at_step_t_of_the_base(tiny):
    """Plant a donor whose step index is written into its values, then read back
    what the model actually received. Catches what the identity test cannot: a
    patch that installs step 0 at every step."""
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    planted = {block: torch.zeros(ids.shape[0], MAX_NEW, 32)}
    for t in range(MAX_NEW):
        planted[block][:, t, head * HEAD_DIM:(head + 1) * HEAD_DIM] = float(t + 1)
    seen = []
    patched_generate(adapter, model, [(block, head)], planted, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                     observers=[(block, lambda t: seen.append(float(t[0, -1, head * HEAD_DIM])))])
    assert seen == [1.0, 2.0, 3.0, 4.0]


def test_patch_touches_only_the_selected_heads_columns(tiny):
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    clean, patched = [], []
    module = adapter.get_attn_head_output_module(model, block)
    h = module.register_forward_pre_hook(lambda _m, args: clean.append(args[0][:, -1].clone()))
    try:
        plain_generate(model, ids, mask, extra, MAX_NEW, pad_id=0)
    finally:
        h.remove()
    planted = {block: torch.full((ids.shape[0], MAX_NEW, 32), 9.0)}
    patched_generate(adapter, model, [(block, head)], planted, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                     observers=[(block, lambda t: patched.append(t[:, -1].clone()))])
    owned = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
    assert torch.allclose(patched[0][:, owned], torch.full((ids.shape[0], HEAD_DIM), 9.0))
    # Step 0 is the prefill and nothing upstream of it changed, so every OTHER head
    # of this block must still read exactly what the clean run produced.
    assert torch.equal(patched[0][:, HEAD_DIM:], clean[0][:, HEAD_DIM:])


def test_the_patch_stays_live_after_the_prefill(tiny):
    """The whole point of the script. head_trace's hook no-ops once the tensor
    collapses to one column; this one must not."""
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    planted = {block: torch.full((ids.shape[0], MAX_NEW, 32), 7.0)}
    widths, values = [], []
    patched_generate(adapter, model, [(block, head)], planted, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                     observers=[(block, lambda t: (widths.append(t.shape[1]),
                                                   values.append(float(t[0, -1, head * HEAD_DIM]))))])
    assert widths[0] > 1 and set(widths[1:]) == {1}, "expected one prefill then single-column decodes"
    assert values == [7.0] * MAX_NEW, "patch died after the prefill"


def test_a_shorter_donor_is_held_not_indexed_out_of_range(tiny):
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    planted = {block: torch.zeros(ids.shape[0], 2, 32)}
    planted[block][:, 1, head * HEAD_DIM:(head + 1) * HEAD_DIM] = 3.0
    seen, stats = [], {}
    patched_generate(adapter, model, [(block, head)], planted, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                     stats=stats,
                     observers=[(block, lambda t: seen.append(float(t[0, -1, head * HEAD_DIM])))])
    assert seen == [0.0, 3.0, 3.0, 3.0], "the last donor step should be held"
    assert stats["steps_beyond_donor"] == 2


def test_readback_self_check_passes_and_catches_a_dead_patch(tiny):
    """verify_readback is the gate the real run opens with. It must pass on a
    live patch and FAIL loudly if the hook stops addressing the right column."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    assert verify_readback(adapter, model, HEADS, z, ids, mask, extra, MAX_NEW, 0, HEAD_DIM) <= 1e-3

    # Simulate the failure it exists to catch: the patch silently lands nowhere.
    # The donor must DIFFER from what the clean run produces, or a dead patch is
    # indistinguishable from a live one -- which is exactly why the identity test
    # above is not sufficient on its own.
    import methods.head_swap_vade as mod
    lying = {b: v + 100.0 for b, v in z.items()}
    real = mod.head_columns
    mod.head_columns = lambda heads, block, head_dim: (torch.empty(0, dtype=torch.long)
                                                       if _in_patch() else real(heads, block, head_dim))
    try:
        with pytest.raises(AssertionError, match="read-back mismatch"):
            mod.verify_readback(adapter, model, HEADS, lying, ids, mask, extra, MAX_NEW, 0, HEAD_DIM)
    finally:
        mod.head_columns = real


def _in_patch():
    """True while called from patched_generate (the write side) and False from
    verify_readback (the compare side), so the write lands nowhere while the
    comparison still looks at the real head columns."""
    import inspect
    return any(f.function == "patched_generate" for f in inspect.stack()[:6])


def test_void_traces_are_refused_but_a_metric_capped_one_is_not(tmp_path):
    """The guard must separate two different low-cause failures: a pre-518cf31
    ranking that is INERT (base answers survive untouched), and a valid trace
    whose ceiling is capped by answer length (base answers destroyed, just not
    transferred). Judging on `cause` alone rejects calling_code's good trace."""
    def write(name, cause, base_kept):
        f = tmp_path / name
        f.write_text(json.dumps({
            "phase1_ranked": [[23, 4], [21, 1], [23, 3]],
            "phase2": {"image_patch_only_cause": 0.95},
            "phase3_sufficiency": {"ceiling_all_traced_heads": cause,
                                   "arms": [{"k": 84, "kind": "all",
                                             "cause": cause, "base_kept": base_kept}]}}))
        return str(f)

    # The real void trace's numbers: patching 196 heads changed nothing.
    with pytest.raises(SystemExit, match="inert"):
        heads_from_trace(write("void.json", 0.0156, 0.953), 2)

    # calling_code's real numbers: low cause, but the patch DID dislodge the answer.
    assert heads_from_trace(write("capped.json", 0.0625, 0.266), 2) == [(23, 4), (21, 1)]
    # language's real numbers.
    assert heads_from_trace(write("good.json", 0.984, 0.0), 3) == [(23, 4), (21, 1), (23, 3)]


def test_head_sets_resolve_from_lists_and_traces(tmp_path):
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps({
        "phase1_ranked": [[23, 4], [21, 1], [23, 3], [22, 19]],
        "phase3_sufficiency": {"arms": [{"k": 84, "kind": "all", "cause": 0.98, "base_kept": 0.0}]}}))
    sets = resolve_head_sets([f"a=21.1,23.4", f"b={trace}#2", f"c={trace}#4"], "flags", top_k=3)
    assert sets["a"] == [(21, 1), (23, 4)]
    assert sets["b"] == [(23, 4), (21, 1)]          # trace order, not sorted
    assert len(resolve_head_sets([f"c={trace}"], "flags", top_k=3)["c"]) == 3   # falls back to --top_k
    assert all_blocks(sets) == [21, 22, 23]         # union across sets, so 22.19 counts
    with pytest.raises(AssertionError, match="NAME=SPEC"):
        resolve_head_sets(["21.1,23.4"], "flags", 3)


def test_per_attribute_sets_expand_and_are_keyed_by_donor_attribute(tmp_path):
    for a in ("capital", "currency", "language", "calling_code"):
        d = tmp_path / a
        d.mkdir()
        (d / "t.json").write_text(json.dumps({
            "phase1_ranked": [[21, hash(a) % 4], [23, 4]],
            "phase3_sufficiency": {"arms": [{"k": 84, "kind": "all", "cause": 0.9, "base_kept": 0.0}]}}))
    sets = resolve_head_sets([f"p={tmp_path}/{{attribute}}/t.json#1"], "flags", top_k=1)
    assert set(sets["p"]) == {"capital", "currency", "language", "calling_code"}
    assert heads_for(sets["p"], "language") == sets["p"]["language"]
    assert heads_for([(21, 1)], "language") == [(21, 1)], "a flat set ignores the attribute"


def test_batches_never_mix_donor_attributes():
    jobs = [{"donor_attribute": a, "n": i} for i in range(7) for a in ("capital", "language")]
    seen = []
    for attribute, chunk in batches_by_donor_attribute(jobs, 4):
        assert all(j["donor_attribute"] == attribute for j in chunk)
        seen += chunk
    assert len(seen) == len(jobs)


# --- subspace patching (the DAS hypothesis, without the training) -------------

def _orth(n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, k, generator=g, dtype=torch.float64))
    return q.float()


def test_full_rank_subspace_equals_a_plain_head_patch(tiny):
    """A subspace spanning everything must reproduce the unrestricted patch
    exactly -- the algebra reduces to it, so a mismatch is a sign error."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    donor = {b: v.flip(0) + 2.0 for b, v in z.items()}
    plain = patched_generate(adapter, model, HEADS, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM)
    full = {b: torch.eye(HEAD_DIM) for b in blocks}          # one head per block here
    sub = patched_generate(adapter, model, HEADS, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                           subspace=full)
    assert torch.equal(plain, sub)


def test_subspace_patch_moves_only_inside_the_subspace(tiny):
    """In-subspace coordinates must become the donor's; the orthogonal
    complement must stay exactly as the base produced it."""
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    heads = [(block, head)]
    P = _orth(HEAD_DIM, 3)
    donor = {block: torch.full((ids.shape[0], MAX_NEW, 32), 5.0)}
    base_seen, sub_seen = [], []
    patched_generate(adapter, model, heads, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                     subspace={block: torch.zeros(HEAD_DIM, 0)},
                     observers=[(block, lambda t: base_seen.append(t[:, -1].clone()))])
    patched_generate(adapter, model, heads, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                     subspace={block: P},
                     observers=[(block, lambda t: sub_seen.append(t[:, -1].clone()))])
    owned = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
    b0, s0 = base_seen[0][:, owned].double(), sub_seen[0][:, owned].double()
    Pd, want = P.double(), donor[block][:, 0, owned].double()
    assert torch.allclose(s0 @ Pd, want @ Pd, atol=1e-4), "in-subspace part did not become the donor's"
    perp = torch.eye(HEAD_DIM, dtype=torch.float64) - Pd @ Pd.T
    assert torch.allclose(s0 @ perp, b0 @ perp, atol=1e-4), "the complement moved"


def test_empty_subspace_is_a_no_op(tiny):
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    donor = {b: v + 50.0 for b, v in z.items()}
    clean = plain_generate(model, ids, mask, extra, MAX_NEW, pad_id=0)
    none = {b: torch.zeros(HEAD_DIM, 0) for b in blocks}
    assert torch.equal(clean, patched_generate(adapter, model, HEADS, donor, ids, mask, extra,
                                               MAX_NEW, 0, HEAD_DIM, subspace=none))


def test_readback_under_a_subspace_checks_only_the_projection(tiny):
    """The read-back must compare the PROJECTION, because only the in-subspace
    component was installed. Asserted together with the fact that makes it
    necessary: the full vector demonstrably did NOT become the donor's."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    donor = {b: v + 3.0 for b, v in z.items()}
    sub = {b: _orth(HEAD_DIM, 2, seed=b) for b in blocks}
    assert verify_readback(adapter, model, HEADS, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                           subspace=sub) <= 1e-2

    block, head = HEADS[0]
    seen = []
    patched_generate(adapter, model, HEADS, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                     subspace=sub, observers=[(block, lambda t: seen.append(t[:, -1].clone()))])
    owned = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
    full_gap = (seen[0][:, owned] - donor[block][:, 0, owned]).abs().max()
    assert full_gap > 1e-2, ("the full vector matched the donor, so this subspace patch was really a "
                             "full patch and the projection check proves nothing")


def test_subspace_needs_an_attribute_whose_values_repeat(tmp_path):
    """capital/calling_code are unique per country, so their 'value centroids'
    are the countries -- an entity subspace wearing an attribute's name."""
    import numpy as np
    from methods.head_swap_vade import build_value_subspace
    cap = tmp_path / "cap"; cap.mkdir()
    vade = tmp_path / "VADE" / "data" / "flags"; vade.mkdir(parents=True)
    attrs, blocks, hd, n = ["language", "capital"], [21], 8, 4
    (cap / "meta.json").write_text(json.dumps(
        {"attributes": attrs, "blocks": blocks, "head_dim": hd, "shape": [n * len(attrs), 1, 1, 2 * hd]}))
    np.save(str(cap / "acts_attn_head_output.npy"),
            np.random.default_rng(0).standard_normal((n * len(attrs), 1, 1, 2 * hd)).astype(np.float32))
    names = [f"C{i}" for i in range(n)]
    (cap / "index.jsonl").write_text("".join(
        json.dumps({"row": i * len(attrs) + j, "item": c, "attribute": a}) + "\n"
        for i, c in enumerate(names) for j, a in enumerate(attrs)))
    (vade / "ground_truth.json").write_text(json.dumps({"countries": {
        c: {"language": "Shared" if i < 2 else "Other", "capital": f"City{i}"} for i, c in enumerate(names)}}))
    heads = [(21, 0)]
    P, n_val, n_it = build_value_subspace(str(cap), heads, "language", 3, str(tmp_path / "VADE"), "flags")
    assert n_val == 2 and P[21].shape[0] == hd and P[21].shape[1] == 1   # capped at n_values-1
    with pytest.raises(AssertionError, match="entity subspace"):
        build_value_subspace(str(cap), heads, "capital", 3, str(tmp_path / "VADE"), "flags")


def test_transform_returning_the_donor_reproduces_a_plain_patch(tiny):
    """head_das.py's trained module rides this path. An identity transform must
    be indistinguishable from the unrestricted overwrite, or every trained
    result is measured on a different generation loop than R10/R11 were."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    donor = {b: v.flip(0) + 3.0 for b, v in z.items()}
    plain = patched_generate(adapter, model, HEADS, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM)
    via = patched_generate(adapter, model, HEADS, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                           transform={b: (lambda have, want: want) for b in blocks})
    assert torch.equal(plain, via)


def test_transform_returning_the_base_is_a_no_op(tiny):
    """The other end of the same check: a transform that keeps `have` must
    leave generation exactly as the unpatched run."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    donor = {b: v.flip(0) + 3.0 for b, v in z.items()}
    clean = plain_generate(model, ids, mask, extra, MAX_NEW, 0)
    via = patched_generate(adapter, model, HEADS, donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                           transform={b: (lambda have, want: have) for b in blocks})
    assert torch.equal(clean, via)


def test_transform_sees_one_column_per_step_with_matching_shapes(tiny):
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    donor = {block: torch.full((ids.shape[0], MAX_NEW, 32), 5.0)}
    seen = []

    def fn(have, want):
        seen.append((tuple(have.shape), tuple(want.shape)))
        return want
    patched_generate(adapter, model, [(block, head)], donor, ids, mask, extra, MAX_NEW, 0,
                     HEAD_DIM, transform={block: fn})
    assert seen, "transform was never called"
    assert all(h == w == (ids.shape[0], 1, HEAD_DIM) for h, w in seen), seen


def test_transform_and_subspace_together_are_rejected(tiny):
    adapter, model, ids, mask, extra = pieces(tiny)
    block = 1
    donor = {block: torch.full((ids.shape[0], MAX_NEW, 32), 5.0)}
    with pytest.raises(AssertionError):
        patched_generate(adapter, model, [(block, 0)], donor, ids, mask, extra, MAX_NEW, 0, HEAD_DIM,
                         subspace={block: torch.eye(HEAD_DIM)},
                         transform={block: (lambda have, want: want)})


# --------------------------------------------------------------------------
# --align: which donor step lands at which base step.
#
# These exist to separate a CONTENT effect from a STEP-ALIGNMENT one. Under
# `matched` a donor asked a different question contributes, at t>=1, its own
# answer's continuation, so a multi-token result cannot tell the two apart.
# Every failure here is silent in a real run: the generation stays fluent and
# the score just moves.
# --------------------------------------------------------------------------

def _planted(n, head, n_steps=MAX_NEW):
    """Donor whose value at step t IS t + 1, so an observer reads the index back."""
    z = torch.zeros(n, n_steps, 32)
    for t in range(n_steps):
        z[:, t, head * HEAD_DIM:(head + 1) * HEAD_DIM] = float(t + 1)
    return z


def _installed(tiny, align, n_steps=MAX_NEW):
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    seen = []
    patched_generate(adapter, model, [(block, head)], {block: _planted(ids.shape[0], head, n_steps)},
                     ids, mask, extra, MAX_NEW, 0, HEAD_DIM, align=align,
                     observers=[(block, lambda t: seen.append(float(t[0, -1, head * HEAD_DIM])))])
    return seen


def test_donor_index_table():
    assert [donor_index("matched", t, 3) for t in range(5)] == [0, 1, 2, 2, 2]
    assert [donor_index("hold0", t, 3) for t in range(5)] == [0, 0, 0, 0, 0]
    assert [donor_index("step0", t, 3) for t in range(5)] == [0, None, None, None, None]
    with pytest.raises(ValueError):
        donor_index("nearest", 0, 3)


def test_every_align_mode_agrees_at_step_zero():
    """The self-test the CLI tells the user to check first: the modes differ only
    in t>=1, so the first generated token must be identical across all three."""
    assert {donor_index(a, 0, 4) for a in ALIGN_MODES} == {0}


def test_hold0_installs_the_readout_column_at_every_step(tiny):
    assert _installed(tiny, "hold0") == [1.0] * MAX_NEW
    assert _installed(tiny, "matched") == [1.0, 2.0, 3.0, 4.0], "matched must be unchanged"


def test_step0_patches_once_and_then_releases(tiny):
    """After t=0 the hook must return None, not write step 0 again -- the
    difference between `step0` and `hold0`, and the whole falsifier."""
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    clean = []
    module = adapter.get_attn_head_output_module(model, block)
    h = module.register_forward_pre_hook(
        lambda _m, args: clean.append(float(args[0][0, -1, head * HEAD_DIM])))
    try:
        plain_generate(model, ids, mask, extra, MAX_NEW, pad_id=0)
    finally:
        h.remove()
    seen = _installed(tiny, "step0")
    assert seen[0] == 1.0, "step 0 must still be patched"
    assert seen[1:] != [1.0] * (MAX_NEW - 1), "step0 held its value -- that is hold0"
    # Released means the model's own value flows through. It is NOT the clean run's
    # value at t>=1 (step 0 was patched, so the base has diverged by then), but it
    # must be whatever this forward pass computed, which the hook never touched.
    assert all(v != 1.0 for v in seen[1:]) or seen[1:] == clean[1:]


def test_step0_ignores_a_donor_longer_than_the_base(tiny):
    """`step0` must never index past 0, so a donor of any length behaves alike and
    `steps_beyond_donor` -- meaningless here -- must stay unset."""
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    stats = {}
    seen = []
    patched_generate(adapter, model, [(block, head)], {block: _planted(ids.shape[0], head, 2)},
                     ids, mask, extra, MAX_NEW, 0, HEAD_DIM, align="step0", stats=stats,
                     observers=[(block, lambda t: seen.append(float(t[0, -1, head * HEAD_DIM])))])
    assert seen[0] == 1.0
    assert "steps_beyond_donor" not in stats, "that counter only means anything under `matched`"


def test_self_replay_is_a_no_op_under_matched_and_step0_but_not_hold0(tiny):
    """The identity test, run per mode. `matched` and `step0` only ever reinstall
    what the base already had, so both must stay bit-identical. `hold0` overwrites
    t>=1 with the prefill column, so it must install something DIFFERENT -- if it
    does not, the patch is dead past step 0 and the mode comparison would read as
    a null for the wrong reason.

    The hold0 half asserts on the INSTALLED VALUES, not the generated tokens:
    argmax is a thresholded readout that absorbs perturbations without moving
    (CLAUDE.md's rule, and verify_sites' 0.25-logit shift with identical text),
    and this tiny model duly generates the same string either way."""
    adapter, model, ids, mask, extra = pieces(tiny)
    block, head = 1, 0
    clean = plain_generate(model, ids, mask, extra, MAX_NEW, pad_id=0)
    z = capture_donor(adapter, model, sorted({b for b, _ in HEADS}), ids, mask, extra,
                      MAX_NEW, pad_id=0)
    for align in ("matched", "step0"):
        assert torch.equal(clean, patched_generate(adapter, model, HEADS, z, ids, mask, extra,
                                                   MAX_NEW, 0, HEAD_DIM, align=align)), align
    own = capture_donor(adapter, model, [block], ids, mask, extra, MAX_NEW, pad_id=0)[block]
    seen = []
    patched_generate(adapter, model, [(block, head)], {block: own}, ids, mask, extra, MAX_NEW, 0,
                     HEAD_DIM, align="hold0",
                     observers=[(block, lambda t: seen.append(t[0, -1, :HEAD_DIM].clone()))])
    assert all(torch.equal(v, seen[0]) for v in seen), "hold0 must install ONE column everywhere"
    assert not all(torch.allclose(own[0, t, :HEAD_DIM], seen[0]) for t in range(1, MAX_NEW)), \
        "the run's own t>=1 already equalled step 0, so this donor cannot test hold0"


def test_readback_checks_every_mode_at_the_steps_it_installed(tiny):
    """verify_readback is the gate head_cross opens with, and it must pass under
    each mode rather than failing on the steps a mode deliberately leaves alone."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    donor = {b: v.flip(0) for b, v in z.items()}
    for align in ALIGN_MODES:
        assert verify_readback(adapter, model, HEADS, donor, ids, mask, extra, MAX_NEW, 0,
                               HEAD_DIM, align=align) <= 1e-3, align


def test_readback_still_catches_a_dead_patch_under_step0(tiny):
    """The escape hatch to close: `step0` skips t>=1, so the check must not become
    vacuous -- a wrong value at t=0 has to still fail."""
    adapter, model, ids, mask, extra = pieces(tiny)
    blocks = sorted({b for b, _ in HEADS})
    z = capture_donor(adapter, model, blocks, ids, mask, extra, MAX_NEW, pad_id=0)
    # A merely WRONG donor is self-consistent -- verify_readback installs and
    # compares the same values, so it would pass. Kill the WRITE instead, exactly
    # as the matched-mode dead-patch test above does.
    import methods.head_swap_vade as mod
    lying = {b: v + 100.0 for b, v in z.items()}
    real = mod.head_columns
    mod.head_columns = lambda heads, block, head_dim: (torch.empty(0, dtype=torch.long)
                                                       if _in_patch() else real(heads, block, head_dim))
    try:
        with pytest.raises(AssertionError, match="read-back mismatch"):
            mod.verify_readback(adapter, model, HEADS, lying, ids, mask, extra, MAX_NEW, 0,
                                HEAD_DIM, align="step0")
    finally:
        mod.head_columns = real
