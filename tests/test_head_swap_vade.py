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
from methods.head_swap_vade import (all_blocks, batches_by_donor_attribute, capture_donor,
                                    head_columns, heads_for, heads_from_trace, parse_heads,
                                    patched_generate, plain_generate, resolve_head_sets,
                                    verify_readback)

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
