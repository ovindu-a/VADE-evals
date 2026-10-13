"""Offline checks for the Tier-2 generality experiments (E7-E11).

Tiny random Qwen2.5-VL from test_head_followups, plus a tiny random InternVL built
here for the second-family adapter. No weights, no processor download.
"""
import json
import os
import random

import pytest
import torch

from test_head_followups import tiny                                    # noqa: F401
from methods import head_overlap as ho
from methods import vrh_ablation as vrh
from methods.head_severed import aggregate, own_hit


# ---------------------------------------------------------------------------
# head_severed --gold own
# ---------------------------------------------------------------------------

def test_own_hit_reads_at_the_divergence_point():
    the_us, the_uk = [1, 2, 3], [1, 2, 4]
    assert own_hit([1, 2, 3, 9], the_us, [the_uk]) is True      # ' the United States'
    assert own_hit([1, 2, 4], the_us, [the_uk]) is False        # base answer
    assert own_hit([1, 7, 3], the_us, [the_uk]) is False        # left the shared prefix early
    assert own_hit([5, 6], [5, 6], [[5]]) is True               # the other is a prefix: index 1 separates
    assert own_hit([5], [5, 6], [[5]]) is False                 # ...and stopping before it is not a hit
    assert own_hit([5, 6], [5], [[5, 6]]) is None               # a prefix of the other: nothing separates
    assert own_hit([1, 2, 3], [1, 2, 3], [[1, 2, 3]]) is None   # identical answers
    # third: must separate from BOTH others
    assert own_hit([1, 8], [1, 8], [[1, 2], [1, 3]]) is True


def test_aggregate_excludes_undistinguished_rows_from_own_rates():
    rows = [{"distinct_first": True, "src_first": True, "base_first": False, "src_text": True,
             "base_text": False, "other_text": False, "own_src_div": True, "own_base_div": False},
            {"distinct_first": False, "src_first": False, "base_first": False, "src_text": False,
             "base_text": False, "other_text": True, "own_src_div": None, "own_base_div": None}]
    a = aggregate(rows)
    assert a["own_src_div"] == 1.0 and a["n_own_src_div"] == 1


# ---------------------------------------------------------------------------
# head_overlap (E7 / E10)
# ---------------------------------------------------------------------------

def test_block_null_preserves_per_block_counts_and_pool_null_stays_in_window():
    rng = random.Random(0)
    spec = {"heads": [(21, 1), (21, 5), (23, 4)], "pool_blocks": [21, 22, 23]}
    for _ in range(50):
        d = ho.draw(rng, spec, 28, 28, "block")
        assert sorted(b for b, _ in d) == [21, 21, 23] and len(d) == 3
        p = ho.draw(rng, spec, 28, 28, "pool")
        assert len(p) == 3 and all(b in (21, 22, 23) for b, _ in p)


def test_overlap_test_identical_sets_are_significant_and_disjoint_blocks_are_not():
    a = {"heads": [(21, h) for h in range(8)], "pool_blocks": [19, 20, 21, 22, 23]}
    r = ho.overlap_test(a, a, 28, 28, "pool", 2000)
    assert r["observed"] == 8 and r["p"] < 0.01
    b = {"heads": [(16, h) for h in range(8)], "pool_blocks": None}
    r = ho.overlap_test(a, b, 28, 28, "block", 200)
    assert r["observed"] == 0 and r["null_mean"] == 0 and r["p"] == 1.0


def test_consensus_counts_heads_in_m_sets():
    sets = [{"heads": [(1, 1), (1, 2)], "pool_blocks": [1]}, {"heads": [(1, 1), (1, 3)], "pool_blocks": [1]},
            {"heads": [(1, 1), (1, 2)], "pool_blocks": [1]}]
    c = ho.consensus_test(sets, 2, 4, 28, "pool", 200)
    assert c["observed"] == 2 and c["heads"] == ["1.1", "1.2"]


def test_published_lists_parse_and_unknown_base_emits_both_readings(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"lists": [
        {"name": "x", "applies_to": ["M"], "layer_base": None, "head_base": 0, "heads": [[16, 0], [19, 17]]},
        {"name": "y", "applies_to": ["M"], "layer_base": 0, "head_base": 0, "heads": [[23, 6]]},
        {"name": "z", "applies_to": ["other"], "layer_base": 0, "head_base": 0, "heads": [[1, 1]]}]}))
    got = ho.load_published(str(path), "M")
    assert set(got) == {"x", "x@1based", "y"}
    assert got["x"]["heads"] == [(16, 0), (19, 17)]
    assert got["x@1based"]["heads"] == [(15, 0), (18, 17)]       # head base is KNOWN 0, only the layer shifts
    assert got["y"]["index_note"] == ""


def test_shipped_published_file_fits_its_models():
    shapes = {"Qwen/Qwen2.5-VL-7B-Instruct": (28, 28), "OpenGVLab/InternVL3-8B-hf": (28, 28),
              "Qwen/Qwen2.5-VL-3B-Instruct": (36, 16)}
    for model, (n_l, n_h) in shapes.items():
        lists = ho.load_published(ho.PUBLISHED, model)
        assert lists, model
        for name, s in lists.items():
            assert all(0 <= b < n_l and 0 <= h < n_h for b, h in s["heads"]), (model, name)
    p = ho.load_published(ho.PUBLISHED, "Qwen/Qwen2.5-VL-7B-Instruct")
    assert len(p["park_vrh"]["heads"]) == 20 and len(p["lietzow_promoting"]["heads"]) == 21


# ---------------------------------------------------------------------------
# vrh_ablation (E9)
# ---------------------------------------------------------------------------

def test_block_matched_random_avoids_list_and_exclusions():
    heads = [(16, 0), (16, 1), (19, 17)]
    out = vrh.block_matched_random(heads, 28, random.Random(1), exclude=[(16, 2)])
    assert sorted(b for b, _ in out) == [16, 16, 19]
    assert not set(out) & {(16, 0), (16, 1), (19, 17), (16, 2)}


def test_box_mass_and_percentile():
    attn = torch.zeros(2, 5, 6)
    attn[0, 3:, 1] = 0.6
    attn[0, 3:, 2] = 0.2
    attn[1, 3:, 4] = 1.0
    m = vrh.box_mass(attn, [3, 4], [1, 2])
    torch.testing.assert_close(m, torch.tensor([0.8, 0.0]))
    per, mean = vrh.percentile_of({(0, 0): 0.8, (0, 1): 0.0, (1, 0): 0.4}, [(0, 0)])
    assert per == {"0.0": 100.0} and mean == 100.0


def test_zero_hook_zeroes_exactly_the_heads_at_every_position(tiny):
    runner, batch = tiny
    adapter, model = runner.adapter, runner.model
    heads, head_dim = [(1, 2), (3, 0)], runner.head_dim
    seen = {}
    handles = vrh.register_zero(adapter, model, heads, head_dim)
    for b in (1, 2, 3):
        handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(
            lambda m, a, _b=b: seen.__setitem__(_b, a[0].detach().clone())))
    try:
        with torch.no_grad():
            model(input_ids=batch["base_input_ids"], **batch["base_extra"])
    finally:
        for h in handles:
            h.remove()
    assert not seen[1][..., 2 * head_dim:3 * head_dim].any() and seen[1][..., :2 * head_dim].abs().sum() > 0
    assert not seen[3][..., :head_dim].any()
    assert seen[2].abs().sum() > 0                                   # untouched block


def test_mover_write_equals_o_proj_of_the_masked_input(tiny):
    runner, batch = tiny
    adapter, model, hd = runner.adapter, runner.model, runner.head_dim
    movers = [(2, 1), (2, 3)]
    sink = {}
    handles = vrh.register_mover_capture(adapter, model, movers, sink)
    try:
        with torch.no_grad():
            model(input_ids=batch["base_input_ids"], **batch["base_extra"])
    finally:
        for h in handles:
            h.remove()
    got = vrh.mover_write(adapter, model, sink, movers, hd)
    z = sink[2].clone()
    keep = torch.zeros(z.shape[-1], dtype=torch.bool)
    for _, h in movers:
        keep[h * hd:(h + 1) * hd] = True
    z[:, ~keep] = 0
    o_proj = adapter.get_attn_head_output_module(model, 2)
    torch.testing.assert_close(got, o_proj(z).float(), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# InternVL adapter (E8)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_internvl():
    from transformers import InternVLConfig, InternVLForConditionalGeneration
    torch.manual_seed(7)
    config = InternVLConfig(
        vision_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
                           image_size=[28, 28], patch_size=[7, 7]),
        text_config=dict(model_type="qwen2", vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, pad_token_id=0),
        image_token_id=9, image_seq_length=4, downsample_ratio=0.5)
    config._attn_implementation = "eager"
    model = InternVLForConditionalGeneration(config).eval()
    from methods.adapters.registry import get_adapter
    return get_adapter("OpenGVLab/InternVL3-8B-hf"), model


def _internvl_inputs():
    ids = torch.tensor([[3, 9, 9, 9, 9, 5, 12, 13, 14]])
    return ids, torch.randn(1, 3, 28, 28)


def test_internvl_adapter_shapes_and_grid(tiny_internvl):
    adapter, model = tiny_internvl
    assert type(adapter).__name__ == "InternVLAdapter" and adapter.shares_object_grid is False
    assert len(adapter.get_decoder_layers(model)) == 3
    assert adapter.n_attention_heads(model) == 4 and adapter.hidden_size(model) == 32
    assert adapter.intermediate_size(model) == 48
    assert adapter.get_mlp_hidden_module(model, 0).in_features == 48
    g = adapter.vision_patch_grid(model, 336, 336)
    assert (g["grid_rows"], g["grid_cols"], g["n_tokens"]) == (2, 2, 4)


def test_internvl_unembed_and_exact_dla_additivity(tiny_internvl):
    adapter, model = tiny_internvl
    ids, px = _internvl_inputs()
    resid = {}
    h = adapter.get_decoder_layers(model)[-1].register_forward_hook(
        lambda m, i, o: resid.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0, -1].detach()))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, pixel_values=px)
    finally:
        h.remove()
    logits = out.logits[0, -1]
    torch.testing.assert_close(adapter.unembed(model, resid["x"]), logits, atol=1e-5, rtol=1e-4)
    toks = torch.arange(10)
    dla = adapter.final_norm_scale(model, resid["x"]) * (adapter.logit_direction(model, toks) @ resid["x"].float())
    torch.testing.assert_close(dla, logits[toks].float(), atol=1e-5, rtol=1e-4)


def test_internvl_head_zeroing_moves_the_logits(tiny_internvl):
    adapter, model = tiny_internvl
    ids, px = _internvl_inputs()
    with torch.no_grad():
        ref = model(input_ids=ids, pixel_values=px).logits[0, -1]
        hs = vrh.register_zero(adapter, model, [(1, 0), (1, 1), (1, 2), (1, 3)], 8)
        try:
            ko = model(input_ids=ids, pixel_values=px).logits[0, -1]
        finally:
            for h in hs:
                h.remove()
        again = model(input_ids=ids, pixel_values=px).logits[0, -1]
    assert (ko - ref).abs().max() > 1e-4
    torch.testing.assert_close(again, ref)


def test_object_sets_refused_for_a_model_off_the_qwen_grid(tiny_internvl):
    from methods.common.entities import load_entity_assets, resolve_position_set
    from methods.head_swap_vade import DEFAULT_VADE_ROOT
    if not os.path.isdir(os.path.join(DEFAULT_VADE_ROOT, "data", "flags")):
        pytest.skip("VADE checkout not present")
    adapter, model = tiny_internvl
    assets = load_entity_assets(DEFAULT_VADE_ROOT, "flags")
    flat, n, last = resolve_position_set("full_image", assets, adapter, model)
    assert n == 4 and flat == [0, 1, 2, 3] and not last
    assert resolve_position_set("last_token", assets, adapter, model)[2]
    with pytest.raises(AssertionError, match="only 'full_image' and 'last_token'"):
        resolve_position_set("flag_only", assets, adapter, model)
