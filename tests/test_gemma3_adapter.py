"""Offline tests for the Gemma-3 adapter (PAPER_RUNBOOK.md X15).

Two kinds:
  * a tiny RANDOM Gemma3ForConditionalGeneration (3 blocks x 4 heads of 16 = o_proj input 64 vs
    hidden 48, so head_dim != hidden // heads exactly as on the real 4B/12B; a 2x2 image grid;
    post-sublayer norms with randomised (1 + weight) gains) for: module paths, the bidirectional
    image mask shim in forward AND generate, the post-norm attn/mlp output sites, exact DLA and
    per-head writes, raster order, the head patch / read-back path, left-pad invariance;
  * the REAL processor/tokenizer of google/gemma-3-4b-it (12b shares it), SKIPPED unless already in
    the HF cache: BOS, the image block, template/encode identities, the computed object box.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods import head_swap_vade as hsv                               # noqa: E402
from methods.adapters.registry import get_adapter                       # noqa: E402

MODEL_ID = "google/gemma-3-4b-it"
IMG, BOI, EOI = 70, 68, 69     # image soft token / start / end of image in the tiny vocab
N_IMG = 4                      # 28px / 7px patches = 4x4, pooled 2x2 -> 2x2 = mm_tokens_per_image 4


def _tiny(seed=11):
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration
    torch.manual_seed(seed)
    cfg = Gemma3Config(
        text_config=dict(vocab_size=80, hidden_size=48, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                         sliding_window=64, layer_types=["sliding_attention", "sliding_attention", "full_attention"],
                         pad_token_id=0, bos_token_id=2, eos_token_id=1),
        vision_config=dict(hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
                           image_size=28, patch_size=7),
        mm_tokens_per_image=N_IMG, image_token_index=IMG, boi_token_index=BOI, eoi_token_index=EOI)
    cfg._attn_implementation = "eager"
    model = Gemma3ForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        # Gemma norms are zero-initialised offsets (make 1 + w matter), and so is the projector's
        # weight (which would hand every image token the same zero embedding)
        for name, p in model.named_parameters():
            if ("norm" in name and "vision" not in name) or "mm_input_projection_weight" in name:
                p.normal_(0, 0.3)
    return model


@pytest.fixture
def tiny_gemma3():
    adapter = get_adapter(MODEL_ID)
    return adapter, adapter.prepare_model(_tiny())


def _inputs(seed=0, batch=1):
    """[BOS] <sot> user \\n\\n <boi> <img>x4 <eoi> question <eot> <sot> model prefill -- the real layout,
    with arbitrary small ids."""
    ids = torch.tensor([[2, 20, 21, 22, BOI] + [IMG] * N_IMG + [EOI, 30, 31, 32, 23, 20, 24, 40, 41]] * batch)
    g = torch.Generator().manual_seed(seed)
    return ids, {"pixel_values": torch.randn(batch, 3, 28, 28, generator=g)}


# ------------------------------------------------------------------ registry / module tree

def test_registry_routes_gemma3_and_leaves_others_alone():
    for mid in ("google/gemma-3-4b-it", "google/gemma-3-12b-it"):
        a = get_adapter(mid)
        assert type(a).__name__ == "Gemma3Adapter" and a.shares_object_grid is False
    assert type(get_adapter("Qwen/Qwen2.5-VL-7B-Instruct")).__name__ == "Qwen25VLAdapter"
    assert type(get_adapter("HuggingFaceM4/Idefics3-8B-Llama3")).__name__ == "Idefics3Adapter"
    with pytest.raises(AssertionError):
        get_adapter("google/gemma-3-1b-it")


def test_module_paths_and_head_dim(tiny_gemma3):
    adapter, model = tiny_gemma3
    layers = adapter.get_decoder_layers(model)
    assert layers is model.model.language_model.layers and len(layers) == 3
    assert adapter.hidden_size(model) == 48 and adapter.n_attention_heads(model) == 4
    assert adapter.head_dim(model) == 16 != 48 // 4        # the trap: hidden // heads would be 12
    assert adapter.get_attn_head_output_module(model, 1).in_features == 64
    assert adapter.get_mlp_hidden_module(model, 1).in_features == 64 == adapter.intermediate_size(model)
    assert adapter.get_attn_output_module(model, 1) is layers[1].post_attention_layernorm
    assert adapter.get_mlp_output_module(model, 1) is layers[1].post_feedforward_layernorm
    assert adapter.get_attn_block(model, 1) is layers[1].self_attn       # .v_proj stays reachable
    assert adapter.image_token_id(model, None) == IMG
    from methods.common.sites import InterventionSite
    assert InterventionSite("attn_head_output").width(adapter, model) == 64
    assert InterventionSite("attn_output").width(adapter, model) == 48


def test_vision_patch_grid_and_real_geometry(tiny_gemma3):
    adapter, model = tiny_gemma3
    g = adapter.vision_patch_grid(model, 336, 336)
    assert (g["grid_rows"], g["grid_cols"], g["n_tokens"]) == (2, 2, N_IMG)
    from types import SimpleNamespace
    real = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(image_size=896, patch_size=14),
                                                  mm_tokens_per_image=256))
    g = adapter.vision_patch_grid(real, 336, 336)
    assert (g["grid_rows"], g["n_tokens"], g["merge_size"], g["canvas_px_per_token"]) == (16, 256, 4, (21.0, 21.0))


# ------------------------------------------------------------------ bidirectional image mask

def test_forward_shim_supplies_the_bidirectional_image_mask():
    adapter = get_adapter(MODEL_ID)
    model = _tiny()
    ids, extra = _inputs(batch=2)
    with torch.no_grad():
        causal = model(input_ids=ids, **extra).logits
        ref = model(input_ids=ids, **extra, token_type_ids=(ids == IMG).long()).logits
    assert (causal - ref).abs().max() > 1e-3, "image tokens are not bidirectional in this tiny model -- test is moot"
    adapter.prepare_model(model)
    with torch.no_grad():
        got = model(input_ids=ids, **extra).logits
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("cache_impl", [None, "hybrid"])
def test_generate_matches_full_recompute_with_bidirectional_images(tiny_gemma3, cache_impl):
    """The checkpoint's generation_config asks for a 'hybrid' cache, which builds the 4-D mask in
    prepare_inputs_for_generation; both paths must reproduce a no-cache greedy loop."""
    adapter, model = tiny_gemma3
    model.generation_config.cache_implementation = cache_impl
    ids, extra = _inputs(seed=3, batch=2)
    with torch.no_grad():
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **extra, max_new_tokens=4,
                             do_sample=False, pad_token_id=0)
        cur = ids
        for _ in range(4):
            nxt = model(input_ids=cur, **extra, token_type_ids=(cur == IMG).long(), use_cache=False).logits[:, -1]
            cur = torch.cat([cur, nxt.argmax(-1, keepdim=True)], 1)
    assert torch.equal(gen, cur)


def test_generation_config_is_greedy(tiny_gemma3):
    _, model = tiny_gemma3
    assert model.generation_config.do_sample is False


# ------------------------------------------------------------------ sites / DLA

def _capture(adapter, model, ids, extra, pos=-1):
    parts, handles = [], []
    for b in range(3):
        handles.append(adapter.get_attn_output_module(model, b).register_forward_hook(
            lambda m, i, o: parts.append(("attn", o[0, pos].detach()))))
        handles.append(adapter.get_mlp_output_module(model, b).register_forward_hook(
            lambda m, i, o: parts.append(("mlp", o[0, pos].detach()))))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, **extra, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    return out, parts


def test_residual_is_embed_plus_every_post_norm_sublayer(tiny_gemma3):
    """dla.py's RECONSTRUCTION identity: with the POST-norm sublayer outputs, embed + sum = the final
    pre-norm residual. With the raw attention/MLP outputs it would not be."""
    adapter, model = tiny_gemma3
    ids, extra = _inputs()
    out, parts = _capture(adapter, model, ids, extra)
    embed = out.hidden_states[0][0, -1]
    final_pre_norm = out.hidden_states[2][0, -1] + parts[-2][1] + parts[-1][1]
    torch.testing.assert_close(embed + sum(p for _, p in parts), final_pre_norm, atol=1e-5, rtol=1e-4)
    for b in range(3):          # each block's output is its input plus its two post-norm sublayers
        torch.testing.assert_close(out.hidden_states[b][0, -1] + parts[2 * b][1] + parts[2 * b + 1][1],
                                   out.hidden_states[b + 1][0, -1] if b < 2 else final_pre_norm,
                                   atol=1e-5, rtol=1e-4)


def test_unembed_and_exact_dla_additivity(tiny_gemma3):
    adapter, model = tiny_gemma3
    ids, extra = _inputs()
    out, parts = _capture(adapter, model, ids, extra)
    resid = out.hidden_states[2][0, -1] + parts[-2][1] + parts[-1][1]
    logits = out.logits[0, -1]
    torch.testing.assert_close(adapter.unembed(model, resid), logits, atol=1e-5, rtol=1e-4)
    toks = torch.arange(30)
    scale = adapter.final_norm_scale(model, resid)
    comps = [out.hidden_states[0][0, -1]] + [p for _, p in parts]
    dla = sum(scale * (adapter.logit_direction(model, toks) @ c.float()) for c in comps)
    torch.testing.assert_close(dla, logits[toks].float(), atol=1e-4, rtol=1e-4)


def test_head_residual_writes_sum_to_the_attn_output_site(tiny_gemma3):
    adapter, model = tiny_gemma3
    ids, extra = _inputs()
    z, o = {}, {}
    h1 = adapter.get_attn_head_output_module(model, 1).register_forward_pre_hook(
        lambda m, a: z.__setitem__("z", a[0].detach()))
    h2 = adapter.get_attn_output_module(model, 1).register_forward_hook(lambda m, i, out: o.__setitem__("o", out.detach()))
    try:
        with torch.no_grad():
            model(input_ids=ids, **extra)
    finally:
        h1.remove()
        h2.remove()
    writes = adapter.head_residual_writes(model, 1, z["z"])
    assert writes.shape == (1, ids.shape[1], 4, 48)
    torch.testing.assert_close(writes.sum(-2), o["o"].float(), atol=1e-5, rtol=1e-4)
    # the base default (no post-norm) is NOT the site here -- what the override exists for
    from methods.adapters.base import ModelAdapter
    plain = ModelAdapter.head_residual_writes(adapter, model, 1, z["z"]).sum(-2)
    assert (plain - o["o"].float()).abs().max() > 1e-3


def test_head_trace_per_head_delta_uses_the_post_norm_writes(tiny_gemma3):
    from methods.head_trace import per_head_delta
    adapter, model = tiny_gemma3
    zb, zp = torch.randn(2, 1, 64), torch.randn(2, 1, 64)
    d = torch.randn(2, 48)
    s = torch.ones(2, 1, 1)
    rows = per_head_delta(adapter, model, zb, zp, 1, 4, 16, d, s, s)
    wb, wp = adapter.head_residual_writes(model, 1, zb), adapter.head_residual_writes(model, 1, zp)
    for h, _, dr, dd in rows:
        assert dr == pytest.approx((wp[:, :, h] - wb[:, :, h]).norm(dim=-1).mean().item(), rel=1e-5)
        assert dd == pytest.approx(((wp[:, :, h] - wb[:, :, h]) * d[:, None]).sum(-1).mean().item(), rel=1e-4, abs=1e-6)


def test_image_tokens_are_raster_ordered():
    """With the SigLIP encoder's attention zeroed, each soft token sees only its own 2x2 pooled
    patch block: changing the bottom-left quarter of the image moves soft token 2 (row 1, col 0) only."""
    adapter = get_adapter(MODEL_ID)
    model = adapter.prepare_model(_tiny())
    with torch.no_grad():
        for layer in model.model.vision_tower.encoder.layers:
            layer.self_attn.out_proj.weight.zero_()
            layer.self_attn.out_proj.bias.zero_()
    feats = lambda px: model.model.get_image_features(px, return_dict=True).pooler_output[0]
    a = torch.randn(1, 3, 28, 28)
    b = a.clone()
    b[:, :, 14:, :14] += 1.0
    with torch.no_grad():
        diff = (feats(a) - feats(b)).abs().amax(-1)
    assert diff[2] > 1e-3 and torch.all(diff[[0, 1, 3]] < 1e-6), diff


def test_left_padding_leaves_the_last_column_unchanged(tiny_gemma3):
    adapter, model = tiny_gemma3
    ids, extra = _inputs()
    pad = 3
    padded = torch.cat([torch.zeros(1, pad, dtype=torch.long), ids], dim=1)
    mask = torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones_like(ids)], dim=1)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits[0, -1]
        got = model(input_ids=padded, attention_mask=mask, **extra).logits[0, -1]
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_generate_capture_patch_and_readback(tiny_gemma3):
    """head_swap_vade's continuous head install with Gemma's 16-wide heads inside a 64-wide o_proj."""
    adapter, model = tiny_gemma3
    ids, base_extra = _inputs(seed=0, batch=2)
    _, donor_extra = _inputs(seed=1, batch=2)
    mask = torch.ones_like(ids)
    heads, head_dim = hsv.site_units(adapter, model, "attn_head_output", heads=[(1, 0), (2, 3)])
    assert head_dim == 16
    z = hsv.capture_donor(adapter, model, [1, 2], ids, mask, donor_extra, 4, 0)
    assert z[1].shape[0] == 2 and z[1].shape[2] == 64
    assert hsv.verify_readback(adapter, model, heads, z, ids, mask, base_extra, 4, 0, head_dim) < 1e-4


def test_head_trace_head_patches_and_readback(tiny_gemma3):
    """head_trace's own patch builder + capture at o_proj's 64-wide input: install every head of
    block 1 at the last column from a donor capture and read the same tensor back."""
    from methods.common.hooks import make_cache_aware_patch_hook  # noqa: F401  (import path check)
    from methods.head_trace import capture_head_outputs, head_patches
    adapter, model = tiny_gemma3
    ids, base_extra = _inputs(seed=0, batch=2)
    _, donor_extra = _inputs(seed=1, batch=2)
    mask = torch.ones_like(ids)
    last = torch.full((2, 1), ids.shape[1] - 1)
    donor_z, _ = capture_head_outputs(adapter, model, [1], last, ids, mask, donor_extra)
    hd = adapter.head_dim(model)
    patches = head_patches([(1, h) for h in range(4)], donor_z, last, 4 * hd, hd, model.device)
    got, _ = capture_head_outputs(adapter, model, [1], last, ids, mask, base_extra, patches=patches)
    torch.testing.assert_close(got[1], donor_z[1], atol=1e-6, rtol=0)


# ------------------------------------------------------------------ the real processor (cached only)

def _cached_processor():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        pytest.skip("huggingface_hub unavailable")
    hit = try_to_load_from_cache(MODEL_ID, "tokenizer.json")
    if not isinstance(hit, str) or try_to_load_from_cache(MODEL_ID, "preprocessor_config.json") is None:
        pytest.skip(f"{MODEL_ID} tokenizer/processor not in the HF cache")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)


@pytest.fixture(scope="module")
def real():
    return get_adapter(MODEL_ID), _cached_processor()


@pytest.fixture(scope="module")
def canvas():
    from PIL import Image
    return Image.new("RGB", (336, 336), (200, 30, 30))


def _cfg(proc):
    """A stand-in model for the methods that only read the config."""
    from types import SimpleNamespace
    return SimpleNamespace(config=SimpleNamespace(
        image_token_id=proc.tokenizer.image_token_id, name_or_path=MODEL_ID, mm_tokens_per_image=256,
        vision_config=SimpleNamespace(image_size=896, patch_size=14)))


def test_real_build_inputs_layout(real, canvas):
    adapter, proc = real
    built = adapter.build_inputs(proc, canvas, "What is the capital city of the country in this image?",
                                 "The capital city is")
    ids = built["input_ids"].tolist()
    tok = proc.tokenizer
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    soft = tok.image_token_id
    assert soft == 262144 and proc.image_token_id == tok.boi_token_id != soft      # the two "image" ids
    img = [i for i, t in enumerate(ids) if t == soft]
    assert len(img) == 256 and img == list(range(img[0], img[0] + 256))
    assert ids[img[0] - 1] == tok.boi_token_id and ids[img[-1] + 1] == tok.eoi_token_id
    assert tok.decode(ids[-4:]) == "The capital city is"
    assert built["extra"]["pixel_values"].shape == (1, 3, 896, 896) and set(built["extra"]) == {"pixel_values"}


def test_real_default_processor_call_doubles_bos(real, canvas):
    adapter, proc = real
    text = adapter._render_template_text(proc, "Q?", "A:")
    ids = proc(text=[text], images=[canvas], return_tensors="pt")["input_ids"][0]
    assert ids.tolist()[:2] == [proc.tokenizer.bos_token_id] * 2


def test_real_tokenize_template_and_encode_image_match_build_inputs(real, canvas):
    adapter, proc = real
    q, p = "Which language is spoken in the country shown in this flag?", "The language is"
    built = adapter.build_inputs(proc, canvas, q, p)
    assert torch.equal(adapter.tokenize_template(proc, q, p, 256), built["input_ids"])
    enc = adapter.encode_image(proc, canvas)
    assert set(enc) == set(built["extra"])
    assert torch.equal(enc["pixel_values"], built["extra"]["pixel_values"])


def test_real_phrase_and_assistant_columns(real):
    adapter, proc = real
    q, p = "Report the currency of this country.", "Answer:"
    ids = adapter.tokenize_template(proc, q, p, 256)
    col = adapter.find_last_phrase_token_col(proc, q, p, 256, ["currency"])
    assert proc.tokenizer.decode(ids[col]).strip() == "currency"
    a = adapter.assistant_start_col(proc, ids)
    assert proc.tokenizer.decode(ids[a:a + 2]) == "<start_of_turn>model"


def test_real_gold_tokens_for_flags_labels(real):
    from methods.common.targets import derive_gold_token_ids
    _, proc = real
    ids = derive_gold_token_ids(proc.tokenizer, "The capital city is", "Paris")
    assert proc.tokenizer.decode(ids).strip() == "Paris" and proc.tokenizer.bos_token_id not in ids


def test_real_entities_build_batch_and_computed_object_box(real):
    """common/entities.py's path: 256 tokens from the vision config, entity_ring1 COMPUTED as the
    smallest 16x16-cell box containing the flag (InternVL3's rows 6-9, cols 5-10: same 21px cells)."""
    from methods.common import entities as ent
    vade_root = os.environ.get("VADE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "VADE"))
    if not os.path.isdir(os.path.join(vade_root, "data", "flags")):
        pytest.skip("no sibling VADE checkout")
    adapter, proc = real
    assets = ent.load_entity_assets(vade_root, "flags")
    model = _cfg(proc)
    flat, n, last = ent.resolve_position_set("full_image", assets, adapter, model)
    assert (n, len(flat), last) == (256, 256, False)
    ring, n2, _ = ent.resolve_position_set("entity_ring1", assets, adapter, model)
    assert n2 == 256 and ring == [r * 16 + c for r in range(6, 10) for c in range(5, 11)]
    assert ring == assets.object_location["models"]["InternVL3-8B-hf"]["object_token_indices"]["entity_ring1"]["flat"]
    rows = [r for r in ent.load_tuples(assets, "language", "test")][:2]
    cache = ent.BuildBatchCache()
    img = ent.build_batch(rows, assets, adapter, model, proc, "full_image", batch_cache=cache)
    lt = ent.build_batch(rows, assets, adapter, model, proc, "last_token", batch_cache=cache)
    assert img["positions"].shape == (2, 256) and lt["positions"].shape == (2, 1)
    assert int(lt["positions"][0, 0]) == img["base_input_ids"].shape[1] - 1
    assert (img["base_input_ids"].gather(1, img["positions"]) == proc.tokenizer.image_token_id).all()
    assert ((img["base_input_ids"] == proc.tokenizer.bos_token_id).sum(1) == 1).all()
    assert set(img["base_extra"]) == {"pixel_values"}
