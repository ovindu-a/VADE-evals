"""Offline tests for the Pixtral-12B adapter (PAPER_RUNBOOK.md X15).

Two kinds:
  * a tiny RANDOM LlavaForConditionalGeneration with a Pixtral vision tower and a Mistral text stack
    (3 blocks x 4 heads of 8 = o_proj input 32 vs hidden 40, so head_dim != hidden // heads exactly
    as on the real 12B; a 3x3 image grid with [IMG_BREAK] after each row) for: module paths, the
    row-broken image block, raster order, exact DLA, the head patch / read-back path, left-pad
    invariance;
  * the REAL processor/tokenizer of mistral-community/pixtral-12b, SKIPPED unless already in the HF
    cache: BOS, the 21x21 image block with row breaks, template/encode identities, the computed
    object box, entities.build_batch.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods import head_swap_vade as hsv                               # noqa: E402
from methods.adapters.registry import get_adapter                       # noqa: E402

MODEL_ID = "mistral-community/pixtral-12b"
IMG, BRK, END = 10, 12, 13     # [IMG] / [IMG_BREAK] / [IMG_END] in the tiny vocab (IMG = the real id)
SIDE = 3                       # 24px image / 8px patches


def _tiny(seed=11):
    from transformers import LlavaConfig, LlavaForConditionalGeneration
    torch.manual_seed(seed)
    cfg = LlavaConfig(
        vision_config=dict(model_type="pixtral", hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                           num_attention_heads=2, head_dim=8, image_size=64, patch_size=8, hidden_act="silu"),
        text_config=dict(model_type="mistral", vocab_size=64, hidden_size=40, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                         sliding_window=None, rms_norm_eps=1e-5),
        image_token_index=IMG, vision_feature_select_strategy="full", vision_feature_layer=-1,
        projector_hidden_act="gelu")
    cfg._attn_implementation = "eager"
    return LlavaForConditionalGeneration(cfg).eval()


@pytest.fixture
def tiny_pixtral():
    return get_adapter(MODEL_ID), _tiny()


def _block(side=SIDE):
    row = [IMG] * side + [BRK]
    out = row * side
    out[-1] = END
    return out


def _inputs(seed=0, batch=1):
    """<s> [INST] {image block} question [/INST] prefill -- the real layout, small ids."""
    ids = torch.tensor([[1, 3] + _block() + [20, 21, 22, 4, 30, 31]] * batch)
    g = torch.Generator().manual_seed(seed)
    return ids, {"pixel_values": torch.randn(batch, 3, 24, 24, generator=g),
                 "image_sizes": torch.tensor([[24, 24]] * batch)}


# ------------------------------------------------------------------ registry / module tree

def test_registry_routes_pixtral():
    a = get_adapter(MODEL_ID)
    assert type(a).__name__ == "PixtralAdapter" and a.shares_object_grid is False and a.image_row_break
    assert type(get_adapter("llava-hf/llava-1.5-7b-hf")).__name__ == "LlavaAdapter"
    with pytest.raises(AssertionError):
        get_adapter("mistralai/Pixtral-12B-2409")


def test_module_paths_and_head_dim(tiny_pixtral):
    adapter, model = tiny_pixtral
    layers = adapter.get_decoder_layers(model)
    assert layers is model.model.language_model.layers and len(layers) == 3
    assert adapter._final_norm(model) is model.model.language_model.norm
    assert adapter.hidden_size(model) == 40 and adapter.n_attention_heads(model) == 4
    assert adapter.head_dim(model) == 8 != 40 // 4           # the trap: hidden // heads would be 10
    assert adapter.get_attn_head_output_module(model, 1).in_features == 32
    assert adapter.get_mlp_hidden_module(model, 1).in_features == 48
    # no post-sublayer norms: the output sites are the modules themselves
    assert adapter.get_attn_output_module(model, 1) is layers[1].self_attn
    assert adapter.get_mlp_output_module(model, 1) is layers[1].mlp
    assert getattr(model.lm_head, "bias", None) is None
    assert adapter.image_token_id(model, None) == IMG


def test_vision_patch_grid_follows_the_canvas(tiny_pixtral):
    adapter, model = tiny_pixtral
    assert adapter.vision_patch_grid(model, 24, 24)["n_tokens"] == 9
    assert adapter.vision_patch_grid(model, 128, 64)["grid_rows"] == 4          # longest edge 64: 128x64 -> 64x32
    from types import SimpleNamespace
    real = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(image_size=1024, patch_size=16),
                                                  vision_feature_select_strategy="full"))
    g = adapter.vision_patch_grid(real, 336, 336)
    assert (g["grid_rows"], g["grid_cols"], g["n_tokens"], g["canvas_px_per_token"]) == (21, 21, 441, (16.0, 16.0))


# ------------------------------------------------------------------ the row-broken image block

def test_object_token_positions_handles_row_breaks():
    from methods.common.entities import object_token_positions
    ids, _ = _inputs()
    pos = object_token_positions(ids[0], IMG, list(range(9)), 9)
    assert pos == [2, 3, 4, 6, 7, 8, 10, 11, 12] and all(int(ids[0, p]) == IMG for p in pos)
    assert object_token_positions(ids[0], IMG, [4], 9) == [7]                  # row 1, col 1
    # contiguous blocks are unchanged
    flat = torch.tensor([5, 5, IMG, IMG, IMG, IMG, 6])
    assert object_token_positions(flat, IMG, [0, 3], 4) == [2, 5]
    # anything else is refused: a missing token, or a gap that is not one break per row
    with pytest.raises(AssertionError):
        object_token_positions(ids[0], IMG, [0], 10)
    with pytest.raises(AssertionError):
        object_token_positions(torch.tensor([IMG, IMG, 7, 7, IMG, IMG]), IMG, [0], 4)


def test_image_features_fill_img_tokens_in_raster_order():
    """With the vision transformer's attention zeroed each patch is processed alone (2-D RoPE only
    acts inside attention): changing the patch at (row 1, col 2) moves the 6th [IMG] token only."""
    model = _tiny()
    with torch.no_grad():
        for layer in model.model.vision_tower.transformer.layers:
            layer.attention.o_proj.weight.zero_()
    ids, a = _inputs(seed=0)
    b = {k: v.clone() for k, v in a.items()}
    b["pixel_values"][:, :, 8:16, 16:24] += 1.0
    with torch.no_grad():
        ha = model(input_ids=ids, **a, output_hidden_states=True).hidden_states[0][0]
        hb = model(input_ids=ids, **b, output_hidden_states=True).hidden_states[0][0]
    img_cols = [i for i, t in enumerate(ids[0].tolist()) if t == IMG]
    moved = [k for k, c in enumerate(img_cols) if (ha[c] - hb[c]).abs().max() > 1e-5]
    assert moved == [1 * SIDE + 2]
    assert torch.equal(ha[[i for i in range(ids.shape[1]) if i not in img_cols]],
                       hb[[i for i in range(ids.shape[1]) if i not in img_cols]])


# ------------------------------------------------------------------ DLA / forward

def test_unembed_and_exact_dla_additivity(tiny_pixtral):
    adapter, model = tiny_pixtral
    ids, extra = _inputs()
    parts, handles = [], []
    for b in range(3):
        handles.append(adapter.get_attn_output_module(model, b).register_forward_hook(
            lambda m, i, o: parts.append(o[0][0, -1].detach())))
        handles.append(adapter.get_mlp_output_module(model, b).register_forward_hook(
            lambda m, i, o: parts.append(o[0, -1].detach())))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, **extra, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    resid = out.hidden_states[2][0, -1] + parts[-2] + parts[-1]
    torch.testing.assert_close(out.hidden_states[0][0, -1] + sum(parts), resid, atol=1e-5, rtol=1e-4)
    logits = out.logits[0, -1]
    torch.testing.assert_close(adapter.unembed(model, resid), logits, atol=1e-5, rtol=1e-4)
    toks = torch.arange(20)
    dla = adapter.final_norm_scale(model, resid) * (adapter.logit_direction(model, toks) @ resid.float())
    torch.testing.assert_close(dla, logits[toks].float(), atol=1e-5, rtol=1e-4)


def test_head_residual_writes_default_sums_to_attn_output(tiny_pixtral):
    adapter, model = tiny_pixtral
    ids, extra = _inputs()
    z, o = {}, {}
    h1 = adapter.get_attn_head_output_module(model, 1).register_forward_pre_hook(
        lambda m, a: z.__setitem__("z", a[0].detach()))
    h2 = adapter.get_attn_output_module(model, 1).register_forward_hook(lambda m, i, out: o.__setitem__("o", out[0].detach()))
    try:
        with torch.no_grad():
            model(input_ids=ids, **extra)
    finally:
        h1.remove()
        h2.remove()
    torch.testing.assert_close(adapter.head_residual_writes(model, 1, z["z"]).sum(-2), o["o"].float(),
                               atol=1e-5, rtol=1e-4)


def test_left_padding_leaves_the_last_column_unchanged(tiny_pixtral):
    adapter, model = tiny_pixtral
    ids, extra = _inputs()
    pad = 3
    padded = torch.cat([torch.zeros(1, pad, dtype=torch.long), ids], dim=1)
    mask = torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones_like(ids)], dim=1)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits[0, -1]
        got = model(input_ids=padded, attention_mask=mask, **extra).logits[0, -1]
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_generate_capture_patch_and_readback(tiny_pixtral):
    adapter, model = tiny_pixtral
    ids, base_extra = _inputs(seed=0, batch=2)
    _, donor_extra = _inputs(seed=1, batch=2)
    mask = torch.ones_like(ids)
    heads, head_dim = hsv.site_units(adapter, model, "attn_head_output", heads=[(1, 0), (2, 3)])
    assert head_dim == 8
    assert hsv.plain_generate(model, ids, mask, base_extra, 3, 0).shape == (2, 3)
    z = hsv.capture_donor(adapter, model, [1, 2], ids, mask, donor_extra, 4, 0)
    assert z[1].shape[2] == 32
    assert hsv.verify_readback(adapter, model, heads, z, ids, mask, base_extra, 4, 0, head_dim) < 1e-4


def test_head_trace_head_patches_and_readback(tiny_pixtral):
    from methods.head_trace import capture_head_outputs, head_patches
    adapter, model = tiny_pixtral
    ids, base_extra = _inputs(seed=0, batch=2)
    _, donor_extra = _inputs(seed=1, batch=2)
    mask = torch.ones_like(ids)
    last = torch.full((2, 1), ids.shape[1] - 1)
    donor_z, _ = capture_head_outputs(adapter, model, [2], last, ids, mask, donor_extra)
    hd = adapter.head_dim(model)
    patches = head_patches([(2, 0), (2, 2)], donor_z, last, 4 * hd, hd, model.device)
    got, _ = capture_head_outputs(adapter, model, [2], last, ids, mask, base_extra, patches=patches)
    for h in (0, 2):
        torch.testing.assert_close(got[2][..., h * hd:(h + 1) * hd], donor_z[2][..., h * hd:(h + 1) * hd],
                                   atol=1e-6, rtol=0)


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
    from types import SimpleNamespace
    return SimpleNamespace(config=SimpleNamespace(
        image_token_id=proc.image_token_id, name_or_path=MODEL_ID, vision_feature_select_strategy="full",
        vision_config=SimpleNamespace(image_size=1024, patch_size=16)))


def test_real_build_inputs_layout(real, canvas):
    adapter, proc = real
    built = adapter.build_inputs(proc, canvas, "What is the capital city of the country in this image?",
                                 "The capital city is")
    ids = built["input_ids"].tolist()
    tok = proc.tokenizer
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    img = [i for i, t in enumerate(ids) if t == proc.image_token_id]
    assert len(img) == 441
    brk, end = proc.image_break_token_id, proc.image_end_token_id
    span = ids[img[0]:img[-1] + 2]
    assert span == ([proc.image_token_id] * 21 + [brk]) * 20 + [proc.image_token_id] * 21 + [end]
    assert tok.decode(ids[-5:]) == "[/INST]The capital city is"
    assert built["extra"]["pixel_values"].shape == (1, 3, 336, 336)
    assert built["extra"]["image_sizes"].tolist() == [[336, 336]]


def test_real_tokenize_template_and_encode_image_match_build_inputs(real, canvas):
    adapter, proc = real
    q, p = "Which language is spoken in the country shown in this flag?", "The language is"
    built = adapter.build_inputs(proc, canvas, q, p)
    assert torch.equal(adapter.tokenize_template(proc, q, p, 441), built["input_ids"])
    enc = adapter.encode_image(proc, canvas)
    assert set(enc) == set(built["extra"])
    for k in enc:
        assert torch.equal(enc[k], built["extra"][k]), k


def test_real_regex_fix_does_not_change_vade_tokenization(real):
    """transformers warns that this checkpoint's pre-tokenizer regex is wrong; on everything a VADE
    run tokenizes the fixed tokenizer gives identical ids, so the stock processor is used."""
    from transformers import AutoProcessor
    _, proc = real
    fixed = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True, fix_mistral_regex=True).tokenizer
    for t in ("[INST][IMG]What is the capital city of the country in this image?[/INST]The capital city is",
              " Buenos Aires", " Guaraní", " +54", " 1995", "Answer: 250 dollars", " Côte d'Ivoire's"):
        assert proc.tokenizer(t, add_special_tokens=False)["input_ids"] == fixed(t, add_special_tokens=False)["input_ids"], t


def test_real_phrase_and_assistant_columns(real):
    adapter, proc = real
    q, p = "Report the currency of this country.", "Answer:"
    ids = adapter.tokenize_template(proc, q, p, 441)
    col = adapter.find_last_phrase_token_col(proc, q, p, 441, ["currency"])
    assert proc.tokenizer.decode(ids[col]).strip() == "currency"
    a = adapter.assistant_start_col(proc, ids)
    assert proc.tokenizer.decode(ids[a]) == "[/INST]" and proc.tokenizer.decode(ids[a + 1:]) == "Answer:"


def test_real_gold_tokens_for_flags_labels(real):
    from methods.common.targets import derive_gold_token_ids
    _, proc = real
    ids = derive_gold_token_ids(proc.tokenizer, "The capital city is", "Paris")
    assert proc.tokenizer.decode(ids) == " Paris" and proc.tokenizer.bos_token_id not in ids


def test_real_entities_build_batch_and_computed_object_box(real):
    """common/entities.py's path with the row-broken block: full_image resolves to the 441 [IMG]
    columns (never a break), and entity_ring1 is COMPUTED as the 16px-cell box containing the flag
    (112..224 x 140..196 -> cols 7-13, rows 8-12)."""
    from methods.common import entities as ent
    vade_root = os.environ.get("VADE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "VADE"))
    if not os.path.isdir(os.path.join(vade_root, "data", "flags")):
        pytest.skip("no sibling VADE checkout")
    adapter, proc = real
    assets = ent.load_entity_assets(vade_root, "flags")
    model = _cfg(proc)
    flat, n, last = ent.resolve_position_set("full_image", assets, adapter, model)
    assert (n, len(flat), last) == (441, 441, False)
    ring, _, _ = ent.resolve_position_set("entity_ring1", assets, adapter, model)
    assert ring == [r * 21 + c for r in range(8, 13) for c in range(7, 14)]
    rows = [r for r in ent.load_tuples(assets, "language", "test")][:2]
    cache = ent.BuildBatchCache()
    img = ent.build_batch(rows, assets, adapter, model, proc, "full_image", batch_cache=cache)
    lt = ent.build_batch(rows, assets, adapter, model, proc, "last_token", batch_cache=cache)
    obj = ent.build_batch(rows, assets, adapter, model, proc, "entity_ring1", batch_cache=cache)
    assert img["positions"].shape == (2, 441) and lt["positions"].shape == (2, 1) and obj["positions"].shape == (2, 35)
    assert (img["base_input_ids"].gather(1, img["positions"]) == proc.image_token_id).all()
    assert (obj["base_input_ids"].gather(1, obj["positions"]) == proc.image_token_id).all()
    assert int(lt["positions"][0, 0]) == img["base_input_ids"].shape[1] - 1
    assert ((img["base_input_ids"] == proc.tokenizer.bos_token_id).sum(1) == 1).all()
    assert set(img["base_extra"]) == {"pixel_values", "image_sizes"}
    assert img["base_extra"]["image_sizes"].dtype == torch.long
