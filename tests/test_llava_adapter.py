"""Offline tests for the LLaVA-1.5 adapter (PAPER_RUNBOOK.md X4).

Two kinds:
  * a tiny RANDOM LlavaForConditionalGeneration (CLIP vision tower, Llama text stack, 3 blocks x
    4 heads, a 2x2 image grid) for module paths, DLA additivity, causality of the image block,
    left-pad invariance, and the head / MLP patch + read-back path with this adapter's `extra`;
  * the REAL processor/tokenizer of llava-hf/llava-1.5-7b-hf, SKIPPED unless it is already in the
    HF cache (no download is attempted): BOS, the 576-token image block, template/encode
    identities, header and phrase columns, the controlled router prompts' alignment.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods import head_swap_vade as hsv                               # noqa: E402
from methods.adapters.registry import get_adapter                       # noqa: E402

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMG = 9            # image_token_index of the tiny model
N_IMG = 4          # 28px image / 14px patches = 2x2, CLS dropped


@pytest.fixture
def tiny_llava():
    from transformers import LlavaConfig, LlavaForConditionalGeneration
    torch.manual_seed(13)
    config = LlavaConfig(
        vision_config=dict(model_type="clip_vision_model", hidden_size=16, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=24, image_size=28, patch_size=14,
                           projection_dim=16),
        text_config=dict(model_type="llama", vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                         rms_norm_eps=1e-5),
        image_token_index=IMG, vision_feature_layer=-2, vision_feature_select_strategy="default")
    config._attn_implementation = "eager"
    model = LlavaForConditionalGeneration(config).eval()
    return get_adapter(MODEL_ID), model


def _inputs(seed=0, batch=1):
    """[BOS] 'USER:' <image>x4 '\\n' question 'ASSISTANT:' prefill -- the real layout, small ids."""
    ids = torch.tensor([[1, 20, 21] + [IMG] * N_IMG + [13, 30, 31, 32, 33, 40, 41, 42]] * batch)
    g = torch.Generator().manual_seed(seed)
    return ids, {"pixel_values": torch.randn(batch, 3, 28, 28, generator=g)}


# ------------------------------------------------------------------ registry / module tree

def test_registry_routes_llava_15_only():
    a = get_adapter(MODEL_ID)
    assert type(a).__name__ == "LlavaAdapter" and a.shares_object_grid is False
    assert type(get_adapter("llava-hf/llava-1.5-13b-hf")).__name__ == "LlavaAdapter"
    for other in ("llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/llava-onevision-qwen2-7b-ov-hf"):
        with pytest.raises(ValueError):
            get_adapter(other)
    with pytest.raises(AssertionError):
        get_adapter("liuhaotian/llava-v1.5-7b")                       # not the transformers-native port
    assert type(get_adapter("Qwen/Qwen2.5-VL-7B-Instruct")).__name__ == "Qwen25VLAdapter"
    assert type(get_adapter("HuggingFaceM4/Idefics3-8B-Llama3")).__name__ == "Idefics3Adapter"


def test_module_paths_and_shapes(tiny_llava):
    adapter, model = tiny_llava
    layers = adapter.get_decoder_layers(model)
    assert layers is model.model.language_model.layers and len(layers) == 3
    assert adapter._final_norm(model) is model.model.language_model.norm
    assert adapter.hidden_size(model) == 32 and adapter.n_attention_heads(model) == 4
    assert adapter.intermediate_size(model) == 48
    assert adapter.get_mlp_hidden_module(model, 1).in_features == 48
    assert adapter.get_attn_head_output_module(model, 1).in_features == 32
    assert getattr(model.lm_head, "bias", None) is None
    assert adapter.image_token_id(model, None) == IMG


def test_vision_patch_grid(tiny_llava):
    adapter, model = tiny_llava
    g = adapter.vision_patch_grid(model, 336, 336)
    assert (g["grid_rows"], g["grid_cols"], g["n_tokens"], g["merge_size"]) == (2, 2, N_IMG, 1)
    real = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(image_size=336, patch_size=14),
                                                  vision_feature_select_strategy="default",
                                                  vision_feature_layer=-2))
    g = adapter.vision_patch_grid(real, 336, 336)
    assert (g["grid_rows"], g["n_tokens"], g["canvas_px_per_token"]) == (24, 576, (14.0, 14.0))
    real.config.vision_feature_select_strategy = "full"
    with pytest.raises(AssertionError):
        adapter.vision_patch_grid(real, 336, 336)


def test_processor_token_count_formula():
    proc = SimpleNamespace(image_processor=SimpleNamespace(crop_size={"height": 336, "width": 336}),
                           patch_size=14, num_additional_image_tokens=1, vision_feature_select_strategy="default")
    assert get_adapter(MODEL_ID).n_processor_image_tokens(proc) == 576


# ------------------------------------------------------------------ DLA / forward

def test_unembed_and_exact_dla_additivity(tiny_llava):
    adapter, model = tiny_llava
    ids, extra = _inputs()
    resid = {}
    h = adapter.get_decoder_layers(model)[-1].register_forward_hook(
        lambda m, i, o: resid.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0, -1].detach()))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, **extra)
    finally:
        h.remove()
    logits = out.logits[0, -1]
    torch.testing.assert_close(adapter.unembed(model, resid["x"]), logits, atol=1e-5, rtol=1e-4)
    toks = torch.arange(20)
    dla = adapter.final_norm_scale(model, resid["x"]) * (adapter.logit_direction(model, toks) @ resid["x"].float())
    torch.testing.assert_close(dla, logits[toks].float(), atol=1e-5, rtol=1e-4)


def test_residual_is_embed_plus_every_sublayer(tiny_llava):
    adapter, model = tiny_llava
    ids, extra = _inputs()
    parts, handles = [], []
    for b in range(3):
        handles.append(adapter.get_attn_block(model, b).register_forward_hook(
            lambda m, i, o: parts.append(o[0][0, -1].detach())))
        handles.append(adapter.get_mlp_block(model, b).register_forward_hook(
            lambda m, i, o: parts.append(o[0, -1].detach())))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, **extra, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    embed = out.hidden_states[0][0, -1]
    final_pre_norm = out.hidden_states[2][0, -1] + parts[-2] + parts[-1]
    torch.testing.assert_close(embed + sum(parts), final_pre_norm, atol=1e-5, rtol=1e-4)


def test_pixels_reach_only_image_and_later_columns(tiny_llava):
    """Causal and contiguous: columns before the image block are bit-identical across images."""
    adapter, model = tiny_llava
    ids, a = _inputs(seed=0)
    _, b = _inputs(seed=1)
    with torch.no_grad():
        ha = model(input_ids=ids, **a, output_hidden_states=True).hidden_states[2][0]
        hb = model(input_ids=ids, **b, output_hidden_states=True).hidden_states[2][0]
    first = int((ids[0] == IMG).nonzero()[0])
    assert torch.equal(ha[:first], hb[:first])
    assert (ha[first] - hb[first]).abs().max() > 1e-4
    assert (ha[-1] - hb[-1]).abs().max() > 1e-4


def test_image_tokens_are_raster_ordered_patch_features(tiny_llava):
    """Token k of the image block is patch k of the vision tower (CLS dropped), row-major --
    so a 24x24 grid index on the real model is row * 24 + col of the 336px canvas."""
    adapter, model = tiny_llava
    _, extra = _inputs()
    with torch.no_grad():
        vt = model.model.vision_tower(extra["pixel_values"], output_hidden_states=True)
        feats = model.model.multi_modal_projector(vt.hidden_states[-2][:, 1:])
        got = model.model.get_image_features(extra["pixel_values"], vision_feature_layer=-2,
                                             vision_feature_select_strategy="default").pooler_output[0]
    torch.testing.assert_close(got, feats[0])
    assert got.shape[0] == N_IMG


def test_left_padding_leaves_the_last_column_unchanged(tiny_llava):
    adapter, model = tiny_llava
    ids, extra = _inputs()
    pad = 3
    padded = torch.cat([torch.zeros(1, pad, dtype=torch.long), ids], dim=1)
    mask = torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones_like(ids)], dim=1)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits[0, -1]
        got = model(input_ids=padded, attention_mask=mask, **extra).logits[0, -1]
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_generate_capture_patch_and_readback(tiny_llava):
    adapter, model = tiny_llava
    ids, base_extra = _inputs(seed=0, batch=2)
    _, donor_extra = _inputs(seed=1, batch=2)
    mask = torch.ones_like(ids)
    heads, head_dim = hsv.site_units(adapter, model, "attn_head_output", heads=[(1, 0), (2, 3)])
    assert head_dim == 8
    assert hsv.plain_generate(model, ids, mask, base_extra, 4, 0).shape == (2, 4)
    z = hsv.capture_donor(adapter, model, [1, 2], ids, mask, donor_extra, 4, 0)
    assert z[1].shape[0] == 2 and z[1].shape[2] == 32
    assert hsv.verify_readback(adapter, model, heads, z, ids, mask, base_extra, 4, 0, head_dim) < 1e-4
    units, width = hsv.site_units(adapter, model, "mlp_hidden", blocks=[1])
    assert width == 48
    zm = hsv.capture_donor(adapter, model, [1], ids, mask, donor_extra, 3, 0, site="mlp_hidden")
    assert hsv.verify_readback(adapter, model, units, zm, ids, mask, base_extra, 3, 0, width,
                               site="mlp_hidden") < 1e-4


# ------------------------------------------------------------------ prompt helpers (no real processor)

class _ToyTokenizer:
    """Decodes ids to fixed pieces, SentencePiece-style: the header is split over three tokens."""
    PIECES = {1: "<s>", 5: " USER", 6: ":", 7: " What", 8: "?", 10: " ASS", 11: "IST", 12: "ANT:", 14: " Answer:"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.PIECES[i] for i in ids)


def test_assistant_start_col_is_the_first_token_of_the_last_header():
    proc = SimpleNamespace(tokenizer=_ToyTokenizer())
    a = get_adapter(MODEL_ID)
    ids = [1, 5, 6, 7, 8, 10, 11, 12, 14]
    assert a.assistant_start_col(proc, ids) == 5
    assert a.assistant_start_col(proc, torch.tensor(ids)) == 5
    twice = [1, 10, 11, 12, 7, 8, 10, 11, 12, 14]
    assert a.assistant_start_col(proc, twice) == 6
    with pytest.raises(AssertionError):
        a.assistant_start_col(proc, [1, 5, 6, 7, 8])


def test_bos_is_added_only_when_the_template_did_not_write_it():
    a = get_adapter(MODEL_ID)
    proc = SimpleNamespace(tokenizer=SimpleNamespace(bos_token="<s>", bos_token_id=1))
    assert a._add_special(proc, "USER: <image>\nQ ASSISTANT: A")
    assert not a._add_special(proc, "<s>USER: <image>\nQ ASSISTANT: A")
    a._check_bos(proc, [1, 5, 6])
    for bad in ([5, 6], [1, 1, 5], [5, 1]):
        with pytest.raises(AssertionError):
            a._check_bos(proc, bad)


# ------------------------------------------------------------------ the real processor (cached only)

def _cached_processor():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        pytest.skip("huggingface_hub unavailable")
    for f in ("tokenizer.json", "preprocessor_config.json", "processor_config.json"):
        if not isinstance(try_to_load_from_cache(MODEL_ID, f), str):
            pytest.skip(f"{MODEL_ID} {f} not in the HF cache")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)


@pytest.fixture(scope="module")
def real():
    return get_adapter(MODEL_ID), _cached_processor()


@pytest.fixture(scope="module")
def canvas():
    from PIL import Image
    return Image.new("RGB", (336, 336), (200, 30, 30))


def test_real_build_inputs_layout(real, canvas):
    adapter, proc = real
    assert adapter.n_processor_image_tokens(proc) == 576
    built = adapter.build_inputs(proc, canvas, "What is the capital city of the country in this image?",
                                 "The capital city is")
    ids = built["input_ids"].tolist()
    tok = proc.tokenizer
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    img = [i for i, t in enumerate(ids) if t == proc.image_token_id]
    assert len(img) == 576 and img == list(range(img[0], img[0] + 576))
    assert tok.decode(ids[img[-1] + 1:]).lstrip().startswith("What is the capital")
    assert tok.decode(ids).rstrip().endswith("ASSISTANT: The capital city is")
    assert built["extra"]["pixel_values"].shape == (1, 3, 336, 336)


def test_real_tokenize_template_and_encode_image_match_build_inputs(real, canvas):
    adapter, proc = real
    q, p = "Which language is spoken in the country shown in this flag?", "The language is"
    built = adapter.build_inputs(proc, canvas, q, p)
    assert torch.equal(adapter.tokenize_template(proc, q, p, 576), built["input_ids"])
    enc = adapter.encode_image(proc, canvas)
    assert set(enc) == set(built["extra"])
    assert torch.equal(enc["pixel_values"], built["extra"]["pixel_values"])


def test_real_336_canvas_is_not_resampled(real, canvas):
    """VADE's 336px canvas is LLaVA-1.5's native size: resize + center crop must be identities,
    so token (r, c) covers canvas pixels [14r, 14r+14) x [14c, 14c+14) exactly."""
    import numpy as np
    from PIL import Image
    adapter, proc = real
    arr = np.zeros((336, 336, 3), dtype=np.uint8)
    arr[:14, :14] = 255                                       # only the top-left patch is white
    px = adapter.encode_image(proc, Image.fromarray(arr))["pixel_values"][0]
    top_left, rest = px[:, :14, :14].mean(), px[:, 14:, 14:].mean()
    assert top_left > rest + 1.0


def test_real_phrase_and_assistant_columns(real):
    adapter, proc = real
    q, p = "Report the currency of this country.", "Answer:"
    ids = adapter.tokenize_template(proc, q, p, 576)
    col = adapter.find_last_phrase_token_col(proc, q, p, 576, ["currency"])
    assert proc.tokenizer.decode(ids[col]).strip().endswith("currency")
    a = adapter.assistant_start_col(proc, ids)
    assert proc.tokenizer.decode(ids[a:]).lstrip().startswith("ASSISTANT:")
    assert not proc.tokenizer.decode(ids[a + 1:]).lstrip().startswith("ASSISTANT:")


def test_real_controlled_router_prompts_align(real, canvas):
    """attribute_switch_sweep / attr_head_trace need equal-length prompts differing in one column
    after the image. Llama-2's tokenizer must give one token per field word, like Qwen's."""
    from methods import attribute_switch_sweep as sw
    adapter, proc = real
    special = proc.tokenizer.all_special_ids
    built = {a: adapter.build_inputs(proc, canvas, sw.question(a), sw.PREFILL)["input_ids"].unsqueeze(0)
             for a in sw.ATTRIBUTES}
    for a in sw.ATTRIBUTES[1:]:
        assert sw.alignment(built["capital"], built[a], proc.image_token_id, special)
    differ = (built["capital"][0] != built["currency"][0]).nonzero().flatten().tolist()
    assert len(differ) == 1


def test_real_entities_build_batch_full_image_and_last_token(real):
    from methods.common import entities as ent
    vade_root = os.environ.get("VADE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "VADE"))
    if not os.path.isdir(os.path.join(vade_root, "data", "flags")):
        pytest.skip("no sibling VADE checkout")
    adapter, proc = real
    assets = ent.load_entity_assets(vade_root, "flags")
    model = SimpleNamespace(config=SimpleNamespace(
        image_token_id=proc.image_token_id, name_or_path=MODEL_ID,
        vision_config=SimpleNamespace(image_size=336, patch_size=14),
        vision_feature_select_strategy="default", vision_feature_layer=-2))
    flat, n, last = ent.resolve_position_set("full_image", assets, adapter, model)
    assert (n, len(flat), last) == (576, 576, False)
    rows = [r for r in ent.load_tuples(assets, "language", "test")][:2]
    cache = ent.BuildBatchCache()
    img = ent.build_batch(rows, assets, adapter, model, proc, "full_image", batch_cache=cache)
    lt = ent.build_batch(rows, assets, adapter, model, proc, "last_token", batch_cache=cache)
    assert img["positions"].shape == (2, 576) and lt["positions"].shape == (2, 1)
    assert ((img["base_input_ids"] == proc.tokenizer.bos_token_id).sum(1) == 1).all()
    assert set(img["base_extra"]) == {"pixel_values"}


# ------------------------------------------------------------------ text-entity prompt (circuit_common)

class _CharTokenizer:
    """One token per character; BOS id 1 prepended when add_special_tokens=True."""
    bos_token, bos_token_id = "<s>", 1

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        ids = [ord(c) for c in text]
        offs = [(i, i + 1) for i in range(len(text))]
        if add_special_tokens:
            ids, offs = [1] + ids, [(0, 0)] + offs
        return {"input_ids": ids, "offset_mapping": offs}


class _FakeProcessor:
    def __init__(self, prefix=""):
        self.tokenizer, self.prefix = _CharTokenizer(), prefix

    def apply_chat_template(self, messages, **kw):
        return f"{self.prefix}USER: {messages[0]['content'][0]['text']} ASSISTANT: {messages[1]['content'][0]['text']}"


def _wrapped(adapter, processor):
    from methods import circuit_common as cc
    m = cc.Model.__new__(cc.Model)
    m.adapter, m.processor = adapter, processor
    return m


def test_text_prompt_gets_one_bos_on_llava_and_is_unchanged_elsewhere():
    from methods import circuit_common as cc
    llava = _wrapped(get_adapter(MODEL_ID), _FakeProcessor())
    ids, cols = cc.Model.text_prompt(llava, "France", "capital")
    assert ids[0].item() == 1 and (ids == 1).sum() == 1
    assert "".join(chr(int(ids[c])) for c in cols) == "France"
    # a template that writes BOS itself: not added again
    ids2, _ = cc.Model.text_prompt(_wrapped(get_adapter(MODEL_ID), _FakeProcessor("<s>")), "France", "capital")
    assert ids2[0].item() == ord("<")
    # Qwen's adapter has no _add_special: tokenized with add_special_tokens=False, as before
    ids3, cols3 = cc.Model.text_prompt(_wrapped(get_adapter("Qwen/Qwen2.5-VL-7B-Instruct"), _FakeProcessor()),
                                       "France", "capital")
    assert ids3[0].item() == ord("U") and "".join(chr(int(ids3[c])) for c in cols3) == "France"


def test_mlp_lookup_top_tokens_runs_on_llava(tiny_llava, monkeypatch):
    """X9: mlp_lookup's logit-lens top-k phase on the LLaVA stack. Its own assertion checks that the
    lens at the last layer (pre-norm hook + the adapter's unembed) reproduces the model's logits."""
    import methods.circuit_common as cc
    from methods import mlp_lookup as ml
    adapter, model = tiny_llava
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=2, decode=lambda ids, **k: " ".join(f"t{int(i)}" for i in ids))
    m = cc.Model.wrap(adapter, model, SimpleNamespace(tokenizer=tok))
    attrs = ["capital", "language"]
    items = {k: {"name": f"N{k}", "capital": f"C{k}", "language": f"L{k}"} for k in ("AA", "BB")}
    ids0, _ = _inputs()

    def image_prompt(entity_dir, item, attribute):
        g = torch.Generator().manual_seed(sum(map(ord, item["name"] + attribute)))
        ids = ids0[0].clone()
        ids[-5] = 34 + attrs.index(attribute)                  # the question token differs per attribute
        return ids, {"pixel_values": torch.randn(1, 3, 28, 28, generator=g)}

    labels = {}
    monkeypatch.setattr(m, "image_prompt", image_prompt)
    monkeypatch.setattr(m, "text_prompt", lambda name, a: (torch.tensor([1, 20, 50 + len(name) % 5, 34 + attrs.index(a),
                                                                         40, 41][:5 + attrs.index(a)]), [2]))
    monkeypatch.setattr(cc, "gold_ids", lambda t, label: [labels.setdefault(label, 44 + len(labels) % 15)])
    tt = ml.top_tokens(m, None, items, sorted(items), attrs, top_items=sorted(items), k=5, k_bottom=3,
                       write_blocks=(0, 1, 2))
    assert tt["n_layers"] == 3 and len(tt["prompts"]) == 2 * 2 * 2
    p0 = tt["prompts"][0]
    assert set(p0["lens"]) == {1, 2, 3} and len(p0["lens"][3]["top"]) == 5
    assert set(p0["writes"]) == {f"{k}{b}" for k in ("attn", "mlp") for b in (0, 1, 2)}
    md = ml.toptokens_markdown(tt)                              # default layers follow the model's depth
    assert "| block |" in md
