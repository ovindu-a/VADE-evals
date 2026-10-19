"""Offline tests for the Idefics3 adapter (PAPER_RUNBOOK.md X3) and the generic non-Qwen fixes.

Two kinds:
  * a tiny RANDOM Idefics3ForConditionalGeneration (Llama text stack, 3 blocks x 4 heads, a
    2x2 image grid) for module paths, DLA additivity, the head patch/read-back path, left-pad
    invariance and generate() with this adapter's `extra`;
  * the REAL processor/tokenizer of HuggingFaceM4/Idefics3-8B-Llama3, SKIPPED unless its
    tokenizer is already in the HF cache (no download is attempted): BOS, the image block,
    template/encode identities, the controlled router prompts' alignment.
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods import head_swap_vade as hsv                               # noqa: E402
from methods.adapters.registry import get_adapter                       # noqa: E402

MODEL_ID = "HuggingFaceM4/Idefics3-8B-Llama3"
IMG = 9            # image_token_id of the tiny model
N_IMG = 4          # 28px tile / 7px patches = 4x4, pixel shuffle 2 -> 2x2


@pytest.fixture
def tiny_idefics3():
    from transformers import Idefics3Config, Idefics3ForConditionalGeneration
    torch.manual_seed(11)
    config = Idefics3Config(
        vision_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
                           image_size=28, patch_size=7),
        text_config=dict(model_type="llama", vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                         rms_norm_eps=1e-5),
        image_token_id=IMG, scale_factor=2, pad_token_id=0)
    config._attn_implementation = "eager"
    model = Idefics3ForConditionalGeneration(config).eval()
    return get_adapter(MODEL_ID), model


def _inputs(seed=0, batch=1):
    """[BOS] 'User:' <fake> <image>x4 <fake> question ... 'Assistant:' prefill -- the real layout,
    with arbitrary small ids (60 stands in for <fake_token_around_image>)."""
    ids = torch.tensor([[1, 20, 21, 60] + [IMG] * N_IMG + [60, 30, 31, 32, 33, 40, 41, 42]] * batch)
    g = torch.Generator().manual_seed(seed)
    px = torch.randn(batch, 1, 3, 28, 28, generator=g)
    return ids, {"pixel_values": px, "pixel_attention_mask": torch.ones(batch, 1, 28, 28, dtype=torch.bool)}


# ------------------------------------------------------------------ registry / module tree

def test_registry_routes_idefics3_and_leaves_qwen_alone():
    a = get_adapter(MODEL_ID)
    assert type(a).__name__ == "Idefics3Adapter" and a.shares_object_grid is False
    assert a.PROCESSOR_KWARGS == {"do_image_splitting": False}
    assert type(get_adapter("Qwen/Qwen2.5-VL-7B-Instruct")).__name__ == "Qwen25VLAdapter"
    assert type(get_adapter("OpenGVLab/InternVL3-8B-hf")).__name__ == "InternVLAdapter"


def test_module_paths_and_shapes(tiny_idefics3):
    adapter, model = tiny_idefics3
    layers = adapter.get_decoder_layers(model)
    assert layers is model.model.text_model.layers and len(layers) == 3
    assert adapter._final_norm(model) is model.model.text_model.norm
    assert adapter.hidden_size(model) == 32 and adapter.n_attention_heads(model) == 4
    assert adapter.intermediate_size(model) == 48
    assert adapter.get_mlp_hidden_module(model, 1).in_features == 48
    # o_proj's input is n_heads x head_dim == hidden, which every head slice assumes
    assert adapter.get_attn_head_output_module(model, 1).in_features == 32
    assert getattr(model.lm_head, "bias", None) is None
    assert adapter.image_token_id(model, None) == IMG


def test_vision_patch_grid_matches_the_models_own_image_seq_len(tiny_idefics3):
    adapter, model = tiny_idefics3
    g = adapter.vision_patch_grid(model, 336, 336)
    assert (g["grid_rows"], g["grid_cols"], g["n_tokens"]) == (2, 2, N_IMG)
    assert g["n_tokens"] == model.model.image_seq_len
    assert g["canvas_px_per_token"] == (168.0, 168.0)


def test_vision_patch_grid_of_the_real_config_is_13x13():
    """Idefics3-8B: SigLIP 364px / 14px patches = 26, pixel shuffle 2 -> 13x13 = 169."""
    from types import SimpleNamespace
    model = SimpleNamespace(config=SimpleNamespace(
        vision_config=SimpleNamespace(image_size=364, patch_size=14), scale_factor=2))
    g = get_adapter(MODEL_ID).vision_patch_grid(model, 336, 336)
    assert (g["grid_rows"], g["grid_cols"], g["n_tokens"], g["merged_patch_size_px"]) == (13, 13, 169, 28)


# ------------------------------------------------------------------ DLA / forward

def test_unembed_and_exact_dla_additivity(tiny_idefics3):
    adapter, model = tiny_idefics3
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


def test_residual_is_embed_plus_every_sublayer(tiny_idefics3):
    """dla.py's RECONSTRUCTION identity, on the Llama stack: no post-sublayer norms."""
    adapter, model = tiny_idefics3
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
    final_pre_norm = out.hidden_states[2][0, -1] + parts[-2] + parts[-1]   # block 2's input + its sublayers
    torch.testing.assert_close(embed + sum(parts), final_pre_norm, atol=1e-5, rtol=1e-4)


def test_pixels_reach_only_image_and_later_columns(tiny_idefics3):
    """The image block is contiguous and causal: columns before it are bit-identical across
    two images, the first image column differs -- the `seq:before` null holds here too."""
    adapter, model = tiny_idefics3
    ids, a = _inputs(seed=0)
    _, b = _inputs(seed=1)
    with torch.no_grad():
        ha = model(input_ids=ids, **a, output_hidden_states=True).hidden_states[2][0]
        hb = model(input_ids=ids, **b, output_hidden_states=True).hidden_states[2][0]
    first = int((ids[0] == IMG).nonzero()[0])
    assert torch.equal(ha[:first], hb[:first])
    assert (ha[first] - hb[first]).abs().max() > 1e-4
    assert (ha[-1] - hb[-1]).abs().max() > 1e-4


def test_left_padding_leaves_the_last_column_unchanged(tiny_idefics3):
    """head_swap_vade / entities.build_batch LEFT-pad. Llama's forward takes positions from
    cache_position, not the mask, so the real tokens shift by the pad count; with 1-D RoPE only
    relative positions enter attention, so the last column must not move."""
    adapter, model = tiny_idefics3
    ids, extra = _inputs()
    pad = 3
    padded = torch.cat([torch.zeros(1, pad, dtype=torch.long), ids], dim=1)
    mask = torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones_like(ids)], dim=1)
    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits[0, -1]
        got = model(input_ids=padded, attention_mask=mask, **extra).logits[0, -1]
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------ head patch / generate path

def test_generate_capture_patch_and_readback(tiny_idefics3):
    """head_swap_vade's continuous head install runs end to end with this adapter's `extra`
    (pixel_values + bool pixel_attention_mask): capture a donor, install it, read it back."""
    adapter, model = tiny_idefics3
    ids, base_extra = _inputs(seed=0, batch=2)
    _, donor_extra = _inputs(seed=1, batch=2)
    mask = torch.ones_like(ids)
    heads, head_dim = hsv.site_units(adapter, model, "attn_head_output", heads=[(1, 0), (2, 3)])
    assert head_dim == 8
    clean = hsv.plain_generate(model, ids, mask, base_extra, 4, 0)
    assert clean.shape == (2, 4)
    z = hsv.capture_donor(adapter, model, [1, 2], ids, mask, donor_extra, 4, 0)
    assert z[1].shape[0] == 2 and z[1].shape[2] == 32
    worst = hsv.verify_readback(adapter, model, heads, z, ids, mask, base_extra, 4, 0, head_dim)
    assert worst < 1e-4
    # the same machinery on down_proj's input (the MLP lookup site)
    units, width = hsv.site_units(adapter, model, "mlp_hidden", blocks=[1])
    assert width == 48
    zm = hsv.capture_donor(adapter, model, [1], ids, mask, donor_extra, 3, 0, site="mlp_hidden")
    assert hsv.verify_readback(adapter, model, units, zm, ids, mask, base_extra, 3, 0, width,
                               site="mlp_hidden") < 1e-4


def test_to_device_keeps_the_pixel_mask_bool():
    out = hsv.to_device({"pixel_attention_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool),
                         "pixel_values": torch.zeros(1, 1, 3, 4, 4)}, "cpu", torch.bfloat16)
    assert out["pixel_attention_mask"].dtype == torch.bool and out["pixel_values"].dtype == torch.bfloat16


# ------------------------------------------------------------------ generic non-Qwen routing

class _FakeAdapter:
    """Any non-Qwen adapter: build_inputs is the only prompt path head_swap_vade may use."""
    def __init__(self):
        self.calls = []

    def build_inputs(self, processor, image, question, prefill):
        self.calls.append((question, prefill))
        n = 5 + len(question) % 3
        return {"input_ids": torch.arange(n) + 100,
                "extra": {"pixel_values": torch.zeros(1, 1, 3, 4, 4),
                          "pixel_attention_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool)}}


def test_prompt_inputs_routes_non_qwen_through_build_inputs(monkeypatch):
    fake = _FakeAdapter()
    monkeypatch.setattr(hsv, "build_prompt", lambda *a: pytest.fail("Qwen path used for a non-Qwen adapter"))
    ids, extra = hsv.prompt_inputs(None, None, "q?", "A:", fake)
    assert fake.calls == [("q?", "A:")] and ids.ndim == 1 and set(extra) == {"pixel_values", "pixel_attention_mask"}


def test_prompt_inputs_keeps_qwen_on_the_legacy_path(monkeypatch):
    seen = []

    def legacy(processor, image, question, prefill):
        seen.append(question)
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3),
                "pixel_values": torch.zeros(4, 6), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    monkeypatch.setattr(hsv, "build_prompt", legacy)
    for adapter in (None, get_adapter("Qwen/Qwen2.5-VL-7B-Instruct")):
        ids, extra = hsv.prompt_inputs(None, None, "q?", "A:", adapter)
        assert ids.tolist() == [1, 2, 3] and list(extra) == ["pixel_values", "image_grid_thw"]
    assert seen == ["q?", "q?"]


def test_build_batch_concatenates_whatever_extra_the_adapter_returns(tmp_path):
    from PIL import Image
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    items = {"A": {"image": "a.png"}, "B": {"image": "a.png"}}
    lookup = {"capital": {"capital_v1": {"question": "Capital?", "prefill": "It is"}}}
    jobs = [{"base": "A", "source": "B", "queried": "capital", "template_id": "capital_v1",
             "donor_template_id": "capital_v1", "donor_attribute": "capital"}] * 3
    batch = hsv.build_batch(jobs, None, items, str(tmp_path), lookup, 0, adapter=_FakeAdapter())
    assert set(batch["base_extra"]) == {"pixel_values", "pixel_attention_mask"}
    assert batch["base_extra"]["pixel_values"].shape == (3, 1, 3, 4, 4)
    assert batch["donor_extra"]["pixel_attention_mask"].dtype == torch.bool
    assert batch["base_ids"].shape[0] == 3


def test_attribute_switch_cached_refuses_non_qwen():
    from methods import attribute_switch_sweep as sw
    with pytest.raises(SystemExit):
        sw.main(["--model_id", MODEL_ID, "--execution", "cached", "--dry_run"])
    assert sw.is_qwen_model("Qwen/Qwen2.5-VL-7B-Instruct") and not sw.is_qwen_model(MODEL_ID)


def test_chatml_default_assistant_start_is_the_last_im_start():
    from types import SimpleNamespace
    from methods.adapters.base import ModelAdapter
    proc = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda t: {"<|im_start|>": 50}[t]))
    assert ModelAdapter().assistant_start_col(proc, [50, 7, 2, 2, 8, 51, 50, 9, 10]) == 6
    with pytest.raises(AssertionError):
        ModelAdapter().assistant_start_col(proc, [7, 2, 2])


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
    proc = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    proc.image_processor.do_image_splitting = False
    return proc


@pytest.fixture(scope="module")
def real():
    return get_adapter(MODEL_ID), _cached_processor()


@pytest.fixture(scope="module")
def canvas():
    from PIL import Image
    return Image.new("RGB", (336, 336), (200, 30, 30))


def _cfg(proc):
    """A stand-in model for the methods that only read the config / image token id."""
    from types import SimpleNamespace
    return SimpleNamespace(config=SimpleNamespace(image_token_id=proc.image_token_id,
                                                  vision_config=SimpleNamespace(image_size=364, patch_size=14),
                                                  scale_factor=2))


def test_real_build_inputs_layout(real, canvas):
    adapter, proc = real
    built = adapter.build_inputs(proc, canvas, "What is the capital city of the country in this image?",
                                 "The capital city is")
    ids = built["input_ids"].tolist()
    tok = proc.tokenizer
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    img = [i for i, t in enumerate(ids) if t == proc.image_token_id]
    assert len(img) == 169 and img == list(range(img[0], img[0] + 169))
    fake = proc.fake_image_token_id
    glob = tok("<global-img>", add_special_tokens=False)["input_ids"]
    assert ids[img[0] - len(glob) - 1] == fake and ids[img[0] - len(glob):img[0]] == glob
    assert ids[img[-1] + 1] == fake
    assert tok.decode(ids[-6:]) == "Assistant: The capital city is"
    assert built["extra"]["pixel_values"].shape == (1, 1, 3, 364, 364)
    assert built["extra"]["pixel_attention_mask"].dtype == torch.bool


def test_real_default_processor_call_doubles_bos(real, canvas):
    """The trap the adapter guards against: if this ever stops being true, _check_bos is moot
    but harmless; if the adapter ever drops add_special_tokens=False, its own assert fires."""
    adapter, proc = real
    text = adapter._render_template_text(proc, "Q?", "A:")
    ids = proc(text=[text], images=[canvas], return_tensors="pt", do_image_splitting=False)["input_ids"][0]
    assert ids.tolist()[:2] == [proc.tokenizer.bos_token_id] * 2


def test_real_tokenize_template_and_encode_image_match_build_inputs(real, canvas):
    adapter, proc = real
    q, p = "Which language is spoken in the country shown in this flag?", "The language is"
    built = adapter.build_inputs(proc, canvas, q, p)
    assert torch.equal(adapter.tokenize_template(proc, q, p, 169), built["input_ids"])
    enc = adapter.encode_image(proc, canvas)
    assert set(enc) == set(built["extra"])
    for k in enc:
        assert torch.equal(enc[k], built["extra"][k]), k


def test_real_phrase_and_assistant_columns(real):
    adapter, proc = real
    q, p = "Report the currency of this country.", "Answer:"
    ids = adapter.tokenize_template(proc, q, p, 169)
    col = adapter.find_last_phrase_token_col(proc, q, p, 169, ["currency"])
    assert proc.tokenizer.decode(ids[col]).strip() == "currency"
    a = adapter.assistant_start_col(proc, ids)
    assert proc.tokenizer.decode(ids[a]).strip() == "Assistant"
    assert proc.tokenizer.decode(ids[a - 1]) == "\n"


def test_real_entities_build_batch_full_image_and_last_token(real):
    """common/entities.py's path (ceiling_sweep / head_trace / the pipeline): the 169-token
    count comes from the vision config, object sets are refused off the Qwen grid."""
    from methods.common import entities as ent
    vade_root = os.environ.get("VADE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "VADE"))
    if not os.path.isdir(os.path.join(vade_root, "data", "flags")):
        pytest.skip("no sibling VADE checkout")
    adapter, proc = real
    assets = ent.load_entity_assets(vade_root, "flags")
    model = _cfg(proc)
    model.config.name_or_path = MODEL_ID
    flat, n, last = ent.resolve_position_set("full_image", assets, adapter, model)
    assert (n, len(flat), last) == (169, 169, False)
    if "Idefics3-8B-Llama3" not in assets.object_location.get("models", {}):
        with pytest.raises(AssertionError):
            ent.resolve_position_set("entity_only", assets, adapter, model)
    rows = [r for r in ent.load_tuples(assets, "language", "test")][:2]
    cache = ent.BuildBatchCache()
    img = ent.build_batch(rows, assets, adapter, model, proc, "full_image", batch_cache=cache)
    lt = ent.build_batch(rows, assets, adapter, model, proc, "last_token", batch_cache=cache)
    assert img["positions"].shape == (2, 169) and lt["positions"].shape == (2, 1)
    assert int(lt["positions"][0, 0]) == img["base_input_ids"].shape[1] - 1
    first_real = img["attention_mask"].argmax(dim=1)          # rows are LEFT-padded
    assert (img["base_input_ids"].gather(1, first_real[:, None]) == proc.tokenizer.bos_token_id).all()
    assert ((img["base_input_ids"] == proc.tokenizer.bos_token_id).sum(1) == 1).all()
    assert set(img["base_extra"]) == {"pixel_values", "pixel_attention_mask"}


def test_real_controlled_router_prompts_align(real, canvas):
    """attribute_switch_sweep / attr_head_trace need equal-length prompts that differ only after
    the image. Llama-3's tokenizer must give one token per field word, like Qwen's."""
    from methods import attribute_switch_sweep as sw
    adapter, proc = real
    special = proc.tokenizer.all_special_ids
    built = {a: adapter.build_inputs(proc, canvas, sw.question(a), sw.PREFILL)["input_ids"].unsqueeze(0)
             for a in sw.ATTRIBUTES}
    para = adapter.build_inputs(proc, canvas, sw.question("capital", True), sw.PREFILL)["input_ids"].unsqueeze(0)
    for a in sw.ATTRIBUTES[1:]:
        positions = sw.alignment(built["capital"], built[a], proc.image_token_id, special)
        assert positions
    sw.alignment(built["capital"], para, proc.image_token_id, special)
    differ = (built["capital"][0] != built["currency"][0]).nonzero().flatten().tolist()
    assert len(differ) == 1


def test_real_gold_tokens_for_flags_labels(real):
    from methods.common.targets import derive_gold_token_ids
    adapter, proc = real
    ids = derive_gold_token_ids(proc.tokenizer, "The capital city is", "Paris")
    assert proc.tokenizer.decode(ids) == " Paris" and proc.tokenizer.bos_token_id not in ids


def test_real_accuracy_sweep_style_call_is_single_tile_single_bos(real, canvas):
    """VADE/models/run_accuracy_sweep.py's batched apply_chat_template, with the
    --processor_kwargs '{"do_image_splitting": false}' it is given for this model (X3)."""
    _, proc = real
    msgs = [[{"role": "user", "content": [{"type": "image", "image": canvas}, {"type": "text", "text": q}]},
             {"role": "assistant", "content": [{"type": "text", "text": "The capital is"}]}]
            for q in ("Name the capital?", "What is the capital of the country in this image?")]
    proc.tokenizer.padding_side = "left"
    out = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True,
                                   return_dict=True, return_tensors="pt",
                                   processor_kwargs={"padding": True, **json.loads('{"do_image_splitting": false}')})
    assert out["pixel_values"].shape[:2] == (2, 1)
    for row, m in zip(out["input_ids"], out["attention_mask"]):
        real_ids = row[m.bool()].tolist()
        assert real_ids[0] == proc.tokenizer.bos_token_id and real_ids.count(proc.tokenizer.bos_token_id) == 1
        assert real_ids.count(proc.image_token_id) == 169


# ------------------------------------------------------------------ X3 argument derivation (CPU)

# ATTRIBUTE_SWITCH_FINDINGS.md section 1 (Qwen2.5-VL-7B, flags, donor_first %), by layer.
QWEN_QUESTION = {**{L: 1.0 for L in range(1, 16)}, 16: .988, 17: .786, 18: .286, 19: .131,
                 **{L: .024 for L in range(20, 29)}}
QWEN_LAST = {**{L: .024 for L in range(1, 17)}, 17: .131, 18: .5, 19: .655, 20: .976,
             **{L: 1.0 for L in range(21, 29)}}


def test_router_rule_reproduces_the_hand_picked_qwen_window():
    from methods.circuit_windows import select_router_window
    sel = select_router_window(QWEN_QUESTION, QWEN_LAST, n_layers=28)
    assert sel["patch_layer"] == 15 and sel["blocks"] == list(range(15, 23)) and not sel["warnings"]


def test_router_rule_reads_a_switch_sweep_directory(tmp_path):
    from methods.circuit_windows import main as cw_main, switch_curves
    cells = []
    for L in range(1, 29):
        for scope, curve in (("earlier_text", QWEN_QUESTION), ("last_token", QWEN_LAST)):
            for control, rate in (("switch", curve[L]), ("self", 0.0)):
                cells.append({"arm": f"{scope}/residual/L{L}/prefill/{control}", "base_attribute": "capital",
                              "donor_attribute": "currency", "all": None,
                              "both_clean_correct": {"n": 10, "donor_first": rate, "donor_full": rate,
                                                     "original_full": 1 - rate, "neither_full": 0.0}})
    cells.append({"arm": "clean", "base_attribute": "capital", "donor_attribute": "currency",
                  "all": None, "both_clean_correct": None})
    (tmp_path / "switch_summary.json").write_text(json.dumps(cells))
    q, t = switch_curves(str(tmp_path))
    assert q[16] == pytest.approx(.988) and t[20] == pytest.approx(.976)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cw_main(["router", "--switch_dir", str(tmp_path), "--n_layers", "28", "--out", str(tmp_path / "w.json")])
    assert buf.getvalue().strip() == "--patch_layer 15 --blocks 15 16 17 18 19 20 21 22"
    assert json.loads((tmp_path / "w.json").read_text())["patch_layer"] == 15


def test_freeze_rule_reproduces_the_qwen_severed_arguments(tmp_path):
    from methods.circuit_windows import main as cw_main, select_freeze_spans
    assert select_freeze_spans([21, 22, 23], 28) == {"mlp_span": "22-27", "mlp_singles": [23, 24, 25, 26],
                                                     "attn_span": "24-27"}
    trace = tmp_path / "head_trace_x.json"
    trace.write_text(json.dumps({"phase1_ranked": [[23, 3], [21, 4], [22, 17], [23, 4], [23, 6], [22, 15],
                                                   [23, 17], [21, 1], [27, 0]]}))
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cw_main(["severed", "--trace", str(trace), "--n_layers", "28"])
    assert buf.getvalue().strip() == ("--heads 23.3,21.4,22.17,23.4,23.6,22.15,23.17,21.1 --freeze_mlp_span 22-27 "
                                      "--freeze_mlp_singles 23 24 25 26 --freeze_attn_span 24-27")
