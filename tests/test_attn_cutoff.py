"""X10 (methods/attn_cutoff.py): the cut primitives on tiny random Llama / Qwen2 stacks with GQA."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods import attn_cutoff as ac  # noqa: E402


def tiny(kind):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    kw = dict(vocab_size=50, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=64)
    cfg = LlamaConfig(**kw) if kind == "llama" else Qwen2Config(**kw)
    cfg._attn_implementation = "eager"
    model = (LlamaForCausalLM if kind == "llama" else Qwen2ForCausalLM)(cfg).eval()
    layers = model.model.layers
    return model, {b: l.self_attn for b, l in enumerate(layers)}, {b: l.mlp for b, l in enumerate(layers)}


def cut(model, attn, mlp, arm, blocks):
    return ac.register_cut(attn, mlp, arm, blocks, 4, ac.n_kv_heads(model))


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_attn_self_equals_a_mask_that_lets_the_last_query_see_only_itself(kind):
    model, attn, mlp = tiny(kind)
    ids = torch.tensor([[1, 7, 9, 11, 13, 5]])
    T = ids.shape[1]
    with torch.no_grad():
        hs = cut(model, attn, mlp, "attn_self", range(3))
        try:
            hooked = model(ids).logits[0, -1]
        finally:
            for h in hs:
                h.remove()
        allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
        allowed[-1, :-1] = False                          # last query: itself only
        mask4 = torch.zeros(1, 1, T, T).masked_fill(~allowed, torch.finfo(torch.float32).min)
        masked = model(ids, attention_mask=mask4).logits[0, -1]
        clean = model(ids).logits[0, -1]
    torch.testing.assert_close(hooked, masked, atol=1e-5, rtol=1e-5)
    assert (hooked - clean).abs().max() > 1e-3, "the cut changed nothing"


@pytest.mark.parametrize("arm", ac.ARMS)
def test_cut_touches_only_the_last_column_and_only_blocks_from_L(arm):
    model, attn, mlp = tiny("llama")
    ids = torch.tensor([[1, 7, 9, 11, 13, 5]])
    with torch.no_grad():
        clean = model(ids, output_hidden_states=True).hidden_states
        hs = cut(model, attn, mlp, arm, [2])
        try:
            got = model(ids, output_hidden_states=True).hidden_states
        finally:
            for h in hs:
                h.remove()
    for L in (0, 1, 2):                                   # before block 2's output: identical
        torch.testing.assert_close(got[L], clean[L])
    torch.testing.assert_close(got[3][:, :-1], clean[3][:, :-1])   # other columns: identical
    assert (got[3][:, -1] - clean[3][:, -1]).abs().max() > 1e-4


def test_generation_cuts_every_generated_token_and_lens_reads_the_prefill_only():
    model, attn, mlp = tiny("llama")
    ids = torch.tensor([[1, 7, 9, 11, 13, 5]])
    seen = []
    hook = attn[1].register_forward_hook(lambda mod, inp, out: seen.append(out[0].shape[1]))
    sink = {}
    hs = ac.register_lens(model.model.layers, sink) + cut(model, attn, mlp, "attn_self", [1, 2])
    try:
        with torch.no_grad():
            model.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    finally:
        for h in hs + [hook]:
            h.remove()
    assert seen[0] == ids.shape[1] and all(t == 1 for t in seen[1:]), seen   # prefill, then one column per step
    assert set(sink) == {1, 2, 3} and sink[3].shape == (1, 32)


def test_dry_run(capsys):
    vade = os.environ.get("VADE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "VADE"))
    if not os.path.isdir(os.path.join(vade, "data", "flags")):
        pytest.skip("no sibling VADE checkout")
    ac.main(["--dry_run", "--vade_root", vade])
    assert "Grid valid." in capsys.readouterr().out


def test_main_end_to_end_on_a_tiny_llava(tmp_path, monkeypatch):
    """The whole script on a tiny random LLaVA wrapped as circuit_common.Model: prompts, cut hooks
    during generate, the lens, the summary. Answers are random, so the matcher is forced to True."""
    import json
    from types import SimpleNamespace
    from transformers import LlavaConfig, LlavaForConditionalGeneration
    import methods.circuit_common as cc
    import methods.common.text_match as tm
    from methods.adapters.registry import get_adapter
    vade = os.environ.get("VADE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "VADE"))
    if not os.path.isdir(os.path.join(vade, "data", "flags")):
        pytest.skip("no sibling VADE checkout")
    torch.manual_seed(1)
    config = LlavaConfig(
        vision_config=dict(model_type="clip_vision_model", hidden_size=16, num_hidden_layers=1,
                           num_attention_heads=2, intermediate_size=24, image_size=28, patch_size=14, projection_dim=16),
        text_config=dict(model_type="llama", vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2),
        image_token_index=9, vision_feature_layer=-2, vision_feature_select_strategy="default")
    model = LlavaForConditionalGeneration(config).eval()
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=2,
                          decode=lambda ids, **k: " ".join(f"t{int(i)}" for i in ([ids] if isinstance(ids, int) else ids)))
    m = cc.Model.wrap(get_adapter("llava-hf/llava-1.5-7b-hf"), model, SimpleNamespace(tokenizer=tok))
    monkeypatch.setattr(m, "decode", lambda toks: [tok.decode(t) for t in toks])

    def image_prompt(entity_dir, item, attribute):
        g = torch.Generator().manual_seed(len(item["name"]))
        return (torch.tensor([1, 20, 21, 9, 9, 9, 9, 13, 30 + len(attribute) % 5, 40, 41]),
                {"pixel_values": torch.randn(1, 3, 28, 28, generator=g)})
    monkeypatch.setattr(m, "image_prompt", image_prompt)
    monkeypatch.setattr(cc, "Model", lambda *a, **k: m)
    monkeypatch.setattr(cc, "gold_ids", lambda t, label: [10 + len(str(label)) % 40])
    monkeypatch.setattr(tm, "contains_label", lambda text, label: True)
    ac.main(["--model_id", "llava-hf/llava-1.5-7b-hf", "--vade_root", vade, "--n_items", "3",
             "--attributes", "capital", "language", "--layer_step", "2", "--max_new_tokens", "3",
             "--out_dir", str(tmp_path)])
    s = json.load(open(tmp_path / "summary.json"))["summary"]
    assert s["clean"]["n_correct"] == 6 and set(s) == {"clean", *ac.ARMS}
    assert set(s["attn_self"]) == {"0", "2"} and 0 <= s["attn_self"]["0"]["same_first_token"] <= 1
    assert set(s["clean"]["value_rank_median"]) == {"1", "2", "3", "4"}
    rows = [json.loads(l) for l in open(tmp_path / "rows.jsonl")]
    assert len(rows) == 6 * (1 + 3 * 2)
