"""ModelAdapter for Gemma-3 (`google/gemma-3-12b-it`; `google/gemma-3-4b-it` for smoke runs),
transformers-native Gemma3ForConditionalGeneration.

Why this model (PAPER_RUNBOOK.md X15): one of Nikankin et al. (NeurIPS'25)'s two non-Qwen VLMs, and
the first Gemma-family language model here (Qwen2.5, Llama-3.1 and Llama-2 are the others). Checked
against transformers 5.16's modeling_gemma3.py / processing_gemma3.py:

  * text stack  model.model.language_model = Gemma3TextModel: .layers[i].{self_attn,mlp}, .norm;
                head model.lm_head, Linear(bias=False), TIED to the token embeddings (fine for DLA:
                the head reads the raw weight; the sqrt(hidden) embedding scale is applied only in
                Gemma3TextScaledWordEmbedding.forward, i.e. it is already inside hidden_states[0]);
  * 12B: 48 blocks x 16 query heads x head_dim 256 (8 KV heads) -> o_proj's INPUT is 4096 wide while
    hidden is 3840. 4B: 34 x 8 x 256 = 2048 vs 2560. head_dim() is overridden from the config; the
    default hidden // heads would slice o_proj's input wrongly and silently;
  * SwiGLU-style Gemma3MLP (gelu_tanh gate), down_proj's input is the neuron vector.

What differs from Qwen2.5-VL / Llama, and is handled here:

  * POST-SUBLAYER NORMS. A Gemma-3 block is
        h = h + post_attention_layernorm(self_attn(input_layernorm(h)))
        h = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
    so what a sublayer ADDS to the residual is the post-norm tensor, not the attention/MLP output.
    get_attn_output_module / get_mlp_output_module return the two post-norms (the `attn_output` /
    `mlp_output` sites, and dla.py's components); get_attn_block / get_mlp_block stay the attention /
    MLP modules, which callers reach into for .v_proj and attention weights.
    head_residual_writes freezes post_attention_layernorm's per-position scale at the value the full
    head sum gets, so per-head writes still sum exactly to `attn_output`.
  * RMSNorm. Gemma3RMSNorm multiplies by (1 + weight) (weights are zero-initialised offsets), and its
    epsilon attribute is `.eps`, not Llama's `.variance_epsilon`. logit_direction folds in 1 + weight.
    Gemma-2 soft-capped its final logits, which would make DLA non-additive; Gemma-3 does not, and
    load() / logit_direction assert final_logit_softcapping is None rather than assume it.
  * IMAGE TOKENS. SigLIP at 896px, 14px patches = 64x64, average-pooled 4x4 -> 16x16 = 256 soft
    tokens in raster order, each covering 21x21 canvas px of VADE's 336px canvas (the same geometry
    as InternVL3's 448px tile). Pan-and-scan is OFF (the processor default; passed explicitly anyway
    and asserted: one 896px image per prompt). The block is
        \\n\\n<start_of_image><image_soft_token>x256<end_of_image>\\n\\n
    (the processor's full_image_sequence). NOTE processor.image_token_id is <start_of_image> (the
    placeholder in the chat template); the token whose positions carry image features is
    model.config.image_token_id = <image_soft_token>, which is what image_token_id() returns.
  * BIDIRECTIONAL IMAGE ATTENTION. Image soft tokens attend to each other in both directions (the
    model was trained that way), signalled by token_type_ids == 1 on them. Without token_type_ids
    transformers silently falls back to a plain causal mask. Nothing in this repo passes them (extra
    is per-row and cannot carry a per-position, padding-dependent tensor), so load() installs two
    shims: a forward pre-hook that derives token_type_ids = (input_ids == image token) on any
    prefill call that lacks them, and a prepare_inputs_for_generation wrapper doing the same, which
    is needed because Gemma-3's generation_config uses a static ("hybrid") cache, so generate()
    builds the 4-D mask BEFORE forward and the pre-hook alone would come too late. Consequence for
    the position sets: the `seq:before` null DOES NOT HOLD here -- background image tokens before the
    object in raster order read the object through bidirectional attention (they are refused anyway,
    since grid specs are refused for any model with shares_object_grid = False).
  * BOS. The chat template writes <bos> and the processor's default call adds a second one
    (checked); every tokenizer call here passes add_special_tokens=False and _check_bos asserts one.
  * Prompt shape (no system turn):
        <bos><start_of_turn>user\\n{image block}{question}<end_of_turn>\\n<start_of_turn>model\\n{prefill}
  * `extra` is pixel_values [1, 3, 896, 896] only.
  * generation_config ships do_sample=True / top_k / top_p; every generate() call here passes
    do_sample=False, and load() also resets the config so a stray call cannot sample.
"""
import os

import torch

from .base import ModelAdapter, containing_box_override


class Gemma3Adapter(ModelAdapter):
    shares_object_grid = False
    PROCESSOR_KWARGS = {"do_pan_and_scan": False}

    def __init__(self, model_id="google/gemma-3-12b-it"):
        self.model_id = model_id
        self._template_cache = {}

    def load(self, device="cuda:0", dtype=torch.bfloat16, attn_implementation=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(self.model_id)
        # VADE_DEVICE: CPU smoke runs of the real CLIs on a tiny random checkpoint (X15 notebook's
        # offline check); unset on a GPU, so every script's default cuda:0 is unchanged.
        device = os.environ.get("VADE_DEVICE") or device
        extra = {"attn_implementation": attn_implementation} if attn_implementation is not None else {}
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, device_map=device, dtype=dtype, **extra)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        self.prepare_model(model, processor)
        return model, processor

    def prepare_model(self, model, processor=None):
        """Everything load() does to a model besides loading it -- split out so the offline tests
        can apply it to a tiny random Gemma3ForConditionalGeneration."""
        cfg = self._text_config(model)
        assert getattr(cfg, "final_logit_softcapping", None) is None, (
            f"final_logit_softcapping={cfg.final_logit_softcapping}: the logits are soft-capped, so DLA's "
            f"frozen-scale decomposition is no longer exact")
        assert not getattr(cfg, "use_bidirectional_attention", False), "text-side bidirectional attention: not a decoder"
        gc = model.generation_config
        gc.do_sample, gc.top_k, gc.top_p = False, None, None
        if processor is not None:
            soft = processor.tokenizer.image_token_id
            assert soft == self.image_token_id(model, processor), (
                f"tokenizer image_token_id {soft} != config image_token_id {self.image_token_id(model, processor)}")
            assert processor.image_seq_length == self.vision_patch_grid(model, 1, 1)["n_tokens"], (
                "the processor and the vision config disagree on the image token count")
        self._install_token_type_ids(model)
        return model

    def _install_token_type_ids(self, model):
        if getattr(model, "_vade_token_type_ids", False):
            return
        img = self.image_token_id(model, None)

        def needs(kwargs):
            pkv = kwargs.get("past_key_values")
            return kwargs.get("token_type_ids") is None and (pkv is None or pkv.get_seq_length() == 0)

        def pre(mod, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is None or not needs(kwargs):
                return None
            kwargs["token_type_ids"] = (ids == img).long()
            return args, kwargs

        model.register_forward_pre_hook(pre, with_kwargs=True)
        orig = model.prepare_inputs_for_generation

        def prepare_inputs_for_generation(input_ids, *args, **kwargs):
            if kwargs.get("token_type_ids") is None:
                kwargs["token_type_ids"] = (input_ids == img).long()
            return orig(input_ids, *args, **kwargs)

        model.prepare_inputs_for_generation = prepare_inputs_for_generation
        model._vade_token_type_ids = True

    # ---- text backbone -------------------------------------------------------

    def _text_config(self, model):
        return model.config.text_config

    def hidden_size(self, model):
        return self._text_config(model).hidden_size

    def intermediate_size(self, model):
        return self._text_config(model).intermediate_size

    def n_attention_heads(self, model):
        return self._text_config(model).num_attention_heads

    def head_dim(self, model):
        cfg = self._text_config(model)
        d = cfg.head_dim
        o_in = self.get_attn_head_output_module(model, 0).in_features
        assert d * cfg.num_attention_heads == o_in, (
            f"head_dim {d} x {cfg.num_attention_heads} heads != o_proj's input width {o_in}")
        return d

    def _text_model(self, model):
        return model.model.language_model

    def get_decoder_layers(self, model):
        return self._text_model(model).layers

    def get_mlp_block(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].mlp

    def get_mlp_hidden_module(self, model, block_idx):
        return self.get_mlp_block(model, block_idx).down_proj

    def get_attn_block(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].self_attn

    def get_attn_head_output_module(self, model, block_idx):
        return self.get_attn_block(model, block_idx).o_proj

    def get_attn_output_module(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].post_attention_layernorm

    def get_mlp_output_module(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].post_feedforward_layernorm

    @staticmethod
    def _gemma_norm(norm):
        eps = getattr(norm, "eps", None)
        assert eps is not None and type(norm).__name__.startswith("Gemma"), (
            f"{type(norm).__name__} is not the Gemma (1 + weight) RMSNorm this adapter assumes")
        return eps, 1.0 + norm.weight.float()

    def head_residual_writes(self, model, block_idx, z):
        """Per-head writes THROUGH post_attention_layernorm, its scale frozen at the full sum's:
        post_norm(sum_h x_h) = s * g * sum_h x_h  with  s = rsqrt(mean((sum_h x_h)^2) + eps)."""
        x = super().head_residual_writes(model, block_idx, z)                     # [..., n, H] float32
        eps, gain = self._gemma_norm(self.get_attn_output_module(model, block_idx))
        s = torch.rsqrt(x.sum(-2).pow(2).mean(-1, keepdim=True) + eps)            # [..., 1]
        return x * (s.unsqueeze(-2) * gain)

    def _final_norm(self, model):
        return self._text_model(model).norm

    def unembed(self, model, hidden_states):
        assert getattr(self._text_config(model), "final_logit_softcapping", None) is None, "soft-capped logits"
        return model.lm_head(self._final_norm(model)(hidden_states))

    def final_norm_scale(self, model, hidden_states):
        eps, _ = self._gemma_norm(self._final_norm(model))
        x = hidden_states.float()
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def logit_direction(self, model, token_ids):
        assert getattr(model.lm_head, "bias", None) is None, "lm_head has a bias -- DLA additivity would break"
        assert getattr(self._text_config(model), "final_logit_softcapping", None) is None, "soft-capped logits"
        _, gain = self._gemma_norm(self._final_norm(model))
        return model.lm_head.weight[token_ids].float() * gain

    # ---- vision side ---------------------------------------------------------

    def image_token_id(self, model, processor):
        """<image_soft_token> (the positions image features are scattered into), NOT
        processor.image_token_id, which is <start_of_image>."""
        tok_id = getattr(model.config, "image_token_id", None)
        if tok_id is None:
            tok_id = getattr(model.config, "image_token_index", None)
        return tok_id if tok_id is not None else processor.tokenizer.image_token_id

    def vision_patch_grid(self, model, canvas_width_px, canvas_height_px):
        vc = model.config.vision_config
        size = vc.image_size if isinstance(vc.image_size, int) else vc.image_size[0]
        patch = vc.patch_size if isinstance(vc.patch_size, int) else vc.patch_size[0]
        side = int(round(model.config.mm_tokens_per_image ** 0.5))
        per_side = size // patch
        assert side * side == model.config.mm_tokens_per_image and per_side % side == 0 and per_side * patch == size, (
            f"{size}px / {patch}px patches = {per_side} per side does not pool evenly to {side}x{side}")
        pool = per_side // side
        return {"model": self.model_id, "patch_size_px": patch, "merge_size": pool,
                "merged_patch_size_px": patch * pool, "tile_px": size, "grid_rows": side, "grid_cols": side,
                "n_tokens": side * side, "flat_index_formula": "row * grid_cols + col (tile space)",
                "canvas_px_per_token": (canvas_width_px / side, canvas_height_px / side)}

    def object_location_override(self, object_location, model):
        canvas = object_location["canvas_px"]
        return containing_box_override(object_location,
                                       self.vision_patch_grid(model, canvas["width"], canvas["height"]))

    def _messages(self, question, prefill, image=None):
        img = {"type": "image", "image": image} if image is not None else {"type": "image"}
        return [{"role": "user", "content": [img, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": prefill}]}]

    def _render_template_text(self, processor, question, prefill):
        key = (question, prefill)
        if key not in self._template_cache:
            self._template_cache[key] = processor.apply_chat_template(
                self._messages(question, prefill), tokenize=False, add_generation_prompt=False,
                continue_final_message=True)
        return self._template_cache[key]

    def _check_bos(self, processor, ids):
        if torch.is_tensor(ids):
            assert ids.dim() == 1 or (ids.dim() == 2 and ids.shape[0] == 1), f"one prompt expected, got {tuple(ids.shape)}"
            ids = ids.reshape(-1).tolist()
        else:
            ids = list(ids)
        bos = processor.tokenizer.bos_token_id
        assert ids and ids[0] == bos and ids.count(bos) == 1, (
            f"expected exactly one BOS ({bos}) at column 0, got {ids.count(bos)} (first ids {ids[:4]}) -- "
            f"the chat template already writes <bos>, so add_special_tokens=True doubles it")

    def _expand(self, processor, text, n_image_tokens):
        """The processor's own expansion of the one <start_of_image> placeholder (no pan-and-scan
        crops). -> (expanded text, char offset of the placeholder in `text`, extra chars inserted)."""
        boi = processor.boi_token
        pos = text.find(boi)
        assert pos != -1 and text.count(boi) == 1, (
            f"expected exactly one {boi!r} in the rendered chat template, found {text.count(boi)}")
        assert n_image_tokens == processor.image_seq_length, (
            f"n_image_tokens {n_image_tokens} != the processor's image_seq_length {processor.image_seq_length}")
        block = processor.replace_image_token({"num_crops": [0]}, 0)
        assert block.count(processor.tokenizer.image_token) == n_image_tokens
        return text.replace(boi, block, 1), pos, len(block) - len(boi)

    def _tokenize(self, processor, text, **kw):
        enc = processor.tokenizer(text, add_special_tokens=False, **kw)
        self._check_bos(processor, enc["input_ids"])
        return enc

    def build_inputs(self, processor, image, question, prefill):
        assert image is not None, "Gemma-3 is a vision-language model here -- image is required"
        text = self._render_template_text(processor, question, prefill)
        out = processor(text=[text], images=[image], return_tensors="pt", padding=False,
                        add_special_tokens=False, **self.PROCESSOR_KWARGS)
        px = out["pixel_values"]
        assert px.shape[0] == 1, (
            f"expected ONE image, got pixel_values {tuple(px.shape)} -- pan-and-scan crops were added, and "
            f"every token-count assumption here would be wrong")
        ids = out["input_ids"][0]
        self._check_bos(processor, ids)
        soft = processor.tokenizer.image_token_id
        assert int((ids == soft).sum()) == processor.image_seq_length
        if "token_type_ids" in out:      # what the forward shim derives must be what the processor says
            assert torch.equal(out["token_type_ids"][0].bool(), ids == soft), "token_type_ids != image-token mask"
        return {"input_ids": ids, "extra": {"pixel_values": px}}

    def encode_image(self, processor, image):
        out = processor.image_processor(images=[image], return_tensors="pt", **self.PROCESSOR_KWARGS)
        assert out["pixel_values"].shape[0] == 1, f"expected ONE image, got {tuple(out['pixel_values'].shape)}"
        return {"pixel_values": out["pixel_values"]}

    def tokenize_template(self, processor, question, prefill, n_image_tokens):
        text, _, _ = self._expand(processor, self._render_template_text(processor, question, prefill),
                                  n_image_tokens)
        return self._tokenize(processor, text, return_tensors="pt")["input_ids"][0]

    def find_last_phrase_token_col(self, processor, question, prefill, n_image_tokens, phrases):
        q_lower = question.lower()
        match = next(((p, q_lower.rfind(p.lower())) for p in phrases if p.lower() in q_lower), None)
        if match is None:
            return None
        phrase, char_start = match
        text = self._render_template_text(processor, question, prefill)
        q_pos = text.find(question)
        assert q_pos != -1, "question text not found verbatim in its own rendered chat template"
        expanded, img_pos, shift = self._expand(processor, text, n_image_tokens)
        abs_char_end = q_pos + char_start + len(phrase) + (shift if img_pos < q_pos else 0)
        offsets = self._tokenize(processor, expanded, return_offsets_mapping=True)["offset_mapping"]
        token_idx = next((i for i, (s, e) in enumerate(offsets) if s < abs_char_end <= e), None)
        assert token_idx is not None, f"phrase {phrase!r} end did not land inside a token"
        return token_idx

    def assistant_start_col(self, processor, input_ids):
        """Column of the LAST <start_of_turn>, which opens the model turn (Gemma's ChatML analogue)."""
        ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
        sot = processor.tokenizer.convert_tokens_to_ids("<start_of_turn>")
        starts = [i for i, t in enumerate(ids) if t == sot]
        assert starts, "no <start_of_turn> in the prompt -- not Gemma's chat shape"
        return starts[-1]
