"""ModelAdapter for LLaVA-1.5 (`llava-hf/llava-1.5-7b-hf`, and the 13B), transformers-native.

Why this model (PAPER_RUNBOOK.md X4): it is the reference VLM of the interpretability work this
project is positioned against (Zhang et al. CVPR 2025, Cohen et al., Venhoff et al. NeurIPS 2025,
NOTICE), and its language model is Vicuna (a Llama-2 fine-tune), not a Qwen2.5. Venhoff et al.
report that LLaVA-style models resolve an image's entity TOO LATE to reuse the backbone's recall
circuit, which makes it the natural contrast to Qwen2.5-VL. Every assumption the head / MLP / DLA
code makes holds, checked against transformers 5.16's modeling_llava.py / modeling_llama.py:

  * text stack  model.model.language_model = LlamaModel: .layers[i].{self_attn,mlp}, .norm
                (LlamaRMSNorm, with .variance_epsilon); head model.lm_head, Linear(bias=False),
                not tied (LlavaConfig.tie_word_embeddings defaults to False);
  * 32 blocks x 32 heads x head_dim 128 = hidden 4096 (7B; no GQA), so o_proj's input is 32
    contiguous 128-wide head slices; SwiGLU LlamaMLP (intermediate 11008), down_proj's input is
    the neuron vector; LlamaAttention returns a tuple, like Qwen's;
  * 1-D RoPE and plain causal attention: nothing position-related in `extra`.

What differs from Qwen2.5-VL, and is handled here:

  * Image tokens. CLIP ViT-L/14 at 336px: (336/14)^2 = 576 patch features, the CLS token dropped
    (vision_feature_select_strategy "default"), projected one-to-one by the MLP connector, in
    raster (row-major) order -- a 24x24 grid of 14 canvas px per token. VADE's canvases are
    336x336, the model's native size, so the image processor's resize and center crop are
    identities: no tiling, no distortion, no processor kwargs (unlike Idefics3). The grid is not
    object_location.json's 28px Qwen grid, so `shares_object_grid = False` and
    common/entities.py accepts only `full_image` and `last_token` unless VADE adds a per-model
    override.
  * The image block. The processor expands the one `<image>` placeholder into 576 copies of
    itself (processing_llava.replace_image_token); there are no wrapper tokens. _expand does the
    same string replacement, so tokenize_template matches build_inputs exactly.
  * BOS. LLaVA-1.5's chat template does NOT emit <s>; the tokenizer adds it. Every tokenizer
    call here therefore decides add_special_tokens from whether the rendered text already starts
    with the BOS token, and _check_bos asserts exactly one BOS, at column 0, whichever way a
    future template goes (Idefics3's template emits it and the default call doubles it).
  * Chat shape (llava-1.5-hf's chat_template.json):
        <s>USER: <image>x576\\n{question} ASSISTANT: {prefill}
    The assistant header is the text 'ASSISTANT:' (assistant_start_col finds its first token).
  * `extra` is pixel_values [1, 3, 336, 336] only.
"""
import torch

from .base import ModelAdapter

ASSISTANT_HEADER = "ASSISTANT:"


class LlavaAdapter(ModelAdapter):
    shares_object_grid = False

    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        self.model_id = model_id
        self._template_cache = {}

    def load(self, device="cuda:0", dtype=torch.bfloat16, attn_implementation=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(self.model_id)
        extra = {"attn_implementation": attn_implementation} if attn_implementation is not None else {}
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, device_map=device, dtype=dtype, **extra)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        assert self.image_token_id(model, processor) == processor.image_token_id, (
            f"config image_token_id {self.image_token_id(model, processor)} != processor's {processor.image_token_id}")
        assert self.n_processor_image_tokens(processor) == self.vision_patch_grid(model, 1, 1)["n_tokens"], (
            "the processor and the vision config disagree on the image token count")
        return model, processor

    # ---- text backbone -------------------------------------------------------

    def _text_config(self, model):
        return model.config.text_config

    def hidden_size(self, model):
        return self._text_config(model).hidden_size

    def intermediate_size(self, model):
        return self._text_config(model).intermediate_size

    def n_attention_heads(self, model):
        cfg = self._text_config(model)
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        assert head_dim * cfg.num_attention_heads == cfg.hidden_size, (
            f"head_dim {head_dim} x {cfg.num_attention_heads} heads != hidden {cfg.hidden_size}: "
            f"o_proj's input is not hidden-wide, and every head slice here would be wrong")
        return cfg.num_attention_heads

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

    def _final_norm(self, model):
        return self._text_model(model).norm

    def unembed(self, model, hidden_states):
        return model.lm_head(self._final_norm(model)(hidden_states))

    def final_norm_scale(self, model, hidden_states):
        norm = self._final_norm(model)
        eps = getattr(norm, "variance_epsilon", None)
        assert eps is not None, f"{type(norm).__name__} has no .variance_epsilon -- not the RMSNorm this assumes"
        x = hidden_states.float()
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def logit_direction(self, model, token_ids):
        assert getattr(model.lm_head, "bias", None) is None, "lm_head has a bias -- DLA additivity would break"
        return model.lm_head.weight[token_ids].float() * self._final_norm(model).weight.float()

    # ---- vision side ---------------------------------------------------------

    def image_token_id(self, model, processor):
        tok_id = getattr(model.config, "image_token_id", None)
        if tok_id is None:
            tok_id = getattr(model.config, "image_token_index", None)
        return tok_id if tok_id is not None else processor.image_token_id

    def vision_patch_grid(self, model, canvas_width_px, canvas_height_px):
        vc = model.config.vision_config
        size = vc.image_size if isinstance(vc.image_size, int) else vc.image_size[0]
        patch = vc.patch_size if isinstance(vc.patch_size, int) else vc.patch_size[0]
        strategy = getattr(model.config, "vision_feature_select_strategy", "default")
        assert strategy == "default", (
            f"vision_feature_select_strategy={strategy!r}: only 'default' (CLS dropped, patches only) "
            f"gives a pure raster grid of image tokens")
        assert isinstance(getattr(model.config, "vision_feature_layer", -2), int), (
            "a multi-layer vision_feature_layer concatenates features; still one token per patch, but unchecked here")
        side = size // patch
        assert side * patch == size, f"image {size} not divisible into {patch}px patches"
        return {"model": self.model_id, "patch_size_px": patch, "merge_size": 1,
                "merged_patch_size_px": patch, "tile_px": size, "grid_rows": side, "grid_cols": side,
                "n_tokens": side * side, "flat_index_formula": "row * grid_cols + col (tile space)",
                "canvas_px_per_token": (canvas_width_px / side, canvas_height_px / side)}

    def n_processor_image_tokens(self, processor):
        """The processor's own count (processing_llava.replace_image_token's formula)."""
        crop = processor.image_processor.crop_size
        h, w = (crop["height"], crop["width"]) if isinstance(crop, dict) else (crop.height, crop.width)
        n = (h // processor.patch_size) * (w // processor.patch_size) + processor.num_additional_image_tokens
        return n - 1 if processor.vision_feature_select_strategy == "default" else n

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

    def _add_special(self, processor, text):
        """Add the tokenizer's BOS only if the rendered template has not already written it."""
        bos = processor.tokenizer.bos_token
        return not (bos and text.startswith(bos))

    def _check_bos(self, processor, ids):
        if torch.is_tensor(ids):
            assert ids.dim() == 1 or (ids.dim() == 2 and ids.shape[0] == 1), f"one prompt expected, got {tuple(ids.shape)}"
            ids = ids.reshape(-1).tolist()
        else:
            ids = list(ids)
        bos = processor.tokenizer.bos_token_id
        assert ids and ids[0] == bos and ids.count(bos) == 1, (
            f"expected exactly one BOS ({bos}) at column 0, got {ids.count(bos)} (first ids {ids[:4]})")

    def _expand(self, processor, text, n_image_tokens):
        """The processor's own expansion: one `<image>` -> n copies of itself.
        -> (expanded text, char offset of the image token in `text`, extra chars inserted)."""
        img = processor.image_token
        pos = text.find(img)
        assert pos != -1 and text.count(img) == 1, (
            f"expected exactly one {img!r} in the rendered chat template, found {text.count(img)}")
        assert n_image_tokens == self.n_processor_image_tokens(processor), (
            f"n_image_tokens {n_image_tokens} != the processor's {self.n_processor_image_tokens(processor)}")
        block = img * n_image_tokens
        return text.replace(img, block, 1), pos, len(block) - len(img)

    def _tokenize(self, processor, text, **kw):
        enc = processor.tokenizer(text, add_special_tokens=self._add_special(processor, text), **kw)
        self._check_bos(processor, enc["input_ids"])
        return enc

    def build_inputs(self, processor, image, question, prefill):
        assert image is not None, "LLaVA is a vision-language model -- image is required"
        text = self._render_template_text(processor, question, prefill)
        out = processor(text=[text], images=[image], return_tensors="pt", padding=False,
                        add_special_tokens=self._add_special(processor, text))
        px = out["pixel_values"]
        n = self.n_processor_image_tokens(processor)
        side = int(round(n ** 0.5)) * processor.patch_size
        assert tuple(px.shape) == (1, 3, side, side), (
            f"expected pixel_values [1, 3, {side}, {side}], got {tuple(px.shape)}")
        ids = out["input_ids"][0]
        self._check_bos(processor, ids)
        n_img = int((ids == processor.image_token_id).sum())
        assert n_img == n, f"{n_img} image tokens in the prompt, expected {n}"
        return {"input_ids": ids, "extra": {"pixel_values": px}}

    def encode_image(self, processor, image):
        out = processor.image_processor(images=[image], return_tensors="pt")
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
        """Column of the first token of the LAST 'ASSISTANT:' header. SentencePiece splits the
        word over several tokens, so there is no single header id: this is the LARGEST column i
        whose decoded suffix ids[i:] starts with the header -- one column later the suffix starts
        mid-word ('ISTANT: ...'). No decode/re-encode round trip, so columns cannot drift."""
        ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
        tok = processor.tokenizer
        for i in range(len(ids) - 1, -1, -1):
            if tok.decode(ids[i:], skip_special_tokens=False).lstrip().startswith(ASSISTANT_HEADER):
                return i
        raise AssertionError(f"no {ASSISTANT_HEADER!r} in the prompt -- not LLaVA-1.5's chat shape")
