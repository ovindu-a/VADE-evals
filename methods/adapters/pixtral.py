"""ModelAdapter for Pixtral-12B in its transformers port (`mistral-community/pixtral-12b`):
a LlavaForConditionalGeneration with a Pixtral vision tower and a Mistral-Nemo-12B language model.

Why this model (PAPER_RUNBOOK.md X15): Nikankin et al. (NeurIPS'25)'s other non-Qwen VLM, and the first
Mistral-family language model here. Checked against transformers 5.16's modeling_llava.py,
modeling_mistral.py and processing_pixtral.py:

  * text stack  model.model.language_model = MistralModel: .layers[i].{self_attn,mlp}, .norm
                (MistralRMSNorm, .variance_epsilon, Llama-style weight); head model.lm_head,
                Linear(bias=False), not tied. Plain pre-norm blocks (no post-sublayer norms), 1-D RoPE,
                causal attention, no sliding window (sliding_window=None in the checkpoint);
  * 40 blocks x 32 query heads x head_dim 128 (8 KV heads) -> o_proj's INPUT is 4096 wide while
    hidden is 5120. head_dim() comes from the config; hidden // heads (160) would slice o_proj's input
    into the wrong columns and every head patch / ranking would be silently wrong;
  * SwiGLU MistralMLP (intermediate 14336), down_proj's input is the neuron vector.

What differs from Qwen2.5-VL, and is handled here:

  * VARIABLE RESOLUTION. Pixtral keeps the image's own size (down-scaled only if its longest edge
    exceeds 1024) and patchifies it at 16px with no merge, so the grid follows the canvas. Every VADE
    canvas is 336x336 = 21 x 16px, so the grid is a FIXED 21x21 = 441 tokens, each covering exactly
    16x16 canvas px; vision_patch_grid recomputes it with the processor's own resize formula from the
    canvas (never from a constant), and build_inputs asserts pixel_values / image_sizes agree.
  * ROW BREAKS. The image block is NOT contiguous: each grid row of [IMG] tokens is followed by one
    [IMG_BREAK], and the LAST row's break is replaced by [IMG_END] -- 21 x 22 = 462 tokens, of which
    441 are [IMG] (= image_token_id, the positions image features fill, in raster order). Flat index
    row * 21 + col is the (row*21 + col)-th [IMG] token, not an offset from the first;
    common/entities.py's object_token_positions accepts exactly this layout (one-token gap after
    every row) and indexes by image-token rank. image_row_break = True says so.
  * BOS. The chat template writes <s> and the tokenizer does not add another (checked); every
    tokenizer call decides add_special_tokens from whether the rendered text already starts with BOS,
    LLaVA-style, and _check_bos asserts exactly one either way.
  * Prompt shape (no system turn, no assistant header -- the answer follows [/INST] directly):
        <s>[INST]{image block}{question}[/INST]{prefill}
    assistant_start_col is therefore the [/INST] column.
  * `extra` is pixel_values [1, 3, 336, 336] plus image_sizes [1, 2] (int, never dtype-cast).
  * The checkpoint's tokenizer.json triggers transformers' "incorrect regex pattern" warning (the
    Mistral pre-tokenizer regex). fix_mistral_regex=True gives identical ids on every VADE-style
    prompt/label tested (prompts, labels with diacritics, digit answers); the tests pin that.
"""
import math
import os

import torch

from .base import ModelAdapter, containing_box_override


class PixtralAdapter(ModelAdapter):
    shares_object_grid = False
    image_row_break = True

    def __init__(self, model_id="mistral-community/pixtral-12b"):
        self.model_id = model_id
        self._template_cache = {}

    def load(self, device="cuda:0", dtype=torch.bfloat16, attn_implementation=None):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(self.model_id)
        # VADE_DEVICE: CPU smoke runs of the real CLIs on a tiny random checkpoint (X15 notebook's
        # offline check); unset on a GPU, so every script's default cuda:0 is unchanged.
        device = os.environ.get("VADE_DEVICE") or device
        extra = {"attn_implementation": attn_implementation} if attn_implementation is not None else {}
        model = LlavaForConditionalGeneration.from_pretrained(self.model_id, device_map=device, dtype=dtype, **extra)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        assert self.image_token_id(model, processor) == processor.image_token_id, (
            f"config image_token_id {self.image_token_id(model, processor)} != processor's {processor.image_token_id}")
        assert processor.patch_size * processor.spatial_merge_size == self._cell_px(model), (
            "the processor and the vision config disagree on the token cell size")
        return model, processor

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
        d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
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

    def _cell_px(self, model):
        vc = model.config.vision_config
        patch = vc.patch_size if isinstance(vc.patch_size, int) else vc.patch_size[0]
        return patch * int(getattr(model.config, "spatial_merge_size", 1) or 1)

    def vision_patch_grid(self, model, canvas_width_px, canvas_height_px):
        """The processor's own resize rule (image_processing_pixtral.get_resize_output_image_size):
        down-scale (floor) only if the longest edge exceeds the vision image_size, then one token per
        started cell. A 336px canvas stays 336px -> 21x21."""
        vc = model.config.vision_config
        longest = vc.image_size if isinstance(vc.image_size, int) else vc.image_size[0]
        strategy = getattr(model.config, "vision_feature_select_strategy", "full")
        assert strategy == "full", f"vision_feature_select_strategy={strategy!r}: Pixtral keeps every patch"
        cell = self._cell_px(model)
        h, w = canvas_height_px, canvas_width_px
        ratio = max(h / longest, w / longest)
        if ratio > 1:
            h, w = int(math.floor(h / ratio)), int(math.floor(w / ratio))
        rows, cols = math.ceil(h / cell), math.ceil(w / cell)
        return {"model": self.model_id, "patch_size_px": cell, "merge_size": 1, "merged_patch_size_px": cell,
                "tile_px": (rows * cell, cols * cell), "grid_rows": rows, "grid_cols": cols,
                "n_tokens": rows * cols,
                "flat_index_formula": "row * grid_cols + col over [IMG] tokens only ([IMG_BREAK]/[IMG_END] skipped)",
                "canvas_px_per_token": (canvas_width_px / cols, canvas_height_px / rows)}

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
        """The processor's own expansion of the one [IMG] placeholder into a SQUARE grid of
        n_image_tokens [IMG] tokens with row breaks (VADE canvases are square).
        -> (expanded text, char offset of the placeholder in `text`, extra chars inserted)."""
        img = processor.image_token
        pos = text.find(img)
        assert pos != -1 and text.count(img) == 1, (
            f"expected exactly one {img!r} in the rendered chat template, found {text.count(img)}")
        side = math.isqrt(n_image_tokens)
        assert side * side == n_image_tokens, f"{n_image_tokens} image tokens is not a square grid"
        px = side * processor.patch_size * processor.spatial_merge_size
        block = processor.replace_image_token({"image_sizes": [[px, px]]}, 0)
        assert block.count(img) == n_image_tokens and block.count(processor.image_break_token) == side - 1
        return text.replace(img, block, 1), pos, len(block) - len(img)

    def _tokenize(self, processor, text, **kw):
        enc = processor.tokenizer(text, add_special_tokens=self._add_special(processor, text), **kw)
        self._check_bos(processor, enc["input_ids"])
        return enc

    def build_inputs(self, processor, image, question, prefill):
        assert image is not None, "Pixtral is a vision-language model -- image is required"
        text = self._render_template_text(processor, question, prefill)
        out = processor(text=[text], images=[image], return_tensors="pt", padding=False,
                        add_special_tokens=self._add_special(processor, text))
        ids = out["input_ids"][0]
        self._check_bos(processor, ids)
        h, w = (int(v) for v in out["image_sizes"][0])
        cell = processor.patch_size * processor.spatial_merge_size
        assert tuple(out["pixel_values"].shape) == (1, 3, h, w) and h % cell == 0 and w % cell == 0, (
            f"pixel_values {tuple(out['pixel_values'].shape)} vs image_sizes {(h, w)}")
        n_img = int((ids == processor.image_token_id).sum())
        assert n_img == (h // cell) * (w // cell), f"{n_img} [IMG] tokens for a {h}x{w} image"
        return {"input_ids": ids, "extra": {"pixel_values": out["pixel_values"], "image_sizes": out["image_sizes"]}}

    def encode_image(self, processor, image):
        out = processor.image_processor(images=[image], return_tensors="pt")
        return {"pixel_values": out["pixel_values"], "image_sizes": out["image_sizes"]}

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
        """Column of the LAST [/INST]: Mistral's template has no assistant header, the reply
        (here the prefill) starts right after it."""
        ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
        end = processor.tokenizer.convert_tokens_to_ids("[/INST]")
        ends = [i for i, t in enumerate(ids) if t == end]
        assert ends, "no [/INST] in the prompt -- not Mistral's chat shape"
        return ends[-1]
