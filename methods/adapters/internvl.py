"""ModelAdapter for InternVL3 in its native transformers port (`OpenGVLab/InternVL3-8B-hf`).

Why this model as the second family (Tier-2 E8): InternVL3-8B's language model is a
Qwen2.5-7B (28 blocks x 28 query heads, hidden 3584, intermediate 18944 -- the same
shapes as Qwen2.5-VL-7B's text side), behind a different vision encoder (InternViT),
a different connector (pixel-shuffle + MLP) and different multimodal training. So
"same circuit at the same (block, head)?" is a meaningful question there, and Park et
al. (arXiv 2608.27417) publish its Visual Retrieval Heads too.

What differs from Qwen2.5-VL, and is handled here:

  * Image tokens. One 448x448 tile (crop_to_patches=False -- VADE's 336x336 canvases
    fit one tile, so dynamic tiling would only add a thumbnail), 14px patches, pixel
    shuffle 0.5 -> 16x16 = 256 tokens wrapped in <img> ... </img>. Each token covers
    21x21 CANVAS px, so object_location.json's 28px Qwen grid does not apply:
    `shares_object_grid = False` makes common/entities.py accept only `full_image` and
    `last_token` for this adapter (the object box does not fall on 21px boundaries,
    so any object set would be an approximation -- refused rather than guessed).
  * No image_grid_thw and no M-RoPE: `extra` is just pixel_values.
  * The text backbone, final norm and head are named exactly as in Qwen2.5-VL
    (model.model.language_model.{layers,norm}, model.lm_head), which is why every
    site / DLA method below is the same code; they are kept explicit rather than
    inherited so a transformers rename fails here, loudly, and not in a hook.
"""
import torch

from .base import ModelAdapter


class InternVLAdapter(ModelAdapter):
    shares_object_grid = False

    def __init__(self, model_id="OpenGVLab/InternVL3-8B-hf"):
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
        return model, processor

    # ---- text backbone -------------------------------------------------------

    def _text_config(self, model):
        return getattr(model.config, "text_config", None) or model.config

    def hidden_size(self, model):
        return self._text_config(model).hidden_size

    def intermediate_size(self, model):
        return self._text_config(model).intermediate_size

    def n_attention_heads(self, model):
        # QUERY heads (28 on InternVL3-8B); num_key_value_heads (4) is the GQA count.
        return self._text_config(model).num_attention_heads

    def get_decoder_layers(self, model):
        return model.model.language_model.layers

    def get_mlp_block(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].mlp

    def get_mlp_hidden_module(self, model, block_idx):
        # Qwen2MLP: down_proj(act_fn(gate_proj(x)) * up_proj(x)) -- down_proj's input is the neuron vector.
        return self.get_mlp_block(model, block_idx).down_proj

    def get_attn_block(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].self_attn

    def get_attn_head_output_module(self, model, block_idx):
        return self.get_attn_block(model, block_idx).o_proj

    def _final_norm(self, model):
        return model.model.language_model.norm

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
        if tok_id is not None:
            return tok_id
        return processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    def vision_patch_grid(self, model, canvas_width_px, canvas_height_px):
        vc = model.config.vision_config
        size = vc.image_size if isinstance(vc.image_size, int) else vc.image_size[0]
        patch = vc.patch_size if isinstance(vc.patch_size, int) else vc.patch_size[0]
        merge = int(round(1 / model.config.downsample_ratio))
        side = size // patch // merge
        assert side * patch * merge == size, f"tile {size} not divisible into {patch}px patches x {merge}"
        return {"model": self.model_id, "patch_size_px": patch, "merge_size": merge,
                "merged_patch_size_px": patch * merge, "tile_px": size, "grid_rows": side, "grid_cols": side,
                "n_tokens": side * side, "flat_index_formula": "row * grid_cols + col (tile space)",
                "canvas_px_per_token": (canvas_width_px / side, canvas_height_px / side)}

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

    def _expand(self, processor, text, n_image_tokens):
        """The processor's own placeholder expansion (processing_internvl._insert_media_placeholders):
        one image_token -> start + image_token * n + end. -> (expanded text, char offset of the image
        token in `text`, extra chars inserted)."""
        img = processor.image_token
        pos = text.find(img)
        assert pos != -1 and text.count(img) == 1, (
            f"expected exactly one {img!r} in the rendered chat template, found {text.count(img)}")
        block = processor.start_image_token + img * n_image_tokens + processor.end_image_token
        return text.replace(img, block, 1), pos, len(block) - len(img)

    def build_inputs(self, processor, image, question, prefill):
        assert image is not None, "InternVL is a vision-language model -- image is required"
        text = self._render_template_text(processor, question, prefill)
        out = processor(text=[text], images=[image], return_tensors="pt", padding=False, crop_to_patches=False)
        assert out["pixel_values"].shape[0] == 1, (
            f"expected ONE 448px tile, got {out['pixel_values'].shape[0]} -- crop_to_patches=False was ignored, "
            f"and every token-count assumption here would be wrong")
        return {"input_ids": out["input_ids"][0], "extra": {"pixel_values": out["pixel_values"]}}

    def encode_image(self, processor, image):
        out = processor.image_processor(images=image, return_tensors="pt", crop_to_patches=False)
        return {"pixel_values": out["pixel_values"]}

    def tokenize_template(self, processor, question, prefill, n_image_tokens):
        text, _, _ = self._expand(processor, self._render_template_text(processor, question, prefill),
                                  n_image_tokens)
        return processor.tokenizer(text, return_tensors="pt")["input_ids"][0]

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
        offsets = processor.tokenizer(expanded, return_offsets_mapping=True)["offset_mapping"]
        token_idx = next((i for i, (s, e) in enumerate(offsets) if s < abs_char_end <= e), None)
        assert token_idx is not None, f"phrase {phrase!r} end did not land inside a token"
        return token_idx
