"""ModelAdapter for the Qwen2.5-VL family. Ported from flag-benchmark's
experiments/image-to-image/scripts/core/collate.py and VADE-evals'
methods/{sae,intervene}.py -- this is the one place that logic now lives,
instead of being copy-pasted a third time.
"""
import torch

from .base import ModelAdapter

IMAGE_TOKEN = "<|image_pad|>"


class Qwen25VLAdapter(ModelAdapter):
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_id = model_id
        self._template_cache = {}  # (question, prefill) -> rendered chat-template text, see build_inputs

    def load(self, device="cuda:0", dtype=torch.bfloat16, attn_implementation=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_id)
        # attn_implementation left at HF's own default (sdpa/flash, whichever is available) unless a
        # caller explicitly asks otherwise -- e.g. methods/attention_maps.py passes "eager", the only
        # implementation whose Qwen2_5_VLAttention.forward actually returns attn_weights instead of None
        # for output_attentions=True (verified against this project's installed transformers version).
        extra = {"attn_implementation": attn_implementation} if attn_implementation is not None else {}
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, device_map=device, dtype=dtype, **extra)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()  # backbone is fully frozen -- no dropout needed, and eval() removes any train/eval divergence
        return model, processor

    def hidden_size(self, model) -> int:
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and getattr(text_config, "hidden_size", None):
            return text_config.hidden_size
        return model.config.hidden_size

    def get_decoder_layers(self, model):
        return model.model.language_model.layers

    def intermediate_size(self, model) -> int:
        # Same text_config-then-top-level lookup order as hidden_size above: a VLM's config nests the
        # language model's dims under text_config, but older/flattened configs put them at the top.
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and getattr(text_config, "intermediate_size", None):
            return text_config.intermediate_size
        return model.config.intermediate_size

    def get_mlp_block(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].mlp

    def get_attn_block(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].self_attn

    def get_attn_head_output_module(self, model, block_idx):
        # o_proj's INPUT is the concatenated per-head outputs: num_attention_heads (28) * head_dim
        # (128) = 3584 = hidden_size on Qwen2.5-VL-7B. Grouped-query attention reduces the number of
        # KEY/VALUE heads (num_key_value_heads=4), not query heads, so o_proj's input width is
        # unaffected by GQA and stays num_attention_heads * head_dim.
        return self.get_attn_block(model, block_idx).o_proj

    def get_mlp_hidden_module(self, model, block_idx):
        # Qwen2.5-VL's decoder MLP is the gated (SwiGLU) shape -- Qwen2MLP.forward is
        # down_proj(act_fn(gate_proj(x)) * up_proj(x)), so down_proj's INPUT is precisely the
        # post-nonlinearity neuron vector [B, T, intermediate_size] that common/sites.py's
        # `mlp_hidden` site reads and patches (as a forward PRE-hook on this module).
        return self.get_mlp_block(model, block_idx).down_proj

    def unembed(self, model, hidden_states):
        # model.model.language_model.norm is the SAME RMSNorm applied at the end of every real
        # forward pass, right before model.lm_head -- confirmed against this project's installed
        # transformers (Qwen2_5_VLTextModel.forward: "hidden_states = self.norm(hidden_states)"
        # immediately before returning last_hidden_state). Applying it again to hidden_states that
        # are ALREADY post-final-norm (as the last entry of a real forward pass's output.hidden_states
        # tuple is, via transformers' capture_outputs(tie_last_hidden_states=True) machinery) would
        # double-normalize and silently corrupt the logit-lens result -- methods/logit_lens.py never
        # calls this on that last entry; it uses the forward pass's own out.logits there instead
        # (see its module docstring for why that's the more robust choice regardless of this comment).
        return model.lm_head(model.model.language_model.norm(hidden_states))

    def image_token_id(self, model, processor):
        tok_id = getattr(model.config, "image_token_id", None)
        if tok_id is not None:
            return tok_id
        image_token = getattr(processor, "image_token", IMAGE_TOKEN)
        return processor.tokenizer.convert_tokens_to_ids(image_token)

    def vision_patch_grid(self, model, canvas_width_px, canvas_height_px) -> dict:
        vc = model.config.vision_config
        patch_size = vc.patch_size
        merge_size = vc.spatial_merge_size
        merged_patch_size = patch_size * merge_size
        assert canvas_width_px % merged_patch_size == 0 and canvas_height_px % merged_patch_size == 0, (
            f"canvas {canvas_width_px}x{canvas_height_px} doesn't divide evenly into "
            f"{merged_patch_size}px merged patches (patch_size={patch_size}, merge_size={merge_size})")
        grid_cols = canvas_width_px // merged_patch_size
        grid_rows = canvas_height_px // merged_patch_size
        return {
            "model": self.model_id, "patch_size_px": patch_size, "merge_size": merge_size,
            "merged_patch_size_px": merged_patch_size, "grid_rows": grid_rows, "grid_cols": grid_cols,
            "n_tokens": grid_rows * grid_cols, "flat_index_formula": "row * grid_cols + col",
        }

    def build_inputs(self, processor, image, question, prefill) -> dict:
        assert image is not None, "Qwen2.5-VL is a vision-language model -- image is required"
        # apply_chat_template(tokenize=True, ...) re-renders the Jinja chat template from scratch on every
        # call -- ~102ms of which only ~18ms is the actual image processing + tokenization (measured), the
        # rest is Jinja/message-parsing overhead. The rendered TEXT only depends on (question, prefill) --
        # the image is just a placeholder marker in the template, not consulted during text rendering -- so
        # for the fixed, small set of (question, prefill) pairs this project draws from, render once per
        # pair and reuse the cached text on every later row through the plain processor() call instead of
        # apply_chat_template. Byte-identical input_ids/pixel_values to the tokenize=True path (verified).
        cache_key = (question, prefill)
        text_template = self._template_cache.get(cache_key)
        if text_template is None:
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
            ]
            text_template = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, continue_final_message=True,
            )
            self._template_cache[cache_key] = text_template
        out = processor(text=[text_template], images=[image], return_tensors="pt", padding=False)
        return {
            "input_ids": out["input_ids"][0],
            "extra": {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]},
        }

    def encode_image(self, processor, image) -> dict:
        out = processor.image_processor(images=image, return_tensors="pt")
        return {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}

    def _render_template_text(self, processor, question, prefill):
        """The (question, prefill)-only half of build_inputs' chat-template rendering, factored out
        so tokenize_template and find_last_phrase_token_col don't each re-implement this project's
        fixed prompt shape (image placeholder + question in a user turn, prefill continuing the
        assistant turn)."""
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                              continue_final_message=True)

    def tokenize_template(self, processor, question, prefill, n_image_tokens):
        text = self._render_template_text(processor, question, prefill)
        image_token = getattr(processor, "image_token", IMAGE_TOKEN)
        text = text.replace(image_token, image_token * n_image_tokens, 1)
        return processor.tokenizer(text, return_tensors="pt")["input_ids"][0]

    def find_last_phrase_token_col(self, processor, question, prefill, n_image_tokens, phrases):
        """Absolute token-column index (within tokenize_template's own output for this exact
        (question, prefill, n_image_tokens)) of the LAST token of the LAST occurrence of whichever
        candidate in `phrases` (tried in order, case-insensitive substring match against `question`
        only -- never prefill) is found, or None if none match. Lets a probe (see
        methods/probe_common.py's attribute_mention_col) ask "which token mentions attribute X"
        without knowing anything about this model's chat-template rendering.

        Verified against a real Qwen2.5-VL-7B-Instruct processor for both a plain single-mention
        question and currency's few-shot-preamble question (where naive first-occurrence matching
        would land on the exemplar text's "currency code" instead of the real question's) --
        rfind against `question` alone, not the full rendered template, is what makes that safe:
        VADE's few-shot preamble text lives INSIDE `question` (see flags/prompt_templates.json's
        currency templates), always followed by the real question containing the same phrase again.
        """
        q_lower = question.lower()
        match = next(((p, q_lower.rfind(p.lower())) for p in phrases if p.lower() in q_lower), None)
        if match is None:
            return None
        phrase, char_start = match
        char_end = char_start + len(phrase)

        text = self._render_template_text(processor, question, prefill)
        q_pos = text.find(question)
        assert q_pos != -1, "question text not found verbatim in its own rendered chat template"
        abs_char_end = q_pos + char_end

        # The image placeholder is expanded from 1 occurrence to n_image_tokens BEFORE tokenizing
        # (see tokenize_template) -- it sits before the question in this project's fixed prompt shape,
        # so every char offset found above needs shifting by exactly that expansion's extra length.
        image_token = getattr(processor, "image_token", IMAGE_TOKEN)
        img_pos = text.find(image_token)
        shift = (n_image_tokens - 1) * len(image_token) if 0 <= img_pos < q_pos else 0
        abs_char_end += shift
        expanded_text = text.replace(image_token, image_token * n_image_tokens, 1)

        offsets = processor.tokenizer(expanded_text, return_offsets_mapping=True)["offset_mapping"]
        token_idx = next((i for i, (s, e) in enumerate(offsets) if s < abs_char_end <= e), None)
        assert token_idx is not None, (
            f"attribute-mention char span (end={abs_char_end}) didn't land inside any token's offset "
            f"range -- phrase={phrase!r} question={question!r}")
        return token_idx
