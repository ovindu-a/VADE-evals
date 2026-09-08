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

    def load(self, device="cuda:0", dtype=torch.bfloat16):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, device_map=device, dtype=dtype)
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

    def tokenize_template(self, processor, question, prefill, n_image_tokens):
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                              continue_final_message=True)
        image_token = getattr(processor, "image_token", IMAGE_TOKEN)
        text = text.replace(image_token, image_token * n_image_tokens, 1)
        return processor.tokenizer(text, return_tensors="pt")["input_ids"][0]
