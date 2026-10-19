"""ModelAdapter for Idefics3 (`HuggingFaceM4/Idefics3-8B-Llama3`), transformers-native.

Why this model (PAPER_RUNBOOK.md X3): its language model is Llama-3.1-8B, so it is the first
model here whose text stack is NOT a Qwen2.5 (InternVL3-8B's is). Every assumption the head /
MLP / DLA code makes still holds, checked against transformers 5.16's modeling_llama.py and
modeling_idefics3.py:

  * text stack  model.model.text_model = LlamaModel: .layers[i].{self_attn,mlp}, .norm
                (LlamaRMSNorm, with .variance_epsilon); head model.lm_head, Linear(bias=False),
                NOT tied to the embeddings (config tie_word_embeddings=False);
  * 32 blocks x 32 query heads x head_dim 128 = hidden 4096 (8 KV heads, GQA), so o_proj's input
    is 32 contiguous 128-wide head slices; SwiGLU LlamaMLP, intermediate 14336, down_proj's input
    is the neuron vector; LlamaAttention returns (attn_output, attn_weights), a tuple like Qwen's;
  * 1-D RoPE (llama3 scaling) and plain causal attention: no M-RoPE, no image_grid_thw, no
    token_type_ids, nothing position-related in `extra`.

What differs from Qwen2.5-VL, and is handled here:

  * Image tokens. With do_image_splitting=False the processor squares the image to one
    364x364 tile (VADE canvases are square, so nothing is distorted); SigLIP 14px patches give
    26x26, the connector's pixel shuffle (scale_factor 2) merges 2x2 -> 13x13 = 169 tokens, in
    row-major order (Idefics3Connector.pixel_shuffle's two transposes cancel). Each token covers
    canvas/13 px, so object_location.json's 28px Qwen grid does not apply:
    `shares_object_grid = False`, and common/entities.py accepts only `full_image` and
    `last_token` for this adapter unless VADE adds a per-model override.
  * The image block. processing_idefics3.replace_image_token (rows = cols = 0) expands the one
    `<image>` placeholder to <fake_token_around_image><global-img><image>x169<fake_token_around_image>.
    `<global-img>` is NOT a vocabulary token in this checkpoint's tokenizer: it tokenizes as plain
    text (4 tokens). _expand calls the processor's own replace_image_token rather than retyping
    the string, so the two can never disagree.
  * BOS. The chat template itself emits <|begin_of_text|>, and the tokenizer ALSO prepends it
    when add_special_tokens=True (the processor's default) -- the plain processor(text=...) call
    therefore produces TWO BOS tokens (checked). apply_chat_template(tokenize=True) avoids that by
    tokenizing with add_special_tokens=False; every tokenizer call here does the same, and
    _check_bos asserts exactly one BOS, at column 0.
  * `extra` is pixel_values [1, 1, 3, 364, 364] (batch, images-per-prompt, ...) plus
    pixel_attention_mask [1, 1, 364, 364], stored as bool (the model's own default when it is
    absent is a bool all-ones mask), so extra_to_device moves it without casting it to bf16.

Prompt shape (rendered, then tokenized; 190 tokens for a short question):
  <|begin_of_text|>User:<fake><global-img><image>x169<fake>{question}<end_of_utterance>\\n
  Assistant: {prefill}
"""
import torch

from .base import ModelAdapter

N_IMAGES_PER_PROMPT = 1


class Idefics3Adapter(ModelAdapter):
    shares_object_grid = False
    # Passed on every processor call AND set on the image processor at load time: the
    # checkpoint's preprocessor_config.json has do_image_splitting=True, which tiles a 336px
    # canvas into 4x4 crops of an upscaled 1456px image plus a global view (~2,900 tokens).
    PROCESSOR_KWARGS = {"do_image_splitting": False}

    def __init__(self, model_id="HuggingFaceM4/Idefics3-8B-Llama3"):
        self.model_id = model_id
        self._template_cache = {}

    def load(self, device="cuda:0", dtype=torch.bfloat16, attn_implementation=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(self.model_id)
        processor.image_processor.do_image_splitting = False
        extra = {"attn_implementation": attn_implementation} if attn_implementation is not None else {}
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, device_map=device, dtype=dtype, **extra)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        assert self.image_token_id(model, processor) == processor.image_token_id, (
            f"config image_token_id {model.config.image_token_id} != processor's {processor.image_token_id}")
        grid = self.vision_patch_grid(model, 1, 1)
        assert grid["n_tokens"] == processor.image_seq_len, (
            f"vision config gives {grid['n_tokens']} image tokens, processor inserts {processor.image_seq_len}")
        return model, processor

    # ---- text backbone -------------------------------------------------------

    def _text_config(self, model):
        return model.config.text_config

    def hidden_size(self, model):
        return self._text_config(model).hidden_size

    def intermediate_size(self, model):
        return self._text_config(model).intermediate_size

    def n_attention_heads(self, model):
        # QUERY heads (32 on Idefics3-8B); num_key_value_heads (8) is the GQA count. Llama's
        # head_dim is hidden/heads here, which every head-slicing caller assumes -- check it.
        cfg = self._text_config(model)
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        assert head_dim * cfg.num_attention_heads == cfg.hidden_size, (
            f"head_dim {head_dim} x {cfg.num_attention_heads} heads != hidden {cfg.hidden_size}: "
            f"o_proj's input is not hidden-wide, and every head slice here would be wrong")
        return cfg.num_attention_heads

    def get_decoder_layers(self, model):
        return model.model.text_model.layers

    def get_mlp_block(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].mlp

    def get_mlp_hidden_module(self, model, block_idx):
        # LlamaMLP: down_proj(act_fn(gate_proj(x)) * up_proj(x)) -- down_proj's input is the neuron vector.
        return self.get_mlp_block(model, block_idx).down_proj

    def get_attn_block(self, model, block_idx):
        return self.get_decoder_layers(model)[block_idx].self_attn

    def get_attn_head_output_module(self, model, block_idx):
        return self.get_attn_block(model, block_idx).o_proj

    def _final_norm(self, model):
        return model.model.text_model.norm

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
        return processor.image_token_id

    def vision_patch_grid(self, model, canvas_width_px, canvas_height_px):
        vc = model.config.vision_config
        size = vc.image_size if isinstance(vc.image_size, int) else vc.image_size[0]
        patch = vc.patch_size if isinstance(vc.patch_size, int) else vc.patch_size[0]
        merge = int(model.config.scale_factor)
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

    def _check_bos(self, processor, ids):
        """ids: ONE prompt, 1-D (a [1, T] tensor from return_tensors="pt" is accepted)."""
        if torch.is_tensor(ids):
            assert ids.dim() == 1 or (ids.dim() == 2 and ids.shape[0] == 1), f"one prompt expected, got {tuple(ids.shape)}"
            ids = ids.reshape(-1).tolist()
        else:
            ids = list(ids)
        bos = processor.tokenizer.bos_token_id
        assert ids and ids[0] == bos and ids.count(bos) == 1, (
            f"expected exactly one BOS ({bos}) at column 0, got {ids.count(bos)} (first ids {ids[:4]}) -- "
            f"the chat template already emits <|begin_of_text|>, so a tokenizer call with "
            f"add_special_tokens=True doubles it")

    def _expand(self, processor, text, n_image_tokens):
        """The processor's own placeholder expansion for one unsplit image (rows = cols = 0).
        -> (expanded text, char offset of the image token in `text`, extra chars inserted)."""
        img = processor.image_token
        pos = text.find(img)
        assert pos != -1 and text.count(img) == 1, (
            f"expected exactly one {img!r} in the rendered chat template, found {text.count(img)}")
        assert n_image_tokens == processor.image_seq_len, (
            f"n_image_tokens {n_image_tokens} != the processor's image_seq_len {processor.image_seq_len}")
        block = processor.replace_image_token({"rows": [[0]], "cols": [[0]]}, 0)
        assert block.count(img) == n_image_tokens, "replace_image_token did not insert image_seq_len image tokens"
        return text.replace(img, block, 1), pos, len(block) - len(img)

    def _tokenize(self, processor, text, **kw):
        enc = processor.tokenizer(text, add_special_tokens=False, **kw)
        self._check_bos(processor, enc["input_ids"])
        return enc

    def build_inputs(self, processor, image, question, prefill):
        assert image is not None, "Idefics3 is a vision-language model -- image is required"
        text = self._render_template_text(processor, question, prefill)
        out = processor(text=[text], images=[image], return_tensors="pt", padding=False,
                        add_special_tokens=False, **self.PROCESSOR_KWARGS)
        assert tuple(out["pixel_values"].shape[:2]) == (1, N_IMAGES_PER_PROMPT), (
            f"expected ONE 364px tile, got pixel_values {tuple(out['pixel_values'].shape)} -- "
            f"do_image_splitting=False was ignored, and every token-count assumption here would be wrong")
        ids = out["input_ids"][0]
        self._check_bos(processor, ids)
        return {"input_ids": ids, "extra": {"pixel_values": out["pixel_values"],
                                            "pixel_attention_mask": out["pixel_attention_mask"].bool()}}

    def encode_image(self, processor, image):
        out = processor.image_processor(images=[image], return_tensors="pt", **self.PROCESSOR_KWARGS)
        assert tuple(out["pixel_values"].shape[:2]) == (1, N_IMAGES_PER_PROMPT), (
            f"expected ONE 364px tile, got pixel_values {tuple(out['pixel_values'].shape)}")
        return {"pixel_values": out["pixel_values"], "pixel_attention_mask": out["pixel_attention_mask"].bool()}

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
        """Column of the assistant turn's 'Assistant' header token: the first token after the
        LAST <end_of_utterance> (which closes the user turn; the template puts a newline between)."""
        ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
        eou = processor.tokenizer.convert_tokens_to_ids(processor.end_of_utterance_token)
        ends = [i for i, t in enumerate(ids) if t == eou]
        assert ends, f"no {processor.end_of_utterance_token} in the prompt -- not this project's chat shape"
        nl = processor.tokenizer("\n", add_special_tokens=False)["input_ids"]
        col = ends[-1] + 1 + (len(nl) if ids[ends[-1] + 1: ends[-1] + 1 + len(nl)] == nl else 0)
        assert col < len(ids), "the prompt ends at the end of the user turn -- no assistant header"
        return col
