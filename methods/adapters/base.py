"""ModelAdapter: the ONLY interface through which methods/ code (common/
and das/) is allowed to know anything about a specific model family. A
method written against this interface runs unchanged on any model that
implements it -- adding a new model means writing one new adapter file,
never touching common/ or das/.
"""
import math


def containing_box_override(object_location, grid):
    """InternVL3's convention (VADE's object_location.json `models.InternVL3-8B-hf` entries), computed
    rather than stored: for a model whose token grid is NOT object_location.json's 28px Qwen grid, the
    one object position set is `entity_ring1` = the smallest whole-cell box CONTAINING object_bbox_px.
    It is a cover, not pixel-exact (the bbox need not fall on this grid's cell edges), and there is no
    separate entity_only -- the same name as Qwen's dilated set, so `--positions entity_ring1` makes the
    same experimental choice across models. grid: the adapter's own vision_patch_grid for the canvas,
    which must expose canvas_px_per_token (cells are uniform over the canvas)."""
    bb = object_location["object_bbox_px"]
    cw, ch = grid["canvas_px_per_token"]
    c0, c1 = int(math.floor(bb["x"] / cw)), int(math.ceil((bb["x"] + bb["width"]) / cw - 1e-9)) - 1
    r0, r1 = int(math.floor(bb["y"] / ch)), int(math.ceil((bb["y"] + bb["height"]) / ch - 1e-9)) - 1
    rows, cols = list(range(r0, r1 + 1)), list(range(c0, c1 + 1))
    assert rows and cols and r1 < grid["grid_rows"] and c1 < grid["grid_cols"], (rows, cols, grid)
    return {"vlm_token_grid": {k: grid[k] for k in ("model", "patch_size_px", "merge_size", "grid_rows",
                                                    "grid_cols", "n_tokens")},
            "object_token_indices": {"entity_ring1": {
                "flat": [r * grid["grid_cols"] + c for r in rows for c in cols], "rows": rows, "cols": cols,
                "description": f"COMPUTED (adapters/base.py containing_box_override): smallest whole-cell "
                               f"{grid['grid_rows']}x{grid['grid_cols']} box containing object_bbox_px "
                               f"({bb['x']},{bb['y']},{bb['width']},{bb['height']}); a cover, not pixel-exact."}}}


class ModelAdapter:
    model_id: str

    def load(self, device, dtype, attn_implementation=None):
        """-> (model, processor), both eval()-mode / ready to use.
        attn_implementation: None leaves HF's own default in place (sdpa/
        flash, whichever is available) -- every training/intervention
        script wants that for speed. Pass "eager" only when you actually
        need real attn_weights back from output_attentions=True (e.g.
        methods/attention_maps.py) -- sdpa/flash silently return None for
        that instead of raising, so getting this wrong fails quietly."""
        raise NotImplementedError

    def hidden_size(self, model) -> int:
        raise NotImplementedError

    def get_decoder_layers(self, model):
        """-> an indexable stack of decoder blocks (e.g. nn.ModuleList)
        supporting register_forward_hook / register_forward_pre_hook --
        see common/hooks.py's register_patch_hook for the exact contract."""
        raise NotImplementedError

    # ---- MLP-internal intervention sites -------------------------------------
    # The three methods below are ADDITIONS in this repo -- they are not part of
    # the sibling VADE repo's own copy of this file (which only ever needed the
    # residual stream, the site DAS/DBM intervene on). They exist so that
    # common/sites.py can address a decoder block's MLP internals while
    # remaining the ONLY module allowed to know a model family's attribute
    # names -- see this file's own header. A text-only or non-gated-MLP model
    # that never uses an MLP site simply doesn't implement them.

    def intermediate_size(self, model) -> int:
        """-> the width of the MLP/FFN's internal hidden state, i.e. the
        post-nonlinearity, pre-down-projection vector's dimension
        (18944 for Qwen2.5-VL-7B-Instruct, vs. hidden_size's 3584). This is
        the mask width for the `mlp_hidden` site."""
        raise NotImplementedError

    def get_mlp_block(self, model, block_idx):
        """-> decoder block block_idx's MLP submodule, whose forward()
        OUTPUT is that block's contribution to the residual stream
        ([B, T, hidden_size]). Hooked for the `mlp_output` site."""
        raise NotImplementedError

    def get_mlp_hidden_module(self, model, block_idx):
        """-> the submodule of decoder block block_idx's MLP whose forward()
        INPUT is the MLP hidden state ([B, T, intermediate_size]). For a
        gated (SwiGLU-style) MLP that is the down-projection, since its
        input is exactly act_fn(gate_proj(x)) * up_proj(x) -- hooking it
        gets the neuron vector without recomputing either branch by hand.
        Hooked (as a forward PRE-hook) for the `mlp_hidden` site."""
        raise NotImplementedError

    def get_attn_block(self, model, block_idx):
        """-> decoder block block_idx's self-attention submodule, whose
        forward() OUTPUT[0] is that sublayer's contribution to the residual
        stream ([B, T, hidden_size]). NOTE it returns a tuple, not a bare
        tensor, unlike the MLP -- common/sites.py's hooks handle both.
        Hooked for the `attn_output` site."""
        raise NotImplementedError

    def n_attention_heads(self, model) -> int:
        """Number of QUERY heads in a decoder block's self-attention. o_proj's input width is
        n_attention_heads * head_dim, so head h occupies the contiguous dim range
        [h*head_dim, (h+1)*head_dim) of the `attn_head_output` site. Grouped-query attention
        reduces KEY/VALUE heads only and must NOT be reported here."""
        raise NotImplementedError

    # ---- additions for models whose block is not Qwen/Llama-shaped (Gemma-3, Pixtral) --------
    # Every default below reproduces the old behaviour exactly, so no existing adapter changes.

    def head_dim(self, model) -> int:
        """Width of one head's slice of o_proj's input. The default, hidden_size // n_heads, is
        right for Qwen2.5 / Llama / Vicuna, where n_heads * head_dim == hidden_size. It is WRONG
        for Gemma-3 (12B: 16 x 256 = 4096 vs hidden 3840) and Mistral-Nemo / Pixtral (32 x 128 =
        4096 vs 5120), whose adapters override it from the config's own head_dim."""
        hidden, n = self.hidden_size(model), self.n_attention_heads(model)
        assert hidden % n == 0, f"hidden_size {hidden} is not divisible by {n} heads -- override head_dim()"
        return hidden // n

    def get_attn_output_module(self, model, block_idx):
        """-> the module whose forward() OUTPUT (element 0 if a tuple) is EXACTLY what the
        attention sublayer adds to the residual stream -- the `attn_output` site. The default is
        the attention module itself. Gemma-3 normalizes the attention output BEFORE the residual
        add (post_attention_layernorm), so its adapter returns that norm instead.
        get_attn_block stays the attention module (callers reach .v_proj / attention weights)."""
        return self.get_attn_block(model, block_idx)

    def get_mlp_output_module(self, model, block_idx):
        """-> the module whose forward() OUTPUT is exactly what the MLP sublayer adds to the
        residual stream -- the `mlp_output` site. Default: the MLP; Gemma-3: post_feedforward_layernorm."""
        return self.get_mlp_block(model, block_idx)

    def head_residual_writes(self, model, block_idx, z):
        """z: [..., n_heads * head_dim] (o_proj's input) -> [..., n_heads, hidden] float32, each
        head's write into the residual stream; summed over heads it is the `attn_output` site.
        Default: z_h @ W_O[:, h]^T (o_proj is bias-free). An adapter with a post-attention norm
        overrides it, freezing that norm's per-position scale at the value the FULL sum gets
        (the same trick as final_norm_scale), so the per-head split stays exact."""
        import torch
        w =self.get_attn_head_output_module(model, block_idx).weight.float()     # [hidden, n*d]
        n, d = self.n_attention_heads(model), self.head_dim(model)
        zh = z.float().reshape(*z.shape[:-1], n, d)
        return torch.einsum("...nd,hnd->...nh", zh, w.reshape(w.shape[0], n, d))

    def object_location_override(self, object_location, model) -> "dict | None":
        """A per-model {vlm_token_grid, object_token_indices} for an entity's object_location.json,
        used by common/entities.py when VADE's own file has no `models.<slug>` entry. Default None
        (only full_image / last_token for a model off the Qwen grid). See containing_box_override."""
        return None


    def get_attn_head_output_module(self, model, block_idx):
        """-> the submodule of decoder block block_idx's attention whose
        forward() INPUT is the concatenated per-head attention outputs
        ([B, T, num_attention_heads * head_dim]) -- i.e. the output
        projection, whose input is the per-head `z` before heads are mixed.
        Contiguous head_dim-sized blocks of that vector are individual
        heads, which is what makes head-granular masking possible.
        Hooked (as a forward PRE-hook) for the `attn_head_output` site."""
        raise NotImplementedError

    def image_token_id(self, model, processor):
        """-> the image-placeholder token id, or None for a text-only model."""
        raise NotImplementedError

    def vision_patch_grid(self, model, canvas_width_px, canvas_height_px) -> dict:
        """Computes the merged-token grid a canvas of this size renders to,
        FROM the model's own vision config -- this is the "how many patches
        does this model produce" mechanism every entity's object_location.
        json is cross-checked against (see common/entities.py's
        resolve_position_set). Returns a dict with at least: patch_size_px,
        merge_size, grid_rows, grid_cols, n_tokens. Text-only models should
        not implement this (never called for them)."""
        raise NotImplementedError

    def build_inputs(self, processor, image, question, prefill) -> dict:
        """Builds one example's model inputs from a (image, question,
        prefill) triple, matching this project's fixed prompt shape (a
        user turn with the image+question, an assistant turn holding
        `prefill` as an in-progress continuation -- continue_final_message
        semantics, not add_generation_prompt).

        Returns {"input_ids": 1D LongTensor, "extra": {...}} -- `extra`
        holds whatever OTHER per-example tensors this model's forward()/
        generate() need (pixel_values, image_grid_thw, ...), each with a
        leading batch dim of 1 so common/entities.py's concat_extra can
        torch.cat them across a batch. `extra` is opaque to every caller
        outside this adapter -- common/ code only ever spreads it via
        **extra into model calls (see common/hooks.py's extra_to_device).
        A text-only adapter returns extra={} and `image` is ignored/asserted
        None."""
        raise NotImplementedError

    def encode_image(self, processor, image) -> dict:
        """The image-only half of build_inputs -- pixel processing (resize/
        patchify), independent of any question/prefill text. Returns the
        same shape as build_inputs' `extra` (pixel_values/image_grid_thw,
        each with a leading batch dim of 1), verified bit-identical to what
        build_inputs produces for the same image (see common/entities.py's
        BuildBatchCache, which uses this to cache the vision side per
        unique image instead of redoing it for every tuple row that
        references it)."""
        raise NotImplementedError

    def tokenize_template(self, processor, question, prefill, n_image_tokens) -> "torch.Tensor":
        """The text-only half of build_inputs -- tokenizes (question,
        prefill) into this project's fixed prompt shape WITHOUT touching any
        actual image, given only the image's already-known token count
        (n_image_tokens -- from common/entities.py's resolve_position_set,
        itself cross-checked against the model's own vision config). Valid
        because every VADE entity's images share one fixed canvas size (see
        each entity's object_location.json), so the image-placeholder token
        count is a property of the ENTITY, not any specific image -- verified
        bit-identical to build_inputs' input_ids for the same image/question/
        prefill. Returns a 1D LongTensor (no leading batch dim, matching
        build_inputs' "input_ids")."""
        raise NotImplementedError

    def find_last_phrase_token_col(self, processor, question, prefill, n_image_tokens, phrases) -> "int | None":
        """-> the absolute token-column index (within tokenize_template's own output for this exact
        (question, prefill, n_image_tokens)) of the LAST token of the LAST occurrence of whichever
        candidate in `phrases` is found in `question` (case-insensitive substring match, tried in the
        given order), or None if none match. Lets a probe locate "the token that names attribute X in
        this question" (see methods/probe_common.py's attribute_mention_col) without knowing this
        model's chat-template format. A model with no chat template at all may just return None
        always."""
        raise NotImplementedError

    def assistant_start_col(self, processor, input_ids) -> int:
        """-> the column where the assistant turn's header begins in one tokenized prompt
        (input_ids: 1-D), i.e. where the "assistant_prefix" token group of
        methods/head_vocab_projection.py starts. This default is the ChatML convention (Qwen2.5-VL,
        InternVL3): the LAST <|im_start|>, which opens the assistant turn. An adapter whose chat
        template is not ChatML must override it (see idefics3.py)."""
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        starts = [i for i, t in enumerate(ids) if t == im_start_id]
        assert starts, "no <|im_start|> in the prompt -- not a ChatML template; override assistant_start_col"
        return starts[-1]

    def unembed(self, model, hidden_states) -> "torch.Tensor":
        """Projects a residual-stream hidden_states tensor ([..., H]) into vocab-space logits
        ([..., V]) using THIS model's own final norm + output head -- the logit-lens operation.
        Must reproduce the model's real output logits when applied to the model's own final
        (post-final-norm) hidden state (see methods/logit_lens.py, which cross-checks this at the
        last layer against the forward pass's own out.logits rather than trusting that invariant
        blindly)."""
        raise NotImplementedError

    def final_norm_scale(self, model, hidden_states) -> "torch.Tensor":
        """-> [..., 1] float32: the per-position SCALAR the model's final norm multiplies by at
        these hidden states. For an RMSNorm that is rsqrt(mean(x^2) + eps); the learned gain is
        NOT included (it is elementwise, not a scalar -- see logit_direction, which folds it in).

        Exists for direct logit attribution (methods/dla.py). A norm is nonlinear, so
        unembed(a + b) != unembed(a) + unembed(b) and the residual stream's additive structure
        does not survive it. FREEZING the scale at the value the real forward pass computed --
        from the FULL final residual, not from the component -- makes the remaining map linear,
        which is what turns "logit of the answer" into an exact sum of per-component terms.
        Pass the model's own final pre-norm residual stream here, nothing else."""
        raise NotImplementedError

    def logit_direction(self, model, token_ids) -> "torch.Tensor":
        """-> [..., H] float32: the residual-stream direction whose dot product with a component,
        times final_norm_scale, IS that component's contribution to token_ids' logit. For an
        RMSNorm + linear head with no bias that is output_head_weight[token] * final_norm.weight
        (the elementwise learned gain folded into the unembedding row).

        Together: contribution = final_norm_scale(resid_final) * (component . logit_direction(t)),
        and summing that over embed + every attn/mlp sublayer output reproduces the model's own
        logit for t exactly -- which methods/dla.py asserts per row rather than assuming."""
        raise NotImplementedError
