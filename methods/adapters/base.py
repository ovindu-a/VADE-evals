"""ModelAdapter: the ONLY interface through which methods/ code (common/
and das/) is allowed to know anything about a specific model family. A
method written against this interface runs unchanged on any model that
implements it -- adding a new model means writing one new adapter file,
never touching common/ or das/.
"""


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

    def unembed(self, model, hidden_states) -> "torch.Tensor":
        """Projects a residual-stream hidden_states tensor ([..., H]) into vocab-space logits
        ([..., V]) using THIS model's own final norm + output head -- the logit-lens operation.
        Must reproduce the model's real output logits when applied to the model's own final
        (post-final-norm) hidden state (see methods/logit_lens.py, which cross-checks this at the
        last layer against the forward pass's own out.logits rather than trusting that invariant
        blindly)."""
        raise NotImplementedError
