"""ModelAdapter: the ONLY interface through which methods/ code (common/
and das/) is allowed to know anything about a specific model family. A
method written against this interface runs unchanged on any model that
implements it -- adding a new model means writing one new adapter file,
never touching common/ or das/.
"""


class ModelAdapter:
    model_id: str

    def load(self, device, dtype):
        """-> (model, processor), both eval()-mode / ready to use."""
        raise NotImplementedError

    def hidden_size(self, model) -> int:
        raise NotImplementedError

    def get_decoder_layers(self, model):
        """-> an indexable stack of decoder blocks (e.g. nn.ModuleList)
        supporting register_forward_hook / register_forward_pre_hook --
        see common/hooks.py's register_patch_hook for the exact contract."""
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
