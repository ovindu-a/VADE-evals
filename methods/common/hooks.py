"""Model-agnostic activation-patching mechanism, shared by every causal
method in methods/ (currently DAS; PCA/SAE's methods/intervene.py duplicates
this by hand and is a candidate to migrate onto this module later).

Nothing here knows about pixel_values, image_grid_thw, or any other
model-specific forward-pass kwarg -- those travel opaquely through an
`extra` dict (see adapters/base.py's ModelAdapter.build_inputs), moved to
the right device/dtype by `extra_to_device` without ever naming a key. The
only real model-specific assumption is `layers`: an indexable stack of
decoder blocks (ModuleList or similar) supporting register_forward_hook /
register_forward_pre_hook with the standard PyTorch hook signatures --
true of every HF decoder-only transformer this project has used so far
(get it from ModelAdapter.get_decoder_layers).
"""
import torch


def extra_to_device(extra, device, dtype):
    """Moves every tensor in `extra` to `device`, casting floating-point
    tensors (e.g. pixel_values) to `dtype` -- integer tensors (e.g.
    image_grid_thw) are moved but never dtype-cast. Non-tensor values pass
    through unchanged. Generic over whatever keys the adapter put in
    `extra`; never inspects key names."""
    out = {}
    for k, v in extra.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out


def make_cache_aware_patch_hook(positions, compute_replacement):
    """positions: [B, n_pos] absolute column indices (only valid against a
    full, un-cached forward pass). compute_replacement(base_vals) -> new_vals,
    both [B, n_pos, H].

    A no-op once hidden_states has collapsed to a single column (generate()'s
    later incremental decode steps, reusing the KV cache) -- the patch's
    effect from the initial multi-token prefill is already baked into the
    KV cache by then, and indexing `positions` into a 1-column tensor would
    be wrong (out of bounds) rather than merely redundant."""
    def _patch(hidden_states):
        if hidden_states.shape[1] == 1:
            return hidden_states
        B = positions.shape[0]
        base_vals = torch.stack([hidden_states[i, positions[i]] for i in range(B)])
        new_vals = compute_replacement(base_vals)
        patched = hidden_states.clone()
        for i in range(B):
            patched[i, positions[i]] = new_vals[i].to(patched.dtype)
        return patched
    return _patch


def register_patch_hook(layers, layer_idx, patch_fn):
    """layer_idx=0 patches the embedding output (pre-hook on layers[0]);
    layer_idx=1..N patches decoder layer layer_idx's output (post-hook on
    layers[layer_idx-1]). Convention: index 0 = embedding output, index i =
    residual stream after decoder layer i -- matches hidden_states'
    convention on every HF causal LM. Returns hook handles -- caller must
    .remove() them (see forward_patched/generate_patched below, which do)."""
    handles = []
    if layer_idx == 0:
        def pre_hook(module, args, kwargs):
            return (patch_fn(args[0]),) + args[1:], kwargs
        handles.append(layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
    else:
        def post_hook(module, inputs, output):
            return patch_fn(output)
        handles.append(layers[layer_idx - 1].register_forward_hook(post_hook))
    return handles


def forward_with_hidden_states(model, input_ids, attention_mask, extra, **fwd_kwargs):
    """No-hook forward pass, hidden_states always requested. Used to cache
    an unpatched (e.g. source-side) activation at some layer/position."""
    extra_dev = extra_to_device(extra, model.device, model.dtype)
    return model(
        input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
        **extra_dev, output_hidden_states=True, **fwd_kwargs,
    )


def cache_layer_hidden(model, input_ids, attention_mask, extra, positions, layer_idx):
    """One no_grad forward pass -> hidden_states[layer_idx] at `positions`,
    per row. positions: [B, n_pos]. Returns [B, n_pos, H], detached."""
    with torch.no_grad():
        out = forward_with_hidden_states(model, input_ids, attention_mask, extra, logits_to_keep=1)
    layer_hidden = out.hidden_states[layer_idx]
    B = positions.shape[0]
    return torch.stack([layer_hidden[i, positions[i]] for i in range(B)]).detach()


def forward_patched(model, layers, layer_idx, patch_fn, input_ids, attention_mask, extra, **fwd_kwargs):
    """Gradient-enabled forward pass with the patch hook active -- for
    training. Caller reads whatever it needs off the returned output
    (typically .logits)."""
    handles = register_patch_hook(layers, layer_idx, patch_fn)
    try:
        extra_dev = extra_to_device(extra, model.device, model.dtype)
        return model(
            input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
            **extra_dev, **fwd_kwargs,
        )
    finally:
        for h in handles:
            h.remove()


def generate_patched(model, layers, layer_idx, patch_fn, input_ids, attention_mask, extra, pad_token_id, max_new_tokens):
    """Free-running greedy generate() with the patch hook active. Returns
    [B, max_new_tokens] generated token ids (prompt stripped), on CPU."""
    handles = register_patch_hook(layers, layer_idx, patch_fn)
    try:
        extra_dev = extra_to_device(extra, model.device, model.dtype)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                **extra_dev, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_token_id,
            )
    finally:
        for h in handles:
            h.remove()
    return gen[:, input_ids.shape[1]:].cpu()
