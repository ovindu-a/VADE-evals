"""Intervention SITES: WHICH tensor inside a decoder block a causal
intervention reads its source values from and patches its blended values
into.

NOT copied from the sibling VADE repo -- new in this repo. common/hooks.py
(which IS a verbatim copy, see this package's docstring in the README) only
ever knows one site: the residual stream at a decoder-block boundary. This
module generalizes that to MLP-internal sites WITHOUT editing hooks.py, so
that file stays a byte-for-byte copy of VADE's own and keeps receiving
upstream fixes cleanly. The residual path here is a straight delegation to
hooks.py's own functions, not a reimplementation -- so `residual` is
guaranteed byte-identical to what DAS/DBM did before this module existed.

The five SINGLE sites, all at the SAME decoder block, differing in what part
of its computation gets swapped (plus one JOINT site, described after them):

    residual     block L-1's OUTPUT              width = hidden_size (3584)
                 -- everything accumulated through the block: embeddings +
                 every prior block's attention and MLP contribution. This
                 is what RAVEL's DBM and VADE's DAS intervene on. No
                 privileged basis: every read/write of the residual stream
                 goes through a matrix, so an arbitrary rotation of it (with
                 the surrounding weights fixed up) leaves the model's
                 function unchanged -- individual coordinates are an
                 artifact of the checkpoint, not a fact about the model.

    mlp_output   block L-1's mlp OUTPUT           width = hidden_size (3584)
                 -- ONLY this block's MLP contribution to the residual
                 stream, before it's added in. Same width as `residual`, so
                 comparing the two isolates LOCALITY (one MLP's contribution
                 vs. the whole accumulated stream) at matched dimensionality.
                 Also has no privileged basis: it's a linear image
                 (down_proj) of a space that does, and privilege does not
                 survive a linear map.

    attn_output  block L-1's self_attn OUTPUT     width = hidden_size (3584)
                 -- the attention sublayer's contribution to the residual
                 stream, before it's added in. The exact analogue of
                 mlp_output for the OTHER sublayer, so attn_output vs
                 mlp_output asks "is a dead local site an MLP-specific fact,
                 or is any single sublayer's additive contribution too small
                 to carry a whole-entity attribute?" No privileged basis
                 (o_proj is a linear map, and privilege does not survive one).

    attn_head_output  block L-1's per-head z       width = hidden_size (3584)
                 -- o_proj's INPUT, i.e. the concatenated per-head attention
                 outputs (28 heads x 128 dims on Qwen2.5-VL-7B). Same width
                 as attn_output but structured: contiguous 128-dim blocks are
                 individual heads, so a mask here can select whole HEADS, and
                 heads are a privileged unit in Elhage et al.'s sense (each
                 has its own softmax and its own OV circuit). This is the one
                 site in this file whose units are both privileged AND known
                 to be functionally specialized in VLMs -- and it is where
                 this repo's own methods/attention_maps.py already ranks
                 image->text conduits correlationally, so a ceiling probe here
                 causally tests heads that probe has already flagged.

    mlp_hidden   block L-1's post-SwiGLU neurons  width = intermediate_size (18944)
                 -- act_fn(gate_proj(x)) * up_proj(x), i.e. the vector
                 down_proj consumes. The elementwise nonlinearity and
                 elementwise gating product mean the only function-preserving
                 transformations of this space are PERMUTATIONS, so its
                 coordinates ("neurons") are real objects rather than a
                 choice of axes -- a privileged basis (Elhage et al., Toy
                 Models of Superposition; the argument Gurnee et al.'s
                 sparse-probing work gives for probing MLP activations
                 specifically). See methods/ndm/ for the method built on it.

JOINT SITES patch several of the above IN THE SAME forward pass, at the same
block. The one that exists, `attn_output+mlp_output`, patches BOTH of block
L-1's sublayer contributions, which yields

    resid_base@L-1 + attn_src@L + mlp_src@L

i.e. exactly what a `residual`@L swap yields except that the ACCUMULATED
prefix stays the base's (`resid_base@L-1` rather than `resid_src@L-1`). So
`residual` minus `attn_output+mlp_output` isolates the prefix, and that
difference is what tells apart "this block computes the answer" from "the
answer was already accumulated and this block only refines it". Needed
because the two single sublayer sites are NOT additive: a full swap of one
sublayer INSERTS source evidence while leaving every bit of base evidence
upstream intact, whereas a residual swap DELETES the base prefix as well --
so `residual` can read 68.8% at a layer where both of its sublayers read
0.0% each, with nothing wrong anywhere. Joint sites are DIAGNOSTIC ONLY
(ceiling_sweep): a trained mask here would need one mask per part with its
own width and its own L1 term, which is a different method, so train.py /
eval.py reject them via ndm/config.py's NDM_SITES whitelist.

LAYER INDEXING follows hooks.py's existing convention exactly: layer_idx=0
is the embedding output and layer_idx=i is the residual stream AFTER decoder
block i-1. An MLP site at layer_idx=i therefore means block i-1's MLP -- the
very MLP whose output lands in residual layer i. That keeps all three sites
describing the same block at the same --layer, which is the whole point of
being able to compare them; getting this off by one would silently compare
different blocks.
"""
import torch

from .hooks import cache_layer_hidden, extra_to_device, forward_patched, generate_patched, register_patch_hook

SITES = ("residual", "attn_output", "attn_head_output", "mlp_output", "mlp_hidden")

# name -> the SINGLE sites it patches simultaneously, at the same block. Kept
# OUT of SITES on purpose: SITES is the set a mask can be TRAINED on, and
# verify_sites.py / ndm/config.py both iterate it.
JOINT_SITES = {
    "attn_output+mlp_output": ("attn_output", "mlp_output"),
}

# Everything ceiling_sweep.py (the diagnostic) accepts.
ALL_SITES = SITES + tuple(JOINT_SITES)

# pre-projection site -> the post-projection site it is INDISTINGUISHABLE from
# under a FULL swap. down_proj(h_src) IS mlp_out_src and o_proj(z_src) IS
# attn_out_src, so replacing the whole pre-projection vector produces the
# identical residual-stream update as replacing the whole post-projection one.
# Measured: identical to the row in every cell of four full sweeps. Hence
# ceiling_sweep.py's default OMITS the keys -- their numbers are exactly their
# values' numbers, so probing both spends a third of the sweep re-deriving a
# column you can copy. The distinction is real only for a SPARSE mask (a
# subset of neurons/heads decodes to a residual update no axis-aligned
# residual mask can express), which is a trained-mask question, not a
# full-swap one.
FULL_SWAP_EQUIVALENT = {
    "attn_head_output": "attn_output",
    "mlp_hidden": "mlp_output",
}

DEFAULT_SITE = "residual"


def _source_forward(model, input_ids, attention_mask, extra):
    """The no-grad forward pass every capture hook rides on. logits_to_keep=1
    because a capture only ever wants activations, never the full [B, T, V]
    logit tensor."""
    extra_dev = extra_to_device(extra, model.device, model.dtype)
    with torch.no_grad():
        model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
              **extra_dev, logits_to_keep=1)


def _assert_fired_once(name, captured):
    assert len(captured) == 1, (
        f"expected the {name!r} capture hook to fire exactly once per forward pass, got "
        f"{len(captured)} -- the hooked module was called more than once (gradient checkpointing? "
        f"a model that reuses the same MLP module across blocks?), which makes 'the' source "
        f"activation ambiguous")


class InterventionSite:
    """Where an intervention attaches. Construct one per run and pass it
    down; every site-specific decision (mask width, which module to hook,
    how to capture the source-side value, whether the shared source cache
    applies) is answered here rather than at each call site."""

    def __init__(self, name=DEFAULT_SITE):
        assert name in SITES, f"unknown site {name!r} -- expected one of {SITES}"
        self.name = name

    def __repr__(self):
        return f"InterventionSite({self.name!r})"

    # Not a property, so `getattr(site, "is_joint", False)` and `site.is_joint`
    # read the same on both classes -- callers branch on this rather than on
    # isinstance, and JointSite deliberately duck-types InterventionSite.
    is_joint = False

    @property
    def is_residual(self):
        return self.name == "residual"

    def width(self, adapter, model) -> int:
        """The mask's embed_dim for this site."""
        if self.name == "mlp_hidden":
            return adapter.intermediate_size(model)
        return adapter.hidden_size(model)

    def _module_and_hook_kind(self, adapter, model, layer_idx):
        """-> (module_to_hook, "pre"|"post") for every non-residual site.
        "pre" means the tensor we want is that module's INPUT: down_proj's
        input IS the post-SwiGLU neuron vector, and o_proj's input IS the
        concatenated per-head attention outputs -- so hooking a projection's
        input is how you read/write the wide pre-projection space without
        recomputing anything by hand."""
        assert layer_idx >= 1, (
            f"site {self.name!r} needs layer_idx>=1 (it addresses a sublayer of decoder block "
            f"layer_idx-1); layer_idx=0 is the embedding output, which has no attention or MLP sublayer")
        block = layer_idx - 1
        if self.name == "mlp_output":
            return adapter.get_mlp_block(model, block), "post"
        if self.name == "mlp_hidden":
            return adapter.get_mlp_hidden_module(model, block), "pre"
        if self.name == "attn_output":
            return adapter.get_attn_block(model, block), "post"
        if self.name == "attn_head_output":
            return adapter.get_attn_head_output_module(model, block), "pre"
        raise AssertionError(f"no module mapping for site {self.name!r}")

    # ---- patching (base side) -------------------------------------------------

    def register(self, adapter, model, layers, layer_idx, patch_fn):
        """Registers the patch hook. Returns handles -- caller must .remove()
        them (forward_patched/generate_patched below do)."""
        if self.is_residual:
            return register_patch_hook(layers, layer_idx, patch_fn)

        module, kind = self._module_and_hook_kind(adapter, model, layer_idx)
        if kind == "post":
            def post_hook(mod, inputs, output):
                # A self_attn module returns a TUPLE ((attn_output, attn_weights), and historically a
                # past_key_value too), whereas an mlp module returns a bare tensor. Patch element 0 and
                # hand the rest back untouched, so the site works for both without the caller caring.
                if isinstance(output, tuple):
                    return (patch_fn(output[0]),) + output[1:]
                return patch_fn(output)
            return [module.register_forward_hook(post_hook)]

        def pre_hook(mod, args, kwargs):
            return (patch_fn(args[0]),) + args[1:], kwargs
        return [module.register_forward_pre_hook(pre_hook, with_kwargs=True)]

    def forward_patched(self, adapter, model, layers, layer_idx, patch_fn, input_ids, attention_mask, extra,
                        **fwd_kwargs):
        """Gradient-enabled patched forward pass (training)."""
        if self.is_residual:
            return forward_patched(model, layers, layer_idx, patch_fn, input_ids, attention_mask, extra, **fwd_kwargs)
        handles = self.register(adapter, model, layers, layer_idx, patch_fn)
        try:
            extra_dev = extra_to_device(extra, model.device, model.dtype)
            return model(
                input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                **extra_dev, **fwd_kwargs,
            )
        finally:
            for h in handles:
                h.remove()

    def generate_patched(self, adapter, model, layers, layer_idx, patch_fn, input_ids, attention_mask, extra,
                         pad_token_id, max_new_tokens):
        """Free-running greedy generation with the patch active (eval).
        Returns [B, max_new_tokens] generated ids (prompt stripped), on CPU.

        The patch is a no-op after the initial multi-token prefill for MLP
        sites for exactly the same reason as for the residual stream: the
        patch changes what block L-1 writes into the residual stream during
        prefill, which is what every later block's K/V is computed from and
        cached -- so by the time generate() is doing single-token decode
        steps, the effect is already baked in. make_cache_aware_patch_hook's
        `shape[1] == 1` guard handles that identically here, since the
        down_proj input is [B, 1, intermediate_size] on a decode step."""
        if self.is_residual:
            return generate_patched(model, layers, layer_idx, patch_fn, input_ids, attention_mask, extra,
                                     pad_token_id, max_new_tokens)
        handles = self.register(adapter, model, layers, layer_idx, patch_fn)
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

    # ---- capturing (source side) ---------------------------------------------

    def capture(self, adapter, model, layer_idx, input_ids, attention_mask, extra, positions):
        """One no-grad forward pass -> this site's activation at `positions`,
        per row. positions: [B, n_pos]. Returns [B, n_pos, width], detached.

        For `residual` this delegates to hooks.py's cache_layer_hidden,
        which gets it free from output_hidden_states. MLP internals are NOT
        in output_hidden_states, so the MLP sites need a capture hook -- it
        indexes `positions` INSIDE the hook rather than stashing the whole
        [B, T, 18944] tensor (which for B=4/T~1500 would be ~227MB held per
        call, vs ~3.6MB for the [B, n_pos, 18944] slice we actually want)."""
        if self.is_residual:
            return cache_layer_hidden(model, input_ids, attention_mask, extra, positions, layer_idx)

        captured = []
        handle = self._register_capture(adapter, model, layer_idx, positions, captured)
        try:
            _source_forward(model, input_ids, attention_mask, extra)
        finally:
            handle.remove()
        _assert_fired_once(self.name, captured)
        return captured[0]

    def _register_capture(self, adapter, model, layer_idx, positions, captured):
        """Registers a hook that appends this site's [B, n_pos, width] slice
        to `captured`, and returns the handle (caller must .remove() it).
        Split out of capture() so JointSite can register EVERY part's capture
        hook before a SINGLE source forward pass instead of paying one
        forward per part. Non-residual sites only -- `residual` reads
        output_hidden_states and needs no hook at all."""
        assert not self.is_residual, "residual capture goes through cache_layer_hidden, not a hook"
        module, kind = self._module_and_hook_kind(adapter, model, layer_idx)
        B = positions.shape[0]

        def grab(t):
            captured.append(torch.stack([t[i, positions[i]] for i in range(B)]).detach())

        if kind == "post":
            # Tuple-aware for the same reason as register()'s post_hook: self_attn returns a tuple.
            return module.register_forward_hook(
                lambda mod, inputs, output: grab(output[0] if isinstance(output, tuple) else output))
        return module.register_forward_pre_hook(lambda mod, args: grab(args[0]))

    def lookup_source(self, cache, batch, layer_idx, positions_name, device, dtype):
        """The cached counterpart of capture(): reads this site's source
        activation out of a prebuilt cache instead of running a forward pass.
        Each site has its OWN cache format and file, so this dispatches:

          residual  -> common/source_cache.py (one file per entity, every
                       layer at once, straight from output_hidden_states)
          mlp_*     -> common/site_source_cache.py (one file per
                       (site, positions, layer), capture-hooked)

        Both caches are keyed by entity+model only, so both live under
        --vade_root/results/ per that repo's own convention.

        `positions_name` is only meaningful for the MLP sites (whose cache is
        keyed by it); the residual cache stores every image token and indexes
        the subset out of it, so it ignores this argument. Passing a cache
        built for a different site/layer/positions raises rather than
        mis-patching -- see lookup_site_source's own note on why shape
        agreement proves nothing here (an mlp_output cache and a residual
        cache have identical shapes)."""
        if self.is_residual:
            from .source_cache import lookup_source_hidden
            return lookup_source_hidden(cache, batch, layer_idx, device, dtype)
        from .site_source_cache import lookup_site_source
        return lookup_site_source(cache, batch, self, layer_idx, positions_name, device, dtype)


class JointSite:
    """Several InterventionSites of the SAME block, patched in ONE forward
    pass. Duck-types InterventionSite for the read-only/diagnostic surface
    (`name`, `is_residual`, `is_joint`, `width`, `capture`, `register`,
    `generate_patched`) and deliberately does NOT implement the training
    surface -- see this module's docstring on why a joint site is not a
    trainable site.

    Where the signatures differ from InterventionSite, they differ PER PART:
    `capture` returns a tuple of activations (one per part, in `self.parts`
    order) and `register`/`generate_patched` take a SEQUENCE of patch_fns in
    that same order. Each part has its own width, so one shared patch_fn
    could not be correct anyway (attn_output is 3584 wide, mlp_hidden 18944).

    Hook ordering inside the block is whatever the forward pass does -- for
    attn_output+mlp_output the attention post-hook necessarily fires before
    the MLP post-hook, so the MLP reads the ALREADY-PATCHED residual. That is
    the intended semantics and it does not matter for a FULL swap (the MLP's
    output is overwritten wholesale regardless of what it computed), but it
    would matter for a partial one."""

    is_residual = False
    is_joint = True

    def __init__(self, name):
        assert name in JOINT_SITES, f"unknown joint site {name!r} -- expected one of {tuple(JOINT_SITES)}"
        self.name = name
        self.parts = tuple(InterventionSite(n) for n in JOINT_SITES[name])
        assert not any(p.is_residual for p in self.parts), (
            f"joint site {name!r} includes `residual`, which is the whole block's OUTPUT -- patching it "
            f"alongside one of its own sublayer inputs would make the result depend on hook order and "
            f"is not a meaningful decomposition")

    def __repr__(self):
        return f"JointSite({self.name!r})"

    def widths(self, adapter, model):
        """Per-part widths, in self.parts order."""
        return [p.width(adapter, model) for p in self.parts]

    def width(self, adapter, model) -> int:
        """The TOTAL number of dimensions a full swap here replaces (the sum
        over parts) -- reported in ceiling_sweep's `width` column so the
        column keeps meaning "how much was swapped". It is NOT a mask
        embed_dim; a joint site has no single mask."""
        return sum(self.widths(adapter, model))

    def register(self, adapter, model, layers, layer_idx, patch_fns):
        """patch_fns: one per part, in self.parts order. Returns the handles
        from every part -- caller must .remove() them all."""
        patch_fns = list(patch_fns)
        assert len(patch_fns) == len(self.parts), (
            f"joint site {self.name!r} has {len(self.parts)} parts but got {len(patch_fns)} patch_fns")
        handles = []
        for part, fn in zip(self.parts, patch_fns):
            handles.extend(part.register(adapter, model, layers, layer_idx, fn))
        return handles

    def capture(self, adapter, model, layer_idx, input_ids, attention_mask, extra, positions):
        """-> tuple of [B, n_pos, width_i], one per part, from a SINGLE
        source forward pass (every part's capture hook is registered before
        it, rather than one forward per part)."""
        sinks = [[] for _ in self.parts]
        handles = [p._register_capture(adapter, model, layer_idx, positions, sink)
                   for p, sink in zip(self.parts, sinks)]
        try:
            _source_forward(model, input_ids, attention_mask, extra)
        finally:
            for h in handles:
                h.remove()
        for part, sink in zip(self.parts, sinks):
            _assert_fired_once(part.name, sink)
        return tuple(sink[0] for sink in sinks)

    def generate_patched(self, adapter, model, layers, layer_idx, patch_fns, input_ids, attention_mask, extra,
                         pad_token_id, max_new_tokens):
        """Free-running greedy generation with EVERY part patched. Returns
        [B, max_new_tokens] generated ids (prompt stripped), on CPU. Same
        prefill-only semantics as InterventionSite.generate_patched -- see
        its docstring."""
        handles = self.register(adapter, model, layers, layer_idx, patch_fns)
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

    def forward_patched(self, *a, **k):
        raise NotImplementedError(
            f"joint site {self.name!r} is diagnostic-only: forward_patched is the TRAINING path, and "
            f"training here would need one mask (and one L1 term) per part. Train on a single site from "
            f"SITES instead.")

    def lookup_source(self, *a, **k):
        raise NotImplementedError(
            f"joint site {self.name!r} has no source cache -- ceiling_sweep calls capture() directly, "
            f"which is a single extra forward pass per batch.")


def resolve_site(name):
    """name -> InterventionSite or JointSite. The one entry point that
    accepts anything in ALL_SITES; callers that must have a trainable site
    should keep using InterventionSite(name) so a joint name fails loudly."""
    if name in JOINT_SITES:
        return JointSite(name)
    return InterventionSite(name)


RESIDUAL_SITE = InterventionSite("residual")
