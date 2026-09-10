"""Source-activation cache for the MLP intervention sites (common/sites.py).

NOT copied from the sibling VADE repo -- new in this repo, and a sibling of
common/source_cache.py rather than an edit to it (that file is a verbatim
copy of VADE's own and stays one).

WHY A SECOND CACHE. common/source_cache.py caches torch.stack(
out.hidden_states) -- the RESIDUAL stream at every layer, in one file per
entity. MLP internals appear nowhere in output_hidden_states, so that cache
cannot serve an MLP site at all. Its entries are the wrong tensor, and for
mlp_hidden also the wrong width (3584 vs 18944). For mlp_output they are the
wrong tensor at the SAME width, which is precisely why lookup below asserts
on recorded metadata rather than trusting shapes.

WHY IT IS SOUND -- the same causal-attention argument that justifies the
residual cache, and it carries over unchanged. In the chat-template layout
every method here uses (adapters/*.build_inputs), the image comes BEFORE the
question/prefill text. A decoder block's MLP at position p reads only the
residual stream at position p, which under causal self-attention depends
only on tokens <= p. So an image token's MLP hidden state never depends on
the question, prefill, template_id, queried attribute, or paired base entity
that follow it -- it is a function of the source image alone, plus the
constant preamble every row shares. See common/source_cache.py's own module
docstring for the empirical verification of the residual version of this
claim (bit-identical hidden states across rows sharing a source), and for
the batch-composition float-drift caveat, which is why entries here are also
built with each item run SOLO at batch size 1.

This does NOT hold for positions="last_token": that position sits at the end
of each row's own prompt, whose length varies with the template, so it
depends on the question text. build_site_source_cache refuses it.

WHAT IS DIFFERENT FROM THE RESIDUAL CACHE'S SHAPE, and why:

  * ONE FILE PER (site, positions, layer), not one file per entity holding
    every layer. The residual cache gets all layers for free from a single
    output_hidden_states=True pass; an MLP site needs a capture hook at one
    specific module, so covering every layer would mean either 28 hooks per
    pass or 28 passes -- and the storage is 5.3x wider per layer. For flags
    at flag_ring1 that is ~76MB per layer (84 items x 24 positions x 18944 x
    bf16) against ~2.1GB for all 28 layers at once. A layer sweep trains one
    layer at a time, so per-layer files are also what actually gets reused.

  * KEYED BY THE POSITION SET, storing only that set's positions rather than
    every image token. Same motivation: at intermediate_size, caching the
    whole image would multiply the file size by (n_image_tokens / n_pos).
    The tradeoff is that changing --positions rebuilds the cache instead of
    re-indexing an existing one; a sweep varies the layer and holds
    positions fixed, so that is the cheap direction to give up.

  * A `meta` header alongside the items, asserted on lookup. The residual
    cache is a flat {item_id: tensor} dict and cannot self-describe; because
    an mlp_output cache and a residual cache have identical shapes, silent
    cross-use is a real failure mode here and worth failing loudly on.

Layout: {"meta": {...}, "items": {item_id: tensor[n_pos, width]}} (bf16, CPU).
"""
import os

import torch

CACHE_DIRNAME = "source_activation_cache"
CACHE_FORMAT_VERSION = 1


def site_source_cache_path(vade_root, model_slug, entity, site_name, positions_name, layer):
    """Same directory as common/source_cache.py's own file (keyed by
    entity+model, not by method -- an NDM cache is equally reusable by any
    future method intervening at the same site), distinguished by filename."""
    return os.path.join(vade_root, "results", model_slug, entity, CACHE_DIRNAME,
                         f"{site_name}_{positions_name}_layer{layer}.pt")


def build_site_source_cache(adapter, model, processor, entity_assets, site, layer, positions_name, cache_path,
                             canonical_attribute=None, canonical_template_id=None, progress_every=20):
    """One solo forward pass per entity item with a capture hook at
    (site, layer), storing that item's activation at `positions_name`'s
    positions. Resumable: items already present are skipped and kept.

    canonical_attribute/canonical_template_id: which (question, prefill) to
    render the image with -- arbitrary per the module docstring, and never
    affects the cached values. Defaults to this entity's first attribute and
    that attribute's first template_id, matching common/source_cache.py."""
    from PIL import Image

    from .entities import object_token_positions, resolve_position_set

    attribute = canonical_attribute or entity_assets.attributes[0]
    template_id = canonical_template_id or next(iter(entity_assets.template_lookup[attribute]))
    entry = entity_assets.template_lookup[attribute][template_id]
    question, prefill = entry["question"], entry["prefill"]

    flat_indices, n_image_tokens, is_last_token = resolve_position_set(
        positions_name, entity_assets, adapter, model)
    assert not is_last_token, (
        f"positions={positions_name!r} resolves to a last_token position, which depends on each row's own "
        f"prompt length and so is NOT cacheable -- see this module's docstring. Run with the cache disabled.")
    # resolve_position_set returns (None, None, True) for a last_token set, so flat_indices is only
    # non-None past the assert above; re-bind it as a concrete list so that's evident to a reader (and
    # to a type checker) rather than implied by the assert.
    flat_indices = list(flat_indices)
    n_pos = len(flat_indices)
    image_token_id = adapter.image_token_id(model, processor)
    width = site.width(adapter, model)

    meta = {
        "format_version": CACHE_FORMAT_VERSION, "site": site.name, "layer": layer,
        "positions": positions_name, "width": width, "n_pos": n_pos,
        "model_id": getattr(adapter, "model_id", None), "entity": entity_assets.entity,
    }

    cache = {"meta": meta, "items": {}}
    if os.path.exists(cache_path):
        existing = torch.load(cache_path, map_location="cpu")
        _assert_meta_compatible(existing.get("meta"), meta, cache_path)
        cache = existing
        print(f"[site_source_cache] resuming: {len(cache['items'])} items already cached at {cache_path}")

    todo = [item_id for item_id in entity_assets.items if item_id not in cache["items"]]
    projected_mb = len(entity_assets.items) * n_pos * width * 2 / 1e6
    print(f"[site_source_cache] entity={entity_assets.entity} site={site.name} layer={layer} "
          f"positions={positions_name} n_pos={n_pos} width={width}: {len(todo)} items to cache "
          f"(of {len(entity_assets.items)}), projected size ~{projected_mb:.0f}MB -> {cache_path}")

    for i, item_id in enumerate(todo):
        image = Image.open(os.path.join(
            entity_assets.entity_dir, entity_assets.items[item_id]["image"])).convert("RGB")
        built = adapter.build_inputs(processor, image, question, prefill)
        input_ids = built["input_ids"].unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        # object_token_positions returns a plain python LIST of absolute column indices (not a tensor --
        # common/source_cache.py gets away with using it directly because torch accepts a list as an
        # index). site.capture instead needs positions as a [B, n_pos] TENSOR, since it does
        # positions.shape[0] and indexes row by row, so convert explicitly here. B=1: each item is run
        # solo with no padding at all -- see the module docstring on why not batched.
        pos = object_token_positions(built["input_ids"], image_token_id, flat_indices, n_image_tokens)
        positions = torch.tensor([pos], dtype=torch.long)
        act = site.capture(adapter, model, layer, input_ids, attention_mask, built["extra"], positions)
        assert act.shape == (1, n_pos, width), (
            f"item {item_id!r}: captured {tuple(act.shape)}, expected {(1, n_pos, width)}")
        cache["items"][item_id] = act[0].to(torch.bfloat16).cpu()

        if (i + 1) % progress_every == 0 or (i + 1) == len(todo):
            print(f"  [site_source_cache] {i + 1}/{len(todo)}", flush=True)
            _save(cache, cache_path)

    if todo:
        _save(cache, cache_path)
    print(f"[site_source_cache] DONE: {len(cache['items'])} items -> {cache_path}")
    return cache_path


def _save(cache, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)


ALL_META_KEYS = ("format_version", "site", "layer", "positions", "width", "n_pos")
# The subset a LOOKUP can meaningfully check. width/n_pos are deliberately excluded there: the lookup
# path has no adapter/model to recompute them from, so comparing them would mean comparing the cache's
# own recorded values against themselves -- a tautology that reads like a real check. They ARE checked
# at build time (against site.width(adapter, model) and the resolved position set), and the four keys
# below already cover every dangerous mix-up: wrong site, wrong layer, wrong position set, or a
# residual cache handed in by mistake (which has no `meta` at all).
LOOKUP_META_KEYS = ("format_version", "site", "layer", "positions")


def _assert_meta_compatible(found, want, cache_path, keys=ALL_META_KEYS):
    assert found is not None, (
        f"{cache_path} has no `meta` header -- it is not a site source cache (a flat {{item_id: tensor}} "
        f"file is common/source_cache.py's RESIDUAL cache format). Delete or rename it.")
    for key in keys:
        assert found.get(key) == want[key], (
            f"{cache_path} was built with {key}={found.get(key)!r}, but this run wants {key}={want[key]!r} -- "
            f"delete the file to rebuild it rather than mixing incompatible entries.")


def load_site_source_cache(cache_path, device="cpu"):
    return torch.load(cache_path, map_location=device)


def get_or_build_site_source_cache(adapter, model, processor, entity_assets, site, layer, positions_name,
                                    vade_root, model_slug):
    """CLI convenience mirroring common/source_cache.py's
    get_or_build_source_cache: build is resumable/idempotent, so calling it
    unconditionally is safe -- an up-to-date cache short-circuits to an empty
    todo list. Returns the loaded cache, or None if this site/positions
    combination isn't cacheable (last_token), so callers can just fall
    through to the live-forward-pass path."""
    from .entities import resolve_position_set

    _, _, is_last_token = resolve_position_set(positions_name, entity_assets, adapter, model)
    if is_last_token:
        print(f"[site_source_cache] positions={positions_name!r} is a last_token set -- not cacheable, "
              f"falling back to a live source forward pass per batch")
        return None
    cache_path = site_source_cache_path(vade_root, model_slug, entity_assets.entity, site.name,
                                         positions_name, layer)
    build_site_source_cache(adapter, model, processor, entity_assets, site, layer, positions_name, cache_path)
    return load_site_source_cache(cache_path)


def lookup_site_source(cache, batch, site, layer, positions_name, device, dtype):
    """-> [B, n_pos, width] on `device`/`dtype`, gathered by indexing each
    row's source item. No forward pass, no flat_indices indexing (unlike the
    residual cache's lookup): entries already hold exactly this position
    set's positions, in order.

    Asserts the cache's `meta` matches this run (LOOKUP_META_KEYS -- see the
    note there on why width/n_pos are checked at build time instead). That
    check is the whole reason the header exists: an mlp_output cache, a
    residual cache and a wrong-layer cache all have compatible SHAPES, so
    shape agreement proves nothing here."""
    _assert_meta_compatible(cache.get("meta"), {
        "format_version": CACHE_FORMAT_VERSION, "site": site.name, "layer": layer,
        "positions": positions_name,
    }, "<in-memory site source cache>", keys=LOOKUP_META_KEYS)

    items = cache["items"]
    vecs = []
    for row in batch["rows"]:
        item_id = row["source"]
        if item_id not in items:
            raise KeyError(f"site source cache missing item {item_id!r} -- rebuild the cache for this entity")
        vecs.append(items[item_id])
    return torch.stack(vecs).to(device=device, dtype=dtype)
