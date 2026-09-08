"""Caches source-side activations across an entity's items, keyed by item id
alone -- eliminates the single biggest source of redundant compute in a
layer sweep.

Why this is sound: in the chat-template layout every method here uses (see
adapters/*.build_inputs), the image comes BEFORE the question/prefill text
in the "user" turn. With causal self-attention, a token's hidden state at
any decoder layer can only depend on itself and tokens *before* it -- so
the hidden state at an image-token position never depends on the question,
prefill, template_id, queried attribute, or paired base entity that follow
it. It depends on the source image alone (plus whatever constant preamble
the chat template always puts before the image, which is the same for
every row). Empirically verified (see scratchpad validate_invariance.py
during development): two real rows sharing a source but differing in
template_id/queried both produced BIT-IDENTICAL hidden states at every
layer checked. This does NOT apply to "last_token" positions -- those sit
at the end of each row's own prompt, whose length varies with the
template, so that position set always falls back to a live forward pass.

Practical effect: instead of one source forward pass per (row, layer) --
repeated across every micro-batch AND every layer of a sweep -- this does
ONE forward pass per entity ITEM, ONCE, covering every layer at once
(output_hidden_states=True already computes all of them), reused for
every attribute/config/layer that entity ever appears as a source in.
For flags (84 items), that's roughly a 400x reduction in source-side
forward passes over a full 28-layer sweep of the ~36k-row train split.

Caveat found during validation: batching an item with OTHER differently-
sized rows shifts its hidden state by a small amount even when the
identity holds exactly in isolation (ordinary GPU batched-matmul/kernel-
selection float non-determinism, not a real dependency -- confirmed by
testing two rows of IDENTICAL length, where no padding differed at all,
and the values still weren't bit-identical). This means live per-batch
computation was never a single canonical value to begin with -- it already
drifts with whatever else happens to share the micro-batch. The cache is
therefore built with each item run SOLO (batch size 1, no padding at all),
which is at least as principled as -- arguably more reproducible than --
today's live values, and the observed drift (median ~0.01-0.02 of typical
hidden-state magnitude in the worst dims) is the same order of magnitude
as that pre-existing batch-composition noise, not a new failure mode.
"""
import os

import torch

from .hooks import extra_to_device

CACHE_DIRNAME = "source_activation_cache"


def source_cache_path(vade_root, model_slug, entity):
    return os.path.join(vade_root, "results", model_slug, entity, CACHE_DIRNAME, "full_image.pt")


def build_source_cache(adapter, model, processor, entity_assets, cache_path, canonical_attribute=None,
                        canonical_template_id=None, progress_every=20):
    """One forward pass per entity item (batch size 1, no padding -- see
    module docstring for why), output_hidden_states=True so every layer is
    captured at once. Stores {item_id: [n_hidden_states, n_image_tokens, H]}
    (bf16, on CPU) to cache_path. Resumable: items already present in an
    existing cache_path are skipped and kept.

    canonical_attribute/canonical_template_id: which (question, prefill)
    text to render the image with -- genuinely arbitrary per the module
    docstring, defaults to this entity's first attribute and that
    attribute's first template_id. Never affects the cached values."""
    from PIL import Image

    from .entities import object_token_positions, resolve_position_set

    attribute = canonical_attribute or entity_assets.attributes[0]
    template_id = canonical_template_id or next(iter(entity_assets.template_lookup[attribute]))
    entry = entity_assets.template_lookup[attribute][template_id]
    question, prefill = entry["question"], entry["prefill"]

    flat_indices, n_image_tokens, is_last_token = resolve_position_set(
        "full_image", entity_assets, adapter, model)
    assert not is_last_token, "full_image should never resolve to last_token"
    image_token_id = adapter.image_token_id(model, processor)

    cache = {}
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        print(f"[source_cache] resuming: {len(cache)} items already cached at {cache_path}")

    todo = [item_id for item_id in entity_assets.items if item_id not in cache]
    print(f"[source_cache] entity={entity_assets.entity} attribute={attribute} template={template_id}: "
          f"{len(todo)} items to cache (of {len(entity_assets.items)})")

    for i, item_id in enumerate(todo):
        image = Image.open(os.path.join(
            entity_assets.entity_dir, entity_assets.items[item_id]["image"])).convert("RGB")
        built = adapter.build_inputs(processor, image, question, prefill)
        input_ids = built["input_ids"].unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        extra_dev = extra_to_device(built["extra"], model.device, model.dtype)
        with torch.no_grad():
            out = model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                        **extra_dev, output_hidden_states=True, logits_to_keep=1)

        pos = object_token_positions(built["input_ids"], image_token_id, flat_indices, n_image_tokens)
        layers = torch.stack([hs[0, pos].detach().cpu() for hs in out.hidden_states])  # [n_hidden_states, n_pos, H]
        cache[item_id] = layers

        if (i + 1) % progress_every == 0 or (i + 1) == len(todo):
            print(f"  [source_cache] {i + 1}/{len(todo)}", flush=True)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(cache, cache_path)

    if todo:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    print(f"[source_cache] DONE: {len(cache)} items -> {cache_path}")
    return cache_path


def load_source_cache(cache_path, device="cpu"):
    return torch.load(cache_path, map_location=device)


def get_or_build_source_cache(adapter, model, processor, entity_assets, vade_root, model_slug):
    """CLI convenience: build_source_cache is resumable/idempotent, so it's
    always safe to call -- a fully up-to-date cache just short-circuits to
    an empty `todo` list. Called by default (opt out with --no_source_cache)
    from every methods/das/*.py CLI, so the cache-location convention lives
    in exactly one place."""
    cache_path = source_cache_path(vade_root, model_slug, entity_assets.entity)
    build_source_cache(adapter, model, processor, entity_assets, cache_path)
    return load_source_cache(cache_path)


def lookup_source_hidden(cache, batch, layer_idx, device, dtype):
    """batch: from entities.build_batch. Returns [B, n_pos, H] on `device`/
    `dtype`, gathered by indexing each row's source item into `cache` --
    no forward pass. Raises KeyError with the offending item id if some
    row's source isn't in the cache (rebuild the cache rather than
    silently falling back, so a stale/partial cache fails loudly)."""
    flat_indices = batch["flat_indices"]
    vecs = []
    for row in batch["rows"]:
        item_id = row["source"]
        if item_id not in cache:
            raise KeyError(f"source_cache missing item {item_id!r} -- rebuild the cache for this entity")
        vecs.append(cache[item_id][layer_idx, flat_indices])
    return torch.stack(vecs).to(device=device, dtype=dtype)
