"""Entity asset loading + generic batch building, shared by every method.
Reads straight from a VADE entity's own ground_truth.json /
object_location.json / prompt_templates.json / tuples/ -- no per-entity
constants baked in here (see ITEMS_KEY_BY_ENTITY's fallback), and no
model-specific code (all model calls go through a ModelAdapter, see
adapters/base.py).

This is the "mechanism to get model data like patch count" for DAS and any
future method: resolve_position_set() calls adapter.vision_patch_grid() to
COMPUTE the merged-token grid from the model's own vision config, and
cross-checks it against the entity's stored object_location.json rather
than trusting that file blindly -- a stale file or a wrong adapter formula
fails loudly instead of silently mis-indexing positions.
"""
import json
import os
from dataclasses import dataclass

import torch
from PIL import Image

from .targets import derive_gold_token_ids, pad_gold_toks

# flags predates the other entities and kept its original "countries" name
# in ground_truth.json even though the concept generalized to "items".
ITEMS_KEY_BY_ENTITY = {"flags": "countries", "brands": "brands", "animals": "species"}


@dataclass
class EntityAssets:
    entity: str
    entity_dir: str
    items: dict            # item_id -> {..., "image": relpath, <attr>: value, ...}
    object_location: dict  # raw parsed object_location.json
    template_lookup: dict  # {attribute: {template_id: {"question", "prefill", ...}}}
    attributes: list       # scored attribute names, e.g. ["capital","currency","language","calling_code"]


def load_entity_assets(vade_root, entity):
    entity_dir = os.path.join(vade_root, "data", entity)
    gt_path = os.path.join(entity_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"No ground_truth.json for entity {entity!r} at {gt_path!r} -- this VADE entity "
            f"likely isn't built yet.")
    gt = json.load(open(gt_path))
    loc = json.load(open(os.path.join(entity_dir, "object_location.json")))
    templates = json.load(open(os.path.join(entity_dir, "prompt_templates.json")))

    items_key = ITEMS_KEY_BY_ENTITY.get(entity)
    if items_key is None or items_key not in gt:
        # Unknown entity: fall back to "the one dict-valued key that isn't metadata",
        # same heuristic as VADE/models/run_accuracy_sweep.py.
        items_key = next(k for k, v in gt.items() if isinstance(v, dict) and k != "coverage")
    items = gt[items_key]
    template_lookup = {attr: {t["template_id"]: t for t in tlist} for attr, tlist in templates.items()}
    attributes = gt.get("attributes") or sorted(template_lookup)
    return EntityAssets(entity, entity_dir, items, loc, template_lookup, attributes)


def load_tuples(entity_assets, attribute, split, tuples_dir=None):
    """tuples_dir overrides the default entity_assets.entity_dir/tuples/ root
    -- e.g. a model-specific pruned copy under models/<model_slug>/<entity>/
    tuples/. Images/object_location/templates always come from entity_assets
    itself regardless (pruning only drops rows, it never touches images)."""
    root = tuples_dir or os.path.join(entity_assets.entity_dir, "tuples")
    path = os.path.join(root, attribute, f"{split}.jsonl")
    return [json.loads(l) for l in open(path)]


def pruned_tuples_dir(vade_root, model_slug, entity):
    return os.path.join(vade_root, "models", model_slug, entity, "tuples")


def require_pruned_tuples(vade_root, model_slug, entity, attribute):
    """Returns the pruned tuples_dir for (entity, attribute) if it exists,
    else raises loudly instead of letting a caller silently fall back to
    data/<entity>/tuples/ (unpruned). Training/eval on unpruned data wastes
    capacity on rows the model can't even answer correctly at baseline --
    see models/prune_tuples.py's docstring -- and that mistake is exactly
    what --allow_unpruned exists to make an explicit, deliberate choice
    instead of an easy-to-forget flag."""
    d = pruned_tuples_dir(vade_root, model_slug, entity)
    train_path = os.path.join(d, attribute, "train.jsonl")
    test_path = os.path.join(d, attribute, "test.jsonl")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"No pruned tuples for entity={entity!r} attribute={attribute!r} at {d}/{attribute}/ -- run "
            f"models/prune_tuples.py --model_id <this model> --entities {entity} --attributes {attribute} "
            f"first, or pass --allow_unpruned to deliberately train/eval on data/{entity}/tuples/ instead.")
    return d


def _validated_n_image_tokens(entity_assets, adapter, model):
    """The number of merged image tokens a VADE entity's (fixed-canvas)
    images expand to -- computed FROM the model's own vision config and
    cross-checked against the entity's stored object_location.json, so a
    stale file or a wrong adapter formula fails loudly instead of silently
    mis-indexing positions. Every image in an entity shares this exact
    count (see each entity's object_location.json note field: one fixed
    canvas size per entity, only the object's pixel content varies) --
    that's what makes it safe for BuildBatchCache to tokenize a template
    once and reuse it for every row regardless of which image."""
    loc = entity_assets.object_location
    grid = loc["vlm_token_grid"]
    canvas = loc["canvas_px"]
    computed = adapter.vision_patch_grid(model, canvas["width"], canvas["height"])
    mismatches = {k: (computed[k], grid[k]) for k in ("grid_rows", "grid_cols", "n_tokens", "patch_size_px", "merge_size")
                  if computed[k] != grid[k]}
    assert not mismatches, (
        f"adapter's computed vision_patch_grid disagrees with {entity_assets.entity}/object_location.json's "
        f"vlm_token_grid: {mismatches} (computed, stored). Either object_location.json is stale for this "
        f"model, or the adapter's patch-geometry formula is wrong.")
    return grid["n_tokens"]


def resolve_position_set(positions_name, entity_assets, adapter, model):
    """Returns (flat_indices, n_image_tokens, is_last_token). flat_indices/
    n_image_tokens are None when is_last_token is True (no image-grid
    reasoning applies to that mode)."""
    if positions_name == "last_token":
        return None, None, True

    n_image_tokens = _validated_n_image_tokens(entity_assets, adapter, model)

    if positions_name == "full_image":
        return list(range(n_image_tokens)), n_image_tokens, False

    available = entity_assets.object_location["object_token_indices"]
    assert positions_name in available, (
        f"Unknown position set {positions_name!r} for entity {entity_assets.entity!r}; "
        f"available: {list(available)} (plus 'last_token', 'full_image')")
    return available[positions_name]["flat"], n_image_tokens, False


def object_token_positions(input_ids_1d, image_token_id, flat_indices, n_image_tokens):
    """Sequence positions (ascending = row-major merged-patch order) that
    hold this image's placeholder tokens within the full templated input,
    offset by flat_indices -- object_location.json's flat indices are
    relative to the image span's own start, row-major (verified against
    real pixel_values diffs when each entity's object_location.json was
    built; see its own "note" field)."""
    ids = input_ids_1d.tolist()
    positions = [i for i, t in enumerate(ids) if t == image_token_id]
    assert positions, "no image tokens found in sequence"
    o = positions[0]
    assert positions == list(range(o, o + n_image_tokens)), \
        f"image token span not contiguous/{n_image_tokens} as expected: {positions[:5]}...{positions[-5:]}"
    return [o + f for f in flat_indices]


def concat_extra(extra_dicts):
    """Concatenates a list of per-row `extra` dicts (adapter.build_inputs'
    model-specific tensors, each with a leading batch dim of 1) along dim 0
    into one batched extra dict -- generic over whatever keys the adapter
    uses (pixel_values/image_grid_thw for a VL model, nothing for a
    text-only one)."""
    if not extra_dicts:
        return {}
    keys = extra_dicts[0].keys()
    return {k: torch.cat([d[k] for d in extra_dicts], dim=0) for k in keys}


class BuildBatchCache:
    """Caches the two things build_batch used to redo on EVERY row despite
    them being pure functions of much smaller keys:

    - image_extra: item_id -> adapter.encode_image's {pixel_values, ...}.
      An entity's ~100 unique images back thousands of tuple rows (each
      image is somebody's base in some rows and somebody's source in
      others), so this collapses a PIL.Image.open + vision-preprocess PER
      ROW into one per unique image.
    - template: (queried, template_id) -> {input_ids, pos_unpadded}, and
      gold: (queried, template_id, label) -> gold token ids. Valid because
      every image in a VADE entity shares one fixed canvas size (see
      _validated_n_image_tokens), so the fully-tokenized prompt (image
      tokens included) and the object-token positions within it depend
      only on the template, never on which image fills it.

    Pass ONE instance into every build_batch call across a training/eval
    run (see train.py/eval.py) to actually get the reuse -- a fresh
    instance per call (the default) just reproduces the old per-call
    behavior with no cross-call caching."""

    def __init__(self):
        self.image_extra = {}
        self.template = {}
        self.gold = {}


def build_batch(rows, entity_assets, adapter, model, processor, positions_name, batch_cache=None):
    """Builds a left-padded base+source batch for DAS (or any method
    needing base/source activations at aligned positions). rows: list of
    tuples-schema dicts (target_attribute, queried, base, source,
    template_id, rule, base_label, source_label, row_index, ...).

    batch_cache (optional): a BuildBatchCache shared across calls -- see
    its docstring. Defaults to a fresh (call-local, no cross-call reuse)
    instance.

    Returns a dict with:
      rows, base_input_ids, source_input_ids, attention_mask, positions,
      base_extra, source_extra (concatenated, model-specific),
      base_gold_toks, base_gold_len, source_gold_toks, source_gold_len.
    """
    image_token_id = adapter.image_token_id(model, processor)
    flat_indices, n_image_tokens, is_last_token = resolve_position_set(positions_name, entity_assets, adapter, model)
    # The real image always occupies n_image_tokens spots in the sequence regardless of
    # positions_name -- resolve_position_set only returns None for last_token because that
    # mode doesn't need the image GRID (no per-patch position reasoning), not because the
    # image itself is smaller. tokenize_template needs the real count either way.
    n_image_tokens_real = n_image_tokens if n_image_tokens is not None else \
        _validated_n_image_tokens(entity_assets, adapter, model)

    cache = batch_cache if batch_cache is not None else BuildBatchCache()

    def get_image_extra(item_id):
        extra = cache.image_extra.get(item_id)
        if extra is None:
            img = Image.open(os.path.join(entity_assets.entity_dir, entity_assets.items[item_id]["image"])).convert("RGB")
            extra = adapter.encode_image(processor, img)
            cache.image_extra[item_id] = extra
        return extra

    def get_template(queried, template_id):
        key = (queried, template_id)
        entry = cache.template.get(key)
        if entry is None:
            tmpl = entity_assets.template_lookup[queried][template_id]
            question, prefill = tmpl["question"], tmpl["prefill"]
            input_ids = adapter.tokenize_template(processor, question, prefill, n_image_tokens_real)
            if is_last_token:
                pos_unpadded = [input_ids.shape[0] - 1]
            else:
                pos_unpadded = object_token_positions(input_ids, image_token_id, flat_indices, n_image_tokens)
            entry = {"input_ids": input_ids, "pos_unpadded": pos_unpadded, "prefill": prefill}
            cache.template[key] = entry
        return entry

    def get_gold(queried, template_id, prefill, label):
        key = (queried, template_id, label)
        ids = cache.gold.get(key)
        if ids is None:
            ids = derive_gold_token_ids(processor.tokenizer, prefill, label)
            cache.gold[key] = ids
        return ids

    per_row = []
    for r in rows:
        tmpl = get_template(r["queried"], r["template_id"])
        base_gold_ids = get_gold(r["queried"], r["template_id"], tmpl["prefill"], r["base_label"])
        source_gold_ids = get_gold(r["queried"], r["template_id"], tmpl["prefill"], r["source_label"])

        per_row.append({
            "row": r, "base_ids": tmpl["input_ids"], "source_ids": tmpl["input_ids"],
            "base_extra": get_image_extra(r["base"]), "source_extra": get_image_extra(r["source"]),
            "pos_unpadded": tmpl["pos_unpadded"],
            "base_gold_ids": base_gold_ids, "source_gold_ids": source_gold_ids,
        })

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    max_len = max(len(x["base_ids"]) for x in per_row)
    B, n_pos = len(per_row), len(per_row[0]["pos_unpadded"])
    base_input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    source_input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    positions = torch.zeros((B, n_pos), dtype=torch.long)

    for i, x in enumerate(per_row):
        L = len(x["base_ids"])
        left_pad = max_len - L
        base_input_ids[i, left_pad:] = x["base_ids"]
        source_input_ids[i, left_pad:] = x["source_ids"]
        attention_mask[i, left_pad:] = 1
        positions[i] = torch.tensor([left_pad + p for p in x["pos_unpadded"]])

    base_gold_toks, base_gold_len = pad_gold_toks([x["base_gold_ids"] for x in per_row], pad_id)
    source_gold_toks, source_gold_len = pad_gold_toks([x["source_gold_ids"] for x in per_row], pad_id)

    return {
        "rows": [x["row"] for x in per_row],
        "base_input_ids": base_input_ids, "source_input_ids": source_input_ids,
        "attention_mask": attention_mask, "positions": positions,
        "base_extra": concat_extra([x["base_extra"] for x in per_row]),
        "source_extra": concat_extra([x["source_extra"] for x in per_row]),
        "base_gold_toks": base_gold_toks, "base_gold_len": base_gold_len,
        "source_gold_toks": source_gold_toks, "source_gold_len": source_gold_len,
        # relative-to-image-span indices (as opposed to `positions`, which are absolute
        # sequence columns) -- source_cache.py needs these to index a per-entity cache
        # that was built once against the image span alone, not any particular row's
        # padded sequence. None/None when is_last_token (that position isn't image-based
        # at all, so it can't be served from an image-keyed cache).
        "flat_indices": flat_indices, "n_image_tokens": n_image_tokens, "is_last_token": is_last_token,
    }
