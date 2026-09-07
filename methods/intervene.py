"""Phase-B step 5: the actual PCA/SAE causal intervention. For every
attribute with a winner in select_features.py's winners.json, patches the
winning dictionary-feature subset between a base and source image's
activations at the winning layer, lets the model generate freely, and
writes predictions in VADE/eval/score.py's format ({"attribute",
"row_index", "generated_text"}).

Adapted from the existing DAS intervention pipeline in the sibling
flag-benchmark repo (experiments/image-to-image/scripts/core/collate.py +
eval/generate_vade_predictions.py) -- verified byte-identical against
VADE's own images and object_location.json token positions (flags:
FLAT_INDICES==[64,65,66,67,76,77,78,79], FLAG_RING1_INDICES match too), so
the same forward-hook/prefill-generation mechanism applies unchanged. This
version generalizes it two ways: (1) entity-agnostic -- reads images,
token positions, and prompt templates straight from VADE's own
ground_truth.json/object_location.json/prompt_templates.json for any
entity, not hardcoded flag positions/filenames; (2) swaps dictionary-
encoded feature dims (PCA components or SAE latents) instead of a learned
DAS rotation -- no training involved, the "intervention" is just
encode(base)/encode(source), copy over the winning dims, decode.

Mechanism (same for every attribute, only (layer, dictionary,
feature_indices) differ):
  1. Run the source image through the model once, cache its hidden state
     at the winning layer, at the token positions covering the object.
  2. Register a forward hook on that layer that, on the ONE multi-token
     prefill forward pass (a no-op on every later single-token incremental
     decode step -- the patch's effect is already baked into the KV cache
     from the prefill by then): encodes both the base's and the cached
     source's hidden state at those positions through the dictionary,
     copies over only the winning feature_indices from source into base,
     decodes back to raw hidden-state space, and substitutes that in.
  3. model.generate() on the base image's prompt with that hook active.

Usage
-----
    python methods/intervene.py --entity flags --token_set flag_only \\
        --dict_method pca --out methods/interventions/flags_flag_only_pca_predictions.jsonl
"""
import argparse
import json
import os

import torch
from PIL import Image

from features import DICTIONARIES_DIR, ITEMS_KEY_BY_ENTITY, REPO_ROOT, dictionary_path, load_dictionary

DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"


# ---------------------------------------------------------------------------
# VADE entity assets
# ---------------------------------------------------------------------------

def load_entity_assets(vade_root, entity):
    entity_dir = os.path.join(vade_root, "data", entity)
    gt = json.load(open(os.path.join(entity_dir, "ground_truth.json")))
    loc = json.load(open(os.path.join(entity_dir, "object_location.json")))
    templates = json.load(open(os.path.join(entity_dir, "prompt_templates.json")))
    items = gt[ITEMS_KEY_BY_ENTITY[entity]]
    template_lookup = {attr: {t["template_id"]: t for t in tlist} for attr, tlist in templates.items()}
    return entity_dir, items, loc, template_lookup


# ---------------------------------------------------------------------------
# Batch building (prompt-only, base+source images, both left-padded to a
# common length so their object-token positions land at the same absolute
# columns) -- ported from flag-benchmark's core/collate.py, generalized off
# object_location.json instead of hardcoded flag indices/filenames.
# ---------------------------------------------------------------------------

def get_decoder_layers(model):
    return model.model.language_model.layers


def object_token_positions(input_ids_1d, image_token_id, flat_indices, n_image_tokens):
    ids = input_ids_1d.tolist()
    positions = [i for i, t in enumerate(ids) if t == image_token_id]
    assert positions, "no image tokens found in sequence"
    o = positions[0]
    assert positions == list(range(o, o + n_image_tokens)), \
        f"image token span not contiguous/{n_image_tokens} as expected: {positions[:5]}...{positions[-5:]}"
    return [o + f for f in flat_indices]


def build_prompt_inputs(processor, image, question, prefill):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": prefill}]},
    ]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt",
    )


def build_batch(rows, template_lookup, processor, items, entity_dir, flat_indices, n_image_tokens, image_token_id):
    per_row = []
    for r in rows:
        entry = template_lookup[r["queried"]][r["template_id"]]
        question, prefill = entry["question"], entry["prefill"]
        base_img = Image.open(os.path.join(entity_dir, items[r["base"]]["image"])).convert("RGB")
        source_img = Image.open(os.path.join(entity_dir, items[r["source"]]["image"])).convert("RGB")
        base_prompt = build_prompt_inputs(processor, base_img, question, prefill)
        source_prompt = build_prompt_inputs(processor, source_img, question, prefill)
        base_pos = object_token_positions(base_prompt["input_ids"][0], image_token_id, flat_indices, n_image_tokens)
        source_pos = object_token_positions(source_prompt["input_ids"][0], image_token_id, flat_indices, n_image_tokens)
        assert base_prompt["input_ids"].shape[1] == source_prompt["input_ids"].shape[1] and base_pos == source_pos, (
            f"{r['base']}/{r['source']} ({r['template_id']}): prompt length or object-token "
            f"position mismatch between base and source")
        per_row.append({
            "row": r, "base_ids": base_prompt["input_ids"][0], "source_ids": source_prompt["input_ids"][0],
            "base_pixel_values": base_prompt["pixel_values"], "source_pixel_values": source_prompt["pixel_values"],
            "base_image_grid_thw": base_prompt["image_grid_thw"], "source_image_grid_thw": source_prompt["image_grid_thw"],
            "pos_unpadded": base_pos,
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
        # base/source share length L per row (asserted above), so the same
        # left_pad amount keeps both rows' object-token positions aligned.
        source_input_ids[i, left_pad:] = x["source_ids"]
        attention_mask[i, left_pad:] = 1
        positions[i] = torch.tensor([left_pad + p for p in x["pos_unpadded"]])

    return {
        "rows": [x["row"] for x in per_row],
        "base_input_ids": base_input_ids, "attention_mask": attention_mask, "positions": positions,
        "base_pixel_values": torch.cat([x["base_pixel_values"] for x in per_row], dim=0),
        "base_image_grid_thw": torch.cat([x["base_image_grid_thw"] for x in per_row], dim=0),
        "source_input_ids": source_input_ids,
        "source_attention_mask": attention_mask,  # same padding as base (asserted equal per-row length above)
        "source_pixel_values": torch.cat([x["source_pixel_values"] for x in per_row], dim=0),
        "source_image_grid_thw": torch.cat([x["source_image_grid_thw"] for x in per_row], dim=0),
    }


# ---------------------------------------------------------------------------
# Hooking + patched generation -- verbatim mechanism from
# flag-benchmark/.../core/collate.py (make_cache_aware_patch_hook,
# register_patch_hook), just not importing across the two repos.
# ---------------------------------------------------------------------------

def cache_source_hidden(model, batch, layer_idx):
    with torch.no_grad():
        out = model(
            input_ids=batch["source_input_ids"].to(model.device),
            attention_mask=batch["source_attention_mask"].to(model.device),
            pixel_values=batch["source_pixel_values"].to(model.device).to(model.dtype),
            image_grid_thw=batch["source_image_grid_thw"].to(model.device),
            output_hidden_states=True,
            logits_to_keep=1,
        )
    layer_hidden = out.hidden_states[layer_idx]
    positions = batch["positions"]
    B = positions.shape[0]
    return torch.stack([layer_hidden[i, positions[i]] for i in range(B)]).detach()


def make_cache_aware_patch_hook(positions, compute_replacement):
    """positions: [B, n_pos] absolute column indices (only valid against a
    full, un-cached forward pass). compute_replacement(base_vals) -> new_vals,
    both [B, n_pos, H]. A no-op once hidden_states has collapsed to a single
    column (generate()'s later incremental decode steps) -- the patch's
    effect from the prefill step is already in the KV cache by then."""
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
    layers[layer_idx-1]) -- matches sae.py's own layer_convention (index 0 =
    embedding output; index i = residual stream after decoder layer i)."""
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


def run_patched_generate(model, layers, layer_idx, patch_fn, input_ids, attention_mask, pixel_values, image_grid_thw,
                          pad_token_id, max_new_tokens):
    handles = register_patch_hook(layers, layer_idx, patch_fn)
    try:
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                pixel_values=pixel_values.to(model.device).to(model.dtype), image_grid_thw=image_grid_thw.to(model.device),
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_token_id,
            )
    finally:
        for h in handles:
            h.remove()
    return gen[:, input_ids.shape[1]:].cpu()


# ---------------------------------------------------------------------------
# The dictionary intervention itself: encode/swap/decode per real position.
# ---------------------------------------------------------------------------

def make_dict_intervention(dictionary, feature_indices):
    F_A = feature_indices

    def intervene(base_vals, source_vals):
        """base_vals, source_vals: [B, n_pos, H] torch tensors (model
        device/dtype). Returns the patched replacement for base_vals, same
        shape/device/dtype."""
        device, dtype = base_vals.device, base_vals.dtype
        B, n_pos, H = base_vals.shape
        base_np = base_vals.detach().float().cpu().numpy().reshape(B * n_pos, H)
        source_np = source_vals.detach().float().cpu().numpy().reshape(B * n_pos, H)
        F_base = dictionary.encode(base_np)
        F_source = dictionary.encode(source_np)
        F_new = F_base.copy()
        F_new[:, F_A] = F_source[:, F_A]
        new_np = dictionary.decode(F_new).reshape(B, n_pos, H)
        return torch.from_numpy(new_np).to(device=device, dtype=dtype)

    return intervene


def generate_patched(model, batch, layer_idx, intervene_fn, source_hidden, pad_token_id, max_new_tokens):
    layers = get_decoder_layers(model)
    positions = batch["positions"]
    patch_fn = make_cache_aware_patch_hook(positions, lambda base_vals: intervene_fn(base_vals, source_hidden))
    return run_patched_generate(
        model, layers, layer_idx, patch_fn, batch["base_input_ids"], batch["attention_mask"],
        batch["base_pixel_values"], batch["base_image_grid_thw"], pad_token_id, max_new_tokens,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def discover_jobs(selections_dir, entities):
    """(entity, token_set, dict_method, winners_path) for every fitted
    combination with a winners.json on disk, across the given entities --
    lets a multi-entity run auto-discover everything select_features.py has
    produced instead of the caller respelling each combo."""
    jobs = []
    for entity in entities:
        entity_dir = os.path.join(selections_dir, entity)
        if not os.path.isdir(entity_dir):
            continue
        for name in sorted(os.listdir(entity_dir)):
            winners_path = os.path.join(entity_dir, name, "winners.json")
            if not os.path.exists(winners_path):
                continue
            for method in ("pca", "sae"):
                if name.endswith(f"_{method}"):
                    jobs.append((entity, name[: -len(f"_{method}")], method, winners_path))
                    break
    return jobs


def run_job(model, processor, image_token_id, pad_token_id, entity, token_set, dict_method, winners,
            attributes, entity_dir, items, template_lookup, flat_indices, n_image_tokens, dictionaries_dir,
            split, max_new_tokens, batch_size, limit, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                rec = json.loads(line)
                done.add((rec["attribute"], rec["row_index"]))
            except Exception:
                continue
    out_f = open(out_path, "a")

    for attribute in attributes:
        winner = winners.get(attribute)
        if winner is None:
            print(f"  [{entity}/{token_set}/{dict_method}] attribute={attribute}: no winner, skipping")
            continue
        layer, dict_size, feature_indices = winner["layer"], winner["dict_size"], winner["feature_indices"]
        dict_path = dictionary_path(entity, token_set, layer, dict_method, dict_size, dictionaries_dir)
        dictionary = load_dictionary(dict_path)
        intervene_fn = make_dict_intervention(dictionary, feature_indices)

        tuples_path = os.path.join(entity_dir, "tuples", attribute, f"{split}.jsonl")
        rows = [json.loads(l) for l in open(tuples_path)]
        if limit:
            rows = rows[:limit]
        todo = [r for r in rows if (attribute, r["row_index"]) not in done]
        print(f"  [{entity}/{token_set}/{dict_method}] attribute={attribute} layer={layer} "
              f"dict_size={dict_size} n_features={len(feature_indices)}: {len(todo)} rows remaining of {len(rows)}")

        bs = batch_size
        n_batches = (len(todo) + bs - 1) // bs
        for bi in range(n_batches):
            batch_rows = todo[bi * bs:(bi + 1) * bs]
            batch = build_batch(batch_rows, template_lookup, processor, items, entity_dir,
                                 flat_indices, n_image_tokens, image_token_id)
            source_hidden = cache_source_hidden(model, batch, layer)
            gen_toks = generate_patched(model, batch, layer, intervene_fn, source_hidden,
                                         pad_token_id, max_new_tokens)
            for j, r in enumerate(batch_rows):
                text = processor.tokenizer.decode(gen_toks[j].tolist(), skip_special_tokens=True)
                rec = {"attribute": attribute, "row_index": r["row_index"], "generated_text": text}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (bi + 1) % 10 == 0 or bi + 1 == n_batches:
                print(f"    {attribute}: batch {bi + 1}/{n_batches}", flush=True)

    out_f.close()
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--entity", required=True, help="Comma-separated entities, or 'all'.")
    ap.add_argument("--token_set", default=None,
                     help="Single explicit token_set (requires --dict_method and a single --entity, plus --out). "
                          "Omit together with --dict_method to auto-discover every fitted token_set/dict_method "
                          "combo with a winners.json for each entity, run in one model load (use --out_dir then).")
    ap.add_argument("--dict_method", default=None, choices=[None, "pca", "sae"])
    ap.add_argument("--winners", default=None,
                     help="Explicit winners.json path, single-job mode only. Default: derived from "
                          "--entity/--token_set/--dict_method under --selections_dir.")
    ap.add_argument("--selections_dir", default=os.path.join(REPO_ROOT, "methods", "selections"))
    ap.add_argument("--dictionaries_dir", default=DICTIONARIES_DIR)
    ap.add_argument("--attributes", default=None, help="Comma list; default: every attribute with a winner.")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N rows per attribute (smoke test).")
    ap.add_argument("--out", default=None, help="Single predictions file path -- single-job mode only.")
    ap.add_argument("--out_dir", default=None,
                     help="Directory to write one predictions file per job (named "
                          "<entity>_<token_set>_<dict_method>_predictions.jsonl) -- multi-job mode only.")
    args = ap.parse_args()

    entities = ["flags", "brands", "animals"] if args.entity == "all" else args.entity.split(",")

    if args.token_set and args.dict_method:
        assert len(entities) == 1 and args.out, "single explicit job needs one --entity and --out"
        winners_path = args.winners or os.path.join(
            args.selections_dir, entities[0], f"{args.token_set}_{args.dict_method}", "winners.json")
        jobs = [(entities[0], args.token_set, args.dict_method, winners_path)]
        out_paths = {(entities[0], args.token_set, args.dict_method): args.out}
    else:
        jobs = discover_jobs(args.selections_dir, entities)
        assert args.out_dir, "--out_dir is required in multi-job/auto-discover mode"
        os.makedirs(args.out_dir, exist_ok=True)
        out_paths = {(e, t, m): os.path.join(args.out_dir, f"{e}_{t}_{m}_predictions.jsonl") for e, t, m, _ in jobs}
    print(f"{len(jobs)} job(s): {[(e, t, m) for e, t, m, _ in jobs]}")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map=args.device, dtype=torch.bfloat16)
    model.eval()
    image_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    entity_assets_cache = {}
    for entity, token_set, dict_method_expected, winners_path in jobs:
        if entity not in entity_assets_cache:
            entity_assets_cache[entity] = load_entity_assets(args.vade_root, entity)
        entity_dir, items, loc, template_lookup = entity_assets_cache[entity]
        with open(winners_path) as f:
            wdata = json.load(f)
        dict_method, winners = wdata["dict_method"], wdata["winners"]
        assert dict_method == dict_method_expected, (
            f"{winners_path} says dict_method={dict_method!r}, expected {dict_method_expected!r}")
        attributes = args.attributes.split(",") if args.attributes else list(winners.keys())
        flat_indices = loc["object_token_indices"][token_set]["flat"]
        n_image_tokens = loc["vlm_token_grid"]["n_tokens"]

        print(f"job: entity={entity} token_set={token_set} dict_method={dict_method} attributes={attributes}")
        run_job(model, processor, image_token_id, pad_token_id, entity, token_set, dict_method, winners,
                attributes, entity_dir, items, template_lookup, flat_indices, n_image_tokens,
                args.dictionaries_dir, args.split, args.max_new_tokens, args.batch_size, args.limit,
                out_paths[(entity, token_set, dict_method_expected)])

    print("ALL_JOBS_DONE")


if __name__ == "__main__":
    main()
