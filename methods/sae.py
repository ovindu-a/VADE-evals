"""Phase-A activation extraction for VADE's SAE baseline.

This repo (VADE-evals) sits as a sibling to the VADE benchmark repo
(../VADE relative to this file's repo root) and holds anything that
actually loads a model -- VADE itself stays a pure benchmark (data +
method-agnostic scorer), per its own score.py docstring: "The scorer
never touches model internals."

Loads Qwen2.5-VL-7B-Instruct, runs it once per entity image (flags by
default), and records the residual stream hidden state at EVERY decoder
layer, at the token positions the object (and a dilated ring around it)
occupies -- e.g. flags' "flag_only" (8 tokens) and "flag_ring1" (24
tokens) sets from VADE's <entity>/object_location.json.

This produces the *raw activation corpus* an SAE (or PCA) gets trained
on downstream -- it does not itself train anything. Feature selection
(L1-SVC + SelectFromModel over the SAE's/PCA's featurized space) is a
separate, later script that consumes this file's output.

Why no real question is in the prompt
--------------------------------------
Qwen2.5-VL's LLM backbone is a causal decoder, and the image tokens sit
*before* any question text in the sequence. Causal self-attention means
a token's activation can only depend on tokens at or before its own
position, so the object's tokens are numerically identical no matter
what question follows them. We still need *some* well-formed chat-
template text to get a valid input (Qwen2.5-VL's processor expects a
user turn), but its content is provably irrelevant to what's being
extracted here -- see DUMMY_QUESTION below. Kept fixed across every run
so results are directly comparable.

Row-major token ordering assumption
------------------------------------
Qwen2.5-VL lays an image's merged-patch tokens into the input sequence
in row-major order (flat_index = row * grid_cols + col), matching
object_location.json's own documented flat_index_formula -- the flags
build pipeline already verified this correspondence by diffing
pixel_values between two images differing only inside the flag region
(see flags/object_location.json's note field). We rely on that same
ordering to map object_location.json's flat token indices onto the
image-placeholder positions found in input_ids.

Multiple token sets, one forward pass
--------------------------------------
"flag_only" (8 tokens) is a strict subset of "flag_ring1" (24 tokens,
flag_only dilated by one merged-token cell in every direction -- pixel
bbox [84,252)x[112,224) vs flag_only's [112,224)x[140,196)). Both sets
are sliced out of the *same* forward pass's hidden_states -- extracting
both costs one extra tensor index per layer, not an extra model call.

Where VADE's data is found
----------------------------
Defaults to a sibling ../VADE directory (relative to this repo's root),
matching this repo's own layout. Override with --vade_root or the
VADE_ROOT environment variable if your checkout lives somewhere else.

Requirements (install on the machine actually running this -- NOT
expected to run on a laptop without a real GPU; this is a 7B-parameter
model and we're pulling hidden states at every layer):
    pip install "transformers>=4.49" torch pillow tqdm accelerate

Usage
-----
    # sanity-check images/metadata line up, without loading the model:
    python methods/sae.py --entity flags --dry_run

    # smoke test on a GPU box, first 3 images only:
    python methods/sae.py --entity flags --limit 3

    # full run (both flag_only and flag_ring1 by default):
    python methods/sae.py --entity flags \
        --output methods/activations/flags_qwen2.5-vl-7b_all_layers.pt

    # only one token set, or a VADE checkout that isn't a sibling dir:
    python methods/sae.py --entity flags --token_sets flag_only
    python methods/sae.py --entity flags --vade_root /path/to/VADE
"""
import argparse
import json
import os

import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience only, not a hard dependency
    def tqdm(iterable, **_kwargs):
        return iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Content is irrelevant to the extracted image-token activations -- see
# module docstring. Keep fixed for reproducibility across runs.
DUMMY_QUESTION = "Describe this image in one sentence."


def load_entity_metadata(vade_root, entity, token_set_names=None):
    """Return (item_codes, image_paths, token_sets, grid) for a VADE entity,
    read from its ground_truth.json + object_location.json.

    token_sets is an ordered dict {set_name: flat_indices} -- e.g. for
    flags, {"flag_only": [64,65,...], "flag_ring1": [51,52,...]}.
    token_set_names, if given, restricts to just those set names
    (must exist in object_location.json's object_token_indices);
    None means "all sets object_location.json defines for this entity".
    """
    entity_dir = os.path.join(vade_root, entity)
    if not os.path.isdir(entity_dir):
        raise FileNotFoundError(
            f"No entity directory at {entity_dir!r}. Pass --vade_root to point at "
            f"your VADE checkout (defaulting to sibling dir {DEFAULT_VADE_ROOT!r}).")
    with open(os.path.join(entity_dir, "ground_truth.json")) as f:
        gt = json.load(f)
    with open(os.path.join(entity_dir, "object_location.json")) as f:
        loc = json.load(f)

    items = gt["countries"]  # {iso_code: {..., "image": "images/XX.png", ...}}
    codes = sorted(items)
    image_paths = [os.path.join(entity_dir, items[c]["image"]) for c in codes]

    available = loc["object_token_indices"]
    names = token_set_names or list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"Unknown token set(s) {unknown} for entity {entity!r}; "
                          f"available: {list(available)}")
    token_sets = {name: available[name]["flat"] for name in names}

    grid = loc["vlm_token_grid"]
    return codes, image_paths, token_sets, grid


def resolve_image_token_id(model, processor):
    """Qwen2.5-VL exposes the image-placeholder token id on the model
    config; fall back to the processor/tokenizer if that ever moves."""
    tok_id = getattr(model.config, "image_token_id", None)
    if tok_id is not None:
        return tok_id
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    return processor.tokenizer.convert_tokens_to_ids(image_token)


def find_image_token_positions(input_ids, image_token_id, n_expected):
    """Sequence positions (ascending = row-major merged-patch order) that
    hold this image's placeholder tokens within the full templated input."""
    positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() != n_expected:
        raise ValueError(
            f"Expected {n_expected} image-placeholder tokens (grid_rows*grid_cols "
            f"from object_location.json), found {positions.numel()}. Check that "
            f"the model/processor's image_token_id and the image's rendered size "
            f"still match object_location.json's assumptions."
        )
    return positions


def extract_one_image(model, processor, image_path, token_sets, grid, device, image_token_id):
    """Returns {set_name: tensor[num_layers+1, n_tokens_in_set, hidden_dim]},
    all sliced from a single forward pass over this image."""
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": DUMMY_QUESTION}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    n_expected = grid["grid_rows"] * grid["grid_cols"]
    positions = find_image_token_positions(inputs["input_ids"][0], image_token_id, n_expected)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # outputs.hidden_states: tuple of (num_decoder_layers + 1) tensors, each
    # [1, seq_len, hidden_dim]. Index 0 = embedding output (pre-layer-1);
    # index i = residual stream AFTER decoder layer i.
    result = {}
    for set_name, flat_indices in token_sets.items():
        object_positions = positions[flat_indices]  # row-major -> object_location.json's flat order
        per_layer = [
            layer_hs[0, object_positions, :].to(torch.float32).cpu()
            for layer_hs in outputs.hidden_states
        ]
        result[set_name] = torch.stack(per_layer, dim=0)  # [num_layers+1, n_tokens, hidden_dim]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT,
                     help="Path to a VADE checkout. Defaults to a sibling ../VADE dir, "
                          "or the VADE_ROOT env var if set.")
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--token_sets", default=None,
                     help="Comma-separated object_token_indices set names to extract "
                          "(e.g. 'flag_only,flag_ring1'). Default: every set object_location.json defines.")
    ap.add_argument("--output", default=None,
                     help="Defaults to methods/activations/<entity>_<model>_all_layers.pt (in this repo)")
    ap.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', or 'mps'")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--attn_implementation", default=None,
                     help="e.g. 'flash_attention_2' if installed; left to the model's default otherwise")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N images (smoke test)")
    ap.add_argument("--dry_run", action="store_true",
                     help="Skip loading the model entirely -- just validate that the entity's "
                          "images/ground_truth.json/object_location.json line up.")
    args = ap.parse_args()

    token_set_names = args.token_sets.split(",") if args.token_sets else None
    codes, image_paths, token_sets, grid = load_entity_metadata(args.vade_root, args.entity, token_set_names)
    if args.limit:
        codes, image_paths = codes[:args.limit], image_paths[:args.limit]

    set_summary = ", ".join(f"{name}={len(idx)}" for name, idx in token_sets.items())
    print(f"VADE root: {args.vade_root}")
    print(f"{args.entity}: {len(codes)} images, token sets: {set_summary} "
          f"(grid {grid['grid_rows']}x{grid['grid_cols']}={grid['n_tokens']})")

    if args.dry_run:
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            print(f"MISSING {len(missing)} image files, e.g. {missing[:3]}")
        else:
            print("All image files found. (Model not loaded -- rerun without --dry_run to actually extract.)")
        return

    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype_map = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model_id} on {device} (dtype={args.dtype}, transformers={transformers.__version__}) ...")
    model_kwargs = {"torch_dtype": dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    image_token_id = resolve_image_token_id(model, processor)

    all_layers_by_set = {name: None for name in token_sets}
    for i, (_code, path) in enumerate(tqdm(list(zip(codes, image_paths)), desc=f"extracting ({args.entity})")):
        per_set = extract_one_image(model, processor, path, token_sets, grid, device, image_token_id)
        for name, acts in per_set.items():
            if all_layers_by_set[name] is None:
                all_layers_by_set[name] = torch.zeros((len(codes),) + tuple(acts.shape), dtype=torch.float32)
            all_layers_by_set[name][i] = acts

    output = args.output or os.path.join(
        REPO_ROOT, "methods", "activations",
        f"{args.entity}_{args.model_id.split('/')[-1]}_all_layers.pt")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    first_set = next(iter(all_layers_by_set.values()))
    torch.save({
        "activations_by_token_set": all_layers_by_set,  # {name: [n_images, n_layers+1, n_tokens, hidden_dim]}
        "item_codes": codes,                            # activations[i] <-> item_codes[i], for any set
        "layer_convention": "index 0 = embedding output; index i = residual stream after decoder layer i",
        "token_flat_indices_by_set": token_sets,         # which merged-grid tokens each set's rows are, in order
        "grid": grid,
        "model_id": args.model_id,
        "vade_root": args.vade_root,
        "entity": args.entity,
        "hidden_dim": first_set.shape[-1],
        "num_layers": first_set.shape[1] - 1,
    }, output)
    shapes = {name: tuple(t.shape) for name, t in all_layers_by_set.items()}
    print(f"wrote {output}  shapes={shapes}")


if __name__ == "__main__":
    main()
