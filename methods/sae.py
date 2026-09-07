"""Phase-A activation extraction for VADE's SAE baseline.

This repo (VADE-evals) sits as a sibling to the VADE benchmark repo
(../VADE relative to this file's repo root) and holds anything that
actually loads a model -- VADE itself stays a pure benchmark (data +
method-agnostic scorer), per its own score.py docstring: "The scorer
never touches model internals."

Loads Qwen2.5-VL-7B-Instruct, runs it once per entity image, and records
the residual stream hidden state at EVERY decoder layer, at the token
positions the object (and a dilated ring around it) occupies -- read
generically from each entity's own <entity>/object_location.json, e.g.:

    flags:   flag_only    (8 tokens)   flag_ring1    (24 tokens)
    brands:  logo_only    (36 tokens)  logo_ring1    (64 tokens)
    animals: entity_only  (36 tokens)  entity_ring1  (64 tokens)

--entity defaults to "flags"; pass --entity brands or --entity animals
for the other two built entities (see ITEMS_KEY_BY_ENTITY below --
"compounds" isn't built yet, so it isn't usable here either).

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
    pip install "transformers>=4.49" torch torchvision pillow tqdm accelerate

    torchvision is required even though we only use images -- Qwen2.5-VL's
    AutoProcessor bundles a video processor too, and building it eagerly
    needs torchvision installed regardless. Match it to your installed
    torch/CUDA build (see https://pytorch.org/get-started/locally/) if pip
    doesn't resolve a compatible wheel automatically.

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

    # batched (start small, raise it while watching VRAM headroom):
    python methods/sae.py --entity flags --batch_size 8

    # the other two built entities:
    python methods/sae.py --entity brands --dry_run
    python methods/sae.py --entity animals --dry_run
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageFilter

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a convenience only, not a hard dependency
    class _NoTqdm:
        def __init__(self, *_a, **_kw):
            pass

        def update(self, *_a, **_kw):
            pass

        def set_postfix(self, *_a, **_kw):
            pass

        def close(self):
            pass

    def tqdm(*_a, **_kw):
        return _NoTqdm()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Content is irrelevant to the extracted image-token activations -- see
# module docstring. Keep fixed for reproducibility across runs.
DUMMY_QUESTION = "Describe this image in one sentence."

# Each VADE entity names its items dict differently in ground_truth.json
# (flags predates the other entities and kept its original "countries"
# name even though the concept generalized). "compounds" is deliberately
# absent -- per VADE/compounds/README.md, nothing is built there yet
# (no ground_truth.json exists at all).
ITEMS_KEY_BY_ENTITY = {"flags": "countries", "brands": "brands", "animals": "species"}


def load_entity_metadata(vade_root, entity, token_set_names=None):
    """Return (item_codes, image_paths, token_sets, grid) for a VADE entity,
    read from its ground_truth.json + object_location.json.

    token_sets is an ordered dict {set_name: flat_indices} -- e.g. for
    flags, {"flag_only": [64,65,...], "flag_ring1": [51,52,...]}.
    token_set_names, if given, restricts to just those set names
    (must exist in object_location.json's object_token_indices);
    None means "all sets object_location.json defines for this entity".
    """
    entity_dir = os.path.join(vade_root, "data", entity)
    if not os.path.isdir(entity_dir):
        raise FileNotFoundError(
            f"No entity directory at {entity_dir!r}. Pass --vade_root to point at "
            f"your VADE checkout (defaulting to sibling dir {DEFAULT_VADE_ROOT!r}).")
    gt_path = os.path.join(entity_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"No ground_truth.json for entity {entity!r} at {gt_path!r} -- this VADE entity "
            f"likely isn't built yet (e.g. 'compounds' has no data as of VADE/compounds/README.md). "
            f"Known-buildable entities: {list(ITEMS_KEY_BY_ENTITY)}.")
    with open(gt_path) as f:
        gt = json.load(f)
    with open(os.path.join(entity_dir, "object_location.json")) as f:
        loc = json.load(f)

    items_key = ITEMS_KEY_BY_ENTITY.get(entity)
    if items_key is None or items_key not in gt:
        raise ValueError(
            f"Don't know the ground_truth.json items-dict key for entity {entity!r} "
            f"(known: {ITEMS_KEY_BY_ENTITY}). If this is a new, real VADE entity, add its "
            f"items-dict key name to ITEMS_KEY_BY_ENTITY above.")
    items = gt[items_key]  # {item_id: {..., "image": "images/XX.png", ...}}
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


def build_augmentation_transform(seed):
    """A seeded, PURELY PIXEL-LEVEL perturbation -- color jitter + slight
    Gaussian noise + a coin-flip of slight blur -- never geometric (no
    crop/rotate/flip/resize/translate), so the object's exact pixel
    footprint, and therefore every downstream token-position assumption
    (object_location.json's flat indices, find_image_token_positions),
    stays exactly valid unchanged.

    This exists only to give PCA/SAE dictionary FITTING (fit_dictionaries.
    py's --augmented_pool) a bigger, more varied pool of activations than
    an entity's real handful of images (84-130) can offer on their own --
    it is never used for feature selection or scoring, both of which stay
    tied to the real images and their real ground-truth labels."""
    from torchvision import transforms as T

    rng = np.random.RandomState(seed)
    jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)
    torch_seed = int(rng.randint(0, 2 ** 31 - 1))
    blur_radius = float(rng.uniform(0.15, 0.6)) if rng.rand() < 0.5 else 0.0
    noise_std = float(rng.uniform(1.0, 4.0))  # 0-255 pixel units

    def _transform(img):
        torch.manual_seed(torch_seed)
        out = jitter(img)
        if blur_radius > 0:
            out = out.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        arr = np.asarray(out).astype(np.float32)
        arr = np.clip(arr + rng.normal(0, noise_std, size=arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    return _transform


def extract_batch(model, processor, image_paths, token_sets, grid, device, image_token_id, transforms=None):
    """Returns {set_name: tensor[batch, num_layers+1, n_tokens_in_set, hidden_dim]},
    all sliced from a single batched forward pass over these images.

    Batching is essentially free here memory-wise: every image in an
    entity shares one fixed canvas size and DUMMY_QUESTION is fixed too,
    so every sequence in the batch has identical length -- no padding,
    no attention-mask bookkeeping. The per-item marginal cost of a larger
    batch is just a few tensors of shape [batch, ~150, hidden_dim] per
    layer (order of 1MB/image/layer) -- utterly dwarfed by the model
    weights themselves, which dominate VRAM use almost entirely. So the
    real question for --batch_size isn't "how much does batching cost"
    but "does the model comfortably fit at all" -- see the module
    docstring / project notes for that discussion.

    transforms, if given: a list (aligned with image_paths) of PIL->PIL
    callables applied right after loading, before the processor ever sees
    the image -- see build_augmentation_transform. Purely pixel-level
    (color/noise/blur), never geometric, so the object's pixel footprint
    and hence every downstream token-position assumption (object_location.
    json, find_image_token_positions) stays exactly valid unchanged.
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]
    if transforms is not None:
        images = [t(img) for t, img in zip(transforms, images)]
    messages_batch = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DUMMY_QUESTION}]}]
        for _ in images
    ]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
    inputs = processor(text=texts, images=images, return_tensors="pt").to(device)

    n_expected = grid["grid_rows"] * grid["grid_cols"]
    input_ids = inputs["input_ids"]  # [batch, seq_len]
    # Fixed canvas + fixed prompt text -> every row should have identical
    # sequence length and identical image-token positions. Verify rather
    # than assume, so a future non-uniform image size fails loudly instead
    # of silently misattributing tokens between images.
    row0_positions = find_image_token_positions(input_ids[0], image_token_id, n_expected)
    for row in range(1, input_ids.shape[0]):
        row_positions = find_image_token_positions(input_ids[row], image_token_id, n_expected)
        if not torch.equal(row_positions, row0_positions):
            raise ValueError(
                "Image-token positions differ within a batch -- these images may not "
                "share identical canvas size/template. Reduce --batch_size to 1, or "
                "check the entity's renders for a size mismatch."
            )

    with torch.no_grad():
        # logits_to_keep=1: only outputs.hidden_states is ever read below --
        # without this, the model computes a full [batch, seq_len, vocab_size]
        # lm_head projection (vocab_size=151936 >> hidden_dim=3584) at EVERY
        # sequence position for nothing, dominating the forward pass's cost
        # for no reason (measured: ~30min for 64 images at batch_size=32
        # without this, vs. the sub-second/image a hidden-states-only forward
        # pass should cost -- see build_external_flag_pool.py's smoke test).
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)

    # outputs.hidden_states: tuple of (num_decoder_layers + 1) tensors, each
    # [batch, seq_len, hidden_dim]. Index 0 = embedding output (pre-layer-1);
    # index i = residual stream AFTER decoder layer i.
    result = {}
    for set_name, flat_indices in token_sets.items():
        object_positions = row0_positions[flat_indices]  # row-major -> object_location.json's flat order
        per_layer = [
            layer_hs[:, object_positions, :].to(torch.float32).cpu()  # [batch, n_tokens, hidden_dim]
            for layer_hs in outputs.hidden_states
        ]
        result[set_name] = torch.stack(per_layer, dim=1)  # [batch, num_layers+1, n_tokens, hidden_dim]
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
    ap.add_argument("--batch_size", type=int, default=1,
                     help="Images per forward pass. Safe to raise well above 1 -- every image in an "
                          "entity shares one fixed canvas size, so batching adds no padding complexity "
                          "and costs little extra VRAM (see extract_batch's docstring). Start small and "
                          "watch nvidia-smi/VRAM headroom, since the model weights themselves are what "
                          "actually dominate GPU memory here, not the batch.")
    ap.add_argument("--dry_run", action="store_true",
                     help="Skip loading the model entirely -- just validate that the entity's "
                          "images/ground_truth.json/object_location.json line up.")
    ap.add_argument("--augment_variants", type=int, default=0,
                     help="If >0, build an AUGMENTED POOL instead of the real per-image activation file: "
                          "each real image contributes this many pixel-perturbed variants (see "
                          "build_augmentation_transform), no unperturbed copy included. Meant only to give "
                          "fit_dictionaries.py a bigger training pool -- output defaults to a distinctly-named "
                          "*_augmented_pool_x<N>.pt file, and item_codes become '<code>__augI' pseudo-codes "
                          "(never matched against ground_truth.json).")
    ap.add_argument("--augment_seed", type=int, default=0)
    args = ap.parse_args()

    token_set_names = args.token_sets.split(",") if args.token_sets else None
    codes, image_paths, token_sets, grid = load_entity_metadata(args.vade_root, args.entity, token_set_names)
    if args.limit:
        codes, image_paths = codes[:args.limit], image_paths[:args.limit]

    transforms_by_idx = None
    if args.augment_variants > 0:
        K = args.augment_variants
        aug_codes, aug_paths, aug_transforms = [], [], []
        for i, (code, path) in enumerate(zip(codes, image_paths)):
            for v in range(K):
                aug_codes.append(f"{code}__aug{v}")
                aug_paths.append(path)
                aug_transforms.append(build_augmentation_transform(seed=args.augment_seed * 1_000_003 + i * K + v))
        codes, image_paths, transforms_by_idx = aug_codes, aug_paths, aug_transforms
        print(f"--augment_variants={K}: expanded to {len(codes)} pixel-perturbed image variants "
              f"(no unperturbed copies) for dictionary-training-pool extraction")

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
    pairs = list(zip(codes, image_paths))
    pbar = tqdm(total=len(pairs), unit="img", desc=f"extracting ({args.entity})")
    idx = 0
    for batch_start in range(0, len(pairs), args.batch_size):
        batch = pairs[batch_start:batch_start + args.batch_size]
        batch_codes, batch_paths = zip(*batch)
        batch_transforms = (transforms_by_idx[batch_start:batch_start + args.batch_size]
                             if transforms_by_idx is not None else None)
        per_set = extract_batch(model, processor, list(batch_paths), token_sets, grid, device, image_token_id,
                                 transforms=batch_transforms)
        bs = len(batch)
        for name, acts in per_set.items():  # acts: [bs, num_layers+1, n_tokens, hidden_dim]
            if all_layers_by_set[name] is None:
                all_layers_by_set[name] = torch.zeros((len(codes),) + tuple(acts.shape[1:]), dtype=torch.float32)
            all_layers_by_set[name][idx:idx + bs] = acts
        idx += bs
        pbar.set_postfix(codes=f"{batch_codes[0]}..{batch_codes[-1]}")
        pbar.update(bs)
    pbar.close()

    model_tag = args.model_id.split('/')[-1]
    if args.augment_variants > 0:
        default_name = f"{args.entity}_{model_tag}_augmented_pool_x{args.augment_variants}.pt"
    else:
        default_name = f"{args.entity}_{model_tag}_all_layers.pt"
    output = args.output or os.path.join(REPO_ROOT, "methods", "activations", default_name)
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
        "is_augmented_pool": args.augment_variants > 0,  # item_codes are '<code>__augI' pseudo-codes if so --
                                                          # never matched against ground_truth.json; training-only.
        "augment_variants": args.augment_variants,
    }, output)
    shapes = {name: tuple(t.shape) for name, t in all_layers_by_set.items()}
    print(f"wrote {output}  shapes={shapes}")


if __name__ == "__main__":
    main()
