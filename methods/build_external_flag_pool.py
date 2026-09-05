"""Builds a flags-only EXTERNAL training pool for PCA/SAE dictionary
fitting from a real "flags in the wild" photo dataset (Kaggle:
sjetley/country-flags-in-the-wild -- github.com/... via
https://www.kaggle.com/datasets/sjetley/country-flags-in-the-wild),
instead of (or alongside) sae.py's synthetic --augment_variants pool.

Why compositing is required
-----------------------------
VADE's own token-position extraction assumes every image is on a fixed
336x336 canvas with the flag occupying an EXACT pixel bbox (see
flags/object_location.json) -- object_location.json's flat token indices
are only valid for that convention. The Kaggle photos are real-world
crops at wildly varying resolutions/aspect ratios (verified: 100x67 up to
450x291 in a random sample) with no such convention. So each wild photo
is resized (stretched, not cropped -- precise real-world proportions
don't matter for an unsupervised dictionary's training data) to exactly
fill VADE's own bbox, pasted onto a copy of VADE's own background color
(sampled from a real VADE flag image, not hardcoded), at VADE's own bbox
coordinates -- read from object_location.json, not hardcoded here either,
so this keeps working if flags' geometry ever changes. That makes every
composited image drop-in equivalent to a real VADE render as far as
sae.py's extraction is concerned: same canvas, same bbox, same token
positions -- only the pixel CONTENT inside the bbox is real-world instead
of a clean icon render.

No country-label mapping needed
----------------------------------
The zip's images are organized as verified_flags_{train,test}/<numeric
class id>/<n>.png with no bundled id->country mapping. That's fine here:
dictionary fitting is entirely unsupervised (see fit_dictionaries.py's
module docstring) -- this script never needs to know which photo is
which country, just that it's photographic flag content. Images are
drawn uniformly at random across the whole zip regardless of class id.

Output schema matches sae.py's --augment_variants pool exactly (same
keys, is_augmented_pool=True) so fit_dictionaries.py's --augmented_pool_paths
can merge it in identically -- item_codes are 'kaggle__<zip member path>'
pseudo-codes, never matched against ground_truth.json.

Usage
-----
    # download once (no Kaggle API credentials needed for this public
    # dataset's direct CDN redirect, verified working from this environment):
    curl -sL -o flags_kaggle.zip \\
        "https://www.kaggle.com/api/v1/datasets/download/sjetley/country-flags-in-the-wild"

    python methods/build_external_flag_pool.py --zip_path flags_kaggle.zip --n_images 2000

    # then fit dictionaries against BOTH the real images and this pool:
    python methods/fit_dictionaries.py --entity flags --method both \\
        --augmented_pool_paths methods/activations/flags_Qwen2.5-VL-7B-Instruct_augmented_pool_kaggle.pt
"""
import argparse
import io
import json
import os
import random
import zipfile

import torch
from PIL import Image

from features import ITEMS_KEY_BY_ENTITY, REPO_ROOT
from sae import DEFAULT_MODEL_ID, extract_batch, resolve_image_token_id

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


def load_flags_geometry(vade_root):
    with open(os.path.join(vade_root, "flags", "object_location.json")) as f:
        loc = json.load(f)
    with open(os.path.join(vade_root, "flags", "ground_truth.json")) as f:
        gt = json.load(f)
    first_code = sorted(gt[ITEMS_KEY_BY_ENTITY["flags"]])[0]
    first_image = os.path.join(vade_root, "flags", gt[ITEMS_KEY_BY_ENTITY["flags"]][first_code]["image"])
    background_rgb = Image.open(first_image).convert("RGB").getpixel((0, 0))
    canvas_size = (loc["canvas_px"]["width"], loc["canvas_px"]["height"])
    bbox = loc["object_bbox_px"]
    return loc, canvas_size, bbox, background_rgb


def composite_into_canvas(wild_img, canvas_size, bbox, background_rgb):
    canvas = Image.new("RGB", canvas_size, background_rgb)
    resized = wild_img.convert("RGB").resize((bbox["width"], bbox["height"]), Image.BILINEAR)
    canvas.paste(resized, (bbox["x"], bbox["y"]))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=os.environ.get("VADE_ROOT") or
                     os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE")))
    ap.add_argument("--zip_path", required=True, help="Path to the downloaded Kaggle dataset zip.")
    ap.add_argument("--token_sets", default=None, help="Default: every set in flags/object_location.json.")
    ap.add_argument("--n_images", type=int, default=2000,
                     help="Random subsample size out of the zip's ~19k images (kept modest -- this is a "
                          "training-pool size question, not a need to use literally every photo).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Further cap after subsampling (smoke test).")
    ap.add_argument("--composited_preview_dir", default=None,
                     help="If set, also save a few composited canvases here so you can eyeball the result.")
    ap.add_argument("--output", default=None,
                     help="Default: methods/activations/flags_<model>_augmented_pool_kaggle.pt")
    ap.add_argument("--dry_run", action="store_true", help="Skip the model -- just build/preview composited canvases.")
    args = ap.parse_args()

    loc, canvas_size, bbox, background_rgb = load_flags_geometry(args.vade_root)
    token_set_names = args.token_sets.split(",") if args.token_sets else list(loc["object_token_indices"])
    token_sets = {name: loc["object_token_indices"][name]["flat"] for name in token_set_names}
    grid = loc["vlm_token_grid"]
    print(f"canvas={canvas_size} bbox={bbox} background_rgb={background_rgb} token_sets={list(token_sets)}")

    z = zipfile.ZipFile(args.zip_path)
    members = [n for n in z.namelist() if n.endswith(".png")]
    rng = random.Random(args.seed)
    rng.shuffle(members)
    members = members[:args.n_images]
    if args.limit:
        members = members[:args.limit]
    print(f"{len(z.namelist())} total images in zip, using {len(members)} (seed={args.seed})")

    if args.composited_preview_dir:
        os.makedirs(args.composited_preview_dir, exist_ok=True)
        for m in members[:8]:
            wild = Image.open(io.BytesIO(z.read(m)))
            composite_into_canvas(wild, canvas_size, bbox, background_rgb).save(
                os.path.join(args.composited_preview_dir, m.replace("/", "_")))
        print(f"wrote preview composites -> {args.composited_preview_dir}")

    if args.dry_run:
        return

    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    print(f"Loading {args.model_id} on {device} (transformers={transformers.__version__}) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype_map[args.dtype]).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    image_token_id = resolve_image_token_id(model, processor)

    # Composite every member canvas up front (cheap, CPU-only) so extract_batch
    # can keep using its existing image_paths-based interface unchanged --
    # written to a scratch dir alongside the output, cleaned up at the end.
    scratch_dir = os.path.join(os.path.dirname(os.path.abspath(
        args.output or os.path.join(REPO_ROOT, "methods", "activations", "x"))), "_kaggle_composite_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    codes, image_paths = [], []
    for m in tqdm(members, desc="compositing"):
        wild = Image.open(io.BytesIO(z.read(m)))
        canvas = composite_into_canvas(wild, canvas_size, bbox, background_rgb)
        path = os.path.join(scratch_dir, m.replace("/", "_"))
        canvas.save(path)
        codes.append(f"kaggle__{m}")
        image_paths.append(path)

    all_layers_by_set = {name: None for name in token_sets}
    pairs = list(zip(codes, image_paths))
    idx = 0
    for batch_start in tqdm(range(0, len(pairs), args.batch_size), desc="extracting"):
        batch = pairs[batch_start:batch_start + args.batch_size]
        batch_codes, batch_paths = zip(*batch)
        per_set = extract_batch(model, processor, list(batch_paths), token_sets, grid, device, image_token_id)
        bs = len(batch)
        for name, acts in per_set.items():
            if all_layers_by_set[name] is None:
                all_layers_by_set[name] = torch.zeros((len(codes),) + tuple(acts.shape[1:]), dtype=torch.float32)
            all_layers_by_set[name][idx:idx + bs] = acts
        idx += bs

    for p in image_paths:
        os.remove(p)
    os.rmdir(scratch_dir)

    output = args.output or os.path.join(REPO_ROOT, "methods", "activations",
                                          f"flags_{args.model_id.split('/')[-1]}_augmented_pool_kaggle.pt")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    first_set = next(iter(all_layers_by_set.values()))
    torch.save({
        "activations_by_token_set": all_layers_by_set,
        "item_codes": codes,
        "layer_convention": "index 0 = embedding output; index i = residual stream after decoder layer i",
        "token_flat_indices_by_set": token_sets,
        "grid": grid,
        "model_id": args.model_id,
        "vade_root": args.vade_root,
        "entity": "flags",
        "hidden_dim": first_set.shape[-1],
        "num_layers": first_set.shape[1] - 1,
        "is_augmented_pool": True,
        "source": "kaggle:sjetley/country-flags-in-the-wild",
        "n_source_images": len(members),
    }, output)
    shapes = {name: tuple(t.shape) for name, t in all_layers_by_set.items()}
    print(f"wrote {output}  shapes={shapes}")


if __name__ == "__main__":
    main()
