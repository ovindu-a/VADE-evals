"""Builds a flags-only EXTERNAL training pool for PCA/SAE dictionary
fitting from a real "flags in the wild" photo dataset (Kaggle:
sjetley/country-flags-in-the-wild -- github.com/... via
https://www.kaggle.com/datasets/sjetley/country-flags-in-the-wild),
instead of (or alongside) sae.py's synthetic --augment_variants pool.

Why compositing is required, and why full-canvas is the default
-------------------------------------------------------------------
VADE's own images put the flag in a small bbox (112x56 out of a 336x336
canvas) surrounded by flat gray -- object_location.json's flag_only/
flag_ring1 token sets exist BECAUSE that's genuinely the only place real
flag content lives in a VADE render, so restricting extraction to those
positions there loses nothing. The Kaggle photos are the opposite: real-
world crops that are themselves already tight, full-frame flag shots
(verified directly: sampled images show the flag content filling the
ENTIRE frame edge to edge, no background border -- see this file's
compositing preview). Squeezing one of those into VADE's small bbox
(this script's original approach) would throw away information for no
reason -- it artificially recreates the "small patch on a gray field"
structure of a VADE render even though the source photo never looked like
that, and if it were fed to VADE's own flag_only/flag_ring1 token
positions afterward it'd genuinely be correct, but wasteful, extraction
(most of the composited canvas really would just be gray, same as the
old approach's memmap files showed).

--fill_mode full_canvas (the default) instead resizes each wild photo
(stretched, not cropped -- precise real-world proportions don't matter
for an unsupervised dictionary's training data) to fill the model's
FULL 336x336 canvas, no bbox, no gray padding -- then extracts ALL 144
merged-grid token positions (the "full_image" token set defined below,
not read from object_location.json since VADE itself never needs it) as
real, meaningful training rows, since real flag content now genuinely
occupies every one of them. --fill_mode bbox keeps the original
behavior (resize into VADE's exact bbox, gray elsewhere, flag_only/
flag_ring1 token sets only) if you want it for comparison.

No country-label mapping needed
----------------------------------
The zip's images are organized as verified_flags_{train,test}/<numeric
class id>/<n>.png with no bundled id->country mapping. That's fine here:
dictionary fitting is entirely unsupervised (see fit_dictionaries.py's
module docstring) -- this script never needs to know which photo is
which country, just that it's photographic flag content. Images are
drawn uniformly at random across the whole zip regardless of class id.

Output schema matches sae.py's --augment_variants pool for small pools, but
for anything sized to cover the dataset's full ~19k images, the per-token-set
activation tensors are written directly to disk-backed memmap .npy files
(one per token_set, next to the small metadata .pt this script's --output
names) as each batch completes, instead of being accumulated in one giant
in-RAM tensor first -- this box has 15GB RAM (~7GB free), and the naive
approach OOMs immediately past a few hundred images (measured: requesting
all ~19k images tries to allocate 63GB for flag_only ALONE, before even
touching flag_ring1's 3x-larger token count). features.load_pool_file()
reconstructs activations_by_token_set as read-only np.memmap arrays from
these sibling files, so fit_dictionaries.py's get_layer_slice() only ever
pages in ONE layer's worth of data at a time (order of a few GB, not the
~235GB the full dataset's activations take up on disk across all 29
layers/both token sets) -- see that function for the memmap-vs-tensor
handling. item_codes are 'kaggle__<zip member path>' pseudo-codes, never
matched against ground_truth.json.

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
import shutil
import tempfile
import zipfile

import numpy as np
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
    with open(os.path.join(vade_root, "data", "flags", "object_location.json")) as f:
        loc = json.load(f)
    with open(os.path.join(vade_root, "data", "flags", "ground_truth.json")) as f:
        gt = json.load(f)
    first_code = sorted(gt[ITEMS_KEY_BY_ENTITY["flags"]])[0]
    first_image = os.path.join(vade_root, "data", "flags", gt[ITEMS_KEY_BY_ENTITY["flags"]][first_code]["image"])
    background_rgb = Image.open(first_image).convert("RGB").getpixel((0, 0))
    canvas_size = (loc["canvas_px"]["width"], loc["canvas_px"]["height"])
    bbox = loc["object_bbox_px"]
    return loc, canvas_size, bbox, background_rgb


def composite_into_bbox(wild_img, canvas_size, bbox, background_rgb):
    """Original behavior: resize into VADE's small bbox, gray elsewhere.
    Only real flag content is the bbox region -- see module docstring for
    why --fill_mode full_canvas is the better fit for this dataset."""
    canvas = Image.new("RGB", canvas_size, background_rgb)
    resized = wild_img.convert("RGB").resize((bbox["width"], bbox["height"]), Image.BILINEAR)
    canvas.paste(resized, (bbox["x"], bbox["y"]))
    return canvas


def composite_full_canvas(wild_img, canvas_size):
    """Default behavior: the wild photo already IS a tight, full-frame flag
    shot (verified on real samples), so just resize it (stretched, not
    cropped) to fill the entire canvas -- no bbox, no gray padding, every
    merged-grid token position holds real flag content."""
    return wild_img.convert("RGB").resize(canvas_size, Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=os.environ.get("VADE_ROOT") or
                     os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE")))
    ap.add_argument("--zip_path", required=True, help="Path to the downloaded Kaggle dataset zip.")
    ap.add_argument("--zip_prefix", default=None,
                     help="Only use zip members starting with this path prefix -- e.g. 'verified_flags_test/' "
                          "to pull only from that split (the dataset's top-level folders are "
                          "verified_flags_train/ and verified_flags_test/; the Kaggle web UI's '?select=' link "
                          "just scrolls the file browser there, it's not a separate download). Default: every "
                          ".png in the zip, both splits mixed -- harmless here since this pool is used purely "
                          "unsupervised (no labels, no train/test leakage concern), but pass this to keep a "
                          "split's pool saved and inspectable separately from the mixed one.")
    ap.add_argument("--fill_mode", default="full_canvas", choices=["full_canvas", "bbox"],
                     help="'full_canvas' (default): resize each wild photo to fill the entire 336x336 canvas, "
                          "extract all 144 token positions (see module docstring for why -- these photos are "
                          "already tight, full-frame flag shots). 'bbox': original behavior, resize into VADE's "
                          "small flag bbox with gray elsewhere, extract only flag_only/flag_ring1 positions.")
    ap.add_argument("--token_sets", default=None,
                     help="Default: 'full_image' (all 144 positions) for --fill_mode full_canvas, or every set "
                          "in flags/object_location.json for --fill_mode bbox.")
    ap.add_argument("--n_images", type=int, default=None,
                     help="Random subsample size out of the zip's ~19k images. Default: use every matching "
                          "image (no subsampling) -- safe now that activations are written to disk-backed "
                          "memmaps instead of one in-RAM tensor. Pass a smaller number for a quick smoke test. "
                          "NOTE: --fill_mode full_canvas stores 144 tokens/image vs bbox's 8+24=32 combined, "
                          "so it needs ~4.5x the disk of an equivalent bbox-mode run -- see the printed "
                          "estimated-disk-usage line before a large run.")
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
    grid = loc["vlm_token_grid"]
    if args.fill_mode == "full_canvas":
        default_token_sets = {"full_image": list(range(grid["n_tokens"]))}
    else:
        default_token_sets = {name: loc["object_token_indices"][name]["flat"] for name in loc["object_token_indices"]}
    token_set_names = args.token_sets.split(",") if args.token_sets else list(default_token_sets)
    token_sets = {name: default_token_sets[name] for name in token_set_names}
    print(f"fill_mode={args.fill_mode} canvas={canvas_size} bbox={bbox} background_rgb={background_rgb} "
          f"token_sets={ {k: len(v) for k, v in token_sets.items()} }")
    # Qwen2.5-VL-7B-Instruct-specific constants (29 hidden-state layers incl.
    # embedding output, hidden_dim=3584) -- just for a rough disk estimate
    # before a large run; the actual write uses whatever shape extract_batch
    # returns, so this estimate being off for a different --model_id is harmless.
    est_bytes_per_image = 29 * sum(len(v) for v in token_sets.values()) * 3584 * 4

    z = zipfile.ZipFile(args.zip_path)
    members = [n for n in z.namelist() if n.endswith(".png")]
    if args.zip_prefix:
        members = [n for n in members if n.startswith(args.zip_prefix)]
        if not members:
            raise SystemExit(f"No .png members start with --zip_prefix {args.zip_prefix!r}. Top-level folders: "
                              f"{sorted(set(n.split('/')[0] for n in z.namelist() if '/' in n))}")
    rng = random.Random(args.seed)
    rng.shuffle(members)
    if args.n_images:
        members = members[:args.n_images]
    if args.limit:
        members = members[:args.limit]
    print(f"{len(z.namelist())} total images in zip"
          f"{f' ({len(members)} under prefix {args.zip_prefix!r})' if args.zip_prefix else ''}, "
          f"using {len(members)} (seed={args.seed}) -- "
          f"est. disk usage ~{est_bytes_per_image * len(members) / 1e9:.1f}GB across {len(token_sets)} memmap file(s)")

    def composite(wild):
        if args.fill_mode == "full_canvas":
            return composite_full_canvas(wild, canvas_size)
        return composite_into_bbox(wild, canvas_size, bbox, background_rgb)

    if args.composited_preview_dir:
        os.makedirs(args.composited_preview_dir, exist_ok=True)
        for m in members[:8]:
            wild = Image.open(io.BytesIO(z.read(m)))
            composite(wild).save(
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
    # written to a unique per-run scratch dir (tempfile.mkdtemp, not a fixed
    # shared name) so two runs (or a crashed run's leftovers) can never
    # collide with each other's files, cleaned up at the end.
    scratch_dir = tempfile.mkdtemp(prefix="kaggle_composite_", dir=os.path.dirname(os.path.abspath(
        args.output or os.path.join(REPO_ROOT, "methods", "activations", "x"))))
    codes, image_paths = [], []
    for m in tqdm(members, desc="compositing"):
        wild = Image.open(io.BytesIO(z.read(m)))
        canvas = composite(wild)
        path = os.path.join(scratch_dir, m.replace("/", "_"))
        canvas.save(path)
        codes.append(f"kaggle__{m}")
        image_paths.append(path)

    pool_tag = "kaggle_" + args.fill_mode
    if args.zip_prefix:
        pool_tag += "_" + args.zip_prefix.strip("/").replace("/", "_")
    output = args.output or os.path.join(REPO_ROOT, "methods", "activations",
                                          f"flags_{args.model_id.split('/')[-1]}_augmented_pool_{pool_tag}.pt")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output_stem = os.path.splitext(output)[0]
    memmap_paths = {name: f"{os.path.basename(output_stem)}__{name}.npy" for name in token_sets}  # relative to output's dir

    # Written directly to disk-backed memmaps as each batch completes --
    # never accumulated in one in-RAM tensor (see module docstring: the
    # naive approach OOMs immediately past a few hundred images on this
    # box's 15GB RAM). Created lazily on the first batch once real shapes
    # are known; flushed and closed once every batch is written.
    memmaps = {}
    pairs = list(zip(codes, image_paths))
    idx = 0
    for batch_start in tqdm(range(0, len(pairs), args.batch_size), desc="extracting"):
        batch = pairs[batch_start:batch_start + args.batch_size]
        batch_codes, batch_paths = zip(*batch)
        per_set = extract_batch(model, processor, list(batch_paths), token_sets, grid, device, image_token_id)
        bs = len(batch)
        for name, acts in per_set.items():
            acts_np = acts.numpy()  # [bs, n_layers+1, n_tokens, hidden_dim] float32
            if name not in memmaps:
                shape = (len(codes),) + acts_np.shape[1:]
                mmpath = os.path.join(os.path.dirname(output), memmap_paths[name])
                memmaps[name] = np.lib.format.open_memmap(mmpath, mode="w+", dtype=np.float32, shape=shape)
            memmaps[name][idx:idx + bs] = acts_np
        idx += bs

    for mm in memmaps.values():
        mm.flush()
    first_shape = next(iter(memmaps.values())).shape

    # Save BEFORE any cleanup -- a cleanup failure (e.g. a leftover file from
    # some earlier crashed run sharing this scratch dir) must never lose
    # extraction results that already succeeded. This is now just small
    # metadata -- the actual activations already live in the memmap files.
    torch.save({
        "is_memmap_pool": True,
        "memmap_paths": memmap_paths,
        "item_codes": codes,
        "layer_convention": "index 0 = embedding output; index i = residual stream after decoder layer i",
        "token_flat_indices_by_set": token_sets,
        "grid": grid,
        "model_id": args.model_id,
        "vade_root": args.vade_root,
        "entity": "flags",
        "hidden_dim": first_shape[-1],
        "num_layers": first_shape[1] - 1,
        "is_augmented_pool": True,
        "source": "kaggle:sjetley/country-flags-in-the-wild",
        "zip_prefix": args.zip_prefix,
        "n_source_images": len(members),
    }, output)
    shapes = {name: tuple(mm.shape) for name, mm in memmaps.items()}
    print(f"wrote {output} (+ {len(memmaps)} memmap file(s) alongside it)  shapes={shapes}")

    shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
