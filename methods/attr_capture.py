"""Capture clean activations over the crossed grid (item x queried attribute).

ONE forward pass per run, no intervention, nothing trained. The same image is
asked about four different ways, so anything that varies WITH the question and
not with the image is attribute content -- which is what makes the CROSSING the
point of the design rather than a convenience. Every item must appear with every
attribute, so that the analysis can subtract each item's own mean and be left
with what the question changed.

FOUR SITES at every traced block, all at hidden_size width, captured in one pass:

    residual          block b's OUTPUT (pre-norm), read through a hook rather
                      than out.hidden_states -- whose LAST entry is tied to
                      last_hidden_state and is POST-final-norm in this project's
                      transformers (the trap logit_lens.py documents).
    attn_output       block b's attention contribution, post-o_proj
    mlp_output        block b's MLP contribution
    attn_head_output  o_proj's INPUT: 28 contiguous 128-dim heads. Stored at
                      FULL width on purpose -- selecting k heads is a slice at
                      analysis time and costs nothing, whereas storing a chosen
                      subset would freeze a head ranking into the data.

LAYOUT. `acts_<site>.npy` is a memmapped [n_runs, n_blocks, n_positions, width]
float32 array; `index.jsonl` gives each run's row, item, attribute and array
offset; `meta.json` carries shapes, the block/site/position spec and the hashes.
Memmap rather than one .pt so the analysis can slice ONE head across every run
without reading the whole file.

float32, not float16: the model computes in bf16, which has float32's exponent
range. Residual-stream outlier coordinates can exceed float16's 65504 and would
silently become inf. The whole grid is well under a gigabyte either way.

Written for methods/attr_directions.py; see its docstring for the analyses this
layout is shaped to support.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.attr_head_trace import ENTITY_KEYS  # noqa: E402
from methods.attribute_switch_sweep import ATTRIBUTES, PREFILL, question  # noqa: E402
from methods.common.hooks import extra_to_device  # noqa: E402
from methods.common.run_logging import tee_to_log  # noqa: E402
from methods.common.sites import InterventionSite  # noqa: E402
from methods.common.targets import derive_gold_token_ids  # noqa: E402
from methods.head_trace import _residual_hook  # noqa: E402

MODULE_SITES = ("attn_output", "mlp_output", "attn_head_output")
SITES = ("residual",) + MODULE_SITES


def capture_all_sites(adapter, model, blocks, positions, input_ids, attention_mask, extra):
    """-> {site: {block: [B, n_pos, hidden]}} from ONE forward pass.

    `register_capture` takes sites.py's layer_idx convention (layer_idx = b + 1
    addresses block b's sublayers), while `_residual_hook` takes the raw block
    index. Getting either wrong shifts a whole table by one block, so the two
    conventions are converted here, once.
    """
    sinks = {s: {b: [] for b in blocks} for s in SITES}
    handles = []
    for b in blocks:
        for site in MODULE_SITES:
            handles.append(InterventionSite(site).register_capture(adapter, model, b + 1, positions,
                                                                   sinks[site][b]))
        handles.append(_residual_hook(adapter, model, b, positions, sinks["residual"][b]))
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                  **extra_to_device(extra, model.device, model.dtype), logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    for site in SITES:
        for b in blocks:
            assert len(sinks[site][b]) == 1, (
                f"{site} at block {b} fired {len(sinks[site][b])} times in one forward pass, expected 1")
    return {s: {b: sinks[s][b][0] for b in blocks} for s in SITES}


def head_projection_norms(adapter, model, blocks, n_heads, head_dim):
    """||W_O_h||_F per (block, head), recorded so the analysis can tell a head
    whose output varies a lot from one whose variation actually LANDS in the
    residual stream. o_proj weights heads very differently -- the same reason
    head_trace ranks by delta_resid rather than delta_z -- and the analysis has
    no model to compute this from."""
    out = {}
    for b in blocks:
        w = adapter.get_attn_head_output_module(model, b).weight.float()
        out[str(b)] = [float(w[:, h * head_dim:(h + 1) * head_dim].norm()) for h in range(n_heads)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--attributes", nargs="+", choices=ATTRIBUTES, default=ATTRIBUTES)
    ap.add_argument("--blocks", type=int, nargs="+", default=list(range(15, 23)),
                    help="Decoder blocks to capture at. Default 15-22, the window "
                         "ATTRIBUTE_SWITCH_FINDINGS.md localizes the question->readout handoff to.")
    ap.add_argument("--items", nargs="+", help="Explicit item IDs; otherwise a deterministic sample.")
    ap.add_argument("--n_items", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate the grid, images and output sizing without torch or a model.")
    args = ap.parse_args(argv)

    from pathlib import Path
    entity_dir = Path(args.vade_root) / "data" / args.entity
    raw = (entity_dir / "ground_truth.json").read_bytes()
    gt = json.loads(raw)
    key = ENTITY_KEYS.get(args.entity)
    if key not in gt:
        key = next(k for k, v in gt.items() if isinstance(v, dict) and k != "coverage")
    items = gt[key]
    blocks = sorted(set(args.blocks))
    attributes = list(dict.fromkeys(args.attributes))
    if len(attributes) < 2:
        ap.error("Need at least two attributes -- the design is the CROSSING of items and attributes")
    chosen = args.items or random.Random(args.seed).sample(sorted(items), min(args.n_items, len(items)))
    if len(set(chosen)) != len(chosen):
        ap.error("Item IDs must be unique")
    # Fully crossed and in a fixed order, so `index.jsonl` row i is item i//A, attribute i%A and the
    # analysis can reshape straight to [n_items, n_attributes, ...] after checking the index.
    grid = list(itertools.product(chosen, attributes))
    for item_id in chosen:
        if not (entity_dir / items[item_id]["image"]).is_file():
            raise FileNotFoundError(entity_dir / items[item_id]["image"])
        missing = [a for a in attributes if a not in items[item_id]]
        if missing:
            raise ValueError(f"{item_id} has no {missing}; the grid must be complete")

    out_dir = Path(args.out_dir or Path(REPO_ROOT) / "results" / "attr_capture" / args.entity /
                   f"blocks{blocks[0]}-{blocks[-1]}_n{len(chosen)}")
    n_runs, n_positions = len(grid), 1
    print(f"[attr_capture] {args.entity}: {len(chosen)} items x {len(attributes)} attributes "
          f"= {n_runs} clean runs; blocks {blocks[0]}-{blocks[-1]}; -> {out_dir}")
    if args.dry_run:
        width = 3584
        size = n_runs * len(blocks) * n_positions * width * 4
        print(f"Grid complete. {len(SITES)} sites x {size / 1e6:.0f} MB = "
              f"{len(SITES) * size / 1e6:.0f} MB total (at hidden_size={width}).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with tee_to_log(str(out_dir / "capture.log")):
        adapter = get_adapter(args.model_id)
        # sdpa: nothing here reads attention weights; every site is a module input/output.
        model, processor = adapter.load(device=args.device, dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
        model.eval().requires_grad_(False)
        n_layers = len(adapter.get_decoder_layers(model))
        n_heads, hidden = adapter.n_attention_heads(model), adapter.hidden_size(model)
        assert hidden % n_heads == 0
        head_dim = hidden // n_heads
        assert all(0 <= b < n_layers for b in blocks), f"blocks must be in 0..{n_layers - 1}"

        meta = {"experiment": "attr_capture", "entity": args.entity, "model_id": args.model_id,
                "attributes": attributes, "items": chosen, "blocks": blocks, "sites": list(SITES),
                "positions": "last_token", "grid_order": "item-major, attribute-minor",
                "shape": [n_runs, len(blocks), n_positions, hidden], "dtype": "float32",
                "n_heads": n_heads, "head_dim": head_dim, "prefill": PREFILL,
                "prompts": {a: question(a) for a in attributes},
                "head_proj_norms": head_projection_norms(adapter, model, blocks, n_heads, head_dim),
                "ground_truth_sha256": hashlib.sha256(raw).hexdigest(),
                "implementation_sha256": hashlib.sha256(
                    (Path(REPO_ROOT) / "methods" / "attr_capture.py").read_bytes()).hexdigest()}
        meta_path = out_dir / "meta.json"
        if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
            raise ValueError(f"{out_dir} holds a different capture; use another --out_dir")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        arrays = {s: np.lib.format.open_memmap(out_dir / f"acts_{s}.npy", mode="w+", dtype=np.float32,
                                               shape=(n_runs, len(blocks), n_positions, hidden))
                  for s in SITES}
        index = []
        for start in range(0, n_runs, args.batch_size):
            chunk = grid[start:start + args.batch_size]
            ids, pixels, grids = [], [], []
            for item_id, attribute in chunk:
                with Image.open(entity_dir / items[item_id]["image"]) as img:
                    built = adapter.build_inputs(processor, img.convert("RGB"), question(attribute),
                                                 PREFILL)
                ids.append(built["input_ids"].unsqueeze(0))
                pixels.append(built["extra"]["pixel_values"])
                grids.append(built["extra"]["image_grid_thw"])
            if len({x.shape[1] for x in ids}) != 1:
                raise ValueError("Prompts in a batch must share a token length")
            batch_ids = torch.cat(ids)
            mask = torch.ones_like(batch_ids)
            positions = torch.full((len(chunk), 1), batch_ids.shape[1] - 1, dtype=torch.long)
            captured = capture_all_sites(adapter, model, blocks, positions, batch_ids, mask,
                                         {"pixel_values": torch.cat(pixels),
                                          "image_grid_thw": torch.cat(grids)})
            for site in SITES:
                stacked = torch.stack([captured[site][b] for b in blocks], dim=1)  # [B, n_blocks, P, H]
                arrays[site][start:start + len(chunk)] = stacked.float().cpu().numpy()
            for offset, (item_id, attribute) in enumerate(chunk):
                index.append({"row": start + offset, "item": item_id, "attribute": attribute,
                              "condition": "clean", "prompt_length": int(batch_ids.shape[1]),
                              "gold_ids": derive_gold_token_ids(processor.tokenizer, PREFILL,
                                                                items[item_id][attribute])})
            print(f"  {start + len(chunk)}/{n_runs} runs captured", flush=True)
        for a in arrays.values():
            a.flush()
        (out_dir / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
        print(f"\nwrote {out_dir}  ({len(SITES)} arrays, {n_runs} runs)")


if __name__ == "__main__":
    main()
