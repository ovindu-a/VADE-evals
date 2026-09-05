"""Phase-B step 2: fit PCA and SAE dictionaries per (entity, token_set, layer)
from sae.py's extracted activations -- unsupervised, no attribute/entity
labels involved. One shared dictionary per layer is reused across every
attribute of that entity downstream (step 4 does the label-aware work).

Each dictionary is fit on FLATTENED per-position rows (every real token
position across every image is one training row -- see features.py's
flatten_positions), not pooled entity vectors, because the eventual
intervention (a later, not-yet-built script) encodes/swaps/decodes each
real token position individually, so the dictionary must itself be able to
encode a single position's activation vector, not just a whole-entity mean.

PCA: cheap enough to fit at every layer, sweeping k in --pca_k_grid (default
32/128/256 -- not RAVEL's 512/2048, since flags/brands/animals only give a
few hundred to a couple thousand training rows once flattened). Each k is
automatically capped to min(k, n_rows, hidden_dim) by fit_pca().

SAE: expensive to sweep exhaustively, and the entity's own activations are
the only training data available (see features.py's module docstring for
why: no larger corpus is used here). Start with representative layers
(--sae_layers, default 4/14/24 -- early/the layer-14 DAS-comparison
point/late) and a modest dict_size (default 2x hidden_dim, not the usual
8x-32x overcomplete, since a few hundred/thousand rows can't support that
many free parameters without degenerate solutions). Inspect the printed
reconstruction-MSE/L0 metrics and re-run with different --sae_layers /
--sae_dict_size / --sae_l1_coef to fine-sweep around whatever region looks
promising -- this script does one sweep round per invocation, it does not
pick "promising" regions automatically.

Augmented training pool (--augmented_pool_variants)
-----------------------------------------------------
An entity's real image count (84-130) may simply be too small a training
set for either dictionary, independent of any layer/hyperparameter choice
-- e.g. flags' worse-than-DAS real intervention results. sae.py's
--augment_variants builds a bigger pool via PURELY PIXEL-LEVEL
perturbations of the real images (color jitter + slight noise/blur, never
geometric, so object_location.json's token positions stay exactly valid --
see sae.py's build_augmentation_transform), written to a separate
*_augmented_pool_x<N>.pt file. Passing --augmented_pool_variants <N> here
concatenates that pool's flattened per-position rows onto the real
entity's own before every dictionary fit -- feature selection
(select_features.py) is entirely unaffected, since it always reads the
real per-image activation file and real ground-truth labels directly.

Usage
-----
    # PCA at every layer, k in {32,128,256}, both flags token sets:
    python methods/fit_dictionaries.py --entity flags --method pca

    # SAE at layers 4,14,24 (default), dict_size=2*hidden_dim:
    python methods/fit_dictionaries.py --entity flags --method sae

    # both, one token set only, custom SAE sweep:
    python methods/fit_dictionaries.py --entity flags --method both \\
        --token_sets flag_only --sae_layers 10,14,18 --sae_dict_size 14336

    # build an 8x pixel-perturbed pool, then fit against it too:
    python methods/sae.py --entity flags --augment_variants 8
    python methods/fit_dictionaries.py --entity flags --method both --augmented_pool_variants 8
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from features import (DICTIONARIES_DIR, dictionary_path, fit_pca, fit_sae, flatten_positions, get_layer_slice,
                       load_activations, load_augmented_pool, slice_pca)


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def build_training_matrix(data, aug_pools, token_set, layer):
    """Flattened per-position training rows for one (token_set, layer):
    the real entity's own rows, plus every extra pool's (if any) --
    concatenated so a bigger, more varied corpus supplements the real
    handful of images. See sae.py's --augment_variants docstring and
    build_external_flag_pool.py."""
    parts = [flatten_positions(get_layer_slice(data, token_set, layer))]
    for pool in aug_pools:
        parts.append(flatten_positions(get_layer_slice(pool, token_set, layer)))
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--model_tag", default="Qwen2.5-VL-7B-Instruct",
                     help="Matches the <model> segment of sae.py's output filename.")
    ap.add_argument("--token_sets", default=None,
                     help="Comma-separated token-set names (e.g. 'flag_only,flag_ring1'). "
                          "Default: every set present in the entity's activations file.")
    ap.add_argument("--method", default="both", choices=["pca", "sae", "both"])
    ap.add_argument("--layers", default="all",
                     help="PCA layers: comma list or 'all' (default) for every layer 0..num_layers.")
    ap.add_argument("--pca_k_grid", default="32,128,256", type=str)
    ap.add_argument("--sae_layers", default="4,14,24", type=str,
                     help="Representative first-round layers for the (expensive) SAE sweep.")
    ap.add_argument("--sae_dict_size", type=int, default=None,
                     help="Default: 2 * hidden_dim (modest, not overcomplete -- see module docstring).")
    ap.add_argument("--sae_l1_coef", type=float, default=1e-3)
    ap.add_argument("--sae_epochs", type=int, default=300)
    ap.add_argument("--sae_lr", type=float, default=1e-3)
    ap.add_argument("--sae_val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu' -- SAE training only (PCA is CPU/sklearn).")
    ap.add_argument("--output_dir", default=DICTIONARIES_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--augmented_pool_variants", type=int, default=0,
                     help="If >0, also load the pixel-perturbed pool sae.py's --augment_variants of this size "
                          "wrote (methods/activations/<entity>_<model>_augmented_pool_x<N>.pt) and concatenate "
                          "its flattened per-position rows onto the real entity's own before fitting EVERY "
                          "PCA/SAE dictionary -- a bigger, more varied training pool for the same real layers/"
                          "token-sets. Feature selection (select_features.py) is unaffected -- it always uses "
                          "only the real per-image activations and their real ground-truth labels.")
    ap.add_argument("--augmented_pool_paths", default=None,
                     help="Comma-separated explicit paths to additional pool files (same schema as sae.py's "
                          "output -- see e.g. build_external_flag_pool.py) to concatenate in as well, on top "
                          "of --augmented_pool_variants if both are given. Lets an external, non-synthetic "
                          "pool (real photos composited into VADE's canvas, say) be mixed in too.")
    args = ap.parse_args()

    data = load_activations(args.entity, args.model_tag)
    aug_pools = []
    if args.augmented_pool_variants > 0:
        aug_pools.append(load_augmented_pool(args.entity, args.augmented_pool_variants, args.model_tag))
    if args.augmented_pool_paths:
        for p in args.augmented_pool_paths.split(","):
            aug_pools.append(torch.load(p, map_location="cpu", weights_only=False))
    token_sets = args.token_sets.split(",") if args.token_sets else list(data["activations_by_token_set"])
    num_layers = data["num_layers"]
    pca_layers = list(range(num_layers + 1)) if args.layers == "all" else parse_int_list(args.layers)
    sae_layers = parse_int_list(args.sae_layers)
    pca_k_grid = parse_int_list(args.pca_k_grid)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if aug_pools:
        aug_note = "+ " + ", ".join(f"extra pool ({len(p['item_codes'])} images/token_set)" for p in aug_pools)
    else:
        aug_note = "(no augmented pool)"
    print(f"entity={args.entity} token_sets={token_sets} num_layers={num_layers} "
          f"hidden_dim={data['hidden_dim']} sae_device={device} {aug_note}")

    metrics_log = []
    for token_set in token_sets:
        for layer in (pca_layers if args.method in ("pca", "both") else []):
            X = build_training_matrix(data, aug_pools, token_set, layer)

            out_paths = {k: dictionary_path(args.entity, token_set, layer, "pca", k, args.output_dir)
                         for k in pca_k_grid}
            if all(os.path.exists(p) for p in out_paths.values()) and not args.overwrite:
                print(f"  skip (exists) pca token_set={token_set} layer={layer} k in {pca_k_grid}")
                continue

            # Fit once at the largest k -- svd_solver="full" computes the
            # complete SVD regardless of n_components, so every smaller k is
            # an exact slice of this same fit (see features.slice_pca).
            t0 = time.time()
            max_dictionary, max_metrics = fit_pca(X, max(pca_k_grid))
            fit_seconds = round(time.time() - t0, 2)
            for k in pca_k_grid:
                out_path = out_paths[k]
                if os.path.exists(out_path) and not args.overwrite:
                    print(f"  skip (exists) {out_path}")
                    continue
                dictionary = max_dictionary if k == max(pca_k_grid) else slice_pca(max_dictionary, k)
                dictionary.save(out_path)
                metrics = {"k": dictionary.k, "cumulative_explained_variance":
                           float(dictionary.explained_variance_ratio_.sum())}
                metrics.update({"entity": args.entity, "token_set": token_set, "layer": layer,
                                 "method": "pca", "n_rows": X.shape[0], "seconds": fit_seconds})
                metrics_log.append(metrics)
                print(f"  pca token_set={token_set} layer={layer:>2} k={metrics['k']:>4} "
                      f"cum_explained_var={metrics['cumulative_explained_variance']:.3f} "
                      f"({fit_seconds}s for the shared fit) -> {out_path}")

        for layer in (sae_layers if args.method in ("sae", "both") else []):
            X = build_training_matrix(data, aug_pools, token_set, layer)
            dict_size = args.sae_dict_size or 2 * data["hidden_dim"]
            out_path = dictionary_path(args.entity, token_set, layer, "sae", dict_size, args.output_dir)
            if os.path.exists(out_path) and not args.overwrite:
                print(f"  skip (exists) {out_path}")
                continue
            t0 = time.time()
            dictionary, metrics = fit_sae(X, dict_size, l1_coef=args.sae_l1_coef, epochs=args.sae_epochs,
                                           lr=args.sae_lr, val_frac=args.sae_val_frac, seed=args.seed,
                                           device=device)
            dictionary.save(out_path)
            metrics.update({"entity": args.entity, "token_set": token_set, "layer": layer,
                             "method": "sae", "n_rows": X.shape[0], "seconds": round(time.time() - t0, 2)})
            metrics_log.append(metrics)
            print(f"  sae token_set={token_set} layer={layer:>2} dict_size={dict_size:>6} "
                  f"train_mse={metrics['train_recon_mse']:.4f} val_mse={metrics.get('val_recon_mse')} "
                  f"val_l0={metrics.get('val_l0'):.1f} ({metrics['seconds']}s) -> {out_path}")

    log_path = os.path.join(args.output_dir, args.entity, "fit_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")
    print(f"appended {len(metrics_log)} rows to {log_path}")


if __name__ == "__main__":
    main()
