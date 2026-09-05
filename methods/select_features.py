"""Phase-B steps 3-4 of the vade-evals plan: pool token positions into one
vector per image (step 3), then two-step L1-SVC + SelectFromModel feature
selection over the dictionary-encoded space, swept over (direction, layer,
C), independently per attribute (step 4).

Step 3 -- pooling
------------------
The dictionary (PCA or SAE, from fit_dictionaries.py) was fit on individual
token positions and can encode a single position -- that's what the later,
not-yet-built intervention script needs. But for *this* script's classifier
fitting, a token-set's positions are mean-pooled into one vector per image
first: attributes are whole-entity recalled facts, not local visual
features, and a token-set's 8/24/36/64 positions are correlated, not
independent samples, so pooling first (see features.py's pool_positions)
avoids the classifier stage silently treating n_images*n_tokens correlated
rows as that many independent examples.

Step 4 -- two-step selection, both directions, swept over layer
------------------------------------------------------------------
Two labels exist for every image: its entity-ID (item_code, exactly one
example per class -- deliberately weak, see below) and each attribute's
value (e.g. flags/capital). A two-step pipeline runs an L1-penalized
LinearSVC + SelectFromModel twice, taking whichever features survive the
first (broad) stage as the input to the second (narrow) stage:

    forward:  entity-ID (broad, all data, no CV -- coarse filter only)
              -> attribute (narrow, real held-out CV, C swept)
    inverse:  attribute (broad, real held-out CV, C swept)
              -> entity-ID (narrow, all data, no CV -- coarse filter only)

Same select_dims_by_l1svc() function both ways; only which label goes first
differs. The entity-ID stage never gets cross-validated regardless of which
position it's in -- it has exactly one example per class, so there is no
statistically meaningful held-out split for it (see plan item 4's own
caveat); it's fit on ALL rows at a fixed --id_c and used purely as a coarse
dimensionality filter. Whichever stage IS the attribute, in either
direction, gets real stratified CV (auto-capped to the rarest surviving
class's count, and rows whose class has <2 members are dropped since they
can never appear in both a train and a test fold).

cause/iso proxy scoring
------------------------
There's no generation-time intervention yet to measure real cause/iso
(eval/score.py's definitions) -- that happens in a later script. Here,
for one attribute A's finally-selected feature set F_A (after both steps
of one direction, at one layer, one C):

    cause_score = held-out CV accuracy predicting A from F_A
    iso_score   = 1 - mean over every OTHER attribute B of the
                  chance-adjusted leakage of B from F_A, i.e.
                  max(0, (acc_B - chance_B) / (1 - chance_B))
    combined    = 0.5 * (cause_score + iso_score)

mirroring eval/score.py's final_score = 1/2(cause + mean(iso)) shape at the
feature level instead of the generation level. The (direction, layer, C)
combination maximizing combined is the winner carried into step 5 for that
attribute -- i.e. direction and C are swept jointly with layer, not fixed
in advance, so "the winning (layer, F_A) for that attribute" ends up a
single choice regardless of which of the two two-step orderings produced it.

Usage
-----
    # sweep every layer with a fitted PCA dictionary, default C grid:
    python methods/select_features.py --entity flags --token_set flag_only --dict_method pca

    # same but against the SAE dictionaries fit at layers 4,14,24:
    python methods/select_features.py --entity flags --token_set flag_only --dict_method sae
"""
import argparse
import json
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold

from features import (DICTIONARIES_DIR, REPO_ROOT, dictionary_path, get_layer_slice, list_fitted_layers,
                       list_fitted_sizes, load_activations, load_dictionary, load_labels, pool_positions)

DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# L1-SVC + SelectFromModel, and the held-out CV probe
# ---------------------------------------------------------------------------

def select_dims_by_l1svc(X, labels, C, threshold=1e-6):
    """Fits one L1-penalized LinearSVC on ALL rows (no split -- this is a
    filter/selection step, not an evaluation) and returns the boolean mask
    of features with a nonzero coefficient for at least one class. None if
    every coefficient came out zero (C too small for this label/feature
    combination -- caller should skip that C rather than crash on an empty
    selection)."""
    y = LabelEncoder().fit_transform(labels)
    if len(np.unique(y)) < 2:
        return None
    clf = LinearSVC(penalty="l1", dual=False, C=C, max_iter=10000)
    clf.fit(X, y)
    mask = SelectFromModel(clf, prefit=True, threshold=threshold).get_support()
    return mask if mask.any() else None


def cv_probe(X, labels, C, n_splits=5, seed=0):
    """Stratified-CV held-out accuracy of an L1-SVC predicting `labels` from
    X, plus the majority-class chance baseline among the rows actually used.
    Drops rows whose class has <2 members first (can never span train+test),
    and caps n_splits to the rarest remaining class's count. Returns None if
    fewer than 2 classes / 4 rows remain -- not enough signal to evaluate."""
    labels = np.asarray(labels)
    keep = labels != ""
    X, labels = X[keep], labels[keep]
    y = LabelEncoder().fit_transform(labels)
    counts = np.bincount(y)
    row_keep = counts[y] >= 2
    X, y = X[row_keep], y[row_keep]
    if len(y) < 4 or len(np.unique(y)) < 2:
        return None
    counts = np.bincount(y)
    n_splits_eff = max(2, min(n_splits, int(counts.min())))
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    correct, total = 0, 0
    for train_idx, test_idx in skf.split(X, y):
        clf = LinearSVC(penalty="l1", dual=False, C=C, max_iter=10000)
        clf.fit(X[train_idx], y[train_idx])
        correct += int((clf.predict(X[test_idx]) == y[test_idx]).sum())
        total += len(test_idx)
    return {"accuracy": correct / total, "chance": float(counts.max() / len(y)),
            "n_used": len(y), "n_classes": int(len(np.unique(y))), "n_splits": n_splits_eff}


def two_step_forward(X, mask_id_broad, attr_labels, C, threshold):
    """mask_id_broad is the entity-ID broad-filter mask, precomputed ONCE per
    (layer, attribute) by the caller -- it depends only on id_c and the
    attribute's valid-row subset, not on C, so refitting it inside this
    per-C loop would be pure waste (the expensive part: entity-ID has as
    many classes as there are images)."""
    if mask_id_broad is None:
        return None
    sub_idx = np.where(mask_id_broad)[0]
    mask_attr = select_dims_by_l1svc(X[:, sub_idx], attr_labels, C, threshold)
    if mask_attr is None:
        return None
    return sub_idx[mask_attr]


def two_step_inverse(X, id_labels, attr_labels, id_c, C, threshold):
    mask_attr = select_dims_by_l1svc(X, attr_labels, C, threshold)
    if mask_attr is None:
        return None
    sub_idx = np.where(mask_attr)[0]
    mask_id = select_dims_by_l1svc(X[:, sub_idx], id_labels, id_c, threshold)
    if mask_id is None:
        return None
    return sub_idx[mask_id]


def cause_iso_score(X, F_A, attribute, attributes, attr_labels, C, n_splits, seed):
    cause = cv_probe(X[:, F_A], attr_labels[attribute], C, n_splits, seed)
    if cause is None:
        return None
    leakages = []
    for other in attributes:
        if other == attribute:
            continue
        res = cv_probe(X[:, F_A], attr_labels[other], C, n_splits, seed)
        if res is None or res["chance"] >= 1:
            continue
        leakages.append(max(0.0, (res["accuracy"] - res["chance"]) / (1 - res["chance"])))
    iso_score = 1 - (float(np.mean(leakages)) if leakages else 0.0)
    return {"cause_score": cause["accuracy"], "cause_chance": cause["chance"], "cause_n_used": cause["n_used"],
            "iso_score": iso_score, "n_iso_attributes_scored": len(leakages),
            "combined_score": 0.5 * (cause["accuracy"] + iso_score)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--model_tag", default="Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--token_set", required=True,
                     help="Single token set to run against (dictionaries are fit per token_set -- "
                          "this isn't swept jointly with layer/C/direction).")
    ap.add_argument("--dict_method", default="pca", choices=["pca", "sae"])
    ap.add_argument("--dict_size", type=int, default=None,
                     help="k (pca) or dict_size (sae) to use at each layer. Default: the largest "
                          "fitted size on disk for that layer.")
    ap.add_argument("--dictionaries_dir", default=DICTIONARIES_DIR)
    ap.add_argument("--layers", default=None,
                     help="Comma list of layers to sweep. Default: every layer with a fitted "
                          "--dict_method dictionary on disk for this entity/token_set.")
    ap.add_argument("--c_grid", default="0.001,0.003,0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--id_c", type=float, default=0.1,
                     help="Fixed C for the entity-ID coarse-filter stage (never swept -- see module docstring).")
    ap.add_argument("--threshold", type=float, default=1e-6, help="SelectFromModel nonzero-coefficient threshold.")
    ap.add_argument("--n_splits", type=int, default=5, help="Stratified CV folds (auto-capped per attribute).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", default=None,
                     help="Default: methods/selections/<entity>/<token_set>_<dict_method>/")
    args = ap.parse_args()

    data = load_activations(args.entity, args.model_tag)
    layers = (parse_int_list(args.layers) if args.layers else
              list_fitted_layers(args.entity, args.token_set, args.dict_method, args.dictionaries_dir))
    if not layers:
        raise SystemExit(f"No fitted {args.dict_method} dictionaries found for entity={args.entity} "
                          f"token_set={args.token_set} under {args.dictionaries_dir} -- run "
                          f"fit_dictionaries.py first.")
    C_grid = parse_float_list(args.c_grid)
    attributes, id_labels, attr_labels = load_labels(args.vade_root, args.entity, data["item_codes"])
    print(f"entity={args.entity} token_set={args.token_set} attributes={attributes} "
          f"layers={layers} dict_method={args.dict_method} C_grid={C_grid}")

    rows = []
    for layer in layers:
        size = args.dict_size or (list_fitted_sizes(args.entity, args.token_set, layer, args.dict_method,
                                                      args.dictionaries_dir) or [None])[0]
        if size is None:
            print(f"  layer={layer}: no fitted {args.dict_method} dictionary, skipping")
            continue
        dict_path = dictionary_path(args.entity, args.token_set, layer, args.dict_method, size, args.dictionaries_dir)
        dictionary = load_dictionary(dict_path)
        pooled = pool_positions(get_layer_slice(data, args.token_set, layer))  # [n_images, hidden_dim]
        X = dictionary.encode(pooled)  # [n_images, n_features]

        for attribute in attributes:
            valid = attr_labels[attribute] != ""
            if valid.sum() < 4:
                continue
            id_sub = id_labels[valid]
            attr_sub = {a: attr_labels[a][valid] for a in attributes}
            X_sub = X[valid]
            # Computed once per (layer, attribute): doesn't depend on C, only
            # on id_c and this attribute's valid-row subset (see two_step_forward).
            mask_id_broad = select_dims_by_l1svc(X_sub, id_sub, args.id_c, args.threshold)

            for direction in ("forward", "inverse"):
                for C in C_grid:
                    if direction == "forward":
                        F_A = two_step_forward(X_sub, mask_id_broad, attr_sub[attribute], C, args.threshold)
                    else:
                        F_A = two_step_inverse(X_sub, id_sub, attr_sub[attribute], args.id_c, C, args.threshold)
                    if F_A is None or len(F_A) == 0:
                        continue
                    scored = cause_iso_score(X_sub, F_A, attribute, attributes, attr_sub, C,
                                              args.n_splits, args.seed)
                    if scored is None:
                        continue
                    rows.append({"attribute": attribute, "layer": layer, "dict_size": size, "direction": direction,
                                 "C": C, "id_c": args.id_c, "n_features_selected": int(len(F_A)),
                                 "feature_indices": [int(i) for i in F_A], **scored})

    out_dir = args.output_dir or os.path.join(REPO_ROOT, "methods", "selections", args.entity,
                                                f"{args.token_set}_{args.dict_method}")
    os.makedirs(out_dir, exist_ok=True)
    sweep_path = os.path.join(out_dir, "sweep.jsonl")
    with open(sweep_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    winners = {}
    for attribute in attributes:
        candidates = [r for r in rows if r["attribute"] == attribute]
        if not candidates:
            print(f"attribute={attribute}: no valid (layer, direction, C) combination scored")
            continue
        best = max(candidates, key=lambda r: r["combined_score"])
        winners[attribute] = best
        print(f"attribute={attribute:>15} winner: layer={best['layer']:>2} direction={best['direction']:<7} "
              f"C={best['C']} n_features={best['n_features_selected']:>4} "
              f"cause={best['cause_score']:.3f} iso={best['iso_score']:.3f} combined={best['combined_score']:.3f}")

    winners_path = os.path.join(out_dir, "winners.json")
    with open(winners_path, "w") as f:
        json.dump({"entity": args.entity, "token_set": args.token_set, "dict_method": args.dict_method,
                    "winners": winners}, f, indent=2)
    print(f"wrote {len(rows)} sweep rows -> {sweep_path}")
    print(f"wrote winners -> {winners_path}")


if __name__ == "__main__":
    main()
