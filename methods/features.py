"""Shared utilities for VADE-evals' dictionary-learning + feature-selection
pipeline (Phase B, steps 2-4 of the vade-evals plan). Sits downstream of
sae.py's activation extraction (Phase A) and upstream of the actual causal
intervention script (Phase C, not yet built).

Three things live here, used by both fit_dictionaries.py (step 2) and
select_features.py (steps 3-4):

  1. Activation loading/slicing -- pulling one (token_set, layer) out of an
     sae.py-produced .pt file, in two shapes:
       - flatten_positions(): [n_images*n_tokens, hidden_dim], one row per
         real token position -- what a dictionary (PCA or SAE) is FIT on,
         and what the eventual per-position intervention will ENCODE with.
       - pool_positions(): [n_images, hidden_dim], mean over the token-set's
         positions -- what the attribute/entity-ID classifiers in step 4
         are fit on. Attributes are whole-entity recalled facts, not local
         visual features, and a token-set's positions are correlated, not
         independent samples, so pooling first avoids inflating n by
         treating correlated positions as independent evidence.

  2. Two interchangeable dictionary classes, PCADictionary and
     SAEDictionary, each with the same encode(X)/decode(F) contract
     (numpy in, numpy out) plus save()/load() -- downstream code never
     needs to know which kind of dictionary it's holding.

  3. fit_pca() / fit_sae() -- the actual fitting routines fit_dictionaries.py
     calls per (entity, token_set, layer).

SAE data-budget note (see fit_dictionaries.py docstring for the fuller
discussion): each entity has only 84-130 images, i.e. a few hundred to a
couple thousand token-position rows once flattened. That's nowhere near
typical SAE training-set sizes, so SAEDictionary is deliberately modest
(dict_size default 2-4x hidden_dim, not the usual 8x-32x overcomplete) and
trained only on that one entity's own activations -- not some larger
external image corpus.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVATIONS_DIR = os.path.join(REPO_ROOT, "methods", "activations")
DICTIONARIES_DIR = os.path.join(REPO_ROOT, "methods", "dictionaries")

DEFAULT_MODEL_TAG = "Qwen2.5-VL-7B-Instruct"


# ---------------------------------------------------------------------------
# Activation loading / reshaping
# ---------------------------------------------------------------------------

def activations_path(entity, model_tag=DEFAULT_MODEL_TAG, activations_dir=ACTIVATIONS_DIR):
    return os.path.join(activations_dir, f"{entity}_{model_tag}_all_layers.pt")


def load_activations(entity, model_tag=DEFAULT_MODEL_TAG, activations_dir=ACTIVATIONS_DIR):
    """Loads the dict written by sae.py's main() for one entity."""
    path = activations_path(entity, model_tag, activations_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No activations file at {path!r} -- run methods/sae.py --entity {entity} first.")
    return load_pool_file(path)


def load_pool_file(path):
    """Loads any pool/activations .pt file, transparently handling both
    small in-RAM pools (activations_by_token_set already holds real
    tensors) and large memmap-backed pools (is_memmap_pool=True, written
    by build_external_flag_pool.py) -- for the latter, activations_by_
    token_set is reconstructed as read-only np.memmap arrays from sibling
    .npy files next to `path`, so a huge pool (order of 100s of GB across
    all layers/token-sets on disk) never needs to fit in RAM at once --
    get_layer_slice() only pages in the one layer it actually reads."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("is_memmap_pool"):
        base_dir = os.path.dirname(os.path.abspath(path))
        data["activations_by_token_set"] = {
            name: np.load(os.path.join(base_dir, relpath), mmap_mode="r")
            for name, relpath in data["memmap_paths"].items()
        }
    return data


def augmented_pool_path(entity, n_variants, model_tag=DEFAULT_MODEL_TAG, activations_dir=ACTIVATIONS_DIR):
    return os.path.join(activations_dir, f"{entity}_{model_tag}_augmented_pool_x{n_variants}.pt")


def load_augmented_pool(entity, n_variants, model_tag=DEFAULT_MODEL_TAG, activations_dir=ACTIVATIONS_DIR):
    """Loads a pixel-perturbed training-only pool written by sae.py's
    --augment_variants (see its module docstring / build_augmentation_transform).
    Same schema as load_activations(), except item_codes are '<code>__augI'
    pseudo-codes -- never meant to be matched against ground_truth.json,
    only flattened over positions for dictionary fitting."""
    path = augmented_pool_path(entity, n_variants, model_tag, activations_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No augmented pool at {path!r} -- run methods/sae.py --entity {entity} "
            f"--augment_variants {n_variants} first.")
    return load_pool_file(path)


def get_layer_slice(data, token_set, layer, image_indices=None):
    """[n_images, n_tokens, hidden_dim] float32 numpy array for one
    (token_set, layer) out of a load_activations()/load_pool_file() dict.

    acts is either a real torch tensor (small pools/entity activations,
    already fully in RAM) or a read-only np.memmap array (large pools --
    see load_pool_file). Either way, only THIS layer's slice gets
    materialized as a real in-RAM array here -- for a memmap that's the
    entire point: a huge pool's other 28 layers are never touched.

    image_indices, if given, subsets the image axis at the same time as the
    layer axis -- for a memmap pool this reads only those images off disk
    rather than materializing every image just to subsample afterwards
    (see build_training_matrix's pool_max_images)."""
    acts = data["activations_by_token_set"][token_set]  # [n_images, n_layers+1, n_tokens, hidden_dim]
    num_layers = data["num_layers"]
    if not (0 <= layer <= num_layers):
        raise ValueError(f"layer {layer} out of range [0, {num_layers}] "
                          f"(0=embedding output, i=after decoder layer i)")
    layer_slice = acts[image_indices, layer, :, :] if image_indices is not None else acts[:, layer, :, :]
    if hasattr(layer_slice, "numpy"):  # torch tensor
        return layer_slice.numpy().astype(np.float32)
    return np.array(layer_slice, dtype=np.float32)  # memmap -> real in-RAM copy of just this slice


def flatten_positions(X):
    """[n_images, n_tokens, hidden_dim] -> [n_images*n_tokens, hidden_dim].
    Every real token position is one training row for dictionary fitting."""
    n_images, n_tokens, hidden_dim = X.shape
    return X.reshape(n_images * n_tokens, hidden_dim)


def pool_positions(X):
    """[n_images, n_tokens, hidden_dim] -> [n_images, hidden_dim], mean over
    the token-set's positions. Used only for the classifier-fitting step
    (step 4) -- the actual intervention stays per-position (see module
    docstring)."""
    return X.mean(axis=1)


# ---------------------------------------------------------------------------
# PCA dictionary
# ---------------------------------------------------------------------------

class PCADictionary:
    """encode(X) = (X - mean) @ components.T ; decode(F) = F @ components + mean.
    Wraps sklearn's PCA but stores raw numpy arrays so load() has no sklearn
    dependency at intervention time."""

    method = "pca"

    def __init__(self, mean_, components_, explained_variance_ratio_, meta):
        self.mean_ = mean_                                     # [hidden_dim]
        self.components_ = components_                         # [k, hidden_dim]
        self.explained_variance_ratio_ = explained_variance_ratio_  # [k]
        self.meta = meta

    @property
    def k(self):
        return self.components_.shape[0]

    def encode(self, X):
        return (X - self.mean_) @ self.components_.T

    def decode(self, F):
        return F @ self.components_ + self.mean_

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "method": self.method,
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        assert d["method"] == cls.method, f"{path} is not a PCA dictionary (method={d['method']!r})"
        return cls(d["mean_"], d["components_"], d["explained_variance_ratio_"], d["meta"])


def fit_pca(X, k):
    """X: [n_rows, hidden_dim] (already flattened over positions/images).
    k is capped to min(k, n_rows, hidden_dim) -- sklearn's PCA would raise
    otherwise, and flags/brands/animals' sample counts (a few hundred to a
    couple thousand rows) don't support RAVEL-scale k like 512/2048.

    svd_solver="full" computes U at shape [n_rows, min(n_rows, hidden_dim)]
    regardless of k -- fine for the real-only case (a few hundred/thousand
    rows: U is at most a few thousand x 3584) but a several-GB allocation
    once an external pool pushes n_rows into the hundreds of thousands,
    on top of X itself -- enough to OOM this box. Randomized SVD only ever
    materializes n_rows x k_eff, independent of n_rows' magnitude, so switch
    once X is large enough for the difference to matter; components are
    still variance-ordered either way, so slice_pca's exact-prefix reuse
    still applies to a "randomized" fit, just an approximate one."""
    from sklearn.decomposition import PCA

    k_eff = min(k, X.shape[0], X.shape[1])
    solver = "full" if X.shape[0] <= 20000 else "randomized"
    pca = PCA(n_components=k_eff, svd_solver=solver, random_state=0)
    pca.fit(X)
    meta = {"k_requested": k, "k_effective": k_eff, "n_rows": X.shape[0], "hidden_dim": X.shape[1],
            "svd_solver": solver}
    dictionary = PCADictionary(pca.mean_.astype(np.float32), pca.components_.astype(np.float32),
                                pca.explained_variance_ratio_.astype(np.float32), meta)
    metrics = {"k": k_eff, "cumulative_explained_variance": float(pca.explained_variance_ratio_.sum())}
    return dictionary, metrics


def slice_pca(dictionary, k):
    """A PCADictionary using only the first k of an already-fit dictionary's
    components. Exact, not an approximation: sklearn's svd_solver="full"
    computes the complete decomposition regardless of the n_components it
    was asked for, so the first k components of a k_max-component fit are
    identical to fitting directly with n_components=k. Fitting once at
    max(k_grid) and slicing down avoids redoing the same full SVD once per
    k in the grid."""
    k_eff = min(k, dictionary.k)
    meta = {**dictionary.meta, "k_requested": k, "k_effective": k_eff, "sliced_from_k": dictionary.k}
    return PCADictionary(dictionary.mean_, dictionary.components_[:k_eff],
                          dictionary.explained_variance_ratio_[:k_eff], meta)


# ---------------------------------------------------------------------------
# SAE dictionary
# ---------------------------------------------------------------------------

class _SAEModule(nn.Module):
    """Standard L1-sparse autoencoder (Anthropic "Towards Monosemanticity"
    formulation): subtract a decoder pre-bias before encoding, ReLU
    nonlinearity, tied-free encoder/decoder weights, decoder columns
    renormalized to unit norm every step so the model can't cheat sparsity
    by shrinking the decoder instead of the code.

        f = ReLU(W_enc (x - b_dec) + b_enc)
        x_hat = W_dec f + b_dec
    """

    def __init__(self, hidden_dim, dict_size):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(dict_size, hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, dict_size))
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._renorm_decoder()

    def _renorm_decoder(self):
        with torch.no_grad():
            norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.W_dec.div_(norms)

    def encode(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


class SAEDictionary:
    method = "sae"

    def __init__(self, module, meta):
        self.module = module
        self.meta = meta

    @property
    def dict_size(self):
        return self.module.W_enc.shape[0]

    def encode(self, X):
        self.module.eval()
        with torch.no_grad():
            f = self.module.encode(torch.from_numpy(X.astype(np.float32)))
        return f.numpy()

    def decode(self, F):
        self.module.eval()
        with torch.no_grad():
            x_hat = self.module.decode(torch.from_numpy(F.astype(np.float32)))
        return x_hat.numpy()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"method": self.method, "state_dict": self.module.state_dict(),
                    "hidden_dim": self.module.W_dec.shape[0], "dict_size": self.dict_size,
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        assert d["method"] == cls.method, f"{path} is not an SAE dictionary (method={d['method']!r})"
        module = _SAEModule(d["hidden_dim"], d["dict_size"])
        module.load_state_dict(d["state_dict"])
        return cls(module, d["meta"])


def fit_sae(X, dict_size, l1_coef=1e-3, epochs=300, lr=1e-3, val_frac=0.15, seed=0, device="cpu"):
    """X: [n_rows, hidden_dim] (already flattened over positions/images).
    Full-batch Adam (n_rows is at most a few thousand here, so mini-batching
    buys nothing). Reports final train/val reconstruction MSE and mean L0
    (average number of nonzero latents per row) so fit_dictionaries.py can
    print a sanity signal per layer without a separate eval script."""
    rng = np.random.RandomState(seed)
    n_rows, hidden_dim = X.shape
    perm = rng.permutation(n_rows)
    n_val = max(1, int(round(n_rows * val_frac))) if n_rows > 1 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train = torch.from_numpy(X[train_idx].astype(np.float32)).to(device)
    X_val = torch.from_numpy(X[val_idx].astype(np.float32)).to(device) if n_val else None

    torch.manual_seed(seed)
    module = _SAEModule(hidden_dim, dict_size).to(device)
    opt = torch.optim.Adam(module.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        module.train()
        opt.zero_grad()
        x_hat, f = module(X_train)
        recon_loss = ((x_hat - X_train) ** 2).mean()
        l1_loss = f.abs().mean()
        loss = recon_loss + l1_coef * l1_loss
        loss.backward()
        opt.step()
        module._renorm_decoder()

        if epoch == epochs - 1 or epoch % max(1, epochs // 10) == 0:
            module.eval()
            with torch.no_grad():
                val_recon = None
                if X_val is not None:
                    x_hat_val, f_val = module(X_val)
                    val_recon = ((x_hat_val - X_val) ** 2).mean().item()
                    val_l0 = (f_val > 0).float().sum(dim=1).mean().item()
                else:
                    val_l0 = (f > 0).float().sum(dim=1).mean().item()
            history.append({"epoch": epoch, "train_recon_mse": recon_loss.item(),
                             "train_l0": (f > 0).float().sum(dim=1).mean().item(),
                             "val_recon_mse": val_recon, "val_l0": val_l0})

    meta = {"dict_size": dict_size, "l1_coef": l1_coef, "epochs": epochs, "lr": lr,
             "val_frac": val_frac, "seed": seed, "n_rows": n_rows, "hidden_dim": hidden_dim,
             "history": history}
    dictionary = SAEDictionary(module.cpu(), meta)
    metrics = {"dict_size": dict_size, **{k: v for k, v in history[-1].items() if k != "epoch"}}
    return dictionary, metrics


# ---------------------------------------------------------------------------
# Dictionary path convention + generic load
# ---------------------------------------------------------------------------

def dictionary_path(entity, token_set, layer, method, size, dictionaries_dir=DICTIONARIES_DIR):
    tag = f"k{size}" if method == "pca" else f"d{size}"
    return os.path.join(dictionaries_dir, entity, token_set, f"layer{layer}_{method}_{tag}.pt")


def load_dictionary(path):
    method = torch.load(path, map_location="cpu", weights_only=False)["method"]
    return {"pca": PCADictionary, "sae": SAEDictionary}[method].load(path)


def list_fitted_layers(entity, token_set, method, dictionaries_dir=DICTIONARIES_DIR):
    """Layers with at least one fitted dictionary on disk for this
    (entity, token_set, method), ascending -- lets select_features.py default
    its --layers sweep to "whatever fit_dictionaries.py already produced"
    instead of requiring the user to respell the same layer list twice."""
    set_dir = os.path.join(dictionaries_dir, entity, token_set)
    if not os.path.isdir(set_dir):
        return []
    prefix = f"_{method}_"
    layers = set()
    for fname in os.listdir(set_dir):
        if fname.startswith("layer") and prefix in fname and fname.endswith(".pt"):
            layers.add(int(fname[len("layer"):fname.index("_")]))
    return sorted(layers)


def list_fitted_sizes(entity, token_set, layer, method, dictionaries_dir=DICTIONARIES_DIR):
    """Sizes (k for pca, dict_size for sae) with a fitted dictionary on disk
    for this (entity, token_set, layer, method), largest first."""
    layer_dir = os.path.join(dictionaries_dir, entity, token_set)
    if not os.path.isdir(layer_dir):
        return []
    prefix, suffix = f"layer{layer}_{method}_", ".pt"
    tag_char = "k" if method == "pca" else "d"
    sizes = []
    for fname in os.listdir(layer_dir):
        if fname.startswith(prefix) and fname.endswith(suffix):
            tag = fname[len(prefix):-len(suffix)]
            if tag.startswith(tag_char):
                sizes.append(int(tag[1:]))
    return sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------
# VADE ground-truth loading (entity-ID + attribute labels for step 4)
# ---------------------------------------------------------------------------

ITEMS_KEY_BY_ENTITY = {"flags": "countries", "brands": "brands", "animals": "species"}


def load_labels(vade_root, entity, item_codes):
    """Returns (attributes, id_labels, attr_labels) where:
      attributes: list of attribute field names, from ground_truth.json's
        own "attributes" list (generic across entities -- no hardcoding).
      id_labels: [n_images] array of item_codes themselves (1 example/class,
        the deliberately-weak entity-ID label -- see plan item 4).
      attr_labels: {attribute: [n_images] object array, "" for items
        missing that attribute} -- callers must filter "" out per-attribute
        before fitting/evaluating (e.g. brands' founded_year is missing for
        36/129 items).
    item_codes must match the order activations were extracted in (both
    ultimately come from sorted(ground_truth.json's items dict), so they
    line up by construction -- this function asserts that rather than
    assuming it silently).
    """
    gt_path = os.path.join(vade_root, "data", entity, "ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)
    items_key = ITEMS_KEY_BY_ENTITY[entity]
    items = gt[items_key]
    expected_codes = sorted(items)
    if list(item_codes) != expected_codes:
        raise ValueError(
            f"item_codes from the activations file don't match sorted(ground_truth.json) for "
            f"entity {entity!r} -- activations and labels are out of sync (re-run sae.py?).")

    attributes = gt["attributes"]
    id_labels = np.array(item_codes, dtype=object)
    attr_labels = {}
    for attr in attributes:
        attr_labels[attr] = np.array([str(items[c].get(attr, "") or "") for c in item_codes], dtype=object)
    return attributes, id_labels, attr_labels
