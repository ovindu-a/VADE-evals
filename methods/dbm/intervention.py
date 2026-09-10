"""DBM (Differential Binary Masking, Cao et al. 2020/2022) intervention
module: a sigmoid-gated binary mask over the RAW residual-stream
dimensions -- identity featurizer, F_A(n) = n, no learned rotation (unlike
DAS/MDAS). Reuses pyvene's own SigmoidMaskIntervention verbatim rather than
reimplementing it: RAVEL's own Appendix B.4 (Huang et al. 2024, ACL,
https://aclanthology.org/2024.acl-long.470/) says "For DBM- and DAS-based
methods, we use the implementation from the pyvene library", and the class
is a byte-for-byte match to the paper's formula:

    n = (1 - sigma(m/T)) . GetVals(M(x), N) + sigma(m/T) . GetVals(M(x'), N)
    L_Cause = CE(tau(M_{N<-n}(x)), A_E') + lambda * ||m||_1

m in R^H (H = hidden_size, e.g. 3584 for Qwen2.5-VL-7B-Instruct) is ONE
mask, shared across every token position in whichever position set this
run targets -- matches how DAS's rotation and PCA/SAE's feature_indices
are already applied uniformly across a token set's positions elsewhere in
this project (see methods/common/entities.py's resolve_position_set and
methods/dbm/train.py, which builds one SigmoidMaskIntervention per
(entity, attribute, layer) and applies it identically at every position
build_batch resolves for that positions_name).

We deliberately do NOT go through pyvene's own IntervenableModel/
IntervenableConfig wrapper -- that machinery is built and tested against
plain text-only HF decoder models, with no documented support for a VLM's
extra forward kwargs (pixel_values/image_grid_thw) or Qwen2.5-VL's mRoPE
position handling. SigmoidMaskIntervention itself is a plain nn.Module
(verified directly against pyvene's own Intervention.__init__: constructing
SigmoidMaskIntervention(embed_dim=H) needs no config object or registry),
so it drops straight into this project's own hook mechanism
(methods/common/hooks.py, already proven against Qwen2.5-VL by
methods/das/) with zero adaptation -- see train.py/eval.py.

Note on temperature: SigmoidMaskIntervention.temperature is an
nn.Parameter, but its forward() rebuilds it via torch.tensor(self.
temperature) every call, which detaches it from the autograd graph -- so
it never receives a gradient (verified empirically: .backward() leaves
temperature.grad as None). This matches the paper's design intent, not a
bug we need to route around: temperature is annealed via an explicit
schedule (temperature_schedule below, called every optimizer step from
train.py), never learned.
"""
import torch

from pyvene.models.interventions import SigmoidMaskIntervention

__all__ = ["SigmoidMaskIntervention", "temperature_schedule", "l1_penalty", "dbm_config_tag", "mask_stats"]


def temperature_schedule(num_steps, temperature_start=1e-2, temperature_end=1e-7, dtype=torch.float32):
    """Geometric temperature anneal, DECREASING start->end -- RAVEL's own
    Appendix B.4: "for DBM and MDBM, we use a starting temperature of 1e-2
    and gradually reducing it to 1e-7." Returns one temperature per
    OPTIMIZER STEP (not per epoch -- with train.py's default of 1 epoch, a
    per-epoch schedule would just be a single flat value for the whole run,
    which isn't annealing at all). Pass schedule[step] to intervention.
    set_temperature(...) once per completed optimizer step; see train.py."""
    ratio = temperature_end / temperature_start
    return (temperature_start * ratio ** (torch.arange(num_steps) / max(num_steps - 1, 1))).to(dtype)


def l1_penalty(intervention):
    """lambda * ||m||_1 from the paper's L_Cause, applied to the RAW mask
    parameter (before the sigmoid/temperature reparameterization) -- matches
    the paper's literal notation. Caller multiplies by a lambda coefficient
    (RAVEL's own reported optimum for DBM: ~0.001, Appendix B.4) and adds it
    to the CE loss -- see train.py."""
    return intervention.mask.abs().sum()


def mask_stats(intervention, epsilon=1e-2):
    """The paper's own discretization of the trained CONTINUOUS mask into the hard feature set it
    actually claims to have found: "FA is the set of dimensions i where 1-sigma(mi/T) < epsilon"
    -- i.e. selected iff sigma(mask_i/T) > 1-epsilon (very close to fully swapping in the source).
    Nothing downstream of training NEEDS this (eval.py's generation-time intervention uses the
    continuous mask directly, which is already near-binary once temperature has annealed down) --
    this exists purely so the trained artifact is actually SUMMARIZED somewhere human/machine-
    readable, the same way PCA/SAE/DAS's own winners.json reports n_features_selected/
    feature_indices. Without it, answering "how many/which dimensions did this layer actually
    select" means manually loading layer{L}_intervention.pt and re-deriving this by hand.

    Returns a JSON-able dict: embed_dim, epsilon, n_selected, selected_indices (sorted python ints
    -- this project's convention elsewhere, e.g. select_features.py's own feature_indices), the
    temperature this was computed at (normally the fully-annealed end-of-training value), and a
    few raw-mask/sigmoid summary stats as a sanity check independent of epsilon (e.g. sigmoid_mean
    close to 0.5 everywhere would mean training never meaningfully differentiated any dimension,
    whatever a given epsilon's selection count claims)."""
    with torch.no_grad():
        temperature = intervention.get_temperature().item()
        sigmoid = torch.sigmoid(intervention.mask / intervention.temperature.detach())
        selected_indices = sorted((1 - sigmoid < epsilon).nonzero(as_tuple=True)[0].tolist())
        return {
            "embed_dim": int(intervention.mask.shape[0]), "epsilon": epsilon,
            "n_selected": len(selected_indices), "selected_indices": selected_indices,
            "temperature": temperature,
            "mask_min": intervention.mask.min().item(), "mask_max": intervention.mask.max().item(),
            "mask_mean": intervention.mask.mean().item(), "sigmoid_mean": sigmoid.mean().item(),
        }


def dbm_config_tag(l1_coef, temperature_start, temperature_end, lr, positions, pruned=False,
                    min_lr_ratio=0.0, grad_clip_norm=1.0):
    """Encodes every hyperparameter that changes the trained artifact -- NOT just l1_coef. An earlier
    version omitted temperature_start/temperature_end here, so two runs differing only in temperature
    schedule would collide into the same results/logs directory (the same class of gotcha this
    project's own select_features.py already hit with --dictionaries_dir -- see its README section).
    lr was added for the same reason the moment --lr became a real, sweep-worthy flag (train.py's
    LR=1e-3 default is copied from DAS, whose D x D/D x K orthogonal rotation is a very different
    parametrization from DBM's plain mask vector -- worth sweeping, not assuming). min_lr_ratio/
    grad_clip_norm added the same day both became real flags (see train.py's train_layer) -- default
    values (0.0, 1.0) match this project's pre-existing hardcoded behavior exactly, so this alone
    changes every existing tag's string (new MLR/CLIP segments appear even at defaults) -- deliberate,
    not an oversight: it's the same "always include, never omit-at-default" convention l1_coef/lr
    already follow here, and the alternative (omit at default) would mean a min_lr_ratio=0 run and a
    min_lr_ratio=0.3 run silently colliding into the same directory the moment someone forgets which
    one was the "default" at the time."""
    return (f"L1_{l1_coef}_T{temperature_start:g}-{temperature_end:g}_LR{lr:g}_MLR{min_lr_ratio:g}_"
            f"CLIP{grad_clip_norm:g}_{positions}") + ("_pruned" if pruned else "")
