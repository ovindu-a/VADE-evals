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

__all__ = ["SigmoidMaskIntervention", "temperature_schedule", "l1_penalty", "dbm_config_tag"]


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


def dbm_config_tag(l1_coef, temperature_start, temperature_end, positions, pruned=False):
    """Encodes every hyperparameter that changes the trained artifact -- NOT just l1_coef. An earlier
    version omitted temperature_start/temperature_end here, so two runs differing only in temperature
    schedule would collide into the same results/logs directory (the same class of gotcha this
    project's own select_features.py already hit with --dictionaries_dir -- see its README section)."""
    return f"L1_{l1_coef}_T{temperature_start:g}-{temperature_end:g}_{positions}" + ("_pruned" if pruned else "")
