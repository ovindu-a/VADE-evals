"""NDM (Native Dictionary Masking) -- shared naming, paths and CLI plumbing.

WHAT NDM IS. Same trainer as DBM (a sigmoid-gated mask, temperature-annealed
toward binary, trained end-to-end against teacher-forced generation CE plus
an L1 sparsity term, used for a source->base interchange intervention), but
the mask is learned over a decoder block's MLP HIDDEN STATE -- the
post-SwiGLU neuron vector act_fn(gate_proj(x)) * up_proj(x), width
intermediate_size (18944 on Qwen2.5-VL-7B-Instruct) -- instead of over the
residual stream (width hidden_size, 3584).

WHY IT IS NOT JUST "DBM AT ANOTHER LAYER". In RAVEL's own framing every
method in this family is characterized by its featurizer F_A:

    DBM        F_A(n) = n                    identity, no parameters
    DAS/MDAS   F_A = learned rotation R      linear, learned
    PCA/SAE    F_A = fitted dictionary       linear, learned offline
    NDM        F_A = the model's own MLP     NONLINEAR, zero parameters

Two things follow, and together they make this a different method rather
than a hyperparameter of DBM:

  1. There is a DECODER between the masked vector and the causal variable.
     DBM's blend lands directly in the residual stream; NDM's blend passes
     through down_proj first. Because down_proj is linear,

         resid = base_resid + down_proj((1-s).h_base + s.h_source)
               = base_resid + (1-s).down_proj(h_base) + s.down_proj(h_source)

     i.e. an axis-aligned binary mask in neuron space induces a
     NON-axis-aligned intervention in residual space -- a linear image of a
     binary mask. That is not the hypothesis class DBM searches; it is much
     closer to DAS, except the "rotation" is down_proj, given by the
     architecture rather than learned (and is an 18944->3584 projection
     rather than a square rotation).

  2. The encoder is nonlinear. act_fn(gate_proj(x)) * up_proj(x) is the
     only featurizer in the table above that isn't linear -- and it is the
     reason the space has a privileged basis at all: an elementwise
     nonlinearity and an elementwise gating product mean the only
     function-preserving transformations of that space are PERMUTATIONS,
     so its coordinates are real objects rather than a choice of axes. The
     residual stream has no such property (rotate it, fix up every read/
     write matrix, and the model computes the same function), which is why
     DBM's selected dimensions are a fact about a particular checkpoint's
     arbitrary coordinate system rather than about the model. See
     common/sites.py's module docstring, and Elhage et al.'s "Toy Models of
     Superposition" / "Privileged Bases in the Transformer Residual Stream"
     for the argument itself.

So: NDM is "SAE-style encode -> mask -> decode patching where the dictionary
is the model's own MLP, and therefore needs no fitting and has no
reconstruction error."

HOW THE CODE IS ORGANIZED. NDM owns its NAME, its CLI and its results
namespace; it does NOT own a copy of the training loop. methods/dbm/'s
train_layer/eval_layer/run_one_layer are the shared engine, parameterized by
a common/sites.py InterventionSite -- because what differs between the two
methods is ~40 lines (hook attachment, mask width, source-side capture)
against ~800 lines of infrastructure whose value is precisely that it has
already been debugged: the gradient-accumulation tail flush (without which
~24% of rows contribute no training signal), the temperature-schedule resume
anchor (without which temperature jumps back UP on resume), checkpoint/
resume, progress logging, the predictions format score.py expects. Forking
that would mean maintaining those fixes twice. See commit 77470b4.

Results therefore land under results/<model_slug>/<entity>/ndm/... -- a
separate tree from dbm/'s, so NDM gets its own summary files and its own row
in every comparison table, and so the existing committed DBM results stay
exactly where they are.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

from methods.common.results import logs_dir, results_dir  # noqa: E402
from methods.common.sites import InterventionSite  # noqa: E402
from methods.dbm.intervention import dbm_config_tag  # noqa: E402
from methods.dbm.train import DBM_L1_COEF, LR, NUM_EPOCHS, TEMP_END, TEMP_START  # noqa: E402

METHOD_NAME = "ndm"

# The sites NDM is *about*. "residual" is deliberately NOT here -- that's DBM, and it has its own
# module/results tree; pointing NDM at it would produce two names for one method. mlp_output is
# included because it's the control arm that makes the comparison interpretable: it has the same
# width as the residual stream but is local to one block's MLP, so residual-vs-mlp_output isolates
# LOCALITY while mlp_output-vs-mlp_hidden isolates the PRIVILEGED BASIS. Without it, a win for
# mlp_hidden over residual confounds the two explanations.
NDM_SITES = ("mlp_hidden", "mlp_output")
DEFAULT_SITE = "mlp_hidden"


def ndm_config_tag(l1_coef, temperature_start, temperature_end, lr, positions, site, pruned=False):
    """DBM's tag plus the site, which MUST be encoded: mlp_hidden and
    mlp_output runs are different artifacts and would otherwise collide in
    one directory. This repo has now been bitten by exactly this class of
    bug twice (select_features.py's --dictionaries_dir, then dbm_config_tag
    itself omitting the temperature schedule -- fixed in 77470b4), so the
    site goes in from the start rather than after a run overwrites another.

    Reads <hparams>_<positions>_<site>[_pruned] -- site before the pruned
    marker, so the pruned suffix stays last as it is everywhere else."""
    assert site in NDM_SITES, f"site {site!r} is not an NDM site -- expected one of {NDM_SITES}"
    return dbm_config_tag(l1_coef, temperature_start, temperature_end, lr, positions, pruned=False) + \
        f"_{site}" + ("_pruned" if pruned else "")


def ndm_results_dir(model_slug, entity, attribute, l1_coef, temperature_start, temperature_end, lr, positions,
                     site, pruned=False):
    return results_dir(REPO_ROOT, model_slug, entity, METHOD_NAME, attribute,
                        ndm_config_tag(l1_coef, temperature_start, temperature_end, lr, positions, site, pruned))


def ndm_logs_dir(model_slug, entity, attribute, l1_coef, temperature_start, temperature_end, lr, positions,
                  site, pruned=False):
    return logs_dir(REPO_ROOT, model_slug, entity, METHOD_NAME, attribute,
                     ndm_config_tag(l1_coef, temperature_start, temperature_end, lr, positions, site, pruned))


def add_shared_args(ap):
    """Every NDM CLI's common arguments -- identical in meaning to DBM's own
    (see methods/dbm/train.py's main() for the long-form help on each) plus
    --site."""
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True)
    ap.add_argument("--layer", type=int, required=True,
                     help="Decoder block index in hooks.py's convention: layer L addresses block L-1's MLP -- "
                          "the same block whose OUTPUT is residual-stream layer L, so an NDM run at --layer L is "
                          "directly comparable to a DBM run at --layer L. L=0 is invalid here (the embedding "
                          "output has no MLP).")
    ap.add_argument("--site", default=DEFAULT_SITE, choices=list(NDM_SITES),
                     help=f"Which tensor of block layer-1 to mask (default {DEFAULT_SITE}). mlp_hidden is the "
                          "post-SwiGLU neuron vector (width intermediate_size, privileged basis) -- the method. "
                          "mlp_output is that MLP's contribution to the residual stream (width hidden_size, no "
                          "privileged basis) -- the control arm that separates locality from basis. See config.py.")
    ap.add_argument("--positions", default="flag_ring1")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--l1_coef", type=float, default=DBM_L1_COEF,
                     help=f"Sparsity coefficient on ||m||_1 (default {DBM_L1_COEF}). NOTE: this default is RAVEL's "
                          "reported optimum for DBM over a ~4096-wide RESIDUAL stream. l1_penalty is a plain "
                          "mask.abs().sum(), so at intermediate_size=18944 the penalty term is ~5.3x larger for "
                          "the same per-dimension mask magnitude -- this value is NOT calibrated for mlp_hidden and "
                          "should be swept, not inherited. The config tag encodes it, so parallel sweeps won't "
                          "collide.")
    ap.add_argument("--temperature_start", type=float, default=TEMP_START)
    ap.add_argument("--temperature_end", type=float, default=TEMP_END)
    ap.add_argument("--lr", type=float, default=LR, help="See methods/dbm/train.py --lr.")
    ap.add_argument("--allow_unpruned", action="store_true",
                     help="See methods/dbm/train.py --allow_unpruned -- pruned tuples required by default.")
    ap.add_argument("--no_source_cache", action="store_true",
                     help="Disable NDM's own per-(site, positions, layer) source-activation cache "
                          "(common/site_source_cache.py, ON by default) and take a live source forward pass per "
                          "micro-batch instead -- roughly doubling train time. This is a DIFFERENT cache from the "
                          "one DBM/DAS share (common/source_cache.py holds the residual stream only, so an MLP "
                          "site cannot read it); both live under --vade_root/results/ since both are keyed by "
                          "entity+model rather than by method. Ignored for positions=last_token, which is not "
                          "cacheable at any site.")
    ap.add_argument("--out_dir", default=None,
                     help="Defaults to results/<model_slug>/<entity>/ndm/<attribute>/<config_tag>/.")
    return ap


def resolve_run(args):
    """The shared (model_slug, pruned, tuples_dir, out_dir, log_dir, site)
    resolution every NDM CLI does identically. Imports of VADE-side helpers
    happen here rather than at module load so --vade_root is already known."""
    from methods.common.entities import require_pruned_tuples

    assert args.layer >= 1, (
        f"--layer {args.layer} is invalid for an NDM site: layer L addresses decoder block L-1's MLP, and "
        f"L=0 is the embedding output, which has no MLP. Use --layer >= 1.")
    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    out_dir = args.out_dir or ndm_results_dir(
        model_slug, args.entity, args.attribute, args.l1_coef, args.temperature_start, args.temperature_end,
        args.lr, args.positions, args.site, pruned)
    log_dir = ndm_logs_dir(
        model_slug, args.entity, args.attribute, args.l1_coef, args.temperature_start, args.temperature_end,
        args.lr, args.positions, args.site, pruned)
    return model_slug, pruned, tuples_dir, out_dir, log_dir, InterventionSite(args.site)


def source_cache_for(args, adapter, model, processor, entity_assets, site, layer, model_slug):
    """This run's source-activation cache for one layer, or None if disabled
    or not cacheable. NDM's cache is per-(site, positions, layer) -- see
    common/site_source_cache.py for why it can't mirror the residual cache's
    all-layers-in-one-file shape -- so a layer sweep calls this once per
    layer (via run_sweep's source_cache_for_layer hook) rather than once up
    front the way DBM's sweep does."""
    if args.no_source_cache:
        print(f"[{METHOD_NAME}] --no_source_cache: taking a live source forward pass per micro-batch")
        return None
    from methods.common.site_source_cache import get_or_build_site_source_cache
    return get_or_build_site_source_cache(adapter, model, processor, entity_assets, site, layer,
                                           args.positions, args.vade_root, model_slug)


__all__ = [
    "METHOD_NAME", "NDM_SITES", "DEFAULT_SITE", "DEFAULT_VADE_ROOT", "REPO_ROOT",
    "DBM_L1_COEF", "LR", "NUM_EPOCHS", "TEMP_END", "TEMP_START",
    "ndm_config_tag", "ndm_results_dir", "ndm_logs_dir", "add_shared_args", "resolve_run",
    "source_cache_for",
]
