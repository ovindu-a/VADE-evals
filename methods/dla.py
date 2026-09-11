"""DIRECT LOGIT ATTRIBUTION: how many logits each component writes toward the
answer, from ONE forward pass, exactly.

The residual stream is a sum -- the final pre-norm state is

    resid_final = embed + sum_b (attn_output[b] + mlp_output[b])

-- and the model's head is a norm followed by a bias-free linear map. A norm
is nonlinear, so that additive structure does not survive it on its own; but
its only nonlinearity is a per-position SCALAR (rsqrt of the mean square).
Freeze that scalar at the value the real forward pass computed, from the FULL
final residual, and the remaining map is linear, so the answer's logit splits
exactly into one term per component:

    logit(t) = sum_c  scale * ( component_c . (W_U[t] * final_norm.weight) )
                      \\____________________  ______________________________/
                                           \\/
                        adapter.final_norm_scale / adapter.logit_direction

This is NOT an approximation and not a counterfactual: no patching, no
gradient, no second run. It is the model's own answer logit, decomposed. That
makes it the cheapest honest way to ask "which components help make the
identification right", and -- unlike attribution patching -- it has no
first-order error term to be wrong about.

WHAT IT DOES AND DOES NOT TELL YOU. It measures DIRECT paths to the logit
only. A component that matters by changing a later attention head's pattern
(so that head reads a different token) contributes through that head, and
shows up under the head, not under itself. So a zero here means "writes
nothing toward the answer direction itself", NOT "causally irrelevant" --
the causal question is what methods/ndm/ceiling_sweep.py answers, and the two
disagree productively. It is also, like every per-component attribution, a
MARGINAL measure: it assigns one number per component and therefore cannot
represent a conjunction. If an attribute is encoded redundantly across depth
(the signature being a live `residual` ceiling at a layer whose individual
sublayers all read 0 -- see ceiling_sweep's module docstring), expect the DLA
mass to be spread thin across many blocks rather than concentrated, and read
that spreading as the finding rather than as a null result.

THE SELF-CHECK IS THE POINT. Two things are verified per row rather than
assumed, because every failure mode here (hooking the wrong tensor,
double-applying the final norm, an off-by-one over blocks, a head with a bias)
produces plausible-looking numbers instead of an error:

  reconstruction  embed + sum of every sublayer output, compared against the
                  model's OWN residual stream at hidden_states[n_layers-1].
                  Isolates hook correctness from the norm/direction algebra.
  additivity      the per-component terms summed, compared against the real
                  logit quantity read off out.logits. End-to-end.

Both are reported as relative errors and asserted. bf16 accumulation means
they are never exactly 0; --tolerance sets the bar (default 2%).

DIRECTIONS (--direction). What "the answer's logit" means is a choice, and it
changes what the numbers mean:

  gold_minus_mean  (default) the gold token's logit minus the mean logit over
                   the vocabulary. Standard: a component that lifts every
                   logit equally gets no credit, only one that lifts the gold
                   token RELATIVE to everything else.
  gold             the raw gold logit. Simplest, but rewards components that
                   just push the whole distribution up.
  base_minus_source  the base's gold token minus the SOURCE row's gold token.
                   The VADE-specific one, and usually the most on-point: it is
                   the exact axis `cause` moves along, so it ranks components
                   by how hard they hold the answer at the base's value and
                   away from the source's.

Built on probe_common.py's real question+prefill machinery and scored at the
same teacher-forced answer positions as methods/logit_lens.py -- and
score_one_row() is pure w.r.t. an already-computed forward pass, so
methods/mech_probe.py runs this off the SAME pass as the logit lens and the
attention maps. The sublayer outputs come from common/sites.py's own capture
hook, i.e. the identical tensors the interventions read and patch.

Usage:
    python methods/dla.py --entity flags --attribute language --dry_run   # no GPU
    python methods/dla.py --entity flags --attribute language --limit 12
    python methods/dla.py --entity flags --attribute language \
        --direction base_minus_source --positions last_token
"""
import argparse
import contextlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

import torch  # noqa: E402

from methods.adapters.registry import get_adapter  # noqa: E402
from methods.common.entities import BuildBatchCache, load_entity_assets, require_pruned_tuples  # noqa: E402
from methods.common.sites import InterventionSite  # noqa: E402
from methods.common.targets import MAX_ANSWER_TOKENS  # noqa: E402
from methods.probe_common import build_probe_batch, load_probe_rows, teacher_forced_forward  # noqa: E402

DIRECTIONS = ("gold_minus_mean", "gold", "base_minus_source")
DEFAULT_DIRECTION = "gold_minus_mean"
DEFAULT_TOLERANCE = 0.02
_VOCAB_CHUNK = 8192


def component_names(n_layers):
    """Every component of the final residual stream, in the order the forward
    pass adds them -- so a running sum over this list IS the residual stream
    being built up, and its partial sums are the logit-lens curve."""
    names = ["embed"]
    for b in range(n_layers):
        names += [f"attn{b}", f"mlp{b}"]
    return names


@contextlib.contextmanager
def capture_sublayers(adapter, model, n_layers, readout_start, n_keep=MAX_ANSWER_TOKENS):
    """Registers every block's attn_output and mlp_output capture hook for the
    duration of ONE forward pass, sliced to the n_keep read-out columns
    starting at readout_start. Yields {component name: [1, n_keep, H]}, filled
    in by the time the block exits.

    Slicing INSIDE the hook is what makes this cheap: 2*28 full [1, T~1500,
    3584] bf16 tensors would be ~600MB held at once, versus ~1.6MB for the
    answer-position windows actually scored.

    Uses common/sites.py's own capture hook rather than a second set: a
    contiguous column range passed as `positions` slices a window instead of a
    scatter, and reusing it means DLA is reading exactly the tensors the
    interventions patch (and inherits its self_attn-returns-a-tuple handling)."""
    cols = torch.arange(readout_start, readout_start + n_keep).unsqueeze(0)  # [1, n_keep]
    sinks, handles = {}, []
    try:
        for b in range(n_layers):
            for site_name, key in (("attn_output", f"attn{b}"), ("mlp_output", f"mlp{b}")):
                sinks[key] = []
                # layer_idx = b + 1: sites.py's convention is that layer_idx addresses block
                # layer_idx-1, the same convention hidden_states uses. Off by one here would
                # attribute every contribution to its neighbour and still sum correctly.
                handles.append(InterventionSite(site_name).register_capture(
                    adapter, model, b + 1, cols, sinks[key]))
        yield sinks
    finally:
        for h in handles:
            h.remove()


def collect_components(adapter, model, out, sinks, n_layers, readout_start, n_keep=MAX_ANSWER_TOKENS):
    """capture_sublayers' sinks + the embedding -> {name: [n_keep, H] float32}."""
    comps = {}
    embed = out.hidden_states[0][0, readout_start:readout_start + n_keep, :]
    comps["embed"] = embed.float()
    for key, sink in sinks.items():
        assert len(sink) == 1, (
            f"component {key!r}'s capture hook fired {len(sink)} times in one forward pass, expected 1 "
            f"-- gradient checkpointing or a reused module would make 'the' activation ambiguous")
        comps[key] = sink[0][0].float()
    missing = [n for n in component_names(n_layers) if n not in comps]
    assert not missing, f"missing components {missing} -- capture_sublayers did not cover every block"
    return comps


def _mean_logit_direction(adapter, model, vocab_size, device):
    """The vocabulary-mean of logit_direction, for --direction gold_minus_mean.
    Chunked over the vocab because a float32 [V, H] materialization is ~2GB on
    a 152k-token vocabulary."""
    total, n = None, 0
    for lo in range(0, vocab_size, _VOCAB_CHUNK):
        ids = torch.arange(lo, min(lo + _VOCAB_CHUNK, vocab_size), device=device)
        chunk = adapter.logit_direction(model, ids).sum(0)
        total = chunk if total is None else total + chunk
        n += len(ids)
    return total / n


def directions_for_row(adapter, model, batch, direction, j, mean_dir=None):
    """-> (direction vector [H], a description of the scalar it decomposes)."""
    device = model.device
    gold = batch["base_gold_toks"][0][j].to(device)
    d = adapter.logit_direction(model, gold)
    if direction == "gold":
        return d, "logit(base gold)"
    if direction == "gold_minus_mean":
        assert mean_dir is not None
        return d - mean_dir, "logit(base gold) - mean logit over vocab"
    if direction == "base_minus_source":
        src = batch["source_gold_toks"][0][j].to(device)
        return d - adapter.logit_direction(model, src), "logit(base gold) - logit(source gold)"
    raise AssertionError(f"unknown direction {direction!r} -- expected one of {DIRECTIONS}")


def target_value(out, batch, direction, j, mean_dir_unused=None):
    """The real logit quantity the per-component terms must sum to, read off
    the forward pass's own logits. This is the ground truth for the additivity
    check -- deliberately NOT recomputed from the components."""
    logits = out.logits[0, j].float()                       # [V]
    gold = int(batch["base_gold_toks"][0][j].item())
    if direction == "gold":
        return logits[gold]
    if direction == "gold_minus_mean":
        return logits[gold] - logits.mean()
    if direction == "base_minus_source":
        return logits[gold] - logits[int(batch["source_gold_toks"][0][j].item())]
    raise AssertionError(f"unknown direction {direction!r}")


def score_one_row(model, processor, adapter, entity_assets, row, batch, out, sinks, n_layers,
                  direction=DEFAULT_DIRECTION, mean_dir=None, tolerance=DEFAULT_TOLERANCE):
    """Scores ONE row against an already-computed forward pass `out` (from
    teacher_forced_forward(..., output_hidden_states=True,
    logits_to_keep=MAX_ANSWER_TOKENS) run inside capture_sublayers(...)).
    Pure w.r.t. `out`/`sinks`: does no forwarding, so methods/mech_probe.py can
    call this against a pass it is ALSO handing to logit_lens and
    attention_maps.

    Returns (row_record, deltas) where deltas is
    {component: {"sum": float, "n": int}} for the caller to fold into a running
    aggregate (see fold_deltas)."""
    readout_start = batch["readout_start_col"]
    gold_len = int(batch["base_gold_len"][0].item())
    names = component_names(n_layers)
    comps = collect_components(adapter, model, out, sinks, n_layers, readout_start)

    # The frozen scale comes from the RECONSTRUCTED final residual, which makes the
    # decomposition exact by construction; whether that reconstruction is the model's
    # real residual is what the two checks below establish independently.
    resid_final = sum(comps[n] for n in names)                       # [K, H]
    scale = adapter.final_norm_scale(model, resid_final)             # [K, 1]

    # CHECK 1 (hooks): embed + every sublayer of blocks 0..n_layers-2 must reproduce the
    # model's own hidden_states[n_layers-1] (block n_layers-2's output). hidden_states[-1]
    # is deliberately NOT used -- in this project's transformers it is tied to
    # last_hidden_state and is therefore POST-final-norm (see methods/logit_lens.py).
    partial = sum(comps[n] for n in component_names(n_layers - 1))
    ref = out.hidden_states[n_layers - 1][0, readout_start:readout_start + MAX_ANSWER_TOKENS, :].float()
    recon_err = (partial - ref).abs().max().item()
    recon_rel = recon_err / max(ref.abs().max().item(), 1e-6)

    per_position, deltas = [], {n: {"sum": 0.0, "n": 0} for n in names}
    worst_add_rel = 0.0
    for j in range(gold_len):
        d, target_desc = directions_for_row(adapter, model, batch, direction, j, mean_dir)
        contrib = {n: float(scale[j] * torch.dot(comps[n][j], d)) for n in names}

        # CHECK 2 (algebra, end to end): the terms must sum to the model's real logit quantity.
        total = sum(contrib.values())
        target = float(target_value(out, batch, direction, j))
        add_rel = abs(total - target) / max(abs(target), 1.0)
        worst_add_rel = max(worst_add_rel, add_rel)

        per_position.append({
            "j": j, "gold_id": int(batch["base_gold_toks"][0][j].item()),
            "target": target, "sum_of_components": total, "additivity_rel_err": add_rel,
            "contributions": contrib,
        })
        for n in names:
            deltas[n]["sum"] += contrib[n]
            deltas[n]["n"] += 1

    assert recon_rel <= tolerance, (
        f"row {row['row_index']}: embed + every captured sublayer differs from the model's own "
        f"hidden_states[{n_layers - 1}] by {recon_rel:.3%} (max abs {recon_err:.4f}) -- the capture hooks "
        f"are not reading what actually gets added to the residual stream. Raise --tolerance only if you "
        f"have a reason to believe this is bf16 accumulation and not a wrong tensor.")
    assert worst_add_rel <= tolerance, (
        f"row {row['row_index']}: per-component terms sum to {worst_add_rel:.3%} away from the model's own "
        f"logit -- the frozen-scale algebra is wrong (a double-applied final norm, a head bias, or a "
        f"logit_direction that does not fold in the norm's learned gain), not a rounding issue.")

    return {
        "attribute": row["target_attribute"], "row_index": row["row_index"], "base": row["base"],
        "template_id": row["template_id"], "gold_label": row["base_label"], "gold_len": gold_len,
        "direction": direction, "target_desc": target_desc if gold_len else None,
        "reconstruction_rel_err": recon_rel, "max_additivity_rel_err": worst_add_rel,
        "positions": per_position,
    }, deltas


def fold_deltas(agg, deltas):
    for n, d in deltas.items():
        agg.setdefault(n, {"sum": 0.0, "n": 0})
        agg[n]["sum"] += d["sum"]
        agg[n]["n"] += d["n"]
    return agg


def finalize_summary(agg, n_layers):
    """-> {component: {"mean": float, "cumulative": float, "n": int}} in forward
    order, so `cumulative` is the answer logit being built up block by block."""
    out, running = {}, 0.0
    for n in component_names(n_layers):
        a = agg.get(n, {"sum": 0.0, "n": 0})
        mean = a["sum"] / a["n"] if a["n"] else float("nan")
        running += mean if a["n"] else 0.0
        out[n] = {"mean": mean, "cumulative": running, "n": a["n"]}
    return out


def print_summary(summary, n_layers, direction, top_n=12):
    print(f"\n=== direct logit attribution (direction={direction}) ===")
    print(f"{'block':>6} {'attn':>10} {'mlp':>10} {'block total':>12} {'cumulative':>12}")
    print(f"{'embed':>6} {'':>10} {'':>10} {summary['embed']['mean']:>12.3f} "
          f"{summary['embed']['cumulative']:>12.3f}")
    for b in range(n_layers):
        a, m = summary[f"attn{b}"], summary[f"mlp{b}"]
        print(f"{b:>6} {a['mean']:>10.3f} {m['mean']:>10.3f} {a['mean'] + m['mean']:>12.3f} "
              f"{m['cumulative']:>12.3f}")

    ranked = sorted(((n, v["mean"]) for n, v in summary.items()), key=lambda kv: -abs(kv[1]))[:top_n]
    print(f"\ntop {top_n} components by |contribution|:")
    for n, v in ranked:
        print(f"  {n:>10} {v:>+9.3f}")
    total = summary[f"mlp{n_layers - 1}"]["cumulative"]
    print(f"\ntotal (== the model's own {direction} value, averaged): {total:+.3f}")


def score_rows(model, processor, adapter, entity_assets, rows, positions, direction=DEFAULT_DIRECTION,
               tolerance=DEFAULT_TOLERANCE):
    """Standalone path: one forward pass per row, hooks registered around each."""
    n_layers = len(adapter.get_decoder_layers(model))
    batch_cache = BuildBatchCache()
    mean_dir = (_mean_logit_direction(adapter, model, model.lm_head.weight.shape[0], model.device)
                if direction == "gold_minus_mean" else None)
    records, agg = [], {}
    for i, row in enumerate(rows):
        batch = build_probe_batch([row], entity_assets, adapter, model, processor, positions,
                                  batch_cache=batch_cache)
        with capture_sublayers(adapter, model, n_layers, batch["readout_start_col"]) as sinks:
            out = teacher_forced_forward(model, batch, output_hidden_states=True,
                                         logits_to_keep=MAX_ANSWER_TOKENS)
        rec, deltas = score_one_row(model, processor, adapter, entity_assets, row, batch, out, sinks,
                                    n_layers, direction=direction, mean_dir=mean_dir, tolerance=tolerance)
        records.append(rec)
        fold_deltas(agg, deltas)
        if (i + 1) % 10 == 0:
            print(f"  scored {i + 1}/{len(rows)} rows", flush=True)
    return records, finalize_summary(agg, n_layers), n_layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="flags")
    ap.add_argument("--attribute", default="language")
    ap.add_argument("--positions", default="flag_ring1",
                    help="Only selects which rows/prompt geometry are built (DLA patches nothing) -- it "
                         "matters because build_probe_batch needs a position set, not because the "
                         "attribution is restricted to it.")
    ap.add_argument("--direction", default=DEFAULT_DIRECTION, choices=list(DIRECTIONS),
                    help="Which logit quantity to decompose -- see the module docstring. "
                         "base_minus_source is the VADE-specific one and usually the most on-point: it "
                         "is the exact axis `cause` moves along.")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--limit", type=int, default=12, help="Distinct example images.")
    ap.add_argument("--questions_per_image", type=int, default=6)
    ap.add_argument("--top_n", type=int, default=12, help="How many components to rank in the printout.")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="Relative error allowed on BOTH self-checks before the run aborts. bf16 "
                         "accumulation over ~57 components puts the honest floor near 1%%; a failure "
                         "well above that is a wrong tensor, not rounding.")
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Validate rows/positions resolve without "
                    "loading the 7B model -- no GPU needed. Same affordance as logit_lens.py/mech_probe.py.")
    ap.add_argument("--out", default=None, help="Defaults to methods/dla/"
                                                "<entity>_<attribute>_<positions>_<direction>_report.jsonl "
                                                "(per-row) + _summary.json")
    args = ap.parse_args()

    entity_assets = load_entity_assets(args.vade_root, args.entity)
    model_slug = args.model_id.split("/")[-1]
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if not args.allow_unpruned else None)
    rows = load_probe_rows(entity_assets, args.attribute, args.split, tuples_dir=tuples_dir, limit=args.limit,
                           one_per_image=True, templates_per_image=args.questions_per_image)
    print(f"[dla] entity={args.entity} attribute={args.attribute} positions={args.positions} "
          f"direction={args.direction}: {len(rows)} example row(s)")

    if args.dry_run:
        from methods.adapters.qwen2_5_vl import Qwen25VLAdapter
        from methods.common.entities import resolve_position_set
        from transformers import AutoConfig

        class _ConfigOnlyModel:
            def __init__(self, config):
                self.config = config

        adapter = Qwen25VLAdapter(args.model_id)
        shim = _ConfigOnlyModel(AutoConfig.from_pretrained(args.model_id))
        flat_indices, n_image_tokens, _is_last = resolve_position_set(args.positions, entity_assets, adapter, shim)
        for r in rows:
            img = os.path.join(entity_assets.entity_dir, entity_assets.items[r["base"]]["image"])
            assert os.path.exists(img), f"missing image {img}"
        n_obj = len(flat_indices) if flat_indices is not None else "last_token"
        n_layers = shim.config.text_config.num_hidden_layers
        print(f"[dry_run] OK -- positions={args.positions!r} resolves to {n_obj} of {n_image_tokens} image "
              f"tokens; all {len(rows)} row images found on disk. DLA would decompose "
              f"{len(component_names(n_layers))} components (embed + {n_layers} blocks x attn/mlp) per "
              f"answer position. (Model not loaded -- rerun without --dry_run, on a GPU box.)")
        return

    adapter = get_adapter(args.model_id)
    model, processor = adapter.load()

    records, summary, n_layers = score_rows(model, processor, adapter, entity_assets, rows, args.positions,
                                            direction=args.direction, tolerance=args.tolerance)
    print_summary(summary, n_layers, args.direction, top_n=args.top_n)
    worst_recon = max(r["reconstruction_rel_err"] for r in records)
    worst_add = max(r["max_additivity_rel_err"] for r in records)
    print(f"\nself-checks (worst over {len(records)} rows): reconstruction={worst_recon:.3%} "
          f"additivity={worst_add:.3%}  (tolerance {args.tolerance:.1%})")

    stem = args.out or os.path.join(REPO_ROOT, "methods", "dla",
                                    f"{args.entity}_{args.attribute}_{args.positions}_{args.direction}_report")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(stem + "_summary.json", "w") as f:
        json.dump({"entity": args.entity, "attribute": args.attribute, "positions": args.positions,
                   "direction": args.direction, "n_rows": len(records), "n_layers": n_layers,
                   "worst_reconstruction_rel_err": worst_recon, "worst_additivity_rel_err": worst_add,
                   "by_component": summary}, f, indent=2)
    print(f"wrote {stem}.jsonl and {stem}_summary.json")


if __name__ == "__main__":
    main()
