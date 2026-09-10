"""DBM training: learns a SigmoidMaskIntervention (methods/dbm/intervention.py,
pyvene's own class) at one decoder layer, for one target attribute, on one
VADE entity -- model-agnostic (goes through a ModelAdapter) and
entity-agnostic (goes through common/entities.py's generic tuple/asset
loading). Structurally a near-verbatim port of VADE's own methods/das/
train.py (same adapters/common infra, same teacher-forced-CE training
shape) with the intervention class swapped and the L1 sparsity term +
continuous temperature annealing added -- see methods/dbm/intervention.py's
module docstring for why DBM needs neither a rotation matrix nor pyvene's
IntervenableModel wrapper.

Where things live: this repo (VADE-evals) is a sibling of the VADE
benchmark repo (--vade_root, default ../VADE) -- entity assets/tuples/
pruned-tuples/the shared source-activation cache are all read from (and,
for the cache, written to) THAT repo, exactly like every other script in
this project. DBM's own trained artifacts (checkpoints/train logs/
predictions), however, are NOT written into VADE -- they land under THIS
repo's own results/ and logs/ trees (methods/common/results.py's
results_dir/logs_dir are generic over whatever root you pass; DAS passes
VADE's own root since it lives there, we pass this repo's root instead).

Usage:
    python methods/dbm/train.py --entity flags --attribute capital --layer 14 \\
        --positions flag_ring1
"""
import argparse
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))

import torch
from transformers import get_linear_schedule_with_warmup

from methods.adapters.registry import get_adapter
from methods.common.entities import BuildBatchCache, build_batch, load_entity_assets, load_tuples, require_pruned_tuples
from methods.common.hooks import make_cache_aware_patch_hook
from methods.common.results import logs_dir, results_dir
from methods.common.run_logging import tee_to_log
from methods.common.sites import RESIDUAL_SITE
from methods.common.source_cache import get_or_build_source_cache
from methods.common.targets import (
    MAX_ANSWER_TOKENS, build_teacher_forced_extension, gold_labels_from_lens, target_gold_toks_and_len,
)
from methods.dbm.intervention import (
    SigmoidMaskIntervention, dbm_config_tag, l1_penalty, mask_stats, temperature_schedule,
)

SEED = 42
LR = 1e-3
NUM_EPOCHS = 1
DBM_L1_COEF = 1e-3     # RAVEL Appendix B.4's reported optimum for DBM (MDBM's is ~0, not our concern here)
TEMP_START = 1e-2      # RAVEL Appendix B.4: "a starting temperature of 1e-2 and gradually reducing it to 1e-7"
TEMP_END = 1e-7
# Same hardware-forced micro-batch/accum defaults as DAS's train.py (single 24GB card, no gradient
# checkpointing -- conflicts with the forward hooks the intervention needs, so full activations for the
# whole decoder stack are held for backprop). DBM has no D x D rotation matrix to hold, only a length-H
# mask vector, so it may well tolerate a larger --batch_size than DAS's BATCH_SIZE=4 on the same card --
# untested here, raise it and watch VRAM headroom rather than assuming.
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16
CHECKPOINT_EVERY_OPT_STEPS = 5
# Unlike DAS's train.py (which this file otherwise ports near-verbatim), this prints a one-line
# progress update every completed optimizer step -- with num_epochs defaulting to 1 and a real
# tuples file running to thousands of rows, DAS's own per-epoch-only printing means total console
# silence for the whole run until it's done. Same gap RESULTS.md documents having to patch into
# select_features.py's layer sweep, for the same reason -- fixed here up front instead.
PROGRESS_EVERY_OPT_STEPS = 1


def dbm_results_dir(model_slug, entity, attribute, l1_coef, temperature_start, temperature_end, lr, positions,
                     pruned=False):
    return results_dir(REPO_ROOT, model_slug, entity, "dbm", attribute,
                        dbm_config_tag(l1_coef, temperature_start, temperature_end, lr, positions, pruned))


def dbm_logs_dir(model_slug, entity, attribute, l1_coef, temperature_start, temperature_end, lr, positions,
                  pruned=False):
    return logs_dir(REPO_ROOT, model_slug, entity, "dbm", attribute,
                     dbm_config_tag(l1_coef, temperature_start, temperature_end, lr, positions, pruned))


def cache_source_layer_hidden(adapter, model, batch, layer_idx, site=RESIDUAL_SITE):
    return site.capture(adapter, model, layer_idx, batch["source_input_ids"], batch["attention_mask"],
                         batch["source_extra"], batch["positions"])


def run_intervened_forward(adapter, model, layers, batch, layer_idx, intervention, source_hidden, ext_ids, ext_mask,
                            randomize_positions=False, site=RESIDUAL_SITE):
    """Teacher-forced forward pass with the intervention patch active at
    layer_idx. Returns logits for the last MAX_ANSWER_TOKENS positions,
    aligned 1:1 with target_toks (see build_teacher_forced_extension).

    `site` selects WHICH tensor of decoder block layer_idx-1 gets patched
    (see common/sites.py); the default residual-stream site delegates
    straight to common/hooks.py, i.e. is byte-identical to the pre-sites
    behavior."""
    positions = batch["positions"]
    if randomize_positions:
        B, n_pos, H = source_hidden.shape
        source_for_patch = torch.stack([source_hidden[i, torch.randperm(n_pos)] for i in range(B)])
    else:
        source_for_patch = source_hidden
    patch_fn = make_cache_aware_patch_hook(positions, lambda base_vals: intervention(base_vals, source_for_patch))
    out = site.forward_patched(adapter, model, layers, layer_idx, patch_fn, ext_ids, ext_mask, batch["base_extra"],
                                logits_to_keep=MAX_ANSWER_TOKENS)
    return out.logits[:, -MAX_ANSWER_TOKENS:, :]


def train_layer(adapter, model, processor, entity_assets, attribute, layer, out_dir,
                 positions="flag_ring1", l1_coef=DBM_L1_COEF, temperature_start=TEMP_START, temperature_end=TEMP_END,
                 lr=LR, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
                 cause_only=False, randomize_positions=False, limit_rows=None, tuples_split="train",
                 tuples_dir=None, cleanup_checkpoint=True, source_cache=None, site=None, method_label="dbm"):
    """Trains one (layer, attribute, positions, l1_coef) DBM run to
    completion (or resumes an interrupted one), writing checkpoints/epoch
    snapshots/train_log under out_dir. Returns the path to the final
    weights-only checkpoint. See methods/das/train.py's train_layer (this
    is a near-verbatim port) for source_cache/cleanup_checkpoint semantics.

    site (common/sites.py's InterventionSite, default residual): WHICH
    tensor of decoder block `layer`-1 the mask is learned over. The default
    is the residual stream -- plain DBM, byte-identical to this function
    before sites existed. An MLP site makes this the shared engine behind
    methods/ndm/ (Native Dictionary Masking) instead; nothing else in this
    loop changes, since the mask, the L1 term, the temperature anneal and
    mask_stats are all dimension-agnostic.

    method_label: prefix for this run's progress prints only (so an NDM run
    doesn't announce itself as "[dbm/train]"). Affects no path or artifact.
    """
    site = site or RESIDUAL_SITE
    os.makedirs(out_dir, exist_ok=True)
    layers = adapter.get_decoder_layers(model)
    embed_dim = site.width(adapter, model)
    batch_cache = BuildBatchCache()

    intervention = SigmoidMaskIntervention(embed_dim=embed_dim).to(model.device)
    optimizer = torch.optim.Adam(intervention.parameters(), lr=lr)

    rows = load_tuples(entity_assets, attribute, tuples_split, tuples_dir=tuples_dir)
    for r in rows:
        r.setdefault("target_attribute", attribute)
    if cause_only:
        rows = [r for r in rows if r["queried"] == r["target_attribute"]]
    if limit_rows:
        rows = rows[:limit_rows]
    n_micro_batches_per_epoch = (len(rows) + batch_size - 1) // batch_size
    # Ceiling division: an epoch whose micro-batch count isn't a multiple of grad_accum_steps still
    # gets one MORE optimizer step per epoch for its trailing partial group (flushed explicitly below,
    # right after the micro-batch loop) -- floor division here would undercount t_total, silently
    # discarding that partial group's real forward+backward-computed gradient once the next epoch's
    # (or, for the last epoch, this run's own final) optimizer.zero_grad() wipes it unapplied. Verified
    # with a regression test: 21 rows / batch_size=4 / grad_accum_steps=4 (6 micro-batches, not a
    # multiple of 4) used to take exactly 1 optimizer.step() instead of the 2 needed to actually use
    # every row -- 5 of 21 rows (~24%) contributed zero training signal despite being computed.
    steps_per_epoch = -(-n_micro_batches_per_epoch // grad_accum_steps)  # ceil division
    t_total = steps_per_epoch * num_epochs
    warmup_steps = int(0.1 * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    log_path = os.path.join(out_dir, f"layer{layer}_train_log.jsonl")
    ckpt_path = os.path.join(out_dir, f"layer{layer}_checkpoint.pt")

    rng = random.Random(SEED)
    global_micro_step = 0
    opt_steps_done = 0
    start_epoch, start_mb, resumed_epoch_rows = 0, 0, None
    ckpt = torch.load(ckpt_path, map_location=model.device) if os.path.exists(ckpt_path) else None
    # The temperature schedule's LENGTH is anchored to whichever t_total the FIRST invocation of this
    # exact run computed, persisted in the checkpoint -- deliberately NOT recomputed from this call's
    # own (possibly different) num_epochs/t_total. Without this anchor, resuming a run after raising
    # --num_epochs (the documented use of --keep_checkpoint) would rebuild a schedule sized to the NEW,
    # larger t_total and re-index into it by opt_steps_done -- since that's an earlier position in a
    # now-longer schedule, temperature would jump back UP at the resume boundary instead of continuing
    # to anneal down (verified with a regression test before this comment was written: a 1-epoch run
    # resumed with num_epochs=2 produced a temperature that fell to 1e-7 then jumped back to ~4.6e-6).
    # A fresh run (no checkpoint yet) has no persisted anchor, so it just uses its own t_total.
    temp_schedule_total_steps = ckpt["temp_schedule_total_steps"] if ckpt is not None else t_total
    temp_schedule = temperature_schedule(max(temp_schedule_total_steps, 1), temperature_start, temperature_end)

    print(f"[{method_label}/train] entity={entity_assets.entity} attribute={attribute} layer={layer} "
          f"site={site.name} embed_dim={embed_dim} "
          f"l1_coef={l1_coef} temperature={temperature_start}->{temperature_end} positions={positions} "
          f"rows={len(rows)} cause_only={cause_only} randomize_positions={randomize_positions} "
          f"micro_batch={batch_size} accum={grad_accum_steps} (effective batch {batch_size * grad_accum_steps}) "
          f"epochs={num_epochs} total optimizer steps={t_total} (temperature schedule steps={temp_schedule_total_steps})")

    if ckpt is not None:
        intervention.load_state_dict(ckpt["intervention"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        rng.setstate(ckpt["rng_state"])
        start_epoch, start_mb = ckpt["epoch"], ckpt["next_mb"]
        resumed_epoch_rows = ckpt["epoch_rows"]
        global_micro_step = ckpt["global_micro_step"]
        opt_steps_done = ckpt["opt_steps_done"]
        assert ckpt["batch_size"] == batch_size and ckpt["grad_accum_steps"] == grad_accum_steps, (
            f"{ckpt_path} was trained with batch_size={ckpt['batch_size']}/grad_accum_steps="
            f"{ckpt['grad_accum_steps']}, but this run passed batch_size={batch_size}/"
            f"grad_accum_steps={grad_accum_steps} -- resuming with a different micro-batch size "
            f"than the saved next_mb/optimizer-step counters silently corrupts training.")
        print(f"resumed from {ckpt_path}: epoch {start_epoch}/{num_epochs}, micro-batch {start_mb}/{n_micro_batches_per_epoch}, "
              f"opt_steps_done={opt_steps_done}")
        log_f = open(log_path, "a")
    else:
        log_f = open(log_path, "w")

    def save_checkpoint(epoch, next_mb, epoch_rows):
        torch.save({
            "intervention": intervention.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng_state": rng.getstate(),
            "epoch": epoch, "next_mb": next_mb, "epoch_rows": epoch_rows,
            "global_micro_step": global_micro_step, "opt_steps_done": opt_steps_done,
            "batch_size": batch_size, "grad_accum_steps": grad_accum_steps,
            "temp_schedule_total_steps": temp_schedule_total_steps,
        }, ckpt_path)

    # Temperature at the FIRST not-yet-taken optimizer step -- set once up front so the very first
    # micro-batches (before the first completed opt step below sets it again) already use the right
    # value, matters most on a resume where opt_steps_done > 0.
    intervention.set_temperature(temp_schedule[min(opt_steps_done, len(temp_schedule) - 1)])

    t_start = time.time()
    opt_steps_this_call = 0  # for ETA below -- distinct from opt_steps_done, which persists across resumes
    ce_loss = None

    def complete_opt_step(epoch, accum_loss, accum_ce, accum_l1):
        """optimizer.step() + all its bookkeeping (temperature anneal, jsonl log, progress print) --
        called both at a normal grad_accum_steps boundary AND to flush a trailing PARTIAL accumulation
        group at an epoch's end (see steps_per_epoch's ceil-division comment above for why the latter
        matters: otherwise that partial group's already-computed gradient is silently discarded)."""
        nonlocal opt_steps_done, opt_steps_this_call
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        opt_steps_done += 1
        opt_steps_this_call += 1
        intervention.set_temperature(temp_schedule[min(opt_steps_done, len(temp_schedule) - 1)])
        log_f.write(json.dumps({
            "epoch": epoch, "micro_step": global_micro_step, "opt_step": opt_steps_done,
            "loss": accum_loss, "ce_loss": accum_ce, "l1_term": accum_l1,
            "temperature": intervention.get_temperature().item(), "lr": scheduler.get_last_lr()[0],
        }) + "\n")
        log_f.flush()
        if opt_steps_done % PROGRESS_EVERY_OPT_STEPS == 0:
            elapsed = time.time() - t_start
            sec_per_opt_step = elapsed / opt_steps_this_call
            remaining_steps = max(t_total - opt_steps_done, 0)
            eta_min = sec_per_opt_step * remaining_steps / 60
            print(f"[progress] epoch={epoch} opt_step={opt_steps_done}/{t_total} "
                  f"loss={accum_loss:.4f} ce={accum_ce:.4f} l1={accum_l1:.1f} "
                  f"temp={intervention.get_temperature().item():.2e} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min", flush=True)

    for epoch in range(start_epoch, num_epochs):
        if epoch == start_epoch and resumed_epoch_rows is not None:
            epoch_rows, mb_start_this_epoch = resumed_epoch_rows, start_mb
        else:
            epoch_rows = rows[:]
            rng.shuffle(epoch_rows)
            mb_start_this_epoch = 0

        optimizer.zero_grad()
        accum_loss, accum_ce, accum_l1 = 0.0, 0.0, 0.0
        micro_steps_since_flush = 0
        groups_completed_this_epoch = 0  # counts opt steps taken THIS epoch, full or trailing-partial alike
        mb_batch_rows = [(mb, epoch_rows[mb * batch_size:(mb + 1) * batch_size])
                          for mb in range(mb_start_this_epoch, n_micro_batches_per_epoch)]
        mb_batch_rows = [(mb, br) for mb, br in mb_batch_rows if br]
        # Resume always lands exactly on a group boundary (checkpoints are only ever saved right after
        # complete_opt_step, never mid-group -- see below), so mb_batch_rows here is always some whole
        # number of remaining full groups plus, at most, the SAME trailing partial group the un-resumed
        # epoch would have had. That makes it safe to derive the trailing group's true size from just
        # this call's own remaining count, on every call, not only a fresh (non-resumed) one.
        n_full_groups, trailing_group_size = divmod(len(mb_batch_rows), grad_accum_steps)
        batch_iter = (build_batch(br, entity_assets, adapter, model, processor, positions, batch_cache=batch_cache)
                      for _, br in mb_batch_rows)
        for i, ((mb, batch_rows), batch) in enumerate(zip(mb_batch_rows, batch_iter)):
            # The group THIS micro-batch belongs to -- grad_accum_steps for every group except a
            # trailing partial one at the very end of the epoch (n_micro_batches_per_epoch isn't
            # always a multiple of grad_accum_steps). Dividing by the group's TRUE size (not always
            # the constant grad_accum_steps) matters for both the actual gradient (loss.backward()
            # below) and the logged/diagnostic values -- verified with a real training log: a run's
            # trailing 5-micro-batch group was previously logged (and gradient-weighted) as if it were
            # a full 16-micro-batch group, diluting both by 5/16 of what a true average would be
            # (29.06 * 5/16 ~= 9.08, matching the erroneous 9.05 that got logged).
            current_group_size = grad_accum_steps if i < n_full_groups * grad_accum_steps else trailing_group_size

            target_toks, target_len = target_gold_toks_and_len(batch)
            ext_ids, ext_mask = build_teacher_forced_extension(batch["base_input_ids"], batch["attention_mask"],
                                                                 target_toks, target_len)

            # This gate is unchanged from the pre-sites version; only the lookup dispatches. Each site
            # has its own cache format and file (residual: common/source_cache.py, one file per entity
            # covering every layer; MLP: common/site_source_cache.py, one file per site/positions/layer)
            # and site.lookup_source asserts the cache it is handed was actually built for THIS
            # (site, layer, positions). That assertion is load-bearing: an mlp_output cache and a
            # residual cache have identical shapes, so a mix-up would not fail on shape alone.
            if source_cache is not None and not batch["is_last_token"]:
                source_hidden = site.lookup_source(source_cache, batch, layer, positions,
                                                    model.device, model.dtype)
            else:
                source_hidden = cache_source_layer_hidden(adapter, model, batch, layer, site=site)
            logits = run_intervened_forward(adapter, model, layers, batch, layer, intervention, source_hidden,
                                             ext_ids, ext_mask, randomize_positions=randomize_positions, site=site)

            labels = gold_labels_from_lens(target_toks, target_len).to(model.device)
            ce_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), ignore_index=-100,
            )
            l1_term = l1_penalty(intervention)
            loss = (ce_loss + l1_coef * l1_term) / current_group_size
            loss.backward()
            accum_loss += loss.item()
            accum_ce += ce_loss.item() / current_group_size
            accum_l1 += l1_term.item() / current_group_size
            global_micro_step += 1
            micro_steps_since_flush += 1

            if micro_steps_since_flush == current_group_size:
                complete_opt_step(epoch, accum_loss, accum_ce, accum_l1)
                accum_loss, accum_ce, accum_l1 = 0.0, 0.0, 0.0
                micro_steps_since_flush = 0
                groups_completed_this_epoch += 1

                if groups_completed_this_epoch % CHECKPOINT_EVERY_OPT_STEPS == 0:
                    save_checkpoint(epoch, mb + 1, epoch_rows)

        elapsed = time.time() - t_start
        last_loss = ce_loss.item() if ce_loss is not None else float("nan")
        print(f"epoch {epoch} done, elapsed={elapsed/60:.1f}min, last ce_loss={last_loss:.4f}, "
              f"temperature={intervention.get_temperature().item():.2e}", flush=True)
        save_checkpoint(epoch + 1, 0, None)

    log_f.close()
    final_path = os.path.join(out_dir, f"layer{layer}_intervention.pt")
    torch.save(intervention.state_dict(), final_path)

    # The trained artifact is saved above, but that alone doesn't say WHICH/HOW MANY dimensions it
    # actually selected -- PCA/SAE/DAS report this for free via winners.json's feature_indices/
    # n_features_selected; DBM needs an explicit discretization step (the paper's own 1-sigma(m/T) <
    # epsilon rule) to get the equivalent. Written next to the checkpoint so it's never orphaned from
    # the artifact it describes.
    stats = mask_stats(intervention)
    stats_path = os.path.join(out_dir, f"layer{layer}_mask_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"mask: {stats['n_selected']}/{stats['embed_dim']} dims selected at epsilon={stats['epsilon']} "
          f"(temperature={stats['temperature']:.2e})")
    print(f"DONE. wrote {final_path}")
    print(f"  wrote {stats_path}")

    if cleanup_checkpoint and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  removed {ckpt_path} (resume state no longer needed -- training finished)")

    return final_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--attribute", required=True, help="Cause attribute -- queried==this flips to source; every "
                                                         "other attribute in this entity becomes an iso pool.")
    ap.add_argument("--layer", type=int, required=True, help="0=embedding output, 1..N=decoder layer N's output")
    ap.add_argument("--positions", default="flag_ring1", help="A named set from the entity's object_location.json "
                                                                "(e.g. flag_ring1/flag_only for flags, logo_ring1/"
                                                                "logo_only for brands), or 'last_token' (the closest "
                                                                "analogue of RAVEL's own text-only intervention "
                                                                "site: the entity mention's single last token) or "
                                                                "'full_image'.")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT, help="Path to a sibling VADE checkout (entity data, "
                                                                     "tuples, pruned tuples, source cache). Defaults "
                                                                     "to ../VADE, or the VADE_ROOT env var.")
    ap.add_argument("--l1_coef", type=float, default=DBM_L1_COEF,
                     help=f"Sparsity coefficient on the raw mask's L1 norm (RAVEL's reported optimum: {DBM_L1_COEF}).")
    ap.add_argument("--temperature_start", type=float, default=TEMP_START)
    ap.add_argument("--temperature_end", type=float, default=TEMP_END)
    ap.add_argument("--lr", type=float, default=LR,
                     help=f"Adam learning rate for the mask (default {LR}, copied from DAS's train.py -- DAS "
                          "learns a D x D/D x K orthogonal rotation, a very different parametrization from DBM's "
                          "unconstrained length-H mask vector, so this default isn't validated for DBM specifically. "
                          "Worth sweeping (e.g. 5e-3, 1e-2) if training loss plateaus early relative to num_epochs "
                          "-- ignored on --keep_checkpoint resume, where the checkpoint's own saved optimizer/"
                          "scheduler state (including lr) takes over regardless of what's passed here.")
    ap.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
    ap.add_argument("--cause_only", action="store_true")
    ap.add_argument("--randomize_positions", action="store_true")
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--allow_unpruned", action="store_true",
                     help="Pruned tuples (VADE's models/<model_slug>/<entity>/tuples/, baseline-pruned by VADE's "
                          "own models/prune_tuples.py) are REQUIRED by default -- same reasoning as DAS's own "
                          "--allow_unpruned (see VADE/methods/das/train.py): training on unpruned data wastes "
                          "capacity on rows the model can't answer correctly even at baseline. Pass this to "
                          "deliberately opt out and train on VADE's data/<entity>/tuples/ directly instead.")
    ap.add_argument("--keep_checkpoint", action="store_true",
                     help="By default, layer{layer}_checkpoint.pt (the resume-state file) is deleted once training "
                          "finishes successfully. Pass this if you might later raise --num_epochs on this exact "
                          "run to continue training past what already ran (that resume path needs the file).")
    ap.add_argument("--out_dir", default=None, help="Defaults to THIS repo's results/<model_slug>/<entity>/dbm/"
                                                       "<attribute>/<config_tag>/ (never under --vade_root).")
    ap.add_argument("--no_source_cache", action="store_true",
                     help="Disable the per-entity source-activation cache (on by default, shared with DAS -- see "
                          "VADE/methods/common/source_cache.py -- lives under --vade_root/results/ since it's "
                          "keyed purely by entity+model, not by method). Ignored for --positions=last_token.")
    args = ap.parse_args()

    model_slug = args.model_id.split("/")[-1]
    pruned = not args.allow_unpruned
    tuples_dir = (require_pruned_tuples(args.vade_root, model_slug, args.entity, args.attribute)
                  if pruned else None)
    log_path = os.path.join(dbm_logs_dir(model_slug, args.entity, args.attribute, args.l1_coef,
                                          args.temperature_start, args.temperature_end, args.lr,
                                          args.positions, pruned),
                             f"layer{args.layer}_train.log")

    with tee_to_log(log_path):
        adapter = get_adapter(args.model_id)
        model, processor = adapter.load()

        entity_assets = load_entity_assets(args.vade_root, args.entity)

        out_dir = args.out_dir or dbm_results_dir(model_slug, args.entity, args.attribute, args.l1_coef,
                                                    args.temperature_start, args.temperature_end, args.lr,
                                                    args.positions, pruned)

        source_cache = None
        if not args.no_source_cache:
            source_cache = get_or_build_source_cache(adapter, model, processor, entity_assets,
                                                       args.vade_root, model_slug)

        train_layer(adapter, model, processor, entity_assets, args.attribute, args.layer, out_dir,
                    positions=args.positions, l1_coef=args.l1_coef, temperature_start=args.temperature_start,
                    temperature_end=args.temperature_end, lr=args.lr, num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    grad_accum_steps=args.grad_accum_steps, cause_only=args.cause_only,
                    randomize_positions=args.randomize_positions, limit_rows=args.limit_rows, tuples_dir=tuples_dir,
                    cleanup_checkpoint=not args.keep_checkpoint, source_cache=source_cache)


if __name__ == "__main__":
    main()
