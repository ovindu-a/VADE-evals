"""ONE command for the image->text head localization done by hand on flags,
for any entity/attribute.

    python methods/image_head_pipeline.py --entity brands              # every attribute
    python methods/image_head_pipeline.py --entity animals celebrities --attributes native_continent
    python methods/image_head_pipeline.py --entity brands --dry_run    # print the commands only
    python methods/image_head_pipeline.py --entity flags --summary_only

THE STAGES (per attribute; each is its own subprocess and one model load):

  1. ceiling   ndm/ceiling_sweep.py --sites residual --positions_list <image> last_token,
               every layer 0..n_layers. A full source swap of the residual stream at the
               IMAGE positions, and separately at the LAST TOKEN.
               -> image curve I(L) falls, last-token curve T(L) rises. The crossover is the
                  read (see head_trace.py's docstring). Picked automatically:
                    patch_layer = deepest layer of I's top plateau (I >= --patch_frac * max)
                    read_end    = first layer where T reaches --read_frac of its range
                    handoff     = blocks patch_layer .. read_end-1
  2. windows   methods/head_window_sweep.py: every head of a block window installed at
               the last token (and the mirror knockout, and a shuffled-donor control),
               for every window of <= --max_len blocks in patch_layer .. read_end-1+--margin,
               plus the handoff window itself and a full-coverage identity window.
               -> the SMALLEST window reaching --window_frac of the best window's score.
  3. heads     methods/head_trace.py --blocks <that window>: per-head ranking, cumulative
               necessity/sufficiency curves with a random-head null, shuffled-donor
               localization.
  4. decode    (opt-in, --stages ... decode) methods/head_decode_trace.py on stage 3's
               ranking: re-applies the top-k heads at every generated token, which is
               what multi-token answers (years, calling codes) need -- stages 2-3 patch the
               last PROMPT column only and so cannot steer past the first answer token.

T(L) USES 1 - base_kept, NOT cause. A last-token residual patch can only steer the first
answer token (CLAUDE.md, swap_trace section): flags/calling_code reads cause=6% at every
layer past the handoff while the answer is moved off base 78% of the time. `cause` would
put the read nowhere for every multi-token attribute. The image curve keeps `cause`: an
image-position patch lands in the prefill and is baked into the KV cache.

Selections, stage outputs and a per-entity SUMMARY.md land under
results/image_head_pipeline/<entity>/. Every stage is resumable -- a stage whose output
exists is skipped (--force to redo). Existing ceiling_sweep JSONs under logs/ that already
cover every layer with the residual site are reused instead of re-run (--no_reuse to
disable), so flags' stage 1 costs nothing. --patch_layer / --blocks override the automatic
choices (stage 2 still runs unless dropped from --stages).

This file imports no torch: every stage runs in its own interpreter (--py), so the
selection logic and --dry_run work on a laptop.
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
STAGES = ("ceiling", "windows", "heads", "decode")
DEFAULT_STAGES = ("ceiling", "windows", "heads")


# ============================================================== selection (pure functions)

def residual_curve(ceiling_json, metric):
    """{layer: value} for the residual site. metric: 'cause' or 'moved' (1 - base_kept)."""
    out = {}
    for r in ceiling_json["by_site_layer"].values():
        if r["site"] != "residual":
            continue
        out[r["layer"]] = r["cause_ceiling"] if metric == "cause" else 1.0 - r["base_kept"]
    return dict(sorted(out.items()))


def select_handoff(image_json, last_json, n_layers, patch_frac=0.95, read_frac=0.9, margin=2,
                   patch_layer=None):
    """-> dict with patch_layer, read_end, image_end, handoff blocks and the window-sweep
    range, plus `warnings`. See the module docstring for the rules."""
    img = residual_curve(image_json, "cause")
    last = residual_curve(last_json, "moved")
    warnings = []
    i_max = max(img.values())
    if i_max < 0.3:
        warnings.append(f"image residual swap never exceeds {i_max:.0%} cause -- there is little image "
                        f"effect to trace for this attribute")
    if patch_layer is None:
        # Deepest layer of the TOP plateau: start at the first layer reaching the max and walk
        # forward while the curve stays within patch_frac of it. Walking the whole curve instead
        # would let a noisy late bump pick a layer past the handoff.
        layers = sorted(img)
        start = next(l for l in layers if img[l] >= i_max - 1e-9)
        patch_layer = start
        for l in layers[layers.index(start):]:
            if img[l] >= patch_frac * i_max:
                patch_layer = l
            else:
                break
    image_end = next((l for l in sorted(img) if l > patch_layer and img[l] <= 0.1 * i_max), None)

    floor = 1.0 - last_json.get("unhooked_cause_matches_base", 1.0)
    t_max = max(last.values())
    if t_max - floor < 0.3:
        warnings.append(f"last-token residual swap moves the answer off base at most {t_max:.0%} "
                        f"(floor {floor:.0%}) -- no clear destination for the read")
    target = floor + read_frac * (t_max - floor)
    read_end = next((l for l in sorted(last) if l > patch_layer and last[l] >= target), None)
    if read_end is None:
        read_end = max(last, key=lambda l: last[l])
        warnings.append(f"last-token curve never reaches {target:.0%} above patch_layer {patch_layer}; "
                        f"using its argmax {read_end}")
    lo = patch_layer
    handoff_hi = max(read_end - 1, lo)
    if read_end - 1 < lo:
        warnings.append(f"read_end {read_end} is not above patch_layer {patch_layer}: the curves do not "
                        f"cross -- handoff collapsed to block {lo}")
    hi = min(handoff_hi + margin, n_layers - 1)
    return {"patch_layer": patch_layer, "image_end": image_end, "read_end": read_end,
            "handoff_blocks": list(range(lo, handoff_hi + 1)), "window_range": [lo, hi],
            "image_curve": img, "last_token_curve": last, "last_token_floor": floor,
            "thresholds": {"patch_frac": patch_frac, "read_frac": read_frac, "margin": margin},
            "warnings": warnings}


def scoring_check(image_json, last_json, min_source_match=0.85):
    """Is the scorer able to recognise the model's CORRECT answers at all?

    The full-image residual swap at layer 0 replaces every image embedding with
    the source's, so it IS the model answering about the source image -- and
    VADE's pruning keeps a cause row only if the model answers the source
    correctly (models/prune_tuples.py checks the SOURCE item on match_source
    rows, never the base). So that cell must be near 100% whenever labels and
    matching agree with how the model writes answers; a low value is a scoring
    failure (brands/hq_country under token scoring: 0%), not a finding.

    The unhooked base match is reported but not gated on: it is legitimately
    below 100% (TOGG -> ' Sweden'; ~44% of brands/founded_year bases wrong),
    and it only compresses the last-token curve, whose floor select_handoff
    already subtracts."""
    img = residual_curve(image_json, "cause")
    source_read = img.get(0, float("nan"))
    clean_base = last_json.get("unhooked_cause_matches_base", float("nan"))
    warnings = []
    if not source_read >= min_source_match:
        warnings.append(f"the model's own reading of the source image scores {source_read:.1%} < "
                        f"{min_source_match:.0%} -- labels/matching disagree with its answers")
    if clean_base < 0.7:
        warnings.append(f"only {clean_base:.1%} of unhooked answers match base_label (the model is wrong about "
                        f"many BASE items, which pruning allows): base_kept and the last-token curve are capped "
                        f"there, so read them relative to that floor")
    return {"ok": bool(source_read >= min_source_match), "source_read": source_read,
            "clean_base": clean_base, "min_source_match": min_source_match, "warnings": warnings}


WINDOW_METRICS = {
    "first_src": lambda w: w["suff"]["first_src"],
    "cause": lambda w: w["suff"]["cause"],
    "moved": lambda w: 1.0 - w["suff"]["base_kept"],
}


def select_window(windows_json, metric="first_src", window_frac=0.95):
    """Smallest window whose sufficiency score reaches window_frac of the best
    non-reference window's; ties -> higher score, then earlier. The full-coverage
    reference is excluded: it is an identity check, and it would always win."""
    n_layers, p = windows_json["n_layers"], windows_json["patch_layer"]
    full_ref = list(range(p, n_layers))
    score = WINDOW_METRICS[metric]
    cands = [w for w in windows_json["windows"] if w["blocks"] != full_ref] or windows_json["windows"]
    best = max(score(w) for w in cands)
    ok = [w for w in cands if score(w) >= window_frac * best - 1e-9]
    pick = min(ok, key=lambda w: (len(w["blocks"]), -score(w), w["blocks"][0]))
    img = windows_json["image_patch_only"]
    denom = img["first_src"] if metric == "first_src" else (
        img["cause"] if metric == "cause" else 1.0 - img["base_kept"])
    return {"blocks": pick["blocks"], "label": pick["label"], "metric": metric, "score": score(pick),
            "best_score": best, "frac_of_image_patch": score(pick) / denom if denom > 0 else None,
            "window_frac": window_frac}


# ============================================================== paths / io

def span_label(blocks):
    if blocks == list(range(blocks[0], blocks[-1] + 1)):
        return str(blocks[0]) if len(blocks) == 1 else f"{blocks[0]}-{blocks[-1]}"
    return ".".join(str(b) for b in blocks)


def path_safe(spec):  # mirrors methods/common/position_sets.py (which imports torch)
    return spec.replace("~", "not-").replace(":", "_").replace("@", "-at-").replace("+", "_plus_")


def rel(path):
    """Repo-relative when inside the repo, absolute otherwise (for printing)."""
    r = os.path.relpath(path, REPO_ROOT)
    return path if r.startswith("..") else r


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def pipeline_attributes(vade_root, model_slug, entity, allow_unpruned):
    root = (os.path.join(vade_root, "data", entity, "tuples") if allow_unpruned
            else os.path.join(vade_root, "models", model_slug, entity, "tuples"))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no tuples for entity {entity!r} at {root}")
    return sorted(a for a in os.listdir(root) if os.path.isfile(os.path.join(root, a, "test.jsonl")))


def ceiling_log_candidates(model_slug, entity, attribute, spec, layers, pruned, text=False):
    tag = "-".join(str(l) for l in layers) + ("_text" if text else "")
    suffix = "_pruned" if pruned else ""
    pat = os.path.join(REPO_ROOT, "logs", model_slug, entity, "ndm", attribute,
                       f"L1_0.0_*_{path_safe(spec)}_mlp_hidden{suffix}", f"ceiling_sweep_layers{tag}.json")
    return sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)


def usable_ceiling(path, layers, min_rows, text=False):
    try:
        d = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if d.get("scoring", "token") != ("text" if text else "token"):
        return False
    have = {r["layer"] for r in d["by_site_layer"].values() if r["site"] == "residual"}
    n = min((r["n_cause"] for r in d["by_site_layer"].values() if r["site"] == "residual"), default=0)
    return set(layers) <= have and n >= min_rows


# ============================================================== runner

class Runner:
    def __init__(self, args):
        self.args = args
        self.model_slug = args.model_id.split("/")[-1]
        self.failures = []

    def run(self, cmd, log_path):
        """Stream a stage's output to the console and to log_path. -> exit code."""
        pretty = " ".join(cmd)
        if self.args.dry_run:
            print(f"  $ {pretty}")
            return 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"  $ {pretty}\n    (log: {rel(log_path)})", flush=True)
        t0 = time.time()
        with open(log_path, "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {pretty}\n")
            try:
                proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
            except OSError as e:
                print(f"    -> could not start: {e}")
                return 127
            for line in proc.stdout:
                log.write(line)
                if not self.args.quiet:
                    sys.stdout.write("    | " + line)
            rc = proc.wait()
        print(f"    -> exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        return rc

    def text(self, entity):
        """TEXT scoring for every entity but flags under --scoring auto: flags' labels are cased
        the way the model writes them (and its earlier results are token-scored), the others'
        are stored lowercase and read 0% unhooked under token scoring."""
        return self.args.scoring == "text" or (self.args.scoring == "auto" and entity != "flags")

    def common(self, entity, attribute):
        a = self.args
        out = ["--entity", entity, "--attribute", attribute, "--model_id", a.model_id, "--vade_root", a.vade_root]
        if a.allow_unpruned:
            out.append("--allow_unpruned")
        if self.text(entity):
            out += ["--text_match", "--max_new_tokens", str(a.text_max_new_tokens)]
        return out

    # ---------------------------------------------------------- per attribute

    def attribute(self, entity, attribute):
        a = self.args
        d = os.path.join(a.out_root, entity, attribute)
        logd = os.path.join(REPO_ROOT, "logs", "image_head_pipeline", entity, attribute)
        state_path = os.path.join(d, "pipeline.json")
        state = load_json(state_path) if os.path.exists(state_path) and not a.force else {}
        state.update({"entity": entity, "attribute": attribute, "model_id": a.model_id,
                      "image_positions": a.image_positions})
        print(f"\n{'=' * 78}\n{entity}/{attribute}\n{'=' * 78}", flush=True)
        py = shlex.split(a.py)
        layers = list(range(0, a.n_layers + 1))
        text = self.text(entity)
        sfx = "_text" if text else ""
        state["scoring"] = "text" if text else "token"
        print(f"  scoring: {state['scoring']}" + (" (VADE text match; first-token metrics against the model's "
                                                 "own unhooked base/source answers)" if text else ""))
        if not a.dry_run:
            os.makedirs(d, exist_ok=True)

        # ---- stage 1: ceiling -------------------------------------------------------------
        specs = {"image": a.image_positions, "last": "last_token"}
        ceil_paths = {k: os.path.join(d, f"ceiling_{path_safe(v)}{sfx}.json") for k, v in specs.items()}
        need = [k for k, p in ceil_paths.items() if a.force or not os.path.exists(p)]
        if need and not a.no_reuse and not a.force:
            for k in list(need):
                for c in ceiling_log_candidates(self.model_slug, entity, attribute, specs[k], layers,
                                                not a.allow_unpruned, text):
                    if usable_ceiling(c, layers, a.n_rows_ceiling, text):
                        print(f"  [ceiling] reusing {rel(c)} for {specs[k]}")
                        if not a.dry_run:
                            shutil.copyfile(c, ceil_paths[k])
                        need.remove(k)
                        break
        if need:
            if "ceiling" not in a.stages:
                print(f"  [ceiling] outputs missing and stage not selected -- stopping this attribute")
                return
            cmd = [*py, "methods/ndm/ceiling_sweep.py", *self.common(entity, attribute),
                   "--sites", "residual", "--positions_list", specs["image"], specs["last"],
                   "--layers", *map(str, layers), "--n_rows", str(a.n_rows_ceiling),
                   "--seed", str(a.seed), "--batch_size", str(a.batch_size)]
            print(f"  [stage 1: ceiling] full residual swap at {specs['image']} and last_token, layers "
                  f"0..{a.n_layers}")
            rc = self.run(cmd, os.path.join(logd, "stage1_ceiling.log"))
            if rc:
                return self.fail(entity, attribute, "ceiling", rc)
            if a.dry_run:
                print("  (dry run: stages 2-3 need stage 1's output to pick layers; showing them with "
                      "placeholders)")
                return self.dry_placeholders(entity, attribute)
            for k in need:
                found = ceiling_log_candidates(self.model_slug, entity, attribute, specs[k], layers,
                                               not a.allow_unpruned, text)
                if not found:
                    return self.fail(entity, attribute, "ceiling", "output JSON not found")
                shutil.copyfile(found[0], ceil_paths[k])
        elif not a.dry_run or all(os.path.exists(p) for p in ceil_paths.values()):
            print(f"  [ceiling] have {', '.join(rel(p) for p in ceil_paths.values())}")

        if a.dry_run and not all(os.path.exists(p) for p in ceil_paths.values()):
            # reused from logs but not copied (dry run): read them where they are
            src = {k: next(c for c in ceiling_log_candidates(self.model_slug, entity, attribute, specs[k],
                                                               layers, not a.allow_unpruned, text)
                           if usable_ceiling(c, layers, a.n_rows_ceiling, text)) for k in specs}
        else:
            src = ceil_paths
        sel = select_handoff(load_json(src["image"]), load_json(src["last"]), a.n_layers,
                             a.patch_frac, a.read_frac, a.margin, patch_layer=a.patch_layer)
        state["handoff"] = sel
        self.print_handoff(sel)
        chk = scoring_check(load_json(src["image"]), load_json(src["last"]), a.min_source_match)
        state["scoring_check"] = chk
        print(f"  [scoring check] image swap @ layer 0 (= the model reading the SOURCE image, which pruning "
              f"guarantees it answers correctly): {chk['source_read']:.1%} match source_label; unhooked "
              f"base match {chk['clean_base']:.1%} (NOT guaranteed: pruning never checks the base on cause rows)")
        for w in chk["warnings"]:
            print(f"  !! {w}")
        if not chk["ok"] and not a.skip_scoring_check:
            print(f"  !! stopping {entity}/{attribute} before stages 2-3: the scorer does not recognise the "
                  f"model's own correct answers, so every number would be capped by scoring. Read them with:\n"
                  f"     python methods/ndm/swap_trace.py --entity {entity} --attribute {attribute} "
                  f"--positions {a.image_positions} --site residual --layers 0 --n_rows 32 --seed {a.seed}"
                  + (" --text_match" if text else "")
                  + "\n     then fix labels/matching, or pass --skip_scoring_check to proceed anyway.")
            state["skipped"] = "scoring check failed"
            self.failures.append((entity, attribute, "scoring_check", f"source_read={chk['source_read']:.1%}"))
            return self.save_state(state_path, state)
        if sel["warnings"] and not a.patch_layer and any("little image effect" in w for w in sel["warnings"]):
            print("  !! skipping stages 2-3: nothing to trace (override with --patch_layer to force)")
            state["skipped"] = "no image effect"
            return self.save_state(state_path, state)
        P = sel["patch_layer"]
        lo, hi = sel["window_range"]
        if a.window_range:
            lo, hi = a.window_range

        # ---- stage 2: windows ---------------------------------------------------------------
        win_path = os.path.join(d, f"windows_patch{P}{sfx}.json")
        if a.blocks:
            blocks = sorted(a.blocks)
            state["window"] = {"blocks": blocks, "label": span_label(blocks), "override": True}
        if "windows" in a.stages and (a.force or not os.path.exists(win_path)):
            # The handoff window itself is always probed, even when longer than --max_len.
            cmd = [*py, "methods/head_window_sweep.py", *self.common(entity, attribute),
                   "--patch_layer", str(P), "--positions", a.image_positions, "--range", str(lo), str(hi),
                   "--max_len", str(a.max_len), "--extra_windows", span_label(sel["handoff_blocks"]),
                   "--n_rows", str(a.n_rows), "--seed", str(a.seed), "--batch_size", str(a.batch_size),
                   "--out", win_path]
            print(f"\n  [stage 2: windows] full attention-block patches, blocks {lo}..{hi}, <= {a.max_len} "
                  f"blocks per window, patch_layer={P}")
            rc = self.run(cmd, os.path.join(logd, "stage2_windows.log"))
            if rc:
                return self.fail(entity, attribute, "windows", rc)
        if os.path.exists(win_path):
            wj = load_json(win_path)
            state["windows_json"] = os.path.relpath(win_path, REPO_ROOT)
            self.print_windows(wj)
            io = wj["image_patch_only"]
            rankable = a.window_metric != "first_src" or io.get("n_first", 0) >= max(4, io.get("n", 0) // 8)
            if not a.blocks and not rankable:
                state["window"] = {"blocks": sel["handoff_blocks"], "label": span_label(sel["handoff_blocks"]),
                                   "fallback": "too few rows differ at the first answer token"}
                print(f"  !! only {io.get('n_first', 0)}/{io.get('n', 0)} rows have base/source answers that "
                      f"differ at the FIRST token, so windows cannot be ranked by first_src; using the handoff "
                      f"window {state['window']['label']}. Stage-3 numbers for this attribute are capped the "
                      f"same way -- run --stages ... decode.")
            elif not a.blocks:
                state["window"] = select_window(wj, a.window_metric, a.window_frac)
                w = state["window"]
                print(f"  -> window {w['label']}: {w['metric']}={w['score']:.1%} "
                      f"(best {w['best_score']:.1%}; "
                      + (f"{w['frac_of_image_patch']:.0%} of the image patch's own effect)"
                         if w["frac_of_image_patch"] is not None else "image patch had no effect)"))
        elif not a.blocks:
            if a.dry_run:
                state["window"] = {"blocks": sel["handoff_blocks"], "label": span_label(sel["handoff_blocks"])}
                print(f"  (dry run: stage 3 shown with the handoff window {state['window']['label']}; the real "
                      f"run uses stage 2's pick)")
            else:
                print("  [windows] no stage-2 output and no --blocks -- stopping this attribute")
                return self.save_state(state_path, state)

        # ---- stage 3: heads -----------------------------------------------------------------
        blocks = state["window"]["blocks"]
        n_all = 28 * len(blocks) if not os.path.exists(win_path) else load_json(win_path)["n_heads"] * len(blocks)
        ks = sorted({k for k in a.knockout_ks if k < n_all} | {n_all})
        trace_path = os.path.join(d, f"head_trace_patch{P}_blocks{span_label(blocks)}{sfx}.json")
        if "heads" in a.stages and (a.force or not os.path.exists(trace_path)):
            cmd = [*py, "methods/head_trace.py", *self.common(entity, attribute),
                   "--patch_layer", str(P), "--positions", a.image_positions, "--blocks", *map(str, blocks),
                   "--n_rows", str(a.n_rows), "--seed", str(a.seed), "--batch_size", str(a.batch_size),
                   "--knockout_ks", *map(str, ks), "--control_k", *map(str, a.control_k),
                   "--n_random", str(a.n_random), "--out", trace_path]
            print(f"\n  [stage 3: heads] head_trace in blocks {span_label(blocks)}")
            rc = self.run(cmd, os.path.join(logd, f"stage3_heads_blocks{span_label(blocks)}.log"))
            if rc:
                return self.fail(entity, attribute, "heads", rc)
        if os.path.exists(trace_path):
            state["head_trace_json"] = os.path.relpath(trace_path, REPO_ROOT)
            self.print_trace(load_json(trace_path))

        # ---- stage 4: decode (opt-in) ---------------------------------------------------------
        dec_dir = os.path.join(d, f"decode_blocks{span_label(blocks)}")
        if "decode" in a.stages and (os.path.exists(trace_path) or a.dry_run):
            if a.force or not os.path.exists(os.path.join(dec_dir, "summary.json")):
                cmd = [*py, "methods/head_decode_trace.py", "--trace", trace_path, "--out_dir", dec_dir,
                       "--model_id", a.model_id, "--vade_root", a.vade_root,
                       "--head_ks", *map(str, a.decode_head_ks)]
                if a.allow_unpruned:
                    cmd.append("--allow_unpruned")
                if a.decode_limit:
                    cmd += ["--limit", str(a.decode_limit)]
                print(f"\n  [stage 4: decode] continuous head substitution, top-k {a.decode_head_ks}")
                if text:
                    print("  !! head_decode_trace scores against the stored gold TOKENS, not text: its accuracy "
                          "columns are wrong for lowercase-label entities (read its generated text instead)")
                rc = self.run(cmd, os.path.join(logd, "stage4_decode.log"))
                if rc:
                    return self.fail(entity, attribute, "decode", rc)
            if os.path.exists(os.path.join(dec_dir, "summary.json")):
                state["decode_dir"] = os.path.relpath(dec_dir, REPO_ROOT)
        self.save_state(state_path, state)

    def dry_placeholders(self, entity, attribute):
        a = self.args
        print(f"  $ {a.py} methods/head_window_sweep.py --entity {entity} --attribute {attribute} "
              f"--patch_layer <P> --positions {a.image_positions} --range <P> <read_end-1+{a.margin}> "
              f"--max_len {a.max_len} --n_rows {a.n_rows} --out ...")
        print(f"  $ {a.py} methods/head_trace.py --entity {entity} --attribute {attribute} --patch_layer <P> "
              f"--positions {a.image_positions} --blocks <stage-2 window> --n_rows {a.n_rows} ...")

    def save_state(self, path, state):
        if not self.args.dry_run:
            save_json(path, state)

    def fail(self, entity, attribute, stage, why):
        self.failures.append((entity, attribute, stage, why))
        print(f"  !! {entity}/{attribute} stage {stage} failed ({why}); moving on")

    # ---------------------------------------------------------- printing

    @staticmethod
    def print_handoff(sel):
        img, last = sel["image_curve"], sel["last_token_curve"]
        P, R = sel["patch_layer"], sel["read_end"]
        show = [l for l in sorted(img) if P - 4 <= l <= R + 2]
        print(f"  [handoff] layer     " + " ".join(f"{l:>5}" for l in show))
        print(f"            image     " + " ".join(f"{img[l]:>5.0%}" for l in show) + "   (cause)")
        print(f"            last_tok  " + " ".join(f"{last.get(l, float('nan')):>5.0%}" for l in show)
              + "   (1 - base_kept)")
        print(f"  -> patch_layer={P}  image_end={sel['image_end']}  read_end={R}  handoff blocks "
              f"{span_label(sel['handoff_blocks'])}  window range {sel['window_range'][0]}-{sel['window_range'][1]}")
        for w in sel["warnings"]:
            print(f"  !! {w}")

    @staticmethod
    def print_windows(wj):
        io = wj["image_patch_only"]
        print(f"  [windows] image patch only: cause={io['cause']:.1%} first_src={io['first_src']:.1%} "
              f"(n_first={io.get('n_first')}/{io.get('n')})"
              + (f" div_src={io['div_src']:.1%} (n_div={io['n_div']})" if "div_src" in io else ""))
        print(f"            {'window':>8} {'heads':>5} {'suff.first':>10} {'suff.div':>8} {'suff.cause':>10} "
              f"{'nec.first':>9} {'shuf.own':>8} {'shuf.donor':>10}")
        for w in wj["windows"]:
            nec = w.get("nec", {}).get("first_src", float("nan"))
            sh = w.get("shuffled", {})
            print(f"            {w['label']:>8} {w['n_heads']:>5} {w['suff']['first_src']:>10.1%} "
                  f"{w['suff'].get('div_src', float('nan')):>8.1%} "
                  f"{w['suff']['cause']:>10.1%} {nec:>9.1%} {sh.get('first_own', float('nan')):>8.1%} "
                  f"{sh.get('first_donor', float('nan')):>10.1%}")
        if wj.get("full_reference_identity_ok") is False:
            print("  !! full-coverage identity FAILED in stage 2 -- window rows are not trustworthy")

    @staticmethod
    def print_trace(tj):
        p2, p3 = tj.get("phase2", {}), tj.get("phase3_sufficiency", {})
        print(f"  [heads] image patch only cause={p2.get('image_patch_only_cause', float('nan')):.1%}  "
              f"all traced heads (suff ceiling)={p3.get('ceiling_all_traced_heads', float('nan')):.1%}")
        for arm in p3.get("arms", []):
            if arm["kind"] in ("top", "random"):
                print(f"          suff {arm['kind']:>6}-{arm['k']:<4} cause={arm['cause']:6.1%}")
        conf = next((c for c in tj.get("phase4_shuffled_donor", []) if c["donor"] > 0.5 and c["own"] < 0.2), None)
        if conf:
            print(f"  -> shuffled-donor CONFIRMED at {conf['label']} (donor={conf['donor']:.1%}, own={conf['own']:.1%})")
        print(f"  top-8 heads: " + " ".join(f"{b}.{h}" for b, h in tj.get("phase1_ranked", [])[:8]))


# ============================================================== summary

def pct(x):
    return "--" if x is None or x != x else f"{x:.1%}"


def write_summary(out_root, entity):
    root = os.path.join(out_root, entity)
    if not os.path.isdir(root):
        return None
    L = [f"# Image->text head pipeline: {entity}", "",
         "Generated by `methods/image_head_pipeline.py`. Stage 1 = full residual swap (image cause / "
         "last-token 1-base_kept), stage 2 = full attention-block patches at the last token, stage 3 = "
         "`head_trace.py` in the chosen window. Stages 2-3 patch the last prompt column only, so read "
         "`first_src` for multi-token attributes.", ""]
    overview = ["| attribute | patch_layer | read_end | handoff | window | window suff (first tok) | "
                "image patch (first tok) | top-8 suff | top-8 random | shuffled-donor confirmed at |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    details, top8 = [], {}
    for attribute in sorted(os.listdir(root)):
        sp = os.path.join(root, attribute, "pipeline.json")
        if not os.path.exists(sp):
            continue
        st = load_json(sp)
        h = st.get("handoff", {})
        win = st.get("window", {})
        wj = load_json(os.path.join(REPO_ROOT, st["windows_json"])) if st.get("windows_json") else None
        tj = load_json(os.path.join(REPO_ROOT, st["head_trace_json"])) if st.get("head_trace_json") else None
        wrow = next((w for w in (wj or {}).get("windows", []) if w["blocks"] == win.get("blocks")), None)
        suff = {a["kind"] + str(a["k"]): a["cause"] for a in (tj or {}).get("phase3_sufficiency", {}).get("arms", [])}
        conf = next((c["label"] for c in (tj or {}).get("phase4_shuffled_donor", [])
                     if c["donor"] > 0.5 and c["own"] < 0.2), "no")
        if tj:
            top8[attribute] = [f"{b}.{hh}" for b, hh in tj["phase1_ranked"][:8]]
        overview.append(
            f"| {attribute} | {h.get('patch_layer', '--')} | {h.get('read_end', '--')} | "
            f"{span_label(h['handoff_blocks']) if h.get('handoff_blocks') else '--'} | {win.get('label', '--')} | "
            f"{pct(wrow['suff']['first_src']) if wrow else '--'} | "
            f"{pct(wj['image_patch_only']['first_src']) if wj else '--'} | "
            f"{pct(suff.get('top8'))} | {pct(suff.get('random8'))} | {conf if tj else '--'} |")

        details += [f"## {attribute}", ""]
        if h:
            show = [l for l in sorted(map(int, h["image_curve"])) if h["patch_layer"] - 4 <= l <= h["read_end"] + 2]
            ic = {int(k): v for k, v in h["image_curve"].items()}
            lc = {int(k): v for k, v in h["last_token_curve"].items()}
            details += ["| layer | " + " | ".join(map(str, show)) + " |", "|---" * (len(show) + 1) + "|",
                        "| image cause | " + " | ".join(pct(ic[l]) for l in show) + " |",
                        "| last-token moved | " + " | ".join(pct(lc.get(l)) for l in show) + " |", ""]
            for w in h.get("warnings", []):
                details.append(f"> warning: {w}")
            details.append("")
        if wj:
            details += ["| window | heads | suff first | suff cause | nec first | shuffled own/donor (first) |",
                        "|---|---|---|---|---|---|"]
            for w in wj["windows"]:
                sh = w.get("shuffled", {})
                details.append(f"| {w['label']} | {w['n_heads']} | {pct(w['suff']['first_src'])} | "
                               f"{pct(w['suff']['cause'])} | {pct(w.get('nec', {}).get('first_src'))} | "
                               f"{pct(sh.get('first_own'))} / {pct(sh.get('first_donor'))} |")
            details.append("")
        if tj:
            details.append(f"head_trace (blocks {span_label(tj['blocks'])}, n={tj['n_rows']}): image patch only "
                           f"cause {pct(tj.get('phase2', {}).get('image_patch_only_cause'))}, suff ceiling "
                           f"{pct(tj.get('phase3_sufficiency', {}).get('ceiling_all_traced_heads'))}; top-8 heads "
                           f"`{' '.join(top8[attribute])}`")
            details.append("")
    if len(overview) == 2:
        return None
    L += overview + [""]
    if len(top8) > 1:
        common = set.intersection(*(set(v) for v in top8.values()))
        L += [f"Heads in EVERY attribute's top-8: `{' '.join(sorted(common)) or 'none'}` "
              f"(the flags analogue was 21.1 22.19 23.3 23.4 23.6 -- an entity conduit, see "
              f"ATTRIBUTE_HEAD_EXPERIMENTS.md R8/R9).", ""]
    L += details
    path = os.path.join(root, "SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


# ============================================================== CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", nargs="+", required=True)
    ap.add_argument("--attributes", nargs="+", default=None, help="Default: every attribute with (pruned) tuples.")
    ap.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES), choices=STAGES)
    ap.add_argument("--image_positions", default="full_image",
                    help="Image position set for the image swap (flags used full_image).")
    ap.add_argument("--n_layers", type=int, default=28, help="Decoder blocks in the model (Qwen2.5-VL-7B: 28).")
    ap.add_argument("--n_rows_ceiling", type=int, default=32, help="Stage 1 rows (flags used 32).")
    ap.add_argument("--n_rows", type=int, default=64, help="Stage 2-3 rows (flags used 64).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--scoring", default="auto", choices=["auto", "token", "text"],
                    help="auto = text for every entity except flags. token = stored-label token exact match "
                         "(what all flags results used; reads 0%% unhooked on lowercase-label entities). "
                         "text = VADE's text matcher, first-token metrics at the model's own divergence token.")
    ap.add_argument("--text_max_new_tokens", type=int, default=8,
                    help="Generation budget under text scoring (' the United States.' is 4 tokens, a year 5).")
    # selection
    ap.add_argument("--min_source_match", type=float, default=0.85,
                    help="Scoring guard: the layer-0 full-image swap (the model reading the source image, which "
                         "pruning guarantees is correct) must score at least this, or the attribute stops "
                         "after stage 1.")
    ap.add_argument("--skip_scoring_check", action="store_true")
    ap.add_argument("--patch_frac", type=float, default=0.95)
    ap.add_argument("--read_frac", type=float, default=0.9)
    ap.add_argument("--margin", type=int, default=2, help="Blocks past the handoff to include in stage 2.")
    ap.add_argument("--max_len", type=int, default=3, help="Longest enumerated stage-2 window.")
    ap.add_argument("--window_metric", default="first_src", choices=list(WINDOW_METRICS))
    ap.add_argument("--window_frac", type=float, default=0.95)
    ap.add_argument("--patch_layer", type=int, default=None, help="Override stage 1's patch_layer.")
    ap.add_argument("--window_range", type=int, nargs=2, default=None, help="Override stage 2's block range.")
    ap.add_argument("--blocks", type=int, nargs="+", default=None, help="Override stage 2's window for stage 3.")
    # stage 3/4
    ap.add_argument("--knockout_ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--control_k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--decode_head_ks", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--decode_limit", type=int, default=None)
    # plumbing
    ap.add_argument("--model_id", default=DEFAULT_MODEL)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--allow_unpruned", action="store_true")
    ap.add_argument("--py", default=sys.executable,
                    help="Interpreter for the stage scripts (shell-split, so 'uv run python' works).")
    ap.add_argument("--out_root", default=os.path.join(REPO_ROOT, "results", "image_head_pipeline"))
    ap.add_argument("--no_reuse", action="store_true", help="Never reuse existing ceiling_sweep logs.")
    ap.add_argument("--force", action="store_true", help="Re-run stages whose outputs exist.")
    ap.add_argument("--dry_run", action="store_true", help="Print commands and selections; run nothing.")
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Do not echo stage output (logs still written).")
    args = ap.parse_args()
    if args.blocks:
        assert args.patch_layer is None or min(args.blocks) >= args.patch_layer, "--blocks must be >= --patch_layer"

    runner = Runner(args)
    for entity in args.entity:
        if not args.summary_only:
            attrs = args.attributes or pipeline_attributes(args.vade_root, runner.model_slug, entity,
                                                           args.allow_unpruned)
            print(f"\n##### {entity}: attributes {attrs} stages {args.stages}")
            for attribute in attrs:
                runner.attribute(entity, attribute)
        if not args.dry_run:
            p = write_summary(args.out_root, entity)
            if p:
                print(f"\nwrote {rel(p)}")
    if runner.failures:
        print("\nFAILED:")
        for f in runner.failures:
            print(f"  {f[0]}/{f[1]} stage {f[2]}: {f[3]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
