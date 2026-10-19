"""Derive the router-trace and MLP-freeze arguments for a NEW model from its own runs.

On Qwen2.5-VL-7B these arguments were picked by hand from that model's measurements:
attr_head_trace's `--patch_layer 15 --blocks 15-22` from the attribute-switch crossover
(ATTRIBUTE_SWITCH_FINDINGS.md section 1), and head_severed's `--freeze_mlp_span 22-27
--freeze_mlp_singles 23 24 25 26 --freeze_attn_span 24-27` from the movers' blocks 21-23.
Reusing those numbers on a model with a different depth (Idefics3-8B-Llama3: 32 blocks) would
trace and freeze the wrong blocks. The rules below are written so that, applied to Qwen's own
numbers, they return exactly the hand-picked arguments (tests/test_idefics3_adapter.py checks
both); applied to a new model's runs, they give that model's equivalents.

ROUTER (`router --switch_dir D`), from an attribute_switch_sweep run with scopes earlier_text
and last_token at the residual site, every layer:
    Q(L) = question-token residual swap, donor_first;  T(L) = last-token residual swap.
    plateau_end = deepest layer of Q's top plateau (Q >= patch_frac * max Q, walking forward
                  from the first layer at the max -- image_head_pipeline.select_handoff's rule)
    patch_layer = plateau_end - 1          (one layer of margin below the plateau's edge)
    read_end    = first layer > patch_layer with T >= T_floor + read_frac * (T_max - T_floor)
    blocks      = patch_layer .. min(read_end + 2, n_layers - 1)
  Qwen: Q is ~100% through L16, so plateau_end 16 -> patch_layer 15; T first reaches 90% of its
  range at L20 -> blocks 15..22.

SEVERED (`severed --trace J`), from one head_trace JSON (the mover trace):
    top    = the trace's phase-1 top-k heads (default 8), b_max = their deepest block
    mlp span    = (b_max - 1) .. n_layers - 1
    mlp singles = b_max .. n_layers - 2
    attn span   = (b_max + 1) .. n_layers - 1
  Qwen (common10 / the flags traces, deepest mover block 23, 28 blocks) -> 22-27, 23..26, 24-27.

DEPTH (`depth --root R:N ...`): each pipeline root's per-attribute handoff and head window as
block indices AND as a fraction of that model's depth N ((block + 1) / N), so models of different
depth compare on one axis. Head IDs themselves are not comparable across models.

router/severed print shell-ready arguments on stdout (and nothing else), plus a JSON record with
the curves and any warnings under --out. No torch, no model: a pure reader.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plateau_end(curve, frac=0.95):
    """Deepest layer of the curve's TOP plateau (see image_head_pipeline.select_handoff)."""
    layers = sorted(curve)
    top = max(curve.values())
    start = next(l for l in layers if curve[l] >= top - 1e-9)
    end = start
    for l in layers[layers.index(start):]:
        if curve[l] >= frac * top:
            end = l
        else:
            break
    return end


def select_router_window(question, last, n_layers, patch_frac=0.95, read_frac=0.9, tail=2):
    """question, last: {layer: donor_first}. -> dict(patch_layer, read_end, blocks, warnings)."""
    warnings = []
    if max(question.values()) < 0.5:
        warnings.append(f"question-token swap never exceeds {max(question.values()):.0%}: the "
                        f"controlled prompts may not be read at all")
    end = plateau_end(question, patch_frac)
    patch_layer = max(end - 1, 1)
    floor, top = min(last.values()), max(last.values())
    if top - floor < 0.3:
        warnings.append(f"last-token swap moves at most {top - floor:.0%} above its floor: no clear readout")
    target = floor + read_frac * (top - floor)
    read_end = next((l for l in sorted(last) if l > patch_layer and last[l] >= target), None)
    if read_end is None:
        read_end = max(last, key=lambda l: last[l])
        warnings.append(f"last-token curve never reaches {target:.0%} above layer {patch_layer}; using its argmax")
    hi = min(max(read_end + tail, patch_layer), n_layers - 1)
    return {"plateau_end": end, "patch_layer": patch_layer, "read_end": read_end,
            "blocks": list(range(patch_layer, hi + 1)), "warnings": warnings}


def select_freeze_spans(head_blocks, n_layers):
    """head_blocks: blocks of the installed heads. -> head_severed's freeze arguments."""
    b = max(head_blocks)
    assert b + 1 <= n_layers - 1, f"deepest head block {b} leaves no later block to freeze"
    return {"mlp_span": f"{b - 1}-{n_layers - 1}", "mlp_singles": list(range(b, n_layers - 1)),
            "attn_span": f"{b + 1}-{n_layers - 1}"}


def switch_curves(switch_dir, metric="first"):
    """-> (question, last) residual curves {layer: rate} off an attribute_switch_sweep run."""
    from methods.attribute_switch_report import crossover, load_cells
    rows = crossover(load_cells(switch_dir), metric=metric)
    q = {r["layer"]: r["question_residual"] for r in rows if r["question_residual"] is not None}
    t = {r["layer"]: r["last_residual"] for r in rows if r["last_residual"] is not None}
    assert q and t, (f"{switch_dir} has no earlier_text/residual or last_token/residual cells -- run the "
                     f"sweep with --sites residual --scopes earlier_text last_token")
    return q, t


def trace_top(trace_path, k=8):
    ranked = json.loads(Path(trace_path).read_text())["phase1_ranked"][:k]
    return [(int(b), int(h)) for b, h in ranked]


def depth_rows(root, n_layers):
    """-> [dict] per pipeline.json under an image_head_pipeline root."""
    rows = []
    for p in sorted(Path(root).glob("*/*/pipeline.json")):
        d = json.loads(p.read_text())
        h, w = d.get("handoff") or {}, d.get("window") or {}
        blocks = w.get("blocks") or []
        rows.append({"root": str(root), "entity": d["entity"], "attribute": d["attribute"], "n_layers": n_layers,
                     "patch_layer": h.get("patch_layer"), "read_end": h.get("read_end"),
                     "window": w.get("label"), "window_score": w.get("score"),
                     "rel_window": [round((b + 1) / n_layers, 3) for b in (blocks[0], blocks[-1])] if blocks else None,
                     "rel_read_end": round(h["read_end"] / n_layers, 3) if h.get("read_end") is not None else None})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("router", help="attr_head_trace's --patch_layer / --blocks")
    r.add_argument("--switch_dir", required=True)
    r.add_argument("--n_layers", type=int, required=True)
    r.add_argument("--out", default=None)
    s = sub.add_parser("severed", help="head_severed's --heads and freeze spans")
    s.add_argument("--trace", required=True, help="a head_trace JSON (image_head_pipeline stage 3)")
    s.add_argument("--n_layers", type=int, required=True)
    s.add_argument("--top_k", type=int, default=8)
    s.add_argument("--out", default=None)
    d = sub.add_parser("depth", help="handoff / head window per model, as a fraction of depth")
    d.add_argument("--root", nargs="+", required=True, help="ROOT:N_LAYERS, e.g. results/image_head_pipeline:28")
    d.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "depth":
        rows = []
        for spec in args.root:
            root, _, n = spec.rpartition(":")
            rows += depth_rows(root, int(n))
        print(f"{'root':<48} {'entity/attribute':<26} {'P':>3} {'read':>4} {'window':>7} {'score':>6} "
              f"{'rel window':>12} {'rel read':>8}")
        for r in rows:
            rw = "-".join(f"{x:.2f}" for x in r["rel_window"]) if r["rel_window"] else "--"
            sc = f"{r['window_score']:.2f}" if r["window_score"] is not None else "--"
            print(f"{r['root']:<48} {r['entity'] + '/' + r['attribute']:<26} {r['patch_layer'] or '--':>3} "
                  f"{r['read_end'] or '--':>4} {r['window'] or '--':>7} {sc:>6} {rw:>12} {r['rel_read_end'] or '--':>8}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        return

    if args.cmd == "router":
        q, t = switch_curves(args.switch_dir)
        sel = select_router_window(q, t, args.n_layers)
        record = {**sel, "question_curve": q, "last_token_curve": t, "switch_dir": args.switch_dir}
        shell = f"--patch_layer {sel['patch_layer']} --blocks {' '.join(map(str, sel['blocks']))}"
    else:
        heads = trace_top(args.trace, args.top_k)
        spans = select_freeze_spans([b for b, _ in heads], args.n_layers)
        record = {"heads": heads, **spans, "trace": args.trace}
        shell = (f"--heads {','.join(f'{b}.{h}' for b, h in heads)} --freeze_mlp_span {spans['mlp_span']} "
                 f"--freeze_mlp_singles {' '.join(map(str, spans['mlp_singles']))} "
                 f"--freeze_attn_span {spans['attn_span']}")
    for w in record.get("warnings", []):
        print(f"WARNING: {w}", file=sys.stderr)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(shell)


if __name__ == "__main__":
    main()
