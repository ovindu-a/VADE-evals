"""E5: which ROUTE carries the entity from the image to the answer?

Salazar et al. (arXiv 2607.03358) find two pathways in VLMs: DIRECT (the answer
position reads the image tokens) and TEXT-MEDIATED (the image is first written into
the question tokens, which the answer position then reads), and that ablating one can
reveal the other as a fallback. Our circuit claims the entity travels DIRECT, through
the blocks 21-23 mover heads at the answer position. This script tests that claim
with their two tools, on our prompts and our readout.

1. ATTENTION-EDGE KNOCKOUT. Remove attention edges (query group -> image tokens) in a
   chosen block range, for every head. Default renormalizes the remaining weights,
   which is exactly an additive -inf on those logits (Geva et al. 2023; Salazar et al.);
   --no_renormalize zeroes them without redistribution instead. Implemented as an exact
   correction to the attention module's output -- out += W_O ((A' - A) V) -- so no HF
   attention code is replaced, and A@V is asserted to reconstruct o_proj's input on the
   first batch (a wrong tensor or GQA layout fails loudly, not plausibly).

      question   queries = the text after the image, EXCLUDING the readout columns
      readout    queries = the last prompt column + the appended divergence prefix
      text       both

2. TEXT PATCHING. Patch the residual at the question columns from the SOURCE run at
   layer L, image untouched. Base and source prompts are token-identical there, so
   they differ ONLY by what the question tokens absorbed from the image: a transfer
   means the question tokens carry the entity at layer L. `qpatch@L+ko:readout_all`
   additionally cuts the direct route, asking whether that copy is enough on its own.

Every arm is also run with the source image patched in at --img_layer (residual, all
image columns, below the handoff), so each knockout is read twice: does the model
still answer about its OWN image (base_kept), and does an entity injected into the
image still reach the answer (transfer)?

PREDICTIONS IF THE CIRCUIT IS RIGHT
  ko:readout_all + img_patch      transfer collapses (the direct route is the route)
  ko:readout_movers + img_patch   most of that collapse, from blocks 21-23 alone
  ko:question_* + img_patch       transfer survives (no text mediation needed)
  qpatch@L                        ~0 at every L (question tokens never hold the entity)
  ko:text_all                     sanity: the model loses the image entirely

A fallback in Salazar's sense shows up as ko:readout_all leaving base_kept high
(the model still knows its own country) while qpatch@L transfers.

Readout, rows and eligibility: methods/circuit_common.py's module docstring.

Usage
-----
    python methods/route_knockout.py --dry_run
    python methods/route_knockout.py --n_pairs 8 --out_dir results/route_knockout/smoke
    python methods/route_knockout.py                       # 64 pairs x 4 attributes
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from methods.circuit_common import (ATTRIBUTES, DEFAULT_VADE_ROOT, MODEL_ID, clean_eligible,  # noqa: E402
                                    load_flags, print_table, sample_pairs, summarize_arms, write_json)

DEFAULT_KO = ["question_all=question:0-27", "question_early=question:0-15", "question_late=question:16-27",
              "readout_all=readout:0-27", "readout_movers=readout:21-23", "readout_pre=readout:0-20",
              "text_all=text:0-27"]
QUERY_GROUPS = ("question", "readout", "text")


def parse_ko(spec, n_layers=None):
    """'name=queries:a-b' -> (name, queries, [blocks])."""
    name, body = spec.split("=", 1)
    queries, span = body.split(":", 1)
    assert queries in QUERY_GROUPS, f"{spec}: queries must be one of {QUERY_GROUPS}"
    a, _, b = span.partition("-")
    blocks = list(range(int(a), int(b or a) + 1))
    if n_layers is not None:
        assert all(0 <= x < n_layers for x in blocks), f"{spec}: blocks outside 0..{n_layers - 1}"
    return name, queries, blocks


class EdgeKnockout:
    """Remove attention edges (queries -> keys) in `blocks`, all heads, batched.

    Needs eager attention and output_attentions=True (the weights are read off the
    attention module's own output). queries/keys are column lists shared by every row,
    which holds for the unpadded, equal-layout batches this script builds."""

    def __init__(self, m, blocks, queries, keys, renormalize=True, check=False):
        self.m, self.blocks = m, sorted(blocks)
        self.queries, self.keys = list(queries), list(keys)
        self.renormalize, self.check = renormalize, check
        self.values, self.z, self.handles = {}, {}, []
        self.max_recon_rel = 0.0
        self.calls = 0

    def __enter__(self):
        adapter, model = self.m.adapter, self.m.model
        try:
            for b in self.blocks:
                attn = adapter.get_attn_block(model, b)
                assert hasattr(attn, "v_proj"), "edge knockout needs a Qwen-style v_proj/o_proj"
                self.handles.append(attn.v_proj.register_forward_hook(
                    lambda mod, i, o, _b=b: self.values.__setitem__(_b, o.detach())))
                if self.check:
                    self.handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(
                        lambda mod, i, _b=b: self.z.__setitem__(_b, i[0].detach())))
                self.handles.append(attn.register_forward_hook(
                    lambda mod, i, o, _b=b: self.modify(_b, o)))
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def modify(self, block, output):
        import torch
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError("attention weights unavailable: load with attn_impl='eager' and pass "
                               "output_attentions=True")
        out, A = output[0], output[1]
        if A.shape[-1] != out.shape[1]:
            raise AssertionError("edge knockout needs a full-prefix forward (no KV cache)")
        Bsz, T, _ = out.shape
        H, d = self.m.n_heads, self.m.head_dim
        V = self.values[block]
        kv = V.shape[-1] // d
        assert V.shape[-1] % d == 0 and H % kv == 0 and A.shape[1] == H, "unexpected GQA layout"
        Vh = V.view(Bsz, T, kv, d).permute(0, 2, 1, 3).float().repeat_interleave(H // kv, dim=1)  # [B,H,T,d]
        Q = torch.tensor(self.queries, device=out.device)
        Aq = A[:, :, Q, :].float()                                                              # [B,H,q,T]
        if self.check:
            z = self.z[block][:, Q, :].float().view(Bsz, len(self.queries), H, d).permute(0, 2, 1, 3)
            rel = float((Aq @ Vh - z).norm() / z.norm().clamp_min(1e-8))
            self.max_recon_rel = max(self.max_recon_rel, rel)
            tol = 1e-4 if out.dtype == torch.float32 else 0.02
            if rel > tol:
                raise AssertionError(f"A@V != o_proj input at block {block}: {rel:.3%}; wrong tensor/GQA layout")
        An = Aq.clone()
        An[..., self.keys] = 0
        if self.renormalize:
            An = An / An.sum(-1, keepdim=True).clamp_min(1e-30)
        dz = ((An - Aq) @ Vh).permute(0, 2, 1, 3).reshape(Bsz, len(self.queries), H * d)
        W = self.m.adapter.get_attn_head_output_module(self.m.model, block).weight.float()      # [hidden, H*d]
        patched = out.clone()
        patched[:, Q] = (out[:, Q].float() + dz @ W.T).to(out.dtype)
        self.calls += 1
        return (patched,) + tuple(output[1:])


def layout(ids, image_id, k):
    """-> (image_cols, question_cols, readout_cols) for an unpadded batch whose last k
    columns are the appended divergence prefix."""
    import torch
    img = ids == image_id
    assert torch.equal(img, img[:1].expand_as(img)), "rows disagree on image positions"
    cols = img[0].nonzero().flatten().tolist()
    assert cols, "no image tokens"
    T = ids.shape[1]
    prompt_last = T - 1 - k
    readout = list(range(prompt_last, T))
    question = list(range(cols[-1] + 1, prompt_last))
    assert question, "no text between the image and the readout"
    return cols, question, readout


def build_batch(m, entity_dir, items, rows):
    import torch
    from methods.circuit_common import cat_extra
    base, src, bx, sx = [], [], [], []
    for r in rows:
        b_ids, b_extra = m.image_prompt(entity_dir, items[r["base"]], r["attribute"])
        s_ids, s_extra = m.image_prompt(entity_dir, items[r["source"]], r["attribute"])
        tail = torch.tensor(r["prefix"], dtype=torch.long)
        base.append(torch.cat([b_ids, tail])); src.append(torch.cat([s_ids, tail]))
        bx.append(b_extra); sx.append(s_extra)
    assert len({len(x) for x in base + src}) == 1, (
        "controlled image prompts must share one length; route_knockout cannot pad")
    ids_b, ids_s = torch.stack(base), torch.stack(src)
    return ids_b, ids_s, torch.ones_like(ids_b), cat_extra(bx), cat_extra(sx)


def run_batch(m, args, kos, rows, ids_b, ids_s, mask, xb, xs, check=False):
    """-> list of records for this batch (one per row x arm)."""
    import contextlib
    import torch
    from methods.circuit_common import gather_cols, readout_metrics, residual_patch
    k = len(rows[0]["prefix"])
    img_cols, q_cols, r_cols = layout(ids_b, m.image_id, k)
    groups = {"question": q_cols, "readout": r_cols, "text": q_cols + r_cols}
    B = len(rows)

    def pos(cols):
        return torch.tensor([cols] * B)

    src_out = m.forward(ids_s, mask, xs, output_hidden_states=True)
    hs = src_out.hidden_states
    img_src = gather_cols(hs[args.img_layer], pos(img_cols))
    q_src = {L: gather_cols(hs[L], pos(q_cols)) for L in args.q_layers}
    src_logits = src_out.logits[:, -1]
    del src_out, hs

    records = []

    def emit(arm, logits):
        for r, met in zip(rows, readout_metrics(logits, rows)):
            records.append({"row_index": r["row_index"], "attribute": r["attribute"], "arm": arm, **met})

    def run(arm, hooks=(), ko=None):
        with (EdgeKnockout(m, ko[2], groups[ko[1]], img_cols, renormalize=not args.no_renormalize,
                           check=check) if ko else contextlib.nullcontext()) as edge:
            out = m.forward(ids_b, mask, xb, hooks=hooks, output_attentions=ko is not None)
            if ko is not None:
                assert edge.calls == len(ko[2]), f"{arm}: knockout fired {edge.calls}x for {len(ko[2])} blocks"
                if check:
                    print(f"    [check] {arm}: A@V reconstructs o_proj input to {edge.max_recon_rel:.2e}")
        emit(arm, out.logits[:, -1])
        return out.logits[:, -1]

    img_patch = residual_patch(args.img_layer, pos(img_cols), img_src)
    clean = run("clean")
    emit("source_clean", src_logits)
    run("img_patch", [img_patch])
    if check:
        # An empty key set must be an exact no-op, through the same code path.
        with EdgeKnockout(m, [0], groups["text"], [], check=True):
            same = m.forward(ids_b, mask, xb, output_attentions=True).logits[:, -1]
        err = float((same.float() - clean.float()).abs().max())
        print(f"    [check] knockout with NO keys vs clean: max |dlogit| = {err:.2e}")
        assert err < 1e-2, "an empty knockout changed the logits; the correction is not exact"
    for ko in kos:
        run(f"ko:{ko[0]}", ko=ko)
        run(f"ko:{ko[0]}+img_patch", [img_patch], ko=ko)
    # The question route ALONE: source question columns patched at L, and the readout cut off
    # from the image in EVERY block. Cutting only blocks >= L would leave whatever the readout
    # read directly from the BASE image in earlier blocks, competing with the patched copy.
    direct_cut = ("readout_every_block", "readout", list(range(m.n_layers)))
    for L in args.q_layers:
        qp = residual_patch(L, pos(q_cols), q_src[L])
        run(f"qpatch@{L}", [qp])
        run(f"qpatch@{L}+ko:readout_all", [qp], ko=direct_cut)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--attributes", nargs="+", default=list(ATTRIBUTES), choices=list(ATTRIBUTES))
    ap.add_argument("--n_pairs", type=int, default=64, help="Ordered (base, source) country pairs.")
    ap.add_argument("--knockouts", nargs="+", default=DEFAULT_KO, metavar="NAME=QUERIES:A-B",
                    help=f"QUERIES in {QUERY_GROUPS}; A-B is an inclusive 0-based block range.")
    ap.add_argument("--img_layer", type=int, default=16,
                    help="Residual layer for the source-image patch. Must sit BELOW the handoff, where the "
                         "image-position swap is still ~100%% (<=22 on flags).")
    ap.add_argument("--q_layers", type=int, nargs="+", default=[4, 8, 12, 16, 18, 20, 22, 24])
    ap.add_argument("--no_renormalize", action="store_true",
                    help="Zero knocked-out weights WITHOUT redistributing the mass (default renormalizes, "
                         "i.e. an additive -inf mask).")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "route_knockout", "flags"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    kos = [parse_ko(s) for s in args.knockouts]
    entity_dir, items, gt_sha = load_flags(args.vade_root)
    pairs = sample_pairs(items, args.n_pairs, args.seed)
    n_arms = 3 + 2 * len(kos) + 2 * len(args.q_layers)
    print(f"[route_knockout] {len(pairs)} pairs x {len(args.attributes)} attributes; img_layer={args.img_layer}; "
          f"q_layers={args.q_layers}; ~{n_arms} forwards per batch")
    for name, q, blocks in kos:
        print(f"  ko:{name:<16} {q:>8} -> image, blocks {blocks[0]}-{blocks[-1]}")
    if args.dry_run:
        print("Metadata valid (tokenization and divergence need the model's tokenizer).")
        return

    from methods.circuit_common import Model, chunks_by_prefix, pair_rows
    m = Model(args.model_id, args.device, attn_impl="eager", vade_root=args.vade_root)
    kos = [parse_ko(s, m.n_layers) for s in args.knockouts]
    assert 0 < args.img_layer < m.n_layers and all(0 < L < m.n_layers for L in args.q_layers)
    rows, dropped = pair_rows(items, m.tokenizer, pairs, args.attributes)
    print(f"  {len(rows)} rows ({dropped} dropped: golds never diverge)")
    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    with open(os.path.join(args.out_dir, "rows.jsonl"), "w") as fh:
        for i, chunk in enumerate(chunks_by_prefix(rows, args.batch_size)):
            ids_b, ids_s, mask, xb, xs = build_batch(m, entity_dir, items, chunk)
            recs = run_batch(m, args, kos, chunk, ids_b, ids_s, mask, xb, xs, check=(i == 0))
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            records += recs
            print(f"  {len({r['row_index'] for r in records})}/{len(rows)} rows", flush=True)

    eligible = clean_eligible(records)
    summary = {"config": {k: v for k, v in vars(args).items() if k != "dry_run"},
               "ground_truth_sha256": gt_sha, "n_rows": len(rows), "n_eligible": len(eligible),
               "all": summarize_arms(records), "eligible": summarize_arms(records, eligible),
               "by_attribute": {a: summarize_arms([r for r in records if r["attribute"] == a], eligible)
                                for a in args.attributes}}
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print_table(summary["eligible"], f"ELIGIBLE rows (both clean runs correct): n={len(eligible)}")
    ref = summary["eligible"].get("img_patch", {}).get("transfer", 0.0)
    if ref < 0.8:
        print(f"\n  WARNING: the image patch itself transfers only {ref:.1%} at --img_layer {args.img_layer}. "
              f"Every '+img_patch' arm is read against it; move --img_layer below the handoff.")
    print("\n  Read: `transfer` under '+img_patch' arms = the injected entity still reaches the answer;\n"
          "  `base_kept` on knockout-only arms = the model still answers about its own image.\n"
          "  normLD = (LD - LD_clean) / (LD_source - LD_clean): 0 no effect, 1 full transfer.")
    print(f"\nwrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
