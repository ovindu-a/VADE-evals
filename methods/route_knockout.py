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

KNOCKOUT SPECS. `name=PART[+PART...]`, PART = `queries:a-b[@group]`:

      queries    question | readout | text (as above)
      a-b        inclusive 0-based block range ('a' alone = one block)
      group      KEYS: image (default -- every image column, the original behaviour),
                 object (the entity's smallest object_token_indices set: flags'
                 entity_only, 8 tokens) or background (image minus object)

`readout_all=readout:0-27` parses exactly as it always has. Stacked parts remove the
UNION of their edges in one EdgeKnockout (see EdgeKnockout.union for why two separate
corrections would not compose), renormalized once per query row over that union.

HEAD INSTALL (--install_heads B.H,... or common10). Those heads' o_proj inputs are
captured at the readout columns in the SOURCE forward and written into the base forward
at the same columns (text_entity_contrast.head_install's pre-hook), adding arms
`install` and `install+ko:<name>` per knockout. One full-prefix forward, so the install
is prefill-only by construction. A knockout whose READOUT queries (readout/text) share a
block with an installed head is refused: its correction would be computed against A@V
that the install has replaced.

ENTITY (--entity brands|celebrities). VADE's own prompt template (--template, by
template_id) instead of the flags-only controlled prompt, golds from the model's OWN
greedy answers (head_severed --gold own: these entities store lowercase labels), rows
kept only where both answers contain their stored label, chunked per attribute.

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

from methods.circuit_common import (ATTRIBUTES, DEFAULT_VADE_ROOT, MODEL_ID, MOVERS, clean_eligible,  # noqa: E402
                                    load_flags, parse_heads, print_table, sample_pairs, summarize_arms,
                                    write_json)

DEFAULT_KO = ["question_all=question:0-27", "question_early=question:0-15", "question_late=question:16-27",
              "readout_all=readout:0-27", "readout_movers=readout:21-23", "readout_pre=readout:0-20",
              "text_all=text:0-27"]
QUERY_GROUPS = ("question", "readout", "text")
KEY_GROUPS = ("image", "object", "background")
READOUT_QUERIES = ("readout", "text")          # query groups that include the readout columns
NON_FLAGS_ITEMS = {"brands": "brands", "celebrities": "people"}
# Options added after the first results: omitted from summary.json's config at these
# defaults, so a default run's outputs stay byte-identical to earlier ones.
ADDED_OPTS = {"install_heads": None, "entity": "flags", "template": 0, "max_new_tokens": 8}


def parse_ko_parts(spec, n_layers=None):
    """'name=PART[+PART...]', PART = 'queries:a-b[@group]' -> (name, [(queries, [blocks], group)])."""
    name, body = spec.split("=", 1)
    parts = []
    for part in body.split("+"):
        part, _, group = part.partition("@")
        group = group or "image"
        assert group in KEY_GROUPS, f"{spec}: key group must be one of {KEY_GROUPS}"
        queries, span = part.split(":", 1)
        assert queries in QUERY_GROUPS, f"{spec}: queries must be one of {QUERY_GROUPS}"
        a, _, b = span.partition("-")
        blocks = list(range(int(a), int(b or a) + 1))
        if n_layers is not None:
            assert all(0 <= x < n_layers for x in blocks), f"{spec}: blocks outside 0..{n_layers - 1}"
        parts.append((queries, blocks, group))
    return name, parts


def parse_ko(spec, n_layers=None):
    """'name=queries:a-b' -> (name, queries, [blocks]): the original single-part,
    image-keyed form. Stacked or @object/@background specs need parse_ko_parts."""
    name, parts = parse_ko_parts(spec, n_layers)
    assert len(parts) == 1 and parts[0][2] == "image", f"{spec}: stacked/grouped spec; use parse_ko_parts"
    return name, parts[0][0], parts[0][1]


def describe_ko(parts):
    return " + ".join(f"{q:>8} -> {g}, blocks {bl[0]}-{bl[-1]}" for q, bl, g in parts)


def object_index(entity_dir):
    """-> (n image tokens, sorted image-relative indices of the entity's smallest
    object_token_indices set), read the way vrh_ablation / head_token_ablation do."""
    loc = json.load(open(os.path.join(entity_dir, "object_location.json")))
    obj = min(loc["object_token_indices"].values(), key=lambda v: len(v["flat"]))["flat"]
    return loc["vlm_token_grid"]["n_tokens"], sorted(obj)


def key_columns(img_cols, obj=None):
    """-> {group: columns}. object/background partition the image columns."""
    out = {"image": list(img_cols)}
    if obj is not None:
        n, idx = obj
        assert n == len(img_cols), f"object geometry is for {n} image tokens, this batch has {len(img_cols)}"
        out["object"] = [img_cols[i] for i in idx]
        out["background"] = [c for i, c in enumerate(img_cols) if i not in set(idx)]
        assert sorted(out["object"] + out["background"]) == out["image"]
    return out


def install_clashes(kos, install):
    """-> [(knockout name, shared blocks)] where a readout-query knockout shares a block
    with an installed head (refused: see the module docstring)."""
    ib = {b for b, _ in install or ()}
    return [(name, sorted(ib & set(bl))) for name, parts in kos for q, bl, _ in parts
            if q in READOUT_QUERIES and ib & set(bl)]


class EdgeKnockout:
    """Remove attention edges (queries -> keys) in `blocks`, all heads, batched.

    Needs eager attention and output_attentions=True (the weights are read off the
    attention module's own output). queries/keys are column lists shared by every row,
    which holds for the unpadded, equal-layout batches this script builds."""

    def __init__(self, m, blocks, queries, keys, renormalize=True, check=False):
        self._setup(m, [(blocks, queries, keys)], renormalize, check)

    @classmethod
    def union(cls, m, parts, renormalize=True, check=False):
        """parts = [(blocks, queries, keys)] -> ONE knockout removing the union of their edges.

        Two EdgeKnockouts on one block do NOT compose: each computes its correction from
        the ORIGINAL A (modify leaves output[1] untouched), so on a query row both touch
        they add (A'_1 - A) + (A'_2 - A) -- an edge in both is subtracted twice, and each
        renormalizes with only its own keys removed. They agree only when no (block,
        query row) is shared. Here each (block, query row) removes the union of its
        parts' keys and, with renormalize, rescales ONCE to the row's original mass --
        exactly stacking additive -inf masks (renormalizing after K1 then K2 equals
        renormalizing once after K1 | K2). Rows a part does not touch are unchanged."""
        self = cls.__new__(cls)
        self._setup(m, parts, renormalize, check)
        return self

    def _setup(self, m, parts, renormalize, check):
        self.m = m
        parts = [(sorted(bl), list(q), list(k)) for bl, q, k in parts]
        self.blocks = sorted({b for bl, _, _ in parts for b in bl})
        self.edges = {b: [(q, k) for bl, q, k in parts if b in bl] for b in self.blocks}
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
        queries = list(dict.fromkeys(q for qs, _ in self.edges[block] for q in qs))  # order kept for one part
        Q = torch.tensor(queries, device=out.device)
        Aq = A[:, :, Q, :].float()                                                              # [B,H,q,T]
        if self.check:
            z = self.z[block][:, Q, :].float().view(Bsz, len(queries), H, d).permute(0, 2, 1, 3)
            rel = float((Aq @ Vh - z).norm() / z.norm().clamp_min(1e-8))
            self.max_recon_rel = max(self.max_recon_rel, rel)
            tol = 1e-4 if out.dtype == torch.float32 else 0.02
            if rel > tol:
                raise AssertionError(f"A@V != o_proj input at block {block}: {rel:.3%}; wrong tensor/GQA layout")
        kill = torch.zeros(len(queries), T, dtype=torch.bool, device=out.device)  # [q,T]: edges to remove
        row = {q: i for i, q in enumerate(queries)}
        for qs, ks in self.edges[block]:
            r = torch.tensor([row[q] for q in qs], dtype=torch.long, device=out.device)
            kill[r[:, None], torch.tensor(ks, dtype=torch.long, device=out.device)[None, :]] = True
        An = Aq.masked_fill(kill, 0)
        if self.renormalize:
            # Back to each row's OWN original mass, not to 1. Identical in exact arithmetic, but
            # bf16 attention rows sum to 1 only within ~1e-3: dividing by the sum anyway rescaled
            # every queried row, so an EMPTY knockout moved the 7B's logits by 1.31 and failed
            # the no-op check. S/S is exactly 1 in IEEE arithmetic, so no keys -> no change.
            An = An * (Aq.sum(-1, keepdim=True) / An.sum(-1, keepdim=True).clamp_min(1e-30))
        dz = ((An - Aq) @ Vh).permute(0, 2, 1, 3).reshape(Bsz, len(queries), H * d)
        W = self.m.adapter.get_attn_head_output_module(self.m.model, block).weight.float()      # [hidden, H*d]
        patched = out.clone()
        patched[:, Q] = (out[:, Q].float() + dz @ W.T).to(out.dtype)
        self.calls += 1
        return (patched,) + tuple(output[1:])


class HeadCapture:
    """o_proj inputs at `cols` (shared by every row) for `blocks` -> self.z {b: [B, len(cols), H*d]}."""

    def __init__(self, m, blocks, cols):
        self.m, self.blocks, self.cols = m, sorted(blocks), list(cols)
        self.z, self.handles = {}, []

    def __enter__(self):
        for b in self.blocks:
            self.handles.append(self.m.adapter.get_attn_head_output_module(self.m.model, b).register_forward_pre_hook(
                lambda mod, i, _b=b: self.z.__setitem__(_b, i[0][:, self.cols].detach())))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class HeadInstall:
    """Replace `heads`' slices of o_proj's input at `cols` with z[b][:, :, slice]:
    text_entity_contrast.head_install's pre-hook, as a context manager so it can wrap a
    forward that also runs an EdgeKnockout. Enter it BEFORE the knockout: PyTorch runs
    pre-hooks in registration order, so the knockout's A@V check then reads the
    installed tensor (identical to A@V at every column the install leaves alone)."""

    def __init__(self, m, heads, cols, z):
        from methods.head_followup_common import grouped_heads
        self.m, self.groups, self.cols, self.z = m, grouped_heads(heads), list(cols), z
        self.handles, self.calls = [], 0

    def __enter__(self):
        for b, hs in self.groups.items():
            self.handles.append(self.m.adapter.get_attn_head_output_module(self.m.model, b).register_forward_pre_hook(
                lambda mod, i, _b=b, _hs=hs: self.pre(_b, _hs, i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def pre(self, block, heads, inp):
        t = inp[0]
        assert t.shape[1] > max(self.cols), "head install needs a full-prefix forward (no KV cache)"
        t = t.clone()
        for h in heads:
            sl = slice(h * self.m.head_dim, (h + 1) * self.m.head_dim)
            t[:, self.cols, sl] = self.z[block][:, :, sl].to(t)
        self.calls += 1
        return (t,) + tuple(inp[1:])


def load_entity(vade_root, entity, template=0):
    """brands/celebrities -> (entity_dir, items, sha256 of ground_truth.json,
    {attribute: (question, prefill)}): VADE's template number `template` (by template_id)."""
    import hashlib
    entity_dir = os.path.join(vade_root, "data", entity)
    raw = open(os.path.join(entity_dir, "ground_truth.json"), "rb").read()
    gt = json.loads(raw)
    items = gt[NON_FLAGS_ITEMS[entity]]
    for key, it in items.items():
        assert os.path.isfile(os.path.join(entity_dir, it["image"])), f"missing image for {key}"
    tpl = json.load(open(os.path.join(entity_dir, "prompt_templates.json")))
    pick = {a: sorted(tpl[a], key=lambda t: t["template_id"])[template] for a in gt["attributes"]}
    return entity_dir, items, hashlib.sha256(raw).hexdigest(), {a: (t["question"], t["prefill"])
                                                                 for a, t in pick.items()}


def entity_prompt(m, entity_dir, item, prompt):
    """Model.image_prompt with a VADE (question, prefill) instead of the controlled flags prompt."""
    from PIL import Image
    path = os.path.join(entity_dir, item["image"])
    if path not in m._images:
        with Image.open(path) as im:
            m._images[path] = im.convert("RGB")
    x = m.adapter.build_inputs(m.processor, m._images[path], *prompt)
    return x["input_ids"], x["extra"]


def own_answers(m, entity_dir, items, keys, attributes, prompts, max_new_tokens, batch_size):
    """{(key, attribute): the model's own greedy answer ids}, cut at the first stop token."""
    import torch
    from methods.circuit_common import cat_extra
    eos = m.model.generation_config.eos_token_id
    stop = {t for t in ([eos] if isinstance(eos, int) else list(eos or [])) + [m.tokenizer.eos_token_id, m.pad_id]
            if t is not None}
    out = {}
    for a in attributes:
        have = [k for k in keys if a in items[k]]      # brands: AIRBUS has no industry label
        for i in range(0, len(have), batch_size):
            ks = have[i:i + batch_size]
            xs = [entity_prompt(m, entity_dir, items[k], prompts[a]) for k in ks]
            assert len({len(x[0]) for x in xs}) == 1, "template prompts must share one length (no padding)"
            ids = torch.stack([x[0] for x in xs])
            for k, g in zip(ks, m.generate(ids, torch.ones_like(ids), cat_extra([x[1] for x in xs]),
                                           max_new_tokens).tolist()):
                out[(k, a)] = g[:next((j for j, t in enumerate(g) if t in stop), len(g))]
    return out


def own_rows(items, pairs, attributes, answers, decode):
    """circuit_common.pair_rows with the model's OWN answers as golds. A row needs
    different stored labels, both answers containing their stored label (VADE's
    matcher: the model is right about both items), and answers that diverge."""
    from methods.circuit_common import divergence
    from methods.common.text_match import contains_label, normalize
    rows, dropped = [], {"no_label": 0, "same_label": 0, "model_wrong": 0, "no_divergence": 0}
    for base, source in pairs:
        for attr in attributes:
            if attr not in items[base] or attr not in items[source]:
                dropped["no_label"] += 1
                continue
            lb, ls = str(items[base][attr]), str(items[source][attr])
            ab, as_ = answers[(base, attr)], answers[(source, attr)]
            why = ("same_label" if normalize(lb) == normalize(ls) else
                   "model_wrong" if not (contains_label(decode(ab), lb) and contains_label(decode(as_), ls)) else
                   None if divergence(ab, as_) else "no_divergence")
            if why:
                dropped[why] += 1
                continue
            prefix, bt, st = divergence(ab, as_)
            rows.append({"row_index": len(rows), "base": base, "source": source, "attribute": attr,
                         "base_label": lb, "source_label": ls, "prefix": prefix, "base_tok": bt, "src_tok": st})
    return rows, dropped


def chunks_by_attribute(rows, batch_size):
    """chunks_by_prefix within each attribute: VADE templates differ in length per attribute."""
    from methods.circuit_common import chunks_by_prefix
    for a in dict.fromkeys(r["attribute"] for r in rows):
        yield from chunks_by_prefix([r for r in rows if r["attribute"] == a], batch_size)


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


def build_batch(m, entity_dir, items, rows, prompts=None):
    import torch
    from methods.circuit_common import cat_extra

    def prompt(key, attr):
        if prompts is None:
            return m.image_prompt(entity_dir, items[key], attr)
        return entity_prompt(m, entity_dir, items[key], prompts[attr])
    base, src, bx, sx = [], [], [], []
    for r in rows:
        b_ids, b_extra = prompt(r["base"], r["attribute"])
        s_ids, s_extra = prompt(r["source"], r["attribute"])
        tail = torch.tensor(r["prefix"], dtype=torch.long)
        base.append(torch.cat([b_ids, tail])); src.append(torch.cat([s_ids, tail]))
        bx.append(b_extra); sx.append(s_extra)
    assert len({len(x) for x in base + src}) == 1, (
        "controlled image prompts must share one length; route_knockout cannot pad")
    ids_b, ids_s = torch.stack(base), torch.stack(src)
    return ids_b, ids_s, torch.ones_like(ids_b), cat_extra(bx), cat_extra(sx)


def run_batch(m, args, kos, rows, ids_b, ids_s, mask, xb, xs, check=False, obj=None, install=None):
    """-> list of records for this batch (one per row x arm). kos = [(name, parts)] from
    parse_ko_parts; obj = object_index(...) when a part keys on object/background;
    install = [(block, head)] adds the `install` / `install+ko:*` arms."""
    import contextlib
    import torch
    from methods.circuit_common import gather_cols, readout_metrics, residual_patch
    k = len(rows[0]["prefix"])
    img_cols, q_cols, r_cols = layout(ids_b, m.image_id, k)
    groups = {"question": q_cols, "readout": r_cols, "text": q_cols + r_cols}
    keys = key_columns(img_cols, obj)
    B = len(rows)
    clash = install_clashes(kos, install)
    if clash:
        raise ValueError(f"knockouts {clash} remove readout-query edges in blocks holding installed heads; "
                         f"the correction would be computed against A@V that the install replaced")
    install_blocks = sorted({b for b, _ in install or ()})

    def pos(cols):
        return torch.tensor([cols] * B)

    with (HeadCapture(m, install_blocks, r_cols) if install else contextlib.nullcontext()) as cap:
        src_out = m.forward(ids_s, mask, xs, output_hidden_states=True)
    z_src = cap.z if install else None
    hs = src_out.hidden_states
    img_src = gather_cols(hs[args.img_layer], pos(img_cols))
    q_src = {L: gather_cols(hs[L], pos(q_cols)) for L in args.q_layers}
    src_logits = src_out.logits[:, -1]
    del src_out, hs

    records = []

    def emit(arm, logits):
        for r, met in zip(rows, readout_metrics(logits, rows)):
            records.append({"row_index": r["row_index"], "attribute": r["attribute"], "arm": arm, **met})

    def knockout(parts, check):
        assert all(g in keys for _, _, g in parts), "object/background keys need obj=object_index(entity_dir)"
        return EdgeKnockout.union(m, [(bl, groups[q], keys[g]) for q, bl, g in parts],
                                  renormalize=not args.no_renormalize, check=check)

    def run(arm, hooks=(), ko=None, z=None):
        with (HeadInstall(m, install, r_cols, z) if z is not None else contextlib.nullcontext()) as inst, \
                (knockout(ko[1], check) if ko else contextlib.nullcontext()) as edge:
            out = m.forward(ids_b, mask, xb, hooks=hooks, output_attentions=ko is not None)
            if ko is not None:
                assert edge.calls == len(edge.blocks), f"{arm}: knockout fired {edge.calls}x for {len(edge.blocks)} blocks"
                if check:
                    print(f"    [check] {arm}: A@V reconstructs o_proj input to {edge.max_recon_rel:.2e}")
            if z is not None:
                assert inst.calls == len(install_blocks), f"{arm}: install fired {inst.calls}x"
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
        if any(len(parts) > 1 or any(g != "image" for _, _, g in parts) for _, parts in kos):
            # ... and through the union path, with two parts sharing query rows.
            with EdgeKnockout.union(m, [([0], groups["text"], []), ([0, 1], r_cols, [])], check=True):
                same = m.forward(ids_b, mask, xb, output_attentions=True).logits[:, -1]
            err = float((same.float() - clean.float()).abs().max())
            print(f"    [check] stacked knockout with NO keys vs clean: max |dlogit| = {err:.2e}")
            assert err < 1e-2, "an empty stacked knockout changed the logits; the union is not exact"
    for ko in kos:
        run(f"ko:{ko[0]}", ko=ko)
        run(f"ko:{ko[0]}+img_patch", [img_patch], ko=ko)
    # The question route ALONE: source question columns patched at L, and the readout cut off
    # from the image in EVERY block. Cutting only blocks >= L would leave whatever the readout
    # read directly from the BASE image in earlier blocks, competing with the patched copy.
    direct_cut = ("readout_every_block", [("readout", list(range(m.n_layers)), "image")])
    for L in args.q_layers:
        qp = residual_patch(L, pos(q_cols), q_src[L])
        run(f"qpatch@{L}", [qp])
        run(f"qpatch@{L}+ko:readout_all", [qp], ko=direct_cut)
    if install:
        inst = run("install", z=z_src)
        if check:
            # Installing the base's OWN values must be an exact no-op, and an empty knockout
            # inside the install must leave the install's logits unchanged.
            with HeadCapture(m, install_blocks, r_cols) as own:
                m.forward(ids_b, mask, xb)
            with HeadInstall(m, install, r_cols, own.z):
                same = m.forward(ids_b, mask, xb).logits[:, -1]
            err = float((same.float() - clean.float()).abs().max())
            print(f"    [check] install of the base's own values vs clean: max |dlogit| = {err:.2e}")
            assert err < 1e-2, "installing the base's own head values changed the logits"
            free = next(b for b in range(m.n_layers) if b not in install_blocks)
            with HeadInstall(m, install, r_cols, z_src), EdgeKnockout(m, [free], groups["text"], [], check=True):
                same = m.forward(ids_b, mask, xb, output_attentions=True).logits[:, -1]
            err = float((same.float() - inst.float()).abs().max())
            print(f"    [check] install + knockout with NO keys vs install: max |dlogit| = {err:.2e}")
            assert err < 1e-2, "an empty knockout changed the installed run's logits"
        for ko in kos:
            run(f"install+ko:{ko[0]}", ko=ko, z=z_src)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--attributes", nargs="+", default=None,
                    help=f"Default: every attribute of --entity (flags: {' '.join(ATTRIBUTES)}).")
    ap.add_argument("--n_pairs", type=int, default=64, help="Ordered (base, source) item pairs.")
    ap.add_argument("--knockouts", nargs="+", default=DEFAULT_KO, metavar="NAME=QUERIES:A-B[@GROUP][+...]",
                    help=f"QUERIES in {QUERY_GROUPS}; A-B is an inclusive 0-based block range; GROUP (the "
                         f"keys) in {KEY_GROUPS}, default image. '+' stacks parts into one knockout that "
                         f"removes the union of their edges.")
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
    ap.add_argument("--out_dir", default=None, help="Default: results/route_knockout/<entity>.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--install_heads", default=None, metavar="B.H,B.H,...",
                    help="Also install these heads' SOURCE o_proj inputs at the readout columns (arms "
                         "`install`, `install+ko:<name>`). 'common10' = the blocks 21-23 movers. A knockout "
                         "with readout/text queries in an installed block is refused, so pass --knockouts "
                         "(e.g. late=readout:24-27) with it.")
    ap.add_argument("--entity", default="flags", choices=["flags", *NON_FLAGS_ITEMS])
    ap.add_argument("--template", type=int, default=0,
                    help="Non-flags: which VADE prompt template per attribute (index by template_id).")
    ap.add_argument("--max_new_tokens", type=int, default=8,
                    help="Non-flags: length of the model's own greedy answers used as golds.")
    args = ap.parse_args()

    if args.entity == "flags":
        entity_dir, items, gt_sha = load_flags(args.vade_root)
        prompts, entity_attrs = None, list(ATTRIBUTES)
    else:
        entity_dir, items, gt_sha, prompts = load_entity(args.vade_root, args.entity, args.template)
        entity_attrs = list(prompts)
    args.attributes = args.attributes or entity_attrs
    assert set(args.attributes) <= set(entity_attrs), f"{args.entity} attributes are {entity_attrs}"
    args.out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "route_knockout", args.entity)
    kos = [parse_ko_parts(s) for s in args.knockouts]
    install = parse_heads(MOVERS if args.install_heads == "common10" else args.install_heads) \
        if args.install_heads else None
    clash = install_clashes(kos, install)
    assert not clash, (f"knockouts {clash} remove readout-query edges in blocks holding installed heads; "
                       f"the correction would be computed against A@V that the install replaced -- refused")
    obj = object_index(entity_dir) if any(g != "image" for _, parts in kos for _, _, g in parts) else None
    pairs = sample_pairs(items, args.n_pairs, args.seed)
    n_arms = 3 + 2 * len(kos) + 2 * len(args.q_layers) + ((1 + len(kos)) if install else 0)
    print(f"[route_knockout] {len(pairs)} pairs x {len(args.attributes)} attributes; img_layer={args.img_layer}; "
          f"q_layers={args.q_layers}; ~{n_arms} forwards per batch")
    for name, parts in kos:
        print(f"  ko:{name:<16} {describe_ko(parts)}")
    if obj is not None:
        print(f"  key groups: object = {len(obj[1])} of {obj[0]} image tokens, background = the other "
              f"{obj[0] - len(obj[1])}")
    if install:
        print(f"  install: {len(install)} heads {' '.join(f'{b}.{h}' for b, h in install)} at the readout columns")
    if prompts is not None:
        print(f"  {args.entity} template #{args.template} prefills: {({a: p for a, (_, p) in prompts.items()})}; "
              f"golds = the model's own greedy answers")
    if args.dry_run:
        print("Metadata valid (tokenization and divergence need the model's tokenizer).")
        return

    from methods.circuit_common import Model, chunks_by_prefix, pair_rows
    m = Model(args.model_id, args.device, attn_impl="eager", vade_root=args.vade_root)
    kos = [parse_ko_parts(s, m.n_layers) for s in args.knockouts]
    assert 0 < args.img_layer < m.n_layers and all(0 < L < m.n_layers for L in args.q_layers)
    assert all(0 <= b < m.n_layers and 0 <= h < m.n_heads for b, h in install or ()), "install head out of range"
    if prompts is None:
        rows, dropped = pair_rows(items, m.tokenizer, pairs, args.attributes)
        print(f"  {len(rows)} rows ({dropped} dropped: golds never diverge)")
        chunks = chunks_by_prefix(rows, args.batch_size)
    else:
        answers = own_answers(m, entity_dir, items, sorted({x for p in pairs for x in p}), args.attributes,
                              prompts, args.max_new_tokens, args.batch_size)
        rows, dropped = own_rows(items, pairs, args.attributes, answers,
                                 lambda t: m.tokenizer.decode(t, skip_special_tokens=True))
        print(f"  {len(rows)} rows (dropped: {dropped})")
        chunks = chunks_by_attribute(rows, args.batch_size)
    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    with open(os.path.join(args.out_dir, "rows.jsonl"), "w") as fh:
        for i, chunk in enumerate(chunks):
            ids_b, ids_s, mask, xb, xs = build_batch(m, entity_dir, items, chunk, prompts)
            recs = run_batch(m, args, kos, chunk, ids_b, ids_s, mask, xb, xs, check=(i == 0), obj=obj,
                             install=install)
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            records += recs
            print(f"  {len({r['row_index'] for r in records})}/{len(rows)} rows", flush=True)

    eligible = clean_eligible(records)
    config = {k: v for k, v in vars(args).items()
              if k != "dry_run" and not (k in ADDED_OPTS and v == ADDED_OPTS[k])}
    summary = {"config": config,
               "ground_truth_sha256": gt_sha, "n_rows": len(rows), "n_eligible": len(eligible),
               "all": summarize_arms(records), "eligible": summarize_arms(records, eligible),
               "by_attribute": {a: summarize_arms([r for r in records if r["attribute"] == a], eligible)
                                for a in args.attributes}}
    if prompts is not None:
        summary.update(templates=prompts, dropped=dropped)
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
