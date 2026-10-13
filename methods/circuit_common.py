"""Shared plumbing for the Tier-1 circuit experiments.

    image_enrichment.py      E1  are attributes already present at the image tokens?
    circuit_compose.py       E4  do router + mover heads compose into a new answer?
    route_knockout.py        E5  direct (image->last) vs text-mediated (image->question) route
    text_entity_contrast.py  E6  the same circuit probes with the entity NAMED instead of shown

All four use the CONTROLLED prompts of attribute_switch_sweep.py (one instruction,
one question shape, the shared prefill 'Answer:') rather than VADE's six templates,
for two reasons. The prefill does not fix the answer TYPE, so an intervention can
move the answer to a different attribute (E4 needs that). And every flag and every
attribute tokenizes to the same length, so image-prompt batches need no padding,
which the attention-edge knockout (E5) requires.

THE READOUT (E5, E6). Every pair row is (base item C, source item B, attribute X).
Golds are tokenized under the prefill exactly as targets.py does. The row is read at
its DIVERGENCE POINT: the shared gold prefix is appended to the prompt and the next-
token logits are compared on the two tokens where C's and B's answers first differ.
Calling codes are why this is necessary rather than a refinement -- every code
tokenizes as ['Ġ', digit, ...], so a first-token readout would score 'Ġ' as a
transfer on 100% of rows. The appended prefix is the SAME tokens in base and source,
so the readout columns line up from the end of the sequence in both.

  src_hit   argmax == the source's divergent token   (transfer)
  base_hit  argmax == the base's divergent token     (no effect)
  ld        logit[src] - logit[base]                 (continuous, no threshold)

Rows whose golds never diverge (one is a prefix of the other, or identical labels:
two countries with the same language) carry no readout and are dropped up front.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(ROOT, "..", "VADE"))
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

from methods.attribute_switch_sweep import ATTRIBUTES, FIELDS, INSTRUCTION, PREFILL, question  # noqa: E402

# R8/R10's image-side entity movers (blocks 21-23) and R1's question-side routers (19-21).
MOVERS = "21.1,21.5,22.13,22.15,22.17,22.19,23.3,23.4,23.6,23.17"
ROUTER_TRACE = os.path.join(ROOT, "logs", "attr_head_trace", "flags",
                            "attr_head_trace_patch15_blocks15-22_all_text_n84.json")

# The text-entity condition (E6). Same instruction with 'shown' -> 'named', and the name
# introduced BEFORE the question, mirroring the image prompt where the image precedes it.
TEXT_INSTRUCTION = INSTRUCTION.replace("For the country shown", "For the country named")
assert TEXT_INSTRUCTION != INSTRUCTION, "INSTRUCTION wording changed; update TEXT_INSTRUCTION"


def text_question(attribute, name):
    return TEXT_INSTRUCTION + f"The country is {name}. Report the {FIELDS[attribute]} of this country."


def parse_heads(spec):
    from methods.head_swap_vade import parse_heads as _parse
    return _parse(spec)


def router_heads(path, k, exclude=()):
    """Top-k of attr_head_trace's phase-1 ranking, skipping heads in `exclude`.

    R1's top-8 contains 21.5, which is also one of the ten movers; in a two-donor
    install one head cannot carry both donors' values, so it is given to the movers
    and the routers take the next-ranked head instead."""
    ranked = [tuple(h) for h in json.load(open(path))["phase1_ranked"]]
    out = [h for h in ranked if h not in set(exclude)][:k]
    assert len(out) == k, f"{path} ranks only {len(out)} heads outside the exclusion set"
    return sorted(out)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_flags(vade_root):
    """-> (entity_dir, items, sha256 of ground_truth.json). Flags only: the controlled
    prompts' instruction defines exactly the four flags attributes."""
    import hashlib
    entity_dir = os.path.join(vade_root, "data", "flags")
    raw = open(os.path.join(entity_dir, "ground_truth.json"), "rb").read()
    items = json.loads(raw)["countries"]
    for key, it in items.items():
        assert os.path.isfile(os.path.join(entity_dir, it["image"])), f"missing image for {key}"
    return entity_dir, items, hashlib.sha256(raw).hexdigest()


def sample_pairs(items, n_pairs, seed):
    """Ordered (base, source) item pairs, deterministic in (items, seed)."""
    names = sorted(items)
    pairs = [(a, b) for a in names for b in names if a != b]
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs]


def gold_ids(tokenizer, label):
    from methods.common.targets import derive_gold_token_ids
    return derive_gold_token_ids(tokenizer, PREFILL, str(label))


def first_content_token(tokenizer, ids):
    """First token that is not pure whitespace ('Ġ' leads every calling code)."""
    for t in ids:
        if tokenizer.decode([t]).strip():
            return t
    return None


def divergence(a, b):
    """-> (shared_prefix, a_tok, b_tok) at the first index where token lists differ,
    or None when one is a prefix of the other (no position separates them)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return list(a[:i]), x, y
    return None


def pair_rows(items, tokenizer, pairs, attributes):
    """-> (rows, dropped). One row per (pair, attribute) whose golds diverge."""
    rows, dropped = [], 0
    for base, source in pairs:
        for attr in attributes:
            gb, gs = gold_ids(tokenizer, items[base][attr]), gold_ids(tokenizer, items[source][attr])
            div = divergence(gb, gs)
            if div is None:
                dropped += 1
                continue
            prefix, bt, st = div
            rows.append({"row_index": len(rows), "base": base, "source": source, "attribute": attr,
                         "base_label": items[base][attr], "source_label": items[source][attr],
                         "prefix": prefix, "base_tok": bt, "src_tok": st})
    return rows, dropped


def chunks_by_prefix(rows, batch_size):
    """Batch rows sharing a divergence-prefix LENGTH, so appended sequences have equal
    length and image batches stay unpadded."""
    groups = {}
    for r in rows:
        groups.setdefault(len(r["prefix"]), []).append(r)
    for k in sorted(groups):
        g = groups[k]
        for i in range(0, len(g), batch_size):
            yield g[i:i + batch_size]


def left_pack(seqs, pad_id):
    import torch
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        s = torch.as_tensor(s, dtype=torch.long)
        ids[i, n - len(s):] = s
        mask[i, n - len(s):] = 1
    return ids, mask


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class Model:
    """Adapter + model + processor, with the two prompt builders and one forward."""

    def __init__(self, model_id=MODEL_ID, device="cuda:0", attn_impl="sdpa", vade_root=DEFAULT_VADE_ROOT):
        import torch
        from methods.adapters.registry import get_adapter
        self.adapter = get_adapter(model_id)
        self.model, self.processor = self.adapter.load(
            device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            attn_implementation=attn_impl)
        self.model.eval().requires_grad_(False)
        self._init_common(vade_root)

    @classmethod
    def wrap(cls, adapter, model, processor, vade_root=DEFAULT_VADE_ROOT):
        """For tests: wrap an already-built (e.g. tiny random) model."""
        self = cls.__new__(cls)
        self.adapter, self.model, self.processor = adapter, model, processor
        self._init_common(vade_root)
        return self

    def _init_common(self, vade_root):
        self.vade_root = vade_root
        self.layers = self.adapter.get_decoder_layers(self.model)
        self.n_layers = len(self.layers)
        self.n_heads = self.adapter.n_attention_heads(self.model)
        self.hidden = self.adapter.hidden_size(self.model)
        self.head_dim = self.hidden // self.n_heads
        self.image_id = self.adapter.image_token_id(self.model, self.processor)
        tok = self.processor.tokenizer if self.processor is not None else None
        self.pad_id = (tok.pad_token_id or tok.eos_token_id or 0) if tok is not None else 0
        self._images = {}

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    # -- prompts ------------------------------------------------------------

    def image_prompt(self, entity_dir, item, attribute):
        """-> (input_ids [T], extra) for the controlled image prompt."""
        from PIL import Image
        path = os.path.join(entity_dir, item["image"])
        if path not in self._images:
            with Image.open(path) as im:
                self._images[path] = im.convert("RGB")
        x = self.adapter.build_inputs(self.processor, self._images[path], question(attribute), PREFILL)
        return x["input_ids"], x["extra"]

    def text_prompt(self, name, attribute):
        """-> (input_ids [T], name_cols) for the text-entity prompt, no image.

        name_cols are the token columns covering the country name, found by character
        offsets in the rendered chat text, so they are exact whatever the tokenizer
        does at the name's boundaries."""
        import torch
        q = text_question(attribute, name)
        messages = [{"role": "user", "content": [{"type": "text", "text": q}]},
                    {"role": "assistant", "content": [{"type": "text", "text": PREFILL}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                                  continue_final_message=True)
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        start = text.find(f"The country is {name}.")
        assert start >= 0, "name sentence not found verbatim in the rendered prompt"
        lo = start + len("The country is ")
        hi = lo + len(name)
        cols = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < hi and e > lo]
        assert cols, f"no token overlaps the name {name!r}"
        return torch.tensor(enc["input_ids"], dtype=torch.long), cols

    # -- forward ------------------------------------------------------------

    def forward(self, ids, mask, extra, hooks=(), output_hidden_states=False, output_attentions=False,
                keep=1):
        """One no-grad forward with `hooks` = [(site, layer_idx, fn)] registered through
        common/sites.py's own registration (the tensor every other experiment patches).
        Returns the HF output; logits cover the last `keep` columns."""
        import torch
        from methods.attribute_switch_sweep import multimodal_position_kwargs
        from methods.common.hooks import extra_to_device
        handles = []
        try:
            for site, layer_idx, fn in hooks:
                handles.extend(site.register(self.adapter, self.model, self.layers, layer_idx, fn))
            ids_d = ids.to(self.model.device)
            with torch.no_grad():
                return self.model(input_ids=ids_d, attention_mask=mask.to(self.model.device),
                                  **extra_to_device(extra or {}, self.model.device, self.model.dtype),
                                  **multimodal_position_kwargs(self.model, ids_d),
                                  output_hidden_states=output_hidden_states,
                                  output_attentions=output_attentions, use_cache=False,
                                  logits_to_keep=keep)
        finally:
            for h in handles:
                h.remove()

    def generate(self, ids, mask, extra, max_new_tokens):
        import torch
        from methods.common.hooks import extra_to_device
        with torch.no_grad():
            out = self.model.generate(input_ids=ids.to(self.model.device), attention_mask=mask.to(self.model.device),
                                      **extra_to_device(extra or {}, self.model.device, self.model.dtype),
                                      max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.pad_id)
        return out[:, ids.shape[1]:].cpu()

    def decode(self, toks):
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in toks]


def cat_extra(extras):
    import torch
    if not extras or not extras[0]:
        return {}
    return {k: torch.cat([e[k] for e in extras]) for k in extras[0]}


# ---------------------------------------------------------------------------
# Patch primitives (all through common/hooks + common/sites)
# ---------------------------------------------------------------------------

def residual_patch(layer, positions, values):
    """-> (RESIDUAL_SITE, layer, fn) writing `values` [B, n_pos, H] at `positions`
    [B, n_pos] of the residual stream after block layer-1 (0 = embeddings)."""
    from methods.common.hooks import make_cache_aware_patch_hook
    from methods.common.sites import RESIDUAL_SITE
    return (RESIDUAL_SITE, layer, make_cache_aware_patch_hook(positions, lambda bv: values.to(bv.dtype)))


def gather_cols(t, positions):
    """t [B, T, H], positions [B, n] -> [B, n, H]."""
    import torch
    return torch.stack([t[i, positions[i]] for i in range(positions.shape[0])])


def readout_metrics(logits, rows):
    """logits [B, V] at the divergence column -> per-row dicts (see module docstring)."""
    import torch
    src = torch.tensor([r["src_tok"] for r in rows], device=logits.device)
    base = torch.tensor([r["base_tok"] for r in rows], device=logits.device)
    pick = logits.argmax(-1)
    ar = torch.arange(len(rows), device=logits.device)
    ld = (logits[ar, src].float() - logits[ar, base].float()).tolist()
    return [{"src_hit": bool(p == s), "base_hit": bool(p == b), "ld": d}
            for p, s, b, d in zip(pick.tolist(), src.tolist(), base.tolist(), ld)]


def wilson(k, n, z=1.96):
    """95% Wilson interval for k/n. Treats rows as independent; rows sharing a country
    are not, so read it as a lower bound on the real uncertainty."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def summarize_arms(records, eligible=None):
    """records: [{row_index, arm, src_hit, base_hit, ld}] -> {arm: rates}. `eligible`
    restricts to rows where both clean runs are correct (see `clean_eligible`)."""
    out = {}
    for arm in dict.fromkeys(r["arm"] for r in records):
        rr = [r for r in records if r["arm"] == arm and (eligible is None or r["row_index"] in eligible)]
        if not rr:
            continue
        n = len(rr)
        hits = sum(r["src_hit"] for r in rr)
        out[arm] = {"n": n, "transfer": hits / n, "transfer_ci95": wilson(hits, n),
                    "base_kept": sum(r["base_hit"] for r in rr) / n,
                    "other": sum(not (r["src_hit"] or r["base_hit"]) for r in rr) / n,
                    "mean_ld": sum(r["ld"] for r in rr) / n}
    return out


def clean_eligible(records, clean_arm="clean", source_arm="source_clean"):
    """Rows where the clean base run prefers the base token AND the source's own run
    prefers the source token -- the population on which transfer is meaningful."""
    base_ok = {r["row_index"] for r in records if r["arm"] == clean_arm and r["base_hit"]}
    src_ok = {r["row_index"] for r in records if r["arm"] == source_arm and r["src_hit"]}
    return base_ok & src_ok


def normalized_ld(summary, arm, clean_arm="clean", source_arm="source_clean"):
    """(LD_arm - LD_clean) / (LD_source - LD_clean): 0 = no effect, 1 = full transfer."""
    c, s, a = (summary.get(x, {}).get("mean_ld") for x in (clean_arm, source_arm, arm))
    if None in (c, s, a) or abs(s - c) < 1e-9:
        return None
    return (a - c) / (s - c)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def print_table(summary, title, normalize=True):
    print(f"\n{title}")
    print(f"  {'arm':<44} {'n':>5} {'transfer':>9} {'95% CI':>13} {'base':>7} {'other':>7} {'meanLD':>8}"
          + (f" {'normLD':>7}" if normalize else ""))
    for arm, s in summary.items():
        nl = normalized_ld(summary, arm) if normalize else None
        lo, hi = s["transfer_ci95"]
        print(f"  {arm:<44} {s['n']:>5} {s['transfer']:>8.1%} {f'{lo:.0%}-{hi:.0%}':>13} "
              f"{s['base_kept']:>6.1%} {s['other']:>6.1%} "
              f"{s['mean_ld']:>8.2f}" + (f" {nl:>7.2f}" if nl is not None else ("        " if normalize else "")))
