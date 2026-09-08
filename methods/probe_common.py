"""Shared row-selection + teacher-forced-forward helper for the read-only
interpretability probes (logit_lens.py, attention_maps.py). Both probes
ask the same question -- "given the model's OWN base-image forward pass
on a real VADE prompt, how does it arrive at its prediction" -- so unlike
DAS/DBM/PCA-SAE's intervene.py, there is no source image, no patching, no
training: this module exists to avoid writing the same row-dedup and
teacher-forced-readout bookkeeping twice.

Deliberately built on methods/common/entities.py's real question+prefill
machinery (build_batch/load_tuples), NOT sae.py's dummy-question
activations -- sae.py's activations are provably independent of the
question text (image tokens precede any question text in a causal
decoder), so they can't answer "how does it predict the answer given the
prompt." That requires the real question.
"""
import torch

from .common.entities import build_batch
from .common.targets import MAX_ANSWER_TOKENS, build_teacher_forced_extension

# Candidate English phrasings that name each flags attribute in its OWN question -- entity/prompt
# content, not a model concern (see adapters/qwen2_5_vl.py's find_last_phrase_token_col, which does
# the model-specific work of turning a matched phrase into a token column). VADE's 6 templates per
# attribute phrase things differently (see flags/prompt_templates.json), so each list tries several
# real wordings in order; the first one present in a given question's text wins. Longer/more specific
# phrases are listed before their substrings only where a shorter one could also mis-match a
# DIFFERENT attribute's wording (calling_code's "country code" vs. currency's unrelated phrasing) --
# order among calling_code's own synonyms doesn't matter, since find_last_phrase_token_col takes
# whichever phrase is actually found, not a longest-match. Not populated for brands/animals (unbuilt
# per CLAUDE.md, or their attribute names/wordings just haven't been added here yet) -- callers must
# treat a missing entry (attribute_mention_col returning None) as "skip this analysis", not an error.
ATTRIBUTE_KEYWORDS = {
    "capital": ["capital city", "capital", "seat of government"],  # capital_prefill_v6 says
    # "Which city is the seat of government..." -- no "capital" anywhere (caught by actually running
    # find_last_phrase_token_col against every real template before trusting this list, not by
    # inspection: capital_prefill_v6 crashed the very first validation pass here).
    "currency": ["currency code", "currency"],
    "language": ["official language", "languages", "language"],
    "calling_code": ["international calling code", "international dialling code", "international dialing code",
                      "country calling code", "phone country code", "country code", "calling code",
                      "dialling code", "dialing code"],
}


def attribute_mention_col(adapter, processor, attribute, question, prefill, n_image_tokens):
    """-> the token-column index of the word/phrase that actually NAMES `attribute` in `question`
    (e.g. the " language" token in "What is the official language..."), or None if this attribute
    has no entry in ATTRIBUTE_KEYWORDS, or none of its registered phrasings occur in this particular
    template's wording (in which case: add the missing phrasing to ATTRIBUTE_KEYWORDS above rather
    than assuming the probe silently degrades gracefully -- see find_last_phrase_token_col for why a
    genuine mismatch against a REGISTERED attribute still asserts loudly instead)."""
    phrases = ATTRIBUTE_KEYWORDS.get(attribute)
    if not phrases:
        return None
    return adapter.find_last_phrase_token_col(processor, question, prefill, n_image_tokens, phrases)


def load_probe_rows(entity_assets, attribute, split, tuples_dir=None, limit=None, one_per_image=False,
                     templates_per_image=1):
    """One row per unique (base image, template_id) whose OWN question is
    being asked (queried == target_attribute -- VADE's tuples also contain
    "iso" rows asking about some OTHER attribute, e.g. to test whether an
    intervention leaves capital alone while patching language; those are
    irrelevant here, since there's no intervention and we only care about
    the model's unpatched answer to the attribute's own question).

    A tuple's base image is reused across many rows (paired with different
    sources / rules), all sharing the identical prompt and gold answer --
    deduping keeps the probe's per-row cost proportional to unique
    (image, template) prompts actually run through the model, not the
    combinatorial tuple count. `source`/`rule` on the surviving rows are
    unused leftovers from the tuple schema (build_batch still populates
    source_extra/source_gold_* for them; the probes below simply never
    read those fields).

    one_per_image: tuples are sorted row_index-first, and a VADE entity's
    tuple generator emits every template for one base image before moving
    to the next -- so plain (base, template_id) dedup plus a small `limit`
    silently returns N phrasings of the SAME image rather than N different
    images (verified against flags/language: row_index 0-4 are all
    base=AO). Pass one_per_image=True so `limit` counts DISTINCT images
    rather than raw rows -- what an ad hoc probe over "a few rows" almost
    always wants (see attention_maps.py).

    templates_per_image: only consulted when one_per_image=True. Keeps up
    to this many distinct template_ids (i.e. differently-worded questions)
    per selected image instead of just the first one seen -- "N images,
    each asked K ways" rather than "N images, one question each". `limit`
    still caps the number of DISTINCT IMAGES selected, not the total row
    count, so the returned list can be up to limit * templates_per_image
    rows long (fewer if an image has less than templates_per_image
    template variants in this split).
    """
    from .common.entities import load_tuples
    rows = load_tuples(entity_assets, attribute, split, tuples_dir=tuples_dir)
    for r in rows:
        r.setdefault("target_attribute", attribute)
    rows = [r for r in rows if r["queried"] == r["target_attribute"]]

    if one_per_image:
        per_image_count = {}
        deduped = []
        for r in sorted(rows, key=lambda r: r["row_index"]):
            base = r["base"]
            count = per_image_count.get(base, 0)
            if count == 0 and limit is not None and len(per_image_count) >= limit:
                continue  # already have `limit` distinct images -- a new image doesn't get in
            if count >= templates_per_image:
                continue  # this image already contributed its quota of question phrasings
            per_image_count[base] = count + 1
            deduped.append(r)
        return deduped

    seen = set()
    deduped = []
    for r in sorted(rows, key=lambda r: r["row_index"]):
        key = (r["base"], r["template_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    if limit:
        deduped = deduped[:limit]
    return deduped


def build_probe_batch(rows, entity_assets, adapter, model, processor, positions, batch_cache=None):
    """Wraps common/entities.py's build_batch (which builds BOTH base and
    source sides -- there is no base-only variant) and appends the
    teacher-forced answer-token extension to the BASE side only. Returns
    the original batch dict plus:
      ext_input_ids, ext_attention_mask: [B, L+K-1] -- base sequence with
        the true preceding gold tokens appended (see targets.py's
        build_teacher_forced_extension); forward this, not base_input_ids.
      readout_start_col: the column index of the first of the K read-out
        positions in ext_input_ids (== ext_input_ids.shape[1] - K) --
        hidden_states/attentions/logits at columns
        [readout_start_col : readout_start_col+K] are what the probes read.
    """
    batch = build_batch(rows, entity_assets, adapter, model, processor, positions, batch_cache=batch_cache)
    ext_ids, ext_mask = build_teacher_forced_extension(
        batch["base_input_ids"], batch["attention_mask"], batch["base_gold_toks"], batch["base_gold_len"])
    batch["ext_input_ids"] = ext_ids
    batch["ext_attention_mask"] = ext_mask
    batch["readout_start_col"] = ext_ids.shape[1] - MAX_ANSWER_TOKENS
    return batch


@torch.no_grad()
def teacher_forced_forward(model, batch, **fwd_kwargs):
    """One no-grad forward pass over batch's extended (teacher-forced) base
    sequence. **fwd_kwargs is spread into the model call unchanged (e.g.
    output_hidden_states=True and/or output_attentions=True) -- this
    function knows nothing about which probe is calling it."""
    from .common.hooks import extra_to_device
    extra_dev = extra_to_device(batch["base_extra"], model.device, model.dtype)
    return model(
        input_ids=batch["ext_input_ids"].to(model.device),
        attention_mask=batch["ext_attention_mask"].to(model.device),
        **extra_dev, **fwd_kwargs,
    )


def token_groups_for_row(seq_len, attention_mask_row, image_token_id, input_ids_row, object_cols, readout_start_col):
    """Returns {group_name: LongTensor of column indices} partitioning
    every REAL (non-left-pad) column of one row's extended sequence into
    exactly one of:
      object       -- this attribute's own token positions (--positions,
                       e.g. flag_ring1), the same columns intervene.py
                       would patch
      image_other  -- image-placeholder tokens outside the object's set
                       (background/canvas patches)
      text         -- everything else before the answer-extension: the
                       question, the image's own start marker if any
                       decoder-visible text tokens sit there, and the
                       prefill
      answer_ctx   -- the appended true-preceding-answer-token columns
                       (only real for multi-token gold answers -- see
                       build_teacher_forced_extension; empty for a
                       single-token gold answer)
    Left-padding columns (attention_mask==0) carry exactly 0 softmax
    attention weight by construction and are dropped from every group
    rather than being counted as "text" -- keeps each group's mass a
    faithful fraction of the K read-out position's REAL attention, and
    group masses across the four groups sum to 1.0 (up to fp error).
    """
    object_set = set(object_cols.tolist() if torch.is_tensor(object_cols) else object_cols)
    ids = input_ids_row.tolist()
    mask = attention_mask_row.tolist()
    groups = {"object": [], "image_other": [], "text": [], "answer_ctx": []}
    for col in range(seq_len):
        if mask[col] == 0:
            continue  # left-pad -- 0 attention weight, excluded from every group
        if col >= readout_start_col:
            groups["answer_ctx"].append(col)
        elif col in object_set:
            groups["object"].append(col)
        elif ids[col] == image_token_id:
            groups["image_other"].append(col)
        else:
            groups["text"].append(col)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in groups.items()}
