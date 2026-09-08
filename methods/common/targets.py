"""Gold-token derivation and teacher-forced-training utilities. Pure
tokenizer/token-id manipulation -- no model architecture dependency at all
(works against any HF tokenizer), ported verbatim from flag-benchmark's
experiments/image-to-image/scripts/core/collate.py.
"""
import torch

MAX_ANSWER_TOKENS = 3  # up to 3 tokens resolves every real first-token collision seen across all four attributes


def derive_gold_token_ids(tokenizer, prefill, target_text):
    """BPE-boundary-correct target tokenization: tokenize prefill alone and
    prefill+target together, take the suffix -- avoids mismatches from
    tokenizing target_text in isolation (e.g. leading-space merges)."""
    sep = "" if prefill.endswith("+") else " "
    ids_prefill = tokenizer(prefill, add_special_tokens=False)["input_ids"]
    ids_with_target = tokenizer(prefill + sep + target_text, add_special_tokens=False)["input_ids"]
    assert ids_with_target[: len(ids_prefill)] == ids_prefill, \
        "prefill retokenized differently once target text follows it"
    return ids_with_target[len(ids_prefill):]


def pad_gold_toks(tok_lists, pad_id, k=MAX_ANSWER_TOKENS):
    """tok_lists: list of variable-length int lists (one per row, from
    derive_gold_token_ids). Returns ([B, k] token ids, [B] real lengths,
    capped at k) -- positions at or beyond a row's real length are filled
    with pad_id and must never be read (their content is meaningless)."""
    B = len(tok_lists)
    out = torch.full((B, k), pad_id, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    for i, ids in enumerate(tok_lists):
        n = min(len(ids), k)
        out[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
        lens[i] = n
    return out, lens


def target_gold_toks_and_len(batch):
    """Per-row supervision target sequence (up to MAX_ANSWER_TOKENS): source's
    gold tokens if rule==match_source (cause pool -- base should flip to
    source), else base's own gold tokens (iso pool -- base should stay put)."""
    is_source = torch.tensor([r["rule"] == "match_source" for r in batch["rows"]])
    toks = torch.where(is_source.unsqueeze(1), batch["source_gold_toks"], batch["base_gold_toks"])
    lens = torch.where(is_source, batch["source_gold_len"], batch["base_gold_len"])
    return toks, lens


def build_teacher_forced_extension(input_ids, attention_mask, target_toks, target_len):
    """Appends target_toks[:, :MAX_ANSWER_TOKENS-1] (the TRUE preceding
    tokens, never the model's own guess) after input_ids' existing content.
    input_ids/attention_mask must already be left-padded to a common length
    with no padding after the real content, so every row's real content
    ends at the same last column -- appended positions therefore land at
    the same absolute column across the whole batch too, with no per-row
    offset bookkeeping needed.

    Returns (extended_input_ids, extended_attention_mask). The K logits
    positions to read afterward are simply the last MAX_ANSWER_TOKENS
    columns of a forward pass over the result (out.logits[:, -MAX_ANSWER_TOKENS:, :]),
    aligned 1:1 with target_toks[:, :MAX_ANSWER_TOKENS] by construction --
    position j predicts target_toks[:, j], no shift arithmetic required.
    """
    ctx_len = MAX_ANSWER_TOKENS - 1
    ctx_toks = target_toks[:, :ctx_len].clone()
    ctx_mask = torch.zeros_like(ctx_toks)
    for j in range(ctx_len):
        ctx_mask[:, j] = (target_len > j).long()
    extended_ids = torch.cat([input_ids, ctx_toks], dim=1)
    extended_mask = torch.cat([attention_mask, ctx_mask], dim=1)
    return extended_ids, extended_mask


def gold_labels_from_lens(target_toks, target_len, k=MAX_ANSWER_TOKENS):
    """[B, k] cross-entropy labels: target_toks where j < target_len[i],
    else -100 (ignore_index) -- padding positions never contribute to loss."""
    labels = target_toks.clone()
    for j in range(k):
        labels[:, j] = torch.where(target_len > j, target_toks[:, j], torch.tensor(-100))
    return labels


def exact_match(pred_toks, gold_toks, gold_len, k=MAX_ANSWER_TOKENS):
    """pred_toks/gold_toks: [B, k] token ids. Returns [B] bool: True iff
    every valid position (j < gold_len[i]) matches -- a full up-to-k-token
    exact match, not just the first token."""
    ok = torch.ones(pred_toks.shape[0], dtype=torch.bool)
    for j in range(k):
        valid = gold_len > j
        ok = ok & (~valid | (pred_toks[:, j] == gold_toks[:, j]))
    return ok
