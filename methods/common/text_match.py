"""Text-level answer matching, the way VADE's eval/score.py (and its pruning) does it.

WHY THIS EXISTS. The token-level scorer (targets.py's exact_match over
derive_gold_token_ids(label)) tokenizes the gold label AS STORED. That is correct
for flags, whose labels are cased like the model writes them ('Buenos Aires'), and
wrong for every entity whose labels are stored lowercase. Measured on
brands/hq_country: gold ' united states' = [' united', ' states'], model says
' the United States.' -- so the UNHOOKED baseline scored matches_base=0.0% on
pruned rows that the model answers correctly, and every ceiling cell read
other=100%. VADE's scorer normalizes (casefold, whitespace, diacritics) and
matches the label as a whole word anywhere in the text, which is what pruning
used; this module is that matcher, copied (not imported -- VADE is a sibling
checkout, not a package) so the two stay in agreement.

A new file rather than an edit to targets.py, which is a verbatim copy of VADE's.
"""
import re
import unicodedata


def normalize(s):
    """Copied from VADE/eval/score.py: whitespace/case normalize, strip diacritics."""
    folded = re.sub(r"\s+", " ", s.strip()).casefold()
    return "".join(c for c in unicodedata.normalize("NFKD", folded) if not unicodedata.combining(c))


def contains_label(generated_text, label):
    """Copied from VADE/eval/score.py: whole-word, case-insensitive match
    ('Iran' does not match inside 'Ireland')."""
    if not generated_text or not label:
        return False
    return re.search(r"\b" + re.escape(normalize(label)) + r"\b", normalize(generated_text)) is not None


def decode_answers(tokenizer, gen_toks):
    """[B, T] generated ids -> list of B strings, special tokens dropped."""
    return tokenizer.batch_decode(gen_toks, skip_special_tokens=True)


def labels_of(batch, role):
    """Per-row gold label strings for role 'source' or 'base'. A batch may carry
    an explicit `<role>_labels` list (used when a caller ROLLS the golds, as
    head_trace's shuffled-donor control does); otherwise the rows' own labels."""
    explicit = batch.get(f"{role}_labels")
    return list(explicit) if explicit is not None else [r[f"{role}_label"] for r in batch["rows"]]


def text_match_rates(tokenizer, gen_toks, batch):
    """-> (matches_source_rate, matches_base_rate), VADE-style. Same return shape
    as ceiling_sweep.score_generation, so it can stand in for it."""
    texts = decode_answers(tokenizer, gen_toks)
    src, base = labels_of(batch, "source"), labels_of(batch, "base")
    ms = sum(contains_label(t, l) for t, l in zip(texts, src)) / max(len(texts), 1)
    mb = sum(contains_label(t, l) for t, l in zip(texts, base)) / max(len(texts), 1)
    return ms, mb
