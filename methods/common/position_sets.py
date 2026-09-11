"""EXTENDED position sets, built by post-processing common/entities.py's own
build_batch rather than by extending it.

NOT copied from the sibling VADE repo. Deliberately a WRAPPER: entities.py
stays a byte-identical copy of VADE's and keeps doing all the real work
(tokenization, left-padding, gold tokens, BuildBatchCache). Everything here
is a re-selection of columns from batches build_batch already produced.

WHY THAT WORKS
  1. `full_image` returns positions in FLAT-INDEX ORDER -- column j is image
     token j -- so ANY subset of the image grid is a column selection.
  2. Batches are LEFT-padded with no padding after the real content, so every
     row's real content ends at the same absolute column; "k tokens back from
     the end" needs no template parsing.

HARD CONSTRAINT: batch["positions"] is a rectangular [B, n_pos] tensor, so a
set must yield the same NUMBER of positions for every row. Fixed windows and
grid subsets are fine; "all question tokens" is not (the six flags templates
have different question lengths).

SPECS
  <name>          anything entities.py knows (flag_only/flag_ring1/full_image/
                  last_token), passed straight through.
  ~<name>         complement of a named image set within the image span.
  ring:K          the K-th Chebyshev dilation band around the object's token
                  bbox. K=0 is the object itself; K=1 is the band touching it
                  (flag_only + ring:1 == flag_ring1). Add @<name> to pick the
                  base set explicitly (default: the entity's smallest one).
  side:left|right|above|below|beside
                  directional background relative to the object's bbox.
                  left/right span only the object's OWN rows ("beside" is
                  their union); above/below span the full grid width.
  seq:before|after|between
                  the background split by SEQUENCE order, which under causal
                  attention is a causal split -- see the note below.
  tok:-K[:N]      N text tokens ending K back from the prompt end; tok:-1 is
                  the final token (== last_token).
  phrase:attribute        the token naming the QUERIED attribute in the
                  question (" language" in "...the official language..."),
                  via probe_common.ATTRIBUTE_KEYWORDS + the adapter's
                  find_last_phrase_token_col.
  phrase:<lit>[,<alt>...] the token ending an arbitrary literal phrase, tried
                  in order -- e.g. phrase:country, phrase:image,flag (the
                  alternatives matter: flags' templates say "in this image"
                  in v1/v3/v4 but "shown in this flag" in v2).
  pre_image[:N]   N tokens immediately BEFORE the image span.
  vision_end[:N]  N tokens immediately AFTER the image span.
  A+B+C           UNION of specs. Sub-specs may mix image and text sets --
                  each is resolved against whichever underlying build it
                  needs and the columns are concatenated, deduped and sorted.
                  So `flag_ring1+~flag_ring1` reconstructs full_image exactly,
                  which is a free self-check of this module.

WHY seq:before IS A GUARANTEED NULL, and why that is worth running.
Every image in a VADE entity shares one fixed canvas render -- object_location
.json says so explicitly ("only the flag's pixel content varies between
images, never its position or size"). So the BACKGROUND pixels are identical
across all 84 flags. Under causal attention an image token can only attend to
tokens at or before its own position, so a background token that comes BEFORE
the object in raster order sees identical pixels and identical context in the
base and the source: its activation is bit-identical between them at every
layer, and patching it is provably a no-op. `seq:before` therefore MUST score
0% cause at every layer and every site. Nothing else in the sweep is a
guaranteed null, which makes it the control that licenses reading the other
zeros as real nulls rather than measurement failure. `seq:after` is the
interesting half: those tokens differ between base and source ONLY through
attention to the object, so their ceiling measures how much of the object's
information has leaked into the background by that layer.

Text-relative sets inherit is_last_token=True from the underlying build,
which is correct and load-bearing: their columns depend on the prompt's
length, so the per-entity source cache (keyed by image alone) is invalid for
them. See common/source_cache.py.
"""
import torch

_TEXT_HEADS = ("tok", "pre_image", "vision_end", "phrase")
_GRID_HEADS = ("ring", "side", "seq")
# Which underlying build a spec needs is decided by what it is ANCHORED ON, not by whether it is
# "text" or "grid". Only tok: is anchored on the end of the prompt; pre_image/vision_end/phrase all
# need the IMAGE SPAN's own columns to offset from, so they build with full_image like the grid
# specs do. Getting this wrong is silent and nasty: an earlier version routed pre_image through a
# last_token build, so it read the final prompt column as if it were the image start and resolved to
# a PREFILL token while still reporting a perfectly plausible 0%.
_END_ANCHORED = ("tok",)


def split_union(spec):
    """'A+B' -> ['A', 'B']. Single specs come back as a one-element list."""
    return [p.strip() for p in spec.split("+") if p.strip()]


def is_extended(spec):
    """True if `spec` needs this module rather than entities.py alone."""
    return any(_is_extended_part(p) for p in split_union(spec))


def _is_extended_part(p):
    return p.startswith("~") or p.split(":")[0] in _TEXT_HEADS + _GRID_HEADS


def _head(p):
    return p.split(":")[0].split("@")[0]


def underlying_positions_name(part):
    """Which set entities.py should build before we re-select from it."""
    if not _is_extended_part(part):
        return part
    return "last_token" if _head(part) in _END_ANCHORED else "full_image"


def path_safe(spec):
    """Directory-name-safe rendering. The spec is recorded verbatim in JSON."""
    return spec.replace("~", "not-").replace(":", "_").replace("@", "-at-").replace("+", "_plus_")


# ---- image-grid geometry --------------------------------------------------

def _grid(entity_assets):
    g = entity_assets.object_location["vlm_token_grid"]
    return g["grid_rows"], g["grid_cols"], g["n_tokens"]


def _object_bbox(entity_assets, base_name=None):
    """-> (r0, r1, c0, c1) of a named object set's token bbox. Defaults to the
    entity's SMALLEST object set (flag_only for flags, logo_only for brands),
    i.e. the object itself rather than one of its dilations."""
    available = entity_assets.object_location["object_token_indices"]
    if base_name is None:
        base_name = min(available, key=lambda k: len(available[k]["flat"]))
    assert base_name in available, f"unknown object set {base_name!r}; available: {list(available)}"
    entry = available[base_name]
    rows, cols = entry.get("rows"), entry.get("cols")
    assert rows and cols, (
        f"object set {base_name!r} has no rows/cols in object_location.json -- grid-geometry specs "
        f"(ring:/side:/seq:) need them")
    return min(rows), max(rows), min(cols), max(cols), base_name


def _chebyshev(r, c, r0, r1, c0, c1):
    """0 inside the bbox, else the ring index around it."""
    return max(0, r0 - r, r - r1, c0 - c, c - c1)


def grid_indices(part, entity_assets):
    """-> sorted list of image flat indices selected by one grid/complement spec."""
    n_rows, n_cols, n_tokens = _grid(entity_assets)
    available = entity_assets.object_location["object_token_indices"]

    if part.startswith("~"):
        excluded = set(available[part[1:]]["flat"])
        assert part[1:] in available, f"unknown object set {part[1:]!r}"
        return [j for j in range(n_tokens) if j not in excluded]

    head = _head(part)
    base_name = part.split("@")[1] if "@" in part else None
    r0, r1, c0, c1, base_name = _object_bbox(entity_assets, base_name)
    body = part.split("@")[0].split(":", 1)[1] if ":" in part.split("@")[0] else ""

    def cells(pred):
        return [r * n_cols + c for r in range(n_rows) for c in range(n_cols) if pred(r, c)]

    if head == "ring":
        k = int(body)
        assert k >= 0, f"{part!r}: ring index must be >= 0"
        out = cells(lambda r, c: _chebyshev(r, c, r0, r1, c0, c1) == k)
        assert out, (f"{part!r} selects no tokens -- the object bbox is at rows {r0}-{r1}, cols {c0}-{c1} "
                      f"on a {n_rows}x{n_cols} grid, so the largest ring is "
                      f"{max(_chebyshev(r, c, r0, r1, c0, c1) for r in range(n_rows) for c in range(n_cols))}")
        return out

    if head == "side":
        preds = {
            "left":   lambda r, c: c < c0 and r0 <= r <= r1,
            "right":  lambda r, c: c > c1 and r0 <= r <= r1,
            "beside": lambda r, c: (c < c0 or c > c1) and r0 <= r <= r1,
            "above":  lambda r, c: r < r0,
            "below":  lambda r, c: r > r1,
        }
        assert body in preds, f"{part!r}: side must be one of {list(preds)}"
        return cells(preds[body])

    if head == "seq":
        obj = available[base_name]["flat"]
        lo, hi = min(obj), max(obj)
        preds = {
            "before":  lambda j: j < lo,
            "after":   lambda j: j > hi,
            "between": lambda j: lo < j < hi and j not in set(obj),
        }
        assert body in preds, f"{part!r}: seq must be one of {list(preds)}"
        return [j for j in range(n_tokens) if preds[body](j)]

    raise AssertionError(f"{part!r} is not a grid spec")


# ---- text-window columns ---------------------------------------------------

def _text_columns(part, batch):
    """-> [B, n] absolute columns for one text spec.

    tok: columns are anchored at the END of the prompt. Left-padding makes
    that column identical for every row, so these are genuinely row-invariant.
    pre_image/vision_end are anchored on the IMAGE SPAN, whose absolute column
    DOES vary per row: rows built from a shorter template get more left
    padding, so their span sits further right. They are therefore computed
    per row, not collapsed with min()/max()."""
    positions = batch["positions"]
    B = positions.shape[0]
    head = _head(part)
    body = part.split(":", 1)[1] if ":" in part else ""

    if head == "tok":
        bits = body.split(":")
        off, n = int(bits[0]), (int(bits[1]) if len(bits) > 1 else 1)
        assert off < 0, f"{part!r}: tok offsets are negative (from the end of the prompt)"
        assert n >= 1, f"{part!r}: window length must be >= 1"
        last = positions[:, -1]
        assert int(last.min()) == int(last.max()), (
            "rows disagree on the final column -- batches are supposed to be left-padded with no "
            "padding after the real content; tok: offsets rely on that")
        # Python-style negative indexing: off=-1 is the LAST token (== last_token), hence the +1.
        # An off-by-one here is silent -- every offset still resolves to a real, plausible column.
        end = int(last[0]) + off + 1
        cols = [end - (n - 1) + j for j in range(n)]
        return torch.tensor([cols] * B, dtype=positions.dtype, device=positions.device)

    n = int(body) if body else 1
    assert n >= 1, f"{part!r}: window length must be >= 1"
    anchor = positions[:, 0] if head == "pre_image" else positions[:, -1]   # per row
    offsets = torch.arange(-n, 0, device=positions.device) if head == "pre_image" \
        else torch.arange(1, n + 1, device=positions.device)
    return (anchor.unsqueeze(1) + offsets.unsqueeze(0)).to(positions.dtype)


def _phrase_columns(part, batch, entity_assets, adapter, model, processor):
    """-> [B, 1] absolute column of the token that NAMES something in each row's QUESTION.

    Reuses methods/probe_common.py's ATTRIBUTE_KEYWORDS and the adapter's own
    find_last_phrase_token_col -- the same machinery the read-only probes use, including its
    validated per-attribute synonym lists (capital_prefill_v6 says "seat of government" and
    contains no "capital" at all, which is exactly the kind of thing that list already encodes).

    Two mechanics worth knowing:
      * find_last_phrase_token_col matches against the QUESTION only, never the prefill, and returns
        an UNPADDED column (an index into tokenize_template's own output). Rows are left-padded by
        differing amounts, so that column is shifted per row by (padded image start - unpadded image
        start), both of which we have.
      * The phrase is looked up per (queried, template_id), so the resolved column genuinely differs
        across rows -- which is fine, positions is a per-row tensor.

    Prefill words are NOT reachable here by design; use tok:-K for those (flags' prefills are 4-5
    tokens, so tok:-2 lands on "language" in "One official language is").
    """
    from .entities import object_token_positions
    from ..probe_common import ATTRIBUTE_KEYWORDS

    body = part.split(":", 1)[1] if ":" in part else ""
    assert body, f"{part!r}: expected phrase:attribute or phrase:<literal>[,<alt>...]"
    n_image_tokens = entity_assets.object_location["vlm_token_grid"]["n_tokens"]
    image_token_id = adapter.image_token_id(model, processor)
    positions = batch["positions"]

    cache = {}

    def unpadded_cols(queried, template_id):
        key = (queried, template_id)
        if key not in cache:
            tmpl = entity_assets.template_lookup[queried][template_id]
            question, prefill = tmpl["question"], tmpl["prefill"]
            ids = adapter.tokenize_template(processor, question, prefill, n_image_tokens)
            img_start = object_token_positions(ids, image_token_id, [0], n_image_tokens)[0]
            phrases = ATTRIBUTE_KEYWORDS.get(queried) if body == "attribute" else body.split(",")
            assert phrases, (
                f"{part!r}: attribute {queried!r} has no entry in probe_common.ATTRIBUTE_KEYWORDS -- "
                f"add its wordings there rather than special-casing here")
            col = adapter.find_last_phrase_token_col(processor, question, prefill, n_image_tokens, phrases)
            assert col is not None, (
                f"{part!r}: none of {phrases} occurs in template {template_id!r}'s question "
                f"({question!r}). Add the missing wording to probe_common.ATTRIBUTE_KEYWORDS, pass "
                f"comma-separated alternatives (phrase:image,flag), or restrict with --template_id.")
            cache[key] = (col, img_start)
        return cache[key]

    cols = []
    for i, row in enumerate(batch["rows"]):
        col, unpadded_img_start = unpadded_cols(row["queried"], row["template_id"])
        pad_shift = int(positions[i, 0]) - int(unpadded_img_start)
        cols.append([col + pad_shift])
    return torch.tensor(cols, dtype=positions.dtype, device=positions.device)


# ---- entry point -----------------------------------------------------------

def describe(spec, entity_assets=None):
    """One-line human summary for logs, including token counts where known."""
    outs = []
    for p in split_union(spec):
        if not _is_extended_part(p):
            outs.append(p)
        elif _head(p) in _TEXT_HEADS:
            outs.append(p)
        else:
            n = len(grid_indices(p, entity_assets)) if entity_assets is not None else "?"
            outs.append(f"{p} ({n} image tokens)")
    return " UNION ".join(outs)


def build_batch_at(spec, rows, entity_assets, adapter, model, processor, batch_cache=None):
    """build_batch + column re-selection. Handles unions by building each
    distinct underlying set once and concatenating the selected columns
    (deduped, sorted) -- the builds differ ONLY in `positions`, so any one of
    them can carry the rest of the batch."""
    from .entities import build_batch

    parts = split_union(spec)
    needed = {underlying_positions_name(p) for p in parts}
    builds = {name: build_batch(rows, entity_assets, adapter, model, processor, name, batch_cache=batch_cache)
              for name in needed}
    primary = builds[underlying_positions_name(parts[0])]

    chunks = []
    for p in parts:
        b = builds[underlying_positions_name(p)]
        if not _is_extended_part(p):
            chunks.append(b["positions"])
        elif _head(p) == "phrase":
            chunks.append(_phrase_columns(p, b, entity_assets, adapter, model, processor))
        elif _head(p) in _TEXT_HEADS:
            chunks.append(_text_columns(p, b))
        else:
            chunks.append(b["positions"][:, grid_indices(p, entity_assets)])

    cat = torch.cat(chunks, dim=1)
    # Dedupe + sort using ROW 0 as the reference, then apply that one permutation to every row.
    # Valid because rows differ only by a per-row left-pad offset applied uniformly to all of a
    # row's columns, so the relative order and the duplicate structure are row-invariant. The
    # assert below enforces exactly that rather than assuming it -- a union that mixes
    # end-anchored (row-invariant) with image-anchored (row-varying) columns could in principle
    # order differently across rows if a template were short enough, and that must fail loudly.
    ref = cat[0].tolist()
    seen, keep = set(), []
    for j, v in enumerate(ref):
        if v not in seen:
            seen.add(v)
            keep.append(j)
    keep.sort(key=lambda j: ref[j])
    out = cat[:, keep].contiguous()
    assert bool((out[:, 1:] > out[:, :-1]).all()) if out.shape[1] > 1 else True, (
        f"{spec!r}: the column ordering derived from row 0 does not sort every row -- the union mixes "
        f"position kinds whose relative order differs between rows. Split it into separate runs.")
    assert out.shape[1] > 0, f"{spec!r} selects no positions"
    assert int(out.min()) >= 0, f"{spec!r} resolves to negative column(s) -- a window runs off the prompt start"
    primary["positions"] = out
    return primary
