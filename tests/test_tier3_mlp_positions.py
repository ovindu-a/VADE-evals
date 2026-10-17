"""Offline checks for the Tier-3 code changes (FOLLOWUP_PLAN.md section 2):

    C2  head_das.py        --mask_lr reaches dbm's `mask`; --init_from loads a saved intervention.pt
    C3  head_severed.py    --freeze_mlp_span takes several comma-list spans, one arm each
    C9a position_sets.py   idx:i,j,... explicit image indices
    C9b probe_common.py    ATTRIBUTE_KEYWORDS for the five animals attributes
    C9c ceiling_sweep.py   --out_suffix (default '' = the historical filenames)

Every default here must stay byte-identical to before the change, so most tests
pin the OLD behaviour as well as the new one. No weights, no network; the
VADE-dependent and tokenizer-dependent tests skip when those are absent.
"""
import json
import os
from types import SimpleNamespace

import pytest
import torch

from methods import head_das as hd
from methods import head_severed as hs
from methods.common import position_sets as ps
from methods.probe_common import ATTRIBUTE_KEYWORDS

VADE_ROOT = hd.DEFAULT_VADE_ROOT
HAVE_DAS = os.path.exists(os.path.join(VADE_ROOT, "methods", "das", "intervention.py"))
FLAGS_LOC = os.path.join(VADE_ROOT, "data", "flags", "object_location.json")
ANIMAL_TEMPLATES = os.path.join(VADE_ROOT, "data", "animals", "prompt_templates.json")
ANIMAL_ATTRS = ["native_continent", "primary_habitat", "taxonomic_class", "reproduction", "body_covering"]
OPTIONS = "answer with exactly one of"


# ---------------------------------------------------------------------------
# C2a: the mask param group
# ---------------------------------------------------------------------------

def dbm_ivs(widths=(8, 12)):
    return {b: hd.make_intervention("dbm", w, 0, VADE_ROOT) for b, w in zip((21, 22), widths)}


def test_pyvene_dbm_names_its_mask_mask_and_temperature_is_a_parameter():
    """The premise of C2a, checked against the installed pyvene rather than assumed."""
    names = {n for n, _ in dbm_ivs()[21].named_parameters()}
    assert names == {"mask", "temperature"}


def test_dbm_mask_lr_gets_its_own_group_and_temperature_stays_out():
    ivs = dbm_ivs()
    opt, params = hd.build_optimizer(ivs, 1e-3, 1e-2)
    assert len(opt.param_groups) == 2
    rest, masks = opt.param_groups
    assert (rest["lr"], masks["lr"]) == (1e-3, 1e-2)
    assert {id(p) for p in masks["params"]} == {id(iv.mask) for iv in ivs.values()}
    assert {id(p) for p in rest["params"]} == {id(iv.temperature) for iv in ivs.values()}
    assert len(params) == 4


def test_without_mask_lr_there_is_one_group_exactly_as_before():
    ivs = dbm_ivs()
    opt, params = hd.build_optimizer(ivs, 1e-3, None)
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["lr"] == 1e-3
    old = [p for iv in ivs.values() for p in iv.parameters() if p.requires_grad]
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in old]
    assert [id(p) for p in params] == [id(p) for p in old]


@pytest.mark.skipif(not HAVE_DAS, reason="sibling VADE repo not present")
def test_das_rotated_grouping_is_unchanged():
    ivs = {21: hd.make_intervention("das_rotated", 16, 0, VADE_ROOT)}
    opt, _ = hd.build_optimizer(ivs, 1e-3, 4.0)
    rest, masks = opt.param_groups
    assert [id(p) for p in masks["params"]] == [id(ivs[21].masks)] and masks["lr"] == 4.0
    assert id(ivs[21].masks) not in {id(p) for p in rest["params"]} and rest["params"]
    opt1, _ = hd.build_optimizer(ivs, 1e-3, None)
    assert len(opt1.param_groups) == 1


@pytest.mark.skipif(not HAVE_DAS, reason="sibling VADE repo not present")
def test_das_fixed_has_no_mask_so_mask_lr_is_a_no_op():
    ivs = {21: hd.make_intervention("das_fixed", 16, 4, VADE_ROOT)}
    opt, _ = hd.build_optimizer(ivs, 1e-3, 1e-2)
    assert len(opt.param_groups) == 1


# ---------------------------------------------------------------------------
# C2b: --init_from
# ---------------------------------------------------------------------------

def test_init_from_round_trips_a_saved_dbm_checkpoint(tmp_path):
    torch.manual_seed(0)
    trained = dbm_ivs()
    with torch.no_grad():
        for iv in trained.values():
            iv.mask.normal_()
            iv.temperature.fill_(0.05)
    path = tmp_path / "intervention.pt"
    torch.save({b: iv.state_dict() for b, iv in trained.items()}, path)      # main()'s own format
    fresh = dbm_ivs()
    assert all(float(iv.mask.detach().abs().max()) == 0.0 for iv in fresh.values()), "fresh dbm mask should be 0"
    hd.load_interventions(fresh, str(path))
    for b in trained:
        for (n1, p1), (n2, p2) in zip(trained[b].named_parameters(), fresh[b].named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2)


def test_init_from_refuses_a_checkpoint_for_other_blocks(tmp_path):
    path = tmp_path / "intervention.pt"
    torch.save({21: dbm_ivs()[21].state_dict()}, path)
    with pytest.raises(AssertionError, match="checkpoint blocks"):
        hd.load_interventions(dbm_ivs(), str(path))


@pytest.mark.skipif(not HAVE_DAS, reason="sibling VADE repo not present")
def test_init_from_round_trips_an_orthogonal_das_fixed_checkpoint(tmp_path):
    torch.manual_seed(1)
    a = {21: hd.make_intervention("das_fixed", 16, 4, VADE_ROOT)}
    b = {21: hd.make_intervention("das_fixed", 16, 4, VADE_ROOT)}
    assert not torch.equal(a[21].proj.weight, b[21].proj.weight)
    path = tmp_path / "intervention.pt"
    torch.save({k: iv.state_dict() for k, iv in a.items()}, path)
    hd.load_interventions(b, str(path))
    torch.testing.assert_close(a[21].proj.weight, b[21].proj.weight)
    x, y = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    torch.testing.assert_close(a[21](x, y), b[21](x, y))


# ---------------------------------------------------------------------------
# C3: head_severed spans
# ---------------------------------------------------------------------------

def test_parse_span_keeps_the_old_forms_and_learns_commas():
    assert hs.parse_span("22-27") == [22, 23, 24, 25, 26, 27]
    assert hs.parse_span("24") == [24]
    assert hs.parse_span("23,25-26") == [23, 25, 26]
    assert hs.parse_span("23-24,26") == [23, 24, 26]
    assert hs.parse_span("26,23-24,24") == [23, 24, 26], "sorted and deduplicated"
    assert hs.parse_span(" 23 - 24 , 26 ") == [23, 24, 26]
    assert hs.parse_span(25) == [25]


@pytest.mark.parametrize("bad", ["", "x", "23-", "-23", "23-24-25", "26-23", "23,,24", "2.5", "23;24"])
def test_parse_span_rejects_garbage(bad):
    with pytest.raises(ValueError):
        hs.parse_span(bad)


def sev_args(span, **kw):
    base = dict(freeze_mlp_span=span, freeze_mlp_singles=[22, 23, 24, 25, 26, 27], freeze_attn_span="24-27",
                freeze_attn_singles=[], full_image=False, freeze_values=["base"])
    base.update(kw)
    return SimpleNamespace(**base)


DEFAULT_ARMS = ["clean", "heads", "heads+mlp[22-27]", "heads+mlp[22]", "heads+mlp[23]", "heads+mlp[24]",
                "heads+mlp[25]", "heads+mlp[26]", "heads+mlp[27]", "heads+attn[24-27]"]


def test_default_single_span_arms_are_unchanged_as_str_and_as_list():
    for span in ("22-27", ["22-27"]):
        arms = hs.build_arms(sev_args(span), head_blocks=[21, 22, 23])
        assert [a[0] for a in arms] == DEFAULT_ARMS
        assert arms[2][3] == [("mlp", b) for b in range(22, 28)]


@pytest.mark.parametrize("empty", ["", [""], []])
def test_empty_span_still_skips_the_span_arm(empty):
    names = [a[0] for a in hs.build_arms(sev_args(empty, freeze_mlp_singles=[25], freeze_attn_span=""),
                                         head_blocks=[21, 22, 23])]
    assert names == ["clean", "heads", "heads+mlp[25]"]


def test_several_spans_give_one_arm_each_per_freeze_value():
    spans = ["23-26", "24-26", "23,25-26", "23-24,26", "23-25"]
    args = sev_args(spans, freeze_mlp_singles=[25, 27], freeze_attn_span="",
                    freeze_values=["base", "mean", "other"])
    arms = hs.build_arms(args, head_blocks=[21, 22, 23])
    names = [(a[0], a[4]) for a in arms]
    expect = [("clean", None), ("heads", None)]
    for n in [f"heads+mlp[{s}]" for s in spans] + ["heads+mlp[25]", "heads+mlp[27]"]:
        expect += [(n, "base"), (f"{n}@mean", "mean"), (f"{n}@other", "other")]
    assert names == expect and len(arms) == 23, "P10's 23 arms"
    freezes = {a[0]: a[3] for a in arms}
    assert freezes["heads+mlp[23,25-26]"] == [("mlp", 23), ("mlp", 25), ("mlp", 26)]
    assert freezes["heads+mlp[23-24,26]"] == [("mlp", 23), ("mlp", 24), ("mlp", 26)]


def test_full_image_uses_the_first_span_only():
    arms = hs.build_arms(sev_args(["23-26", "24-26"], freeze_mlp_singles=[], freeze_attn_span="",
                                  full_image=True), head_blocks=[21, 22, 23])
    names = [a[0] for a in arms]
    assert names[-2:] == ["full_image", "full_image+mlp[23-26]"]
    assert "full_image+mlp[24-26]" not in names
    one = hs.build_arms(sev_args("22-27", full_image=True), head_blocks=[21, 22, 23])
    assert [a[0] for a in one] == DEFAULT_ARMS + ["full_image", "full_image+mlp[22-27]"]


def test_a_span_colliding_with_another_arm_is_refused():
    with pytest.raises(AssertionError, match="duplicate arm"):
        hs.build_arms(sev_args(["23-26", "23-26"]), head_blocks=[21, 22, 23])
    with pytest.raises(AssertionError, match="duplicate arm"):
        hs.build_arms(sev_args(["25"], freeze_mlp_singles=[25]), head_blocks=[21, 22, 23])


# ---------------------------------------------------------------------------
# C9a: idx: positions
# ---------------------------------------------------------------------------

@pytest.fixture
def flags_assets():
    if not os.path.exists(FLAGS_LOC):
        pytest.skip("sibling VADE repo not present")
    return SimpleNamespace(object_location=json.load(open(FLAGS_LOC)))


def test_idx_parses_sorted_unique(flags_assets):
    assert ps.is_extended("idx:3") and ps.underlying_positions_name("idx:3") == "full_image"
    assert ps.grid_indices("idx:79,64, 65,64", flags_assets) == [64, 65, 79]
    assert ps.grid_indices("idx:0,143", flags_assets) == [0, 143]


@pytest.mark.parametrize("bad", ["idx:", "idx:,", "idx:144", "idx:-1", "idx:a,b", "idx:1.5"])
def test_idx_rejects_out_of_grid_and_garbage(flags_assets, bad):
    with pytest.raises(AssertionError):
        ps.grid_indices(bad, flags_assets)


def test_idx_of_entity_only_equals_entity_only(flags_assets):
    flat = flags_assets.object_location["object_token_indices"]["entity_only"]["flat"]
    assert ps.grid_indices("idx:" + ",".join(map(str, flat)), flags_assets) == sorted(flat)
    assert ps.grid_indices("idx:" + ",".join(map(str, flat)), flags_assets) == ps.grid_indices("ring:0", flags_assets)


def fake_build_batch(n_tokens, starts):
    """entities.build_batch's contract for full_image: column j is image token j,
    each row's span starting at its own (left-pad-shifted) column."""
    def build(rows, entity_assets, adapter, model, processor, name, batch_cache=None):
        assert name == "full_image", name
        return {"positions": torch.stack([torch.arange(s, s + n_tokens) for s in starts]), "rows": rows}
    return build


def test_idx_composes_in_unions(flags_assets, monkeypatch):
    import methods.common.entities as ent
    monkeypatch.setattr(ent, "build_batch", fake_build_batch(144, [10, 13]))
    qwen = SimpleNamespace(shares_object_grid=True, model_id="qwen")
    out = ps.build_batch_at("idx:52,53,64+ring:0", [{}, {}], flags_assets, qwen, None, None)["positions"]
    ring0 = flags_assets.object_location["object_token_indices"]["entity_only"]["flat"]
    want = sorted(set([52, 53, 64]) | set(ring0))
    assert out.tolist() == [[10 + j for j in want], [13 + j for j in want]]
    alone = ps.build_batch_at("idx:52,53", [{}, {}], flags_assets, qwen, None, None)["positions"]
    assert alone.tolist() == [[62, 63], [65, 66]]


def test_idx_is_refused_where_grid_specs_are(flags_assets, monkeypatch):
    import methods.common.entities as ent
    monkeypatch.setattr(ent, "build_batch", fake_build_batch(144, [10]))
    internvl = SimpleNamespace(shares_object_grid=False, model_id="OpenGVLab/InternVL3-8B-hf")
    for spec in ("idx:64", "full_image+idx:64", "tok:-1+idx:64"):
        with pytest.raises(AssertionError, match="does not share"):
            ps.build_batch_at(spec, [{}], flags_assets, internvl, None, None)


# ---------------------------------------------------------------------------
# C9b: animals ATTRIBUTE_KEYWORDS
# ---------------------------------------------------------------------------

@pytest.fixture
def animal_templates():
    if not os.path.exists(ANIMAL_TEMPLATES):
        pytest.skip("sibling VADE repo not present")
    return json.load(open(ANIMAL_TEMPLATES))


def matched(question, phrases):
    """find_last_phrase_token_col's own string rule: the FIRST listed phrase present
    anywhere in the (lowercased) question, at its LAST occurrence."""
    q = question.lower()
    return next(((p, q.rfind(p.lower())) for p in phrases if p.lower() in q), None)


def test_animals_keywords_are_exactly_the_planned_lists():
    assert ATTRIBUTE_KEYWORDS["native_continent"] == ["continent"]
    assert ATTRIBUTE_KEYWORDS["primary_habitat"] == ["habitat", "environment", "live"]
    assert ATTRIBUTE_KEYWORDS["taxonomic_class"] == ["taxonomic class", "classes", "class"]
    assert ATTRIBUTE_KEYWORDS["reproduction"] == ["reproductive mode", "reproductive strategy", "reproduce",
                                                  "offspring", "lay eggs"]
    assert ATTRIBUTE_KEYWORDS["body_covering"] == ["body covering", "outer covering", "covering", "covers",
                                                   "covered"]


def test_every_animals_template_matches_before_its_options_list(animal_templates):
    seen = 0
    for attr in ANIMAL_ATTRS:
        for t in animal_templates[attr]:
            q = t["question"]
            m = matched(q, ATTRIBUTE_KEYWORDS[attr])
            assert m is not None, f"{t['template_id']}: no keyword in {q!r}"
            phrase, start = m
            opts = q.lower().rfind(OPTIONS)
            assert opts > 0, f"{t['template_id']}: no options list -- this check's premise is gone"
            assert start + len(phrase) <= opts, f"{t['template_id']}: {phrase!r} matched inside the options list"
            # and no keyword occurs in the options list at all, so list order can never pull it there
            assert not any(p.lower() in q.lower()[opts:] for p in ATTRIBUTE_KEYWORDS[attr]), t["template_id"]
            seen += 1
    assert seen == 30


def test_animals_keywords_resolve_to_a_question_token_with_the_real_tokenizer(animal_templates):
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5-VL processor not cached locally")
    from methods.adapters.qwen2_5_vl import Qwen25VLAdapter
    adapter, n_img = Qwen25VLAdapter(), 4
    for attr in ANIMAL_ATTRS:
        for t in animal_templates[attr]:
            q, pre = t["question"], t["prefill"]
            col = adapter.find_last_phrase_token_col(processor, q, pre, n_img, ATTRIBUTE_KEYWORDS[attr])
            opts = adapter.find_last_phrase_token_col(processor, q, pre, n_img, ["Answer with"])
            ids = adapter.tokenize_template(processor, q, pre, n_img)
            assert col is not None and col < opts - 1, (t["template_id"], col, opts)
            phrase, _ = matched(q, ATTRIBUTE_KEYWORDS[attr])
            tok = processor.tokenizer.decode(ids[col:col + 1]).strip().lower()
            assert tok and tok in phrase, (t["template_id"], tok, phrase)


# ---------------------------------------------------------------------------
# C9c: ceiling_sweep --out_suffix
# ---------------------------------------------------------------------------

def test_out_suffix_default_keeps_the_historical_filenames():
    from methods.ndm import ceiling_sweep as cs
    tag = "0-1-2"
    assert cs.json_filename(tag) == f"ceiling_sweep_layers{tag}.json"
    assert cs.json_filename(tag, True) == f"ceiling_sweep_layers{tag}_text.json"
    assert cs.json_filename(tag, True, "") == f"ceiling_sweep_layers{tag}_text.json"
    assert cs.log_filename("ring_0__ring_1", tag, False) == f"ceiling_sweep_ring_0__ring_1_layers{tag}.log"
    assert cs.log_filename("x", tag, True) == f"ceiling_sweep_x_layers{tag}_text.log"


def test_out_suffix_is_appended_after_text():
    from methods.ndm import ceiling_sweep as cs
    assert cs.json_filename("2-6", True, "_rings") == "ceiling_sweep_layers2-6_text_rings.json"
    assert cs.json_filename("2-6", False, "_rings") == "ceiling_sweep_layers2-6_rings.json"
    assert cs.log_filename("ring_0", "2-6", False, "_rings") == "ceiling_sweep_ring_0_layers2-6_rings.log"
