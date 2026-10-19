"""Offline checks for X2b: the router / composition / text-contrast scripts on brands and
celebrities (attribute_switch_sweep.ENTITY_SPECS, own-answer golds, --entity plumbing).

Flags must be byte-identical to before X2b, so its prompts are checked against literal
copies of the pre-X2b strings. Tiny random Qwen2.5-VL from test_head_followups;
tokenizer-dependent checks use the real Qwen2.5-VL processor if it is cached locally and
skip otherwise; VADE-dependent checks skip without a sibling VADE checkout.
"""
import json
import os
import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from test_head_followups import tiny                                    # noqa: F401
from methods import attribute_switch_sweep as sw
from methods import circuit_common as cc
from methods import circuit_compose as comp
from methods import text_entity_contrast as tec
from methods.head_cross import labelled
from methods.head_followup_common import Results

OLD_FLAGS_INSTRUCTION = ('For the country shown, capital means its capital city; currency means its '
                         'three-letter currency code; language means one official language; calling '
                         'means its international calling code, digits only without a plus sign. '
                         'Give only the requested value. ')
OLD_FIELDS = dict(capital='capital', currency='currency', language='language', calling_code='calling')
NON_FLAGS = ("brands", "celebrities")


def real_processor():
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(cc.MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5-VL processor not cached locally")


def vade_entity(entity):
    path = os.path.join(cc.DEFAULT_VADE_ROOT, "data", entity, "ground_truth.json")
    if not os.path.exists(path):
        pytest.skip("VADE checkout not present")
    return cc.load_entity(cc.DEFAULT_VADE_ROOT, entity)


# ---------------------------------------------------------------------------
# Flags stays byte-identical
# ---------------------------------------------------------------------------

def test_flags_prompts_are_the_pre_x2b_strings():
    assert sw.INSTRUCTION == OLD_FLAGS_INSTRUCTION
    for a in sw.ATTRIBUTES:
        assert sw.question(a) == OLD_FLAGS_INSTRUCTION + f"Report the {OLD_FIELDS[a]} of this country."
        assert sw.question(a, True) == OLD_FLAGS_INSTRUCTION + f"Return the {OLD_FIELDS[a]} of this country."
    old_text = OLD_FLAGS_INSTRUCTION.replace("For the country shown", "For the country named")
    assert cc.TEXT_INSTRUCTION == old_text == cc.text_instruction("flags")
    assert cc.text_question("capital", "France") == old_text + "The country is France. Report the capital of this country."
    assert sw.entity_attributes("flags") == sw.ATTRIBUTES and not sw.uses_own_golds("flags")


def test_flags_defaults_resolve_to_the_old_values():
    ap = SimpleNamespace(error=lambda msg: pytest.fail(msg))
    args = SimpleNamespace(entity="flags", movers=None, routers=None, router_trace=None, attributes=None,
                           out_dir=None)
    comp.resolve_entity_args(ap, args)
    assert args.movers == cc.MOVERS and args.router_trace == cc.ROUTER_TRACE
    assert args.attributes == list(cc.ATTRIBUTES)
    assert args.out_dir == os.path.join(comp.REPO_ROOT, "results", "circuit_compose", "flags")


# ---------------------------------------------------------------------------
# Entity specs
# ---------------------------------------------------------------------------

def test_attribute_names_are_unique_and_select_their_entity():
    assert len(sw.ALL_ATTRIBUTES) == len(set(sw.ALL_ATTRIBUTES))
    for entity, spec in sw.ENTITY_SPECS.items():
        for a in spec["attributes"]:
            q = sw.question(a)
            assert q.startswith(spec["instruction"]) and q.endswith(f"of this {spec['noun']}.")
            assert f" {spec['fields'][a]} means " in spec["instruction"]    # every field word is defined
        assert len(set(spec["fields"].values())) == len(spec["attributes"]), f"{entity}: two attributes share a field"


@pytest.mark.parametrize("entity", NON_FLAGS)
def test_entity_attributes_match_vade(entity):
    entity_dir, items, _ = vade_entity(entity)
    gt = json.load(open(os.path.join(entity_dir, "ground_truth.json")))
    assert sorted(gt["attributes"]) == sorted(sw.entity_attributes(entity))
    assert sw.load_entity_items(cc.DEFAULT_VADE_ROOT, entity)[1] == items
    assert all("name" in it for it in items.values()), "text_entity_contrast needs every item's name"


@pytest.mark.parametrize("entity", sw.ENTITIES)
def test_field_words_are_one_token_and_prompts_differ_in_one_column(entity):
    tok = real_processor().tokenizer
    spec = sw.ENTITY_SPECS[entity]
    for field in spec["fields"].values():
        assert len(tok(" " + field, add_special_tokens=False)["input_ids"]) == 1, field
    for paraphrase in (False, True):
        enc = [tok(sw.question(a, paraphrase), add_special_tokens=False)["input_ids"] for a in spec["attributes"]]
        assert len({len(e) for e in enc}) == 1, f"{entity}: controlled prompts differ in length"
        for e in enc[1:]:
            assert sum(x != y for x, y in zip(enc[0], e)) == 1


@pytest.mark.parametrize("entity", NON_FLAGS)
def test_alignment_and_field_scope_on_real_image_inputs(entity):
    proc = real_processor()
    entity_dir, items, _ = vade_entity(entity)
    from methods.adapters.registry import get_adapter
    adapter = get_adapter(cc.MODEL_ID)
    image_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    key = sorted(items)[0]
    with Image.open(os.path.join(entity_dir, items[key]["image"])) as im:
        img = im.convert("RGB")
    built = {a: adapter.build_inputs(proc, img, sw.question(a), sw.PREFILL) for a in sw.entity_attributes(entity)}
    attrs = list(built)
    for b in attrs:
        for d in attrs:
            if b == d:
                continue
            base, donor = built[b]["input_ids"].unsqueeze(0), built[d]["input_ids"].unsqueeze(0)
            assert sw.alignment(base, donor, image_id, proc.tokenizer.all_special_ids)
            col = sw.field_position({"base_input_ids": base, "source_input_ids": donor})
            assert proc.tokenizer.decode(base[0, col]).strip() == sw.ENTITY_SPECS[entity]["fields"][b]
    assert proc.tokenizer.decode(built[attrs[0]]["input_ids"][-1:]) == ":"     # ends on the prefill


@pytest.mark.parametrize("entity", NON_FLAGS)
def test_text_prompt_name_columns_cover_every_name(entity):
    proc = real_processor()
    _, items, _ = vade_entity(entity)
    m = cc.Model.__new__(cc.Model)
    m.processor = proc
    attr = sw.entity_attributes(entity)[0]
    for key, it in items.items():
        ids, cols = cc.Model.text_prompt(m, it["name"], attr)
        assert proc.tokenizer.decode(ids[cols]).strip() == it["name"], key
        assert cols == list(range(cols[0], cols[-1] + 1))
    q = cc.text_question(attr, "X")
    noun = sw.ENTITY_SPECS[entity]["noun"]
    assert f"For the {noun} named" in q and f"The {noun} is X. " in q and "shown" not in q
    assert f"The {noun} is Apple Inc. Report" in cc.text_question(attr, "Apple Inc.")   # no '..' token


# ---------------------------------------------------------------------------
# Own-answer golds
# ---------------------------------------------------------------------------

class FakeTok:
    eos_token_id, pad_token_id = 99, 0

    def __init__(self, words):
        self.words = words

    def decode(self, tokens, **kwargs):
        return " ".join(self.words.get(int(t), "?") for t in (tokens if isinstance(tokens, list) else [tokens]))

    def convert_ids_to_tokens(self, tokens):
        return [self.words.get(int(t), "?") for t in tokens]


def test_own_gold_cuts_at_the_first_stop_and_matcher_is_casefold_whole_word():
    assert sw.own_gold([5, 6, 99, 7], {99}) == [5, 6]
    assert sw.own_gold([5, 6], {99}) == [5, 6]
    assert sw.own_gold([99, 5], {99}) == []
    tok = FakeTok({1: "The", 2: "United", 3: "States.", 4: "Ireland"})
    assert sw.own_correct(tok, [1, 2, 3], "united states")
    assert not sw.own_correct(tok, [4], "iran") and not sw.own_correct(tok, [], "x")
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[7, 8]))
    assert sw.stop_ids(model, tok) == {7, 8, 99, 0}
    model.generation_config.eos_token_id = None
    assert sw.stop_ids(model, tok) == {99, 0}


def switch_setup(tiny, words):
    original, batch = tiny
    runner = sw.SwitchRunner.__new__(sw.SwitchRunner)
    runner.__dict__.update(original.__dict__)
    runner.layers = runner.adapter.get_decoder_layers(runner.model)
    batch['source_input_ids'][0, 4] = 15
    batch['paraphrase_ids'] = batch['base_input_ids'].clone()
    batch['paraphrase_ids'][0, 3] = 16
    batch['earlier_positions'] = sw.alignment(batch['base_input_ids'], batch['source_input_ids'], 2, [1, 3])
    runner.processor = SimpleNamespace(tokenizer=FakeTok(words))
    runner.batch = lambda image, base, donor: batch
    return runner, batch


@pytest.mark.parametrize("correct", [True, False])
def test_switch_execute_uses_own_answers_and_skips_rows_the_model_gets_wrong(tiny, tmp_path, correct):
    # The tiny model's greedy tokens decode to the label only if `correct`.
    words = {t: ("japan" if correct else "nowhere") for t in range(64)}
    runner, batch = switch_setup(tiny, words)
    Image.new('RGB', (4, 4)).save(tmp_path / 'logo.png')
    args = SimpleNamespace(entity='brands', sites=['last_residual'], block_spans=[], layers=[4],
                           modes=['prefill'], controls=['self'], max_new_tokens=4,
                           execution='serial', batch_size=2)
    row = dict(row_index=0, base='SONY', source='SONY', base_attribute='hq_country',
               donor_attribute='industry', base_label='japan', source_label='japan',
               template_id='controlled_report_v1')
    results = Results(tmp_path / 'results', {})
    sw.execute(runner, args, [row], {'SONY': {'image': 'logo.png'}}, tmp_path, results)
    clean = next(r for r in results.records if r['arm'] == 'clean')
    donor = next(r for r in results.records if r['arm'] == 'donor_clean')
    # Golds ARE the clean runs: base gold = clean's tokens, source gold = donor_clean's.
    assert clean['gold_ids']['base'] == clean['generated_ids'] and clean['base_score']['full_match']
    assert clean['gold_ids']['source'] == donor['generated_ids'] and donor['source_score']['full_match']
    assert clean['own_correct'] is correct and donor['own_correct'] is correct
    assert clean['golds'] == 'own_answer' and 'earlier_positions' in clean
    cells = json.loads((results.path / 'switch_summary.json').read_text())
    if correct:
        assert len(results.records) == 2 + len(sw.sweep_arms(args))
        assert all(r['gold_ids'] == clean['gold_ids'] for r in results.records)
        assert all(c['both_clean_correct']['n'] == 1 for c in cells)
    else:
        assert len(results.records) == 2, "a row the model gets wrong must run no sweep arms"
        assert all(c['both_clean_correct'] is None for c in cells)
    n = len(results.records)
    sw.execute(runner, args, [row], {'SONY': {'image': 'logo.png'}}, tmp_path, Results(results.path, {}))
    assert len(Results(results.path, {}).records) == n, "resume must not duplicate or regenerate"


def test_switch_flags_path_never_reads_own_answers(tiny, tmp_path, monkeypatch):
    runner, batch = switch_setup(tiny, {})
    monkeypatch.setattr(sw, 'own_golds_row', lambda *a: pytest.fail("flags used own-answer golds"))
    class Tokenizer(FakeTok):
        def __call__(self, text, **kwargs):
            return {'input_ids': [14] + ([20, 21] if 'Paris' in text else [24, 25] if 'EUR' in text else [])}
    runner.processor = SimpleNamespace(tokenizer=Tokenizer({}))
    Image.new('RGB', (4, 4)).save(tmp_path / 'flag.png')
    args = SimpleNamespace(sites=['last_residual'], block_spans=[], layers=[4], modes=['prefill'],
                           controls=[], max_new_tokens=4, execution='serial', batch_size=2)
    row = dict(row_index=0, base='FR', source='FR', base_attribute='capital', donor_attribute='currency',
               base_label='Paris', source_label='EUR', template_id='controlled')
    results = Results(tmp_path / 'results', {})
    sw.execute(runner, args, [row], {'FR': {'image': 'flag.png'}}, tmp_path, results)
    assert all('own_correct' not in r for r in results.records)
    assert results.records[0]['gold_ids'] == {'base': [20, 21], 'source': [24, 25]}


def test_flags_own_golds_is_opt_in():
    assert not sw.uses_own_golds("flags") and sw.uses_own_golds("flags", own_golds=True)
    assert sw.uses_own_golds("brands") and sw.uses_own_golds("brands", own_golds=True)


def test_switch_flags_own_golds_flag_takes_the_own_answer_path(tiny, tmp_path):
    # Idefics3 writes flags answers lowercase (' luxembourg.'); with --own_golds the clean runs become the
    # golds and VADE's casefold matcher decides correctness, exactly as for brands.
    runner, batch = switch_setup(tiny, {t: "luxembourg" for t in range(64)})
    Image.new('RGB', (4, 4)).save(tmp_path / 'flag.png')
    args = SimpleNamespace(own_golds=True, sites=['last_residual'], block_spans=[], layers=[4],
                           modes=['prefill'], controls=['self'], max_new_tokens=4,
                           execution='serial', batch_size=2)
    row = dict(row_index=0, base='LU', source='LU', base_attribute='capital', donor_attribute='currency',
               base_label='Luxembourg', source_label='Luxembourg', template_id='controlled_report_v1')
    results = Results(tmp_path / 'results', {})
    sw.execute(runner, args, [row], {'LU': {'image': 'flag.png'}}, tmp_path, results)
    clean = next(r for r in results.records if r['arm'] == 'clean')
    assert clean['golds'] == 'own_answer' and clean['own_correct'] is True
    assert clean['gold_ids']['base'] == clean['generated_ids']
    assert len(results.records) == 2 + len(sw.sweep_arms(args))


def test_switch_flags_config_is_unchanged_without_own_golds(tmp_path, monkeypatch):
    # --own_golds off must not add a key to a flags config (that would refuse every old resume).
    seen = {}
    class Stop(Exception):
        pass
    def fake_results(out_dir, config):
        seen.update(config)
        raise Stop
    monkeypatch.setattr(sw, 'Results', fake_results)
    for flag, expect in (([], False), (['--own_golds'], True)):
        seen.clear()
        with pytest.raises(Stop):
            sw.main(['--n_countries', '1', '--layers', '4', '--out_dir', str(tmp_path / 'o'), *flag])
        assert ('own_golds' in seen) is expect and (seen.get('golds') == 'own_answer') is expect


def test_attr_head_trace_own_answer_rows_keep_only_rows_the_model_gets_right(tmp_path, monkeypatch):
    from methods import attr_head_trace as aht
    Image.new('RGB', (4, 4)).save(tmp_path / 'a.png')
    items = {"A": {"image": "a.png", "hq_country": "japan", "industry": "automotive", "founded_year": "1937"}}
    _, rows = aht.sample_rows(tmp_path, items, ["hq_country", "industry", "founded_year"], ["A"], 1, 0)
    assert len(rows) == 6
    answer = {"hq_country": [11, 12, 99], "industry": [13, 99, 5], "founded_year": [14, 99]}
    built = []

    def build_inputs(processor, image, q, prefill):
        a = next(x for x in answer if sw.question(x) == q)
        built.append(a)
        return {"input_ids": torch.tensor([1, 2, 3]), "extra": {"pixel_values": torch.zeros(1, 2)}, "attr": a}

    def fake_generate(model, ids, mask, extra, pad, max_new_tokens):
        assert max_new_tokens == 12 and ids.shape[0] == mask.shape[0] == extra["pixel_values"].shape[0]
        return torch.tensor([(answer[a] + [0, 0])[:3] for a in built[-ids.shape[0]:]])
    monkeypatch.setattr(aht, "generate_unhooked", fake_generate)
    words = {11: "Japan", 12: ".", 13: "Automotive", 14: "1950"}          # the year is WRONG
    proc = SimpleNamespace(tokenizer=FakeTok(words))
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None))
    kept, dropped = aht.own_answer_rows(rows, tmp_path, items, SimpleNamespace(build_inputs=build_inputs),
                                        model, proc, 2, 12)
    assert dropped == 4 and {(r["base_attribute"], r["donor_attribute"]) for r in kept} == {
        ("hq_country", "industry"), ("industry", "hq_country")}
    hq = next(r for r in kept if r["base_attribute"] == "hq_country")
    assert hq["base_gold_ids"] == [11, 12] and hq["source_gold_ids"] == [13]
    assert sorted(built) == sorted(answer), "one generation per (item, attribute), not per row"


def test_sample_rows_skip_items_missing_an_attribute(tmp_path):
    from methods import attr_head_trace as aht
    Image.new('RGB', (4, 4)).save(tmp_path / 'a.png')
    items = {"AIRBUS": {"image": "a.png", "hq_country": "france", "founded_year": "1970"}}
    _, rows = aht.sample_rows(tmp_path, items, ["hq_country", "founded_year", "industry"], ["AIRBUS"], 1, 0)
    assert {(r["base_attribute"], r["donor_attribute"]) for r in rows} == {
        ("hq_country", "founded_year"), ("founded_year", "hq_country")}


def test_modality_agree_keeps_rows_whose_readout_tokens_match_in_both_conditions():
    rows = [{"row_index": 0, "base": "A", "source": "B", "attribute": "x", "prefix": [1], "base_tok": 2, "src_tok": 3},
            {"row_index": 1, "base": "A", "source": "C", "attribute": "x", "prefix": [], "base_tok": 1, "src_tok": 4}]
    img = {("A", "x"): [1, 2, 9], ("B", "x"): [1, 3], ("C", "x"): [4]}
    # A's text answer differs only AFTER row 0's divergence token: still agrees. C's text answer
    # differs at its first token, so row 1's readout token is not C's text answer.
    txt = {("A", "x"): [1, 2, 7], ("B", "x"): [1, 3, 5], ("C", "x"): [6]}
    kept, dropped = tec.modality_agree(rows, txt, img)
    assert dropped == 1 and [r["source"] for r in kept] == ["B"] and kept[0]["row_index"] == 0
    kept, dropped = tec.modality_agree(rows, {}, img)
    assert kept == [] and dropped == 2


# ---------------------------------------------------------------------------
# Missing labels (G and friends): AIRBUS has no industry
# ---------------------------------------------------------------------------

def test_head_cross_skips_rows_whose_golds_need_a_missing_label():
    truth = {"A": {"hq_country": "x", "industry": "y"}, "AIRBUS": {"hq_country": "z"}}
    row = dict(base="A", donor_flag="AIRBUS", q1="hq_country", q2="hq_country")
    assert labelled(row, truth)
    assert not labelled({**row, "q2": "industry"}, truth)
    assert not labelled({**row, "base": "AIRBUS", "donor_flag": "A", "q1": "industry"}, truth)


def test_head_cross_brands_n128_draws_airbus_and_skips_it():
    _, items, _ = vade_entity("brands")
    from methods.head_cross import build_rows
    rows = build_rows(items, sw.entity_attributes("brands"), 128, 0)
    kept = [r for r in rows if labelled(r, items)]
    assert 0 < len(rows) - len(kept) < len(rows)
    assert all("AIRBUS" not in (r["base"], r["donor_flag"]) or "industry" not in (r["q1"], r["q2"]) for r in kept)


def test_compose_rows_skip_missing_labels_and_non_flags_need_their_own_heads():
    items = {"A": {"hq_country": "x", "industry": "y", "founded_year": "1"},
             "AIRBUS": {"hq_country": "z", "founded_year": "2"}}
    rows = comp.build_rows(items, ["hq_country", "industry", "founded_year"], 2, 0)
    assert rows and all(all(k in items[r[i]] for i in ("C", "B") for k in (r["X"], r["A"])) for r in rows)
    errors = []
    ap = SimpleNamespace(error=lambda msg: errors.append(msg) or (_ for _ in ()).throw(SystemExit))
    args = SimpleNamespace(entity="brands", movers=None, routers=None, router_trace=None, attributes=None,
                           out_dir=None)
    with pytest.raises(SystemExit):
        comp.resolve_entity_args(ap, args)
    assert "FLAGS heads" in errors[0]
    args.movers, args.routers = "21.1", "19.2"
    comp.resolve_entity_args(ap, args)
    assert args.attributes == sw.entity_attributes("brands") and args.out_dir.endswith("circuit_compose/brands")


def test_attr_capture_pool_excludes_items_missing_an_attribute(capsys):
    vade_entity("brands")
    from methods import attr_capture
    attr_capture.main(["--entity", "brands", "--n_items", "128", "--dry_run"])
    out = capsys.readouterr().out
    assert "brands: 127 items x 3 attributes" in out          # every brand but AIRBUS


def test_route_knockout_own_rows_on_controlled_prompt_answers():
    """entity_pair_rows reuses route_knockout's own_rows: different labels, both answers right,
    answers diverge."""
    from methods.route_knockout import own_rows
    items = {"A": {"hq_country": "japan"}, "B": {"hq_country": "united states"}, "C": {"hq_country": "japan"}}
    words = {1: "Japan", 2: "The", 3: "United", 4: "States", 5: "France"}
    answers = {("A", "hq_country"): [1], ("B", "hq_country"): [2, 3, 4], ("C", "hq_country"): [5]}
    rows, dropped = own_rows(items, [("A", "B"), ("A", "C"), ("C", "B")], ["hq_country"], answers,
                             FakeTok(words).decode)
    assert [(r["base"], r["source"]) for r in rows] == [("A", "B")]
    assert rows[0]["base_tok"] == 1 and rows[0]["src_tok"] == 2 and rows[0]["prefix"] == []
    assert dropped["same_label"] == 1 and dropped["model_wrong"] == 1


# ---------------------------------------------------------------------------
# CLI dry runs (no model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entity", NON_FLAGS)
def test_switch_sweep_dry_run_builds_rows_with_entity_labels(entity, capsys):
    _, items, _ = vade_entity(entity)
    sw.main(["--entity", entity, "--n_countries", "4", "--sites", "residual", "--block_spans", "--dry_run"])
    out = capsys.readouterr().out
    assert "own clean answers" in out and "4 items" in out
    chosen = random.Random(0).sample(sorted(items), 4)
    n_attr = len(sw.entity_attributes(entity))
    expect = sum(sum(1 for b in sw.entity_attributes(entity) for d in sw.entity_attributes(entity)
                     if b != d and b in items[c] and d in items[c]) for c in chosen)
    assert f"{expect} directed question pairs" in out and expect <= 4 * n_attr * (n_attr - 1)


# ---------------------------------------------------------------------------
# attr_head_trace's window, picked from the sweep by a fixed rule
# ---------------------------------------------------------------------------

# ATTRIBUTE_SWITCH_FINDINGS.md section 1, flags, donor_first by BLOCK (layer = block + 1).
FLAGS_QUESTION = {**{b: 1.0 for b in range(0, 16)}, 16: .988, 17: .786, 18: .286, 19: .131, 20: .024,
                  **{b: .024 for b in range(21, 28)}}
FLAGS_LAST = {**{b: .024 for b in range(0, 17)}, 17: .131, 18: .50, 19: .655, 20: .976,
              **{b: 1.0 for b in range(21, 28)}}


def test_pick_window_reproduces_the_hand_picked_flags_window():
    from methods.attribute_switch_report import pick_window
    cross = [{"layer": b + 1, "question_residual": FLAGS_QUESTION[b], "last_residual": FLAGS_LAST[b]}
             for b in range(28)]
    assert pick_window(cross) == (15, list(range(15, 23)))
    with pytest.raises(ValueError, match="no handoff"):
        pick_window([{**c, "last_residual": .02} for c in cross])
    with pytest.raises(ValueError, match="--scopes"):
        pick_window([{**c, "question_residual": None} for c in cross])


def test_report_baselines_use_own_correct_for_own_answer_rows(tmp_path):
    from methods.attribute_switch_report import baselines
    def rec(i, arm, ok, full=True):
        return {"row_index": i, "arm": arm, "base_attribute": "hq_country", "own_correct": ok,
                "base_score": {"full_match": full, "first_token": full},
                "source_score": {"full_match": full, "first_token": full}}
    lines = [rec(0, "clean", True), rec(0, "donor_clean", True), rec(1, "clean", False), rec(1, "donor_clean", True)]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (tmp_path / "config.json").write_text(json.dumps({"rows": [0, 1]}))
    b = baselines(tmp_path)
    assert b["eligible"] == 1 and b["clean_by_attribute"]["hq_country"]["full"] == 0.5
