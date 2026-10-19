"""The cross-prompt classifier decides the whole result, so it is tested alone.

Four golds per row, several of which COINCIDE in three of the four cells. Getting
that collapse wrong would report a no-op as an entity transfer.
"""
import pytest

from methods.head_cross import build_rows, classify

TRUTH = {
    "AO": {"capital": "Luanda", "language": "Portuguese", "currency": "AOA", "calling_code": "244"},
    "AR": {"capital": "Buenos Aires", "language": "Spanish", "currency": "ARS", "calling_code": "54"},
    "IR": {"capital": "Tehran", "language": "Persian", "currency": "IRR", "calling_code": "98"},
}


def match(text, label):
    """Stand-in with VADE's whole-word semantics, so 'Iran' cannot hit 'Ireland'."""
    import re
    return re.search(r"\b" + re.escape(label.casefold()) + r"\b", text.casefold()) is not None


def row(base="AO", donor="AR", q1="capital", q2="language", cell="both"):
    return {"base": base, "donor_flag": donor, "q1": q1, "q2": q2, "cell": cell}


def test_build_rows_covers_the_full_2x2():
    rows = build_rows(TRUTH, ["capital", "language"], n_pairs=2, seed=0)
    from collections import Counter
    c = Counter(r["cell"] for r in rows)
    # 2 pairs x 2x2 questions x 2 donors; per pair: 2 self, 2 flag, 2 question, 2 both
    assert c == {"self": 4, "flag": 4, "question": 4, "both": 4}
    for r in rows:
        same_flag, same_q = r["donor_flag"] == r["base"], r["q1"] == r["q2"]
        expect = ("self" if same_flag and same_q else "question" if same_flag
                  else "flag" if same_q else "both")
        assert r["cell"] == expect


@pytest.mark.parametrize("text,expect", [
    ("The capital city is Luanda.", "base_q1"),        # patch did nothing
    ("The capital city is Buenos Aires.", "source_q1"),  # ENTITY transferred
    ("The capital city is Spanish.", "source_q2"),     # ATTRIBUTE content transferred
    ("The capital city is Portuguese.", "base_q2"),    # question transferred, entity did not
    ("The capital city is Tehran.", "other"),
])
def test_the_four_outcomes_are_distinguished(text, expect):
    assert classify(text, row(), TRUTH, match) == expect


def test_coinciding_golds_are_collapsed_not_called_ambiguous():
    """In `self` all four golds are the same string; in `flag` they pair up. A
    naive multi-hit check would flag every one of those rows ambiguous."""
    assert classify("The capital city is Luanda.", row(donor="AO", q2="capital", cell="self"),
                    TRUTH, match) == "base_q1"
    # flag cell: q1 == q2, so base_q1 == base_q2 and source_q1 == source_q2
    assert classify("The capital city is Buenos Aires.", row(q2="capital", cell="flag"),
                    TRUTH, match) == "source_q1"
    # question cell: base == donor, so base_q1 == source_q1
    assert classify("The capital city is Portuguese.", row(donor="AO", cell="question"),
                    TRUTH, match) == "source_q2"


def test_two_genuinely_different_golds_in_one_generation_is_ambiguous():
    assert classify("The capital city is Buenos Aires, where they speak Spanish.",
                    row(), TRUTH, match) == "ambiguous"


def test_whole_word_matching_is_used():
    """A substring match would credit 'AOA' inside 'AOAX' and quietly inflate
    whichever label happened to be a prefix."""
    assert classify("The currency code is AOAX.", row(q1="currency", q2="currency", donor="AO",
                                                      cell="self"), TRUTH, match) == "other"


# --gold own (brands/celebrities: stored labels are lowercase, the model adds articles)

from methods.head_cross import GOLD_KEYS, classify_own, informative, norm_answer, own_keys  # noqa: E402

OWN = {("A", "hq"): " the United States.", ("B", "hq"): " Italy.", ("A", "year"): " 1968.",
       ("B", "year"): " 1963.", ("C", "hq"): " The United States", ("D", "hq"): " Japan."}
VALID = {k: True for k in OWN}


def orow(base="A", donor="B", q1="hq", q2="year", cell="both"):
    return {"base": base, "donor_flag": donor, "q1": q1, "q2": q2, "cell": cell}


def test_norm_answer_ignores_case_space_and_trailing_punctuation_only():
    assert norm_answer(" the United States.") == norm_answer("The  United States") == "the united states"
    assert norm_answer(" the UK.") != norm_answer(" the United Kingdom.")


def test_classify_own_distinguishes_the_four_outcomes_and_other():
    assert classify_own(" the United States.", orow(), OWN, VALID) == "base_q1"
    assert classify_own(" Italy.", orow(), OWN, VALID) == "source_q1"
    assert classify_own(" 1963.", orow(), OWN, VALID) == "source_q2"
    assert classify_own(" 1968.", orow(), OWN, VALID) == "base_q2"
    assert classify_own(" France.", orow(), OWN, VALID) == "other"


def test_classify_own_collapses_coinciding_golds_and_excludes_invalid_ones():
    # self cell: all four golds are A's own hq answer
    assert classify_own(" the United States.", orow(donor="A", q2="hq", cell="self"), OWN, VALID) == "base_q1"
    # two US brands: base_q1 and source_q1 are the same answer -> collapsed, not ambiguous
    assert classify_own(" the United States.", orow(donor="C", q2="hq", cell="flag"), OWN, VALID) == "base_q1"
    bad = {**VALID, ("B", "year"): False}
    assert classify_own(" Italy.", orow(), OWN, bad) == "excluded"       # needs B's year too
    assert classify_own(" Italy.", orow(q2="hq", cell="flag"), OWN, bad) == "source_q1"


def test_informative_requires_different_answers_to_q1():
    gold = OWN.__getitem__
    assert informative(orow(), gold)
    assert not informative(orow(donor="C"), gold)                         # both US


def test_own_keys_cover_every_gold_a_row_needs():
    keys = own_keys([orow(), orow(base="D", donor="A", q2="hq", cell="flag")])
    assert keys == sorted({("A", "hq"), ("B", "hq"), ("B", "year"), ("A", "year"), ("D", "hq")})
    assert len(GOLD_KEYS) == 4
