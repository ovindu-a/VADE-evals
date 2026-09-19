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
