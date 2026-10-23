"""PROMPT_STYLES: `default` must reproduce the controlled prompts every earlier run used."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods import attribute_switch_sweep as sw  # noqa: E402
from methods import circuit_common as cc  # noqa: E402
from methods import prompt_style_check  # noqa: E402,F401


def test_default_style_is_the_existing_prompt():
    for a in sw.ALL_ATTRIBUTES:
        assert sw.style_question(a) == sw.question(a)
        assert sw.style_prefill(a) == sw.PREFILL
        for name in ("France", "Apple Inc."):
            assert sw.style_question(a, text_name=name) == cc.text_question(a, name)


def test_every_style_formats_for_every_attribute():
    for style in sw.PROMPT_STYLES:
        for a in sw.ALL_ATTRIBUTES:
            q, p = sw.style_question(a, style), sw.style_prefill(a, style)
            assert "{" not in q + p and q and p


def test_check_script_dry_run(capsys):
    vade = os.environ.get("VADE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "VADE"))
    if not os.path.isdir(os.path.join(vade, "data", "flags")):
        import pytest
        pytest.skip("no sibling VADE checkout")
    prompt_style_check.main(["--dry_run", "--vade_root", vade])
    assert "Grid valid." in capsys.readouterr().out
