"""Offline checks for methods/vade_accuracy_sweep.py: VADE's accuracy sweep run unmodified,
with --processor_kwargs injected from this repo. Skips without a sibling VADE checkout."""
import json
import os

import pytest
import torch
from PIL import Image

from methods import vade_accuracy_sweep as vas

VADE_SWEEP = os.path.join(vas.DEFAULT_VADE_ROOT, "models", "run_accuracy_sweep.py")
needs_vade = pytest.mark.skipif(not os.path.exists(VADE_SWEEP), reason="no sibling VADE checkout")


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append(kw)
        return FakeInputs(input_ids=torch.zeros(len(messages), 5, dtype=torch.long))

    def batch_decode(self, toks, skip_special_tokens=True):
        return [f"ans{i}" for i in range(len(toks))]


class FakeModel:
    device = "cpu"

    def generate(self, input_ids, **kw):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1] + 3, dtype=torch.long)


@needs_vade
def test_loading_vade_sweep_does_not_patch_anything_by_itself():
    mod = vas.load_vade_sweep(vas.DEFAULT_VADE_ROOT)
    assert mod.run_batch.__module__ == mod.__name__
    assert callable(mod.main) and callable(mod.already_done_keys)


@needs_vade
def test_installed_run_batch_passes_the_dict_with_padding_inside_processor_kwargs(tmp_path):
    mod = vas.load_vade_sweep(vas.DEFAULT_VADE_ROOT)
    vas.install(mod, {"do_image_splitting": False})
    img = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(img)
    batch = [{"image_path": str(img), "question": "Q?", "prefill": "A:"}] * 2
    proc = FakeProcessor()
    out = mod.run_batch(FakeModel(), proc, batch, max_new_tokens=3, pad_token_id=0)
    assert out == ["ans0", "ans1"]
    (kw,) = proc.calls
    assert kw["processor_kwargs"] == {"padding": True, "do_image_splitting": False}
    assert "padding" not in kw          # a top-level padding would REPLACE processor_kwargs (transformers 5.16)
    assert kw["continue_final_message"] and not kw["add_generation_prompt"]


def test_sidecar_is_written_and_a_mismatched_resume_is_refused(tmp_path):
    out = str(tmp_path / "flags_predictions.jsonl")
    vas.check_processor_kwargs(out, {"do_image_splitting": False})
    assert json.load(open(vas.sidecar_path(out))) == {"do_image_splitting": False}
    open(out, "w").write("{}\n")
    vas.check_processor_kwargs(out, {"do_image_splitting": False})            # same dict: resume ok
    with pytest.raises(SystemExit):
        vas.check_processor_kwargs(out, {"do_image_splitting": True})


def test_existing_file_without_sidecar_refuses_a_dict(tmp_path):
    out = str(tmp_path / "brands_predictions.jsonl")
    open(out, "w").write("{}\n")                                               # made by plain VADE
    with pytest.raises(SystemExit):
        vas.check_processor_kwargs(out, {"do_image_splitting": False})
    vas.check_processor_kwargs(out, None)                                      # unset: no check, no file
    assert not os.path.exists(vas.sidecar_path(out))


@needs_vade
def test_main_strips_its_own_flags_and_runs_vade_main(monkeypatch):
    seen = {}
    real_load = vas.load_vade_sweep

    def fake_load(root):
        mod = real_load(root)
        mod.main = lambda: seen.update(argv=list(vas.sys.argv[1:]), run_batch=mod.run_batch.__qualname__)
        return mod
    monkeypatch.setattr(vas, "load_vade_sweep", fake_load)
    monkeypatch.setattr(vas.sys, "argv", ["x"])
    vas.main(["--model_id", "M", "--processor_kwargs", '{"do_image_splitting": false}', "--entities", "flags"])
    assert seen["argv"] == ["--model_id", "M", "--entities", "flags"]
    assert "make_run_batch" in seen["run_batch"]
    vas.main(["--model_id", "M"])
    assert seen["argv"] == ["--model_id", "M"] and seen["run_batch"] == "run_batch"
