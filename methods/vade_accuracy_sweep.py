"""VADE's models/run_accuracy_sweep.py, run unmodified, plus --processor_kwargs.

Some models' default image pipeline differs from the input this repo's adapter feeds them.
Idefics3 tiles a 336 px canvas into 4x4 crops of an upscaled image unless
`do_image_splitting` is off, while adapters/idefics3.py uses ONE 364 px tile. Pruning
(VADE's prune_tuples.py, which reads this sweep's predictions) must judge the model on the
same input the experiments use.

VADE is a separate repository and is not edited from here. This wrapper loads its
run_accuracy_sweep.py by path, replaces the module's `run_batch` with one that passes the
dict to the processor (through apply_chat_template's processor_kwargs), and calls the
module's own main() with every other argument unchanged. VADE's main() looks `run_batch`
up as a module global at call time, so the swap reaches every batch. With no
--processor_kwargs, nothing is replaced, and the run is exactly VADE's.

The dict is recorded in <out_dir>/<entity>_predictions.processor_kwargs.json. A resume into
a predictions file made under a different dict is refused, rather than mixing two image
pipelines in one file.

Usage (any VADE argument passes through):
    python methods/vade_accuracy_sweep.py --model_id HuggingFaceM4/Idefics3-8B-Llama3 \\
        --processor_kwargs '{"do_image_splitting": false}' --entities flags brands
"""
import argparse
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VADE_ROOT = os.environ.get("VADE_ROOT") or os.path.normpath(os.path.join(REPO_ROOT, "..", "VADE"))


def load_vade_sweep(vade_root):
    """-> VADE's run_accuracy_sweep module, loaded from its file (VADE itself untouched)."""
    path = os.path.join(vade_root, "models", "run_accuracy_sweep.py")
    for d in (os.path.dirname(path), os.path.join(vade_root, "eval"), vade_root):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("vade_run_accuracy_sweep", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sidecar_path(out_path):
    return out_path[:-len(".jsonl")] + ".processor_kwargs.json"


def check_processor_kwargs(out_path, processor_kwargs):
    """Refuse to append to a predictions file made under a different dict; record the dict
    next to the file when one is set (nothing is written when it is unset)."""
    sidecar = sidecar_path(out_path)
    want = processor_kwargs or {}
    have = json.load(open(sidecar)) if os.path.exists(sidecar) else {}
    if os.path.exists(out_path) and have != want:
        raise SystemExit(f"{out_path} was generated with processor_kwargs={have}, this run has {want}: "
                         f"use another --out_dir rather than mixing two image pipelines in one file")
    if want:
        with open(sidecar, "w") as f:
            json.dump(want, f, indent=2, sort_keys=True)


def make_run_batch(mod, processor_kwargs):
    """VADE's run_batch with `processor_kwargs` passed to the processor.

    `padding` moves INTO processor_kwargs: apply_chat_template REPLACES an explicit
    processor_kwargs with any processor kwargs found in **kwargs (transformers 5.16), so
    passing both would silently drop the dict."""
    def run_batch(model, processor, batch, max_new_tokens, pad_token_id):
        import torch
        from PIL import Image
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        messages_batch = [mod.build_messages(img, batch[0]["question"], batch[0]["prefill"]) for img in images]
        inputs = processor.apply_chat_template(
            messages_batch, tokenize=True, add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True, **processor_kwargs},
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad_token_id)
        return processor.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
    return run_batch


def install(mod, processor_kwargs):
    """Patch `mod` in place: run_batch passes the dict; already_done_keys (called once per
    entity, with that entity's predictions path, right before generating) checks the sidecar
    first."""
    mod.run_batch = make_run_batch(mod, processor_kwargs)
    done_keys = mod.already_done_keys

    def already_done_keys(out_path):
        check_processor_kwargs(out_path, processor_kwargs)
        return done_keys(out_path)
    mod.already_done_keys = already_done_keys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--vade_root", default=DEFAULT_VADE_ROOT)
    ap.add_argument("--processor_kwargs", type=json.loads, default=None)
    known, rest = ap.parse_known_args(argv)
    mod = load_vade_sweep(known.vade_root)
    if known.processor_kwargs:
        install(mod, known.processor_kwargs)
        print(f"[vade_accuracy_sweep] processor_kwargs={known.processor_kwargs} (VADE's run_batch replaced)")
    sys.argv = [mod.__file__] + rest
    mod.main()


if __name__ == "__main__":
    main()
