"""Where every method writes its outputs. Deliberately NOT under models/ --
models/<model_slug>/ has its own git convention (Qwen2.5-VL-7B-Instruct is
checked in as a worked example; see VADE/.gitignore), so anything written
there risks getting committed. results/ is entirely gitignored: run
artifacts (checkpoints, predictions, train logs) are meant to be copied off
a pod/box by hand, never committed. See run_logging.py for the SEPARATE
logs/ tree (live-progress logs for monitoring a run, not for copying off).
"""
import os

RESULTS_DIRNAME = "results"
LOGS_DIRNAME = "logs"


def _tagged_dir(dirname, vade_root, model_slug, entity, method, attribute, config_tag):
    return os.path.join(vade_root, dirname, model_slug, entity, method, attribute, config_tag)


def results_dir(vade_root, model_slug, entity, method, attribute, config_tag):
    """results/<model_slug>/<entity>/<method>/<attribute>/<config_tag>/ --
    config_tag is whatever hyperparameters distinguish one run of `method`
    from another (DAS: subspace dim + position set, e.g. 'K512_flagring1')."""
    return _tagged_dir(RESULTS_DIRNAME, vade_root, model_slug, entity, method, attribute, config_tag)


def logs_dir(vade_root, model_slug, entity, method, attribute, config_tag):
    """logs/<model_slug>/<entity>/<method>/<attribute>/<config_tag>/ --
    same shape as results_dir, so a run's log sits right next to its
    checkpoint/predictions by config, just under the separate logs/ root."""
    return _tagged_dir(LOGS_DIRNAME, vade_root, model_slug, entity, method, attribute, config_tag)


def das_config_tag(subspace_dim, positions, pruned=False):
    return f"K{subspace_dim}_{positions}" + ("_pruned" if pruned else "")
