"""X11: vrh_ablation --ablation mean (offline, CPU)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")

from methods import vrh_ablation as vrh  # noqa: E402
from methods.head_swap_vade import COMMON10, parse_heads  # noqa: E402

MOVERS = parse_heads(COMMON10)
PUBLISHED = {"park_vrh": {"heads": [(14, 1), (15, 2), (16, 3)]}}


def test_default_arm_list_unchanged():
    old = vrh.build_arms(["park_vrh"], PUBLISHED, MOVERS, 2, 28, 0, n_random_movers=3)
    assert vrh.build_arms(["park_vrh"], PUBLISHED, MOVERS, 2, 28, 0, n_random_movers=3,
                          ablations=("zero",)) == old
    assert not any(n.startswith("mean:") for n, _ in old)


def test_mean_arms_reuse_the_zero_draws_and_come_last():
    arms = vrh.build_arms(["park_vrh"], PUBLISHED, MOVERS, 2, 28, 0, n_random_movers=3,
                          ablations=("zero", "mean"))
    d = dict(arms)
    assert d["mean:common10"] == MOVERS
    for i in range(3):
        assert d[f"mean:common10:random{i}"] == d[f"zero:common10:random{i}"]
    names = [n for n, _ in arms]
    assert names.index("mean:common10") > max(i for i, n in enumerate(names) if n.startswith("zero:"))
    only_mean = [n for n, _ in vrh.build_arms(["park_vrh"], PUBLISHED, MOVERS, 2, 28, 0, n_random_movers=3,
                                              ablations=("mean",))]
    assert "zero:common10" not in only_mean and "mean:common10" in only_mean
    assert "zero:park_vrh" in only_mean      # the Park arms are not affected


def test_position_means_left_padding_and_steps():
    pm = vrh.PositionMeans()
    W, P = 4, 3
    a = torch.arange(P * W, dtype=torch.float32).view(P, W)
    b = a + 10
    pad = torch.full((2, W), 999.0)
    pm.add_prefill("k", torch.cat([pad, a]), P, n_steps=2)      # left-padded row
    pm.add_prefill("k", b, P, n_steps=2)
    pm.add_step("k", torch.ones(W), 1)
    pm.add_step("k", 3 * torch.ones(W), 1)
    pm.finalize()
    assert torch.allclose(pm.prefill("k"), (a + b) / 2)        # pad columns never enter
    assert torch.allclose(pm.step("k", 1), 2 * torch.ones(W))
    assert torch.allclose(pm.step("k", 2), 2 * torch.ones(W))   # unseen step -> last seen
    assert torch.allclose(pm.step("k", 50), 2 * torch.ones(W))  # clamped


def test_position_means_rejects_a_length_change():
    pm = vrh.PositionMeans()
    pm.add_prefill("k", torch.zeros(3, 2), 3, 1)
    with pytest.raises(AssertionError):
        pm.add_prefill("k", torch.zeros(4, 2), 4, 1)


class _Attn(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.o_proj = torch.nn.Linear(w, w, bias=False)


class _Adapter:
    def __init__(self, mods):
        self.mods = mods

    def get_attn_head_output_module(self, model, b):
        return self.mods[b]


def test_register_mean_replaces_only_the_heads_columns():
    head_dim, n_heads = 2, 3
    W = head_dim * n_heads
    mods = {21: torch.nn.Linear(W, W, bias=False)}
    adapter = _Adapter(mods)
    pm = vrh.PositionMeans()
    clean = torch.randn(2, 4, W)
    means = {21: pm}
    h = vrh.register_mean_capture(adapter, None, [21], means, ["k", "k"], [4, 4], 2)
    mods[21](clean)
    for t in range(2):
        mods[21](torch.randn(2, 1, W))
    for x in h:
        x.remove()
    pm.finalize()
    seen = {}
    hs = vrh.register_mean(adapter, None, [(21, 1)], head_dim, means, ["k", "k"], [4, 4])
    # registered AFTER the mean hook: pre-hooks run in registration order
    hs.append(mods[21].register_forward_pre_hook(lambda m, a: seen.__setitem__(len(seen), a[0].clone())))
    x = torch.randn(2, 4, W)
    mods[21](x)
    got = seen[0]
    cols = [2, 3]
    keep = [0, 1, 4, 5]
    assert torch.allclose(got[:, :, keep], x[:, :, keep])
    assert torch.allclose(got[0, :, cols], clean.mean(0)[:, cols])
    assert torch.allclose(got[1, :, cols], clean.mean(0)[:, cols])
    for x in hs:
        x.remove()
