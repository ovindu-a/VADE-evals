"""Offline checks for text_entity_contrast.py's `crossmodal` phase (image vs text mover content)."""
import torch

from test_head_followups import tiny                                    # noqa: F401
from test_tier1_circuit import wrapped
from methods import text_entity_contrast as tec


def test_retrieval_finds_same_entity_through_a_modality_offset():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(40, 16, generator=g)
    B = A + 5.0 + 0.05 * torch.randn(40, 16, generator=g)     # big offset, small noise
    groups = ["capital"] * 20 + ["language"] * 20
    r = tec.retrieval(A, B, groups)
    assert r["n"] == 40 and r["top1"] == 1.0 and r["mean_rank"] == 1.0
    assert abs(r["chance_top1"] - 1 / 20) < 1e-9
    assert r["same_cos"] > 0.9 and abs(r["diff_cos"]) < 0.2


def test_retrieval_is_near_chance_for_unrelated_conditions():
    g = torch.Generator().manual_seed(1)
    A, B = torch.randn(200, 32, generator=g), torch.randn(200, 32, generator=g)
    r = tec.retrieval(A, B, ["x"] * 200)
    assert r["top1"] < 0.05 and 60 < r["mean_rank"] < 140


def test_retrieval_scores_identical_inputs_as_chance_not_perfect():
    A = torch.ones(10, 4)                                      # centered -> all zeros, every cosine ties
    r = tec.retrieval(A, A.clone(), ["g"] * 10)
    assert r["top1"] == 0.0 and r["mean_rank"] == 5.5


def test_ridge_retrieval_on_identical_inputs_reads_chance_instead_of_raising():
    # Layer 0 on the pod: every row's last-column residual is the same token embedding.
    A = torch.ones(40, 8) * 3.0
    B = torch.ones(40, 8) * -1.0
    ents = list(range(20)) * 2
    groups = ["a"] * 20 + ["b"] * 20
    for r in (tec.ridge_retrieval(A, B, ents, groups), tec.ridge_retrieval(B, A, ents, groups)):
        assert r["n"] == 40 and r["top1"] == 0.0
        assert abs(r["mean_rank"] - 5.5) < 1e-9                # 10 test rows per group, all tied


def test_ridge_retrieval_recovers_a_rotation_on_held_out_entities():
    g = torch.Generator().manual_seed(2)
    n_ent, d = 60, 8
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    base = torch.randn(n_ent, d, generator=g)
    A = torch.cat([base, base + 1.0])                          # two attributes per entity
    B = A @ Q + 3.0
    ents = list(range(n_ent)) * 2
    groups = ["a"] * n_ent + ["b"] * n_ent
    assert tec.retrieval(A, B, groups)["top1"] < 0.5           # a rotation defeats raw cosine...
    r = tec.ridge_retrieval(A, B, ents, groups, alpha=1e-3)
    assert r["n"] == 2 * n_ent and r["top1"] > 0.95            # ...but not a held-out linear map


def test_ridge_retrieval_never_trains_on_a_test_entity(monkeypatch):
    seen = []
    real = tec.retrieval

    def spy(A, B, groups):
        seen.append(A.shape[0])
        return real(A, B, groups)
    monkeypatch.setattr(tec, "retrieval", spy)
    A = torch.randn(30, 4)
    tec.ridge_retrieval(A, A.clone(), [i % 10 for i in range(30)], ["g"] * 30, folds=2)
    assert sum(seen) == 30                                     # every row tested exactly once


def test_norm_match_rescales_only_listed_heads():
    hd = 4
    zd = {3: torch.randn(2, 1, 3 * hd)}
    zr = {3: torch.randn(2, 1, 3 * hd)}
    out = tec.norm_match(zd, zr, [(3, 1)], hd)
    sl = slice(hd, 2 * hd)
    torch.testing.assert_close(out[3][..., sl].norm(dim=-1), zr[3][..., sl].norm(dim=-1))
    cos = torch.nn.functional.cosine_similarity(out[3][..., sl], zd[3][..., sl], dim=-1)
    torch.testing.assert_close(cos, torch.ones_like(cos))      # direction kept
    torch.testing.assert_close(out[3][..., :hd], zd[3][..., :hd])
    assert out[3] is not zd[3]                                  # donor left untouched


def test_crossmodal_capture_shapes_and_write(tiny):
    m, _, batch = wrapped(tiny)
    ids = batch["base_input_ids"]
    T = ids.shape[1]
    side = {"ids": ids, "mask": torch.ones_like(ids), "extra": batch["base_extra"],
            "readout": torch.tensor([[T - 1]]), "entity": torch.tensor([[1, 2]])}
    movers = [(1, 0), (2, 1)]
    z, write, resid, ent = tec.crossmodal_capture(m, side, movers)
    assert z.shape == (1, 2, m.head_dim)
    assert write.shape == (1, m.hidden)
    assert resid.shape == ent.shape == (1, m.n_layers, m.hidden)
    hs = m.forward(ids, side["mask"], side["extra"], output_hidden_states=True).hidden_states
    torch.testing.assert_close(resid[:, 3].float(), hs[3][:, T - 1].half().float())
    torch.testing.assert_close(ent[:, 3].float(), hs[3][:, 1:3].float().mean(1).half().float())
    zfull, _ = tec.head_capture(m, side, [1, 2])
    W1 = m.adapter.get_attn_head_output_module(m.model, 1).weight[:, :m.head_dim].float()
    W2 = m.adapter.get_attn_head_output_module(m.model, 2).weight[:, m.head_dim:2 * m.head_dim].float()
    want = zfull[1][:, -1, :m.head_dim].float() @ W1.T + zfull[2][:, -1, m.head_dim:2 * m.head_dim].float() @ W2.T
    torch.testing.assert_close(write.float(), want.half().float())
