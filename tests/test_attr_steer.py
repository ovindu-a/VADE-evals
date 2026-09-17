"""Direction steering: the patch arithmetic, checked where the answer is known.

The generation and attention phases need a model; the parts that can silently be
wrong without erroring are the per-lane direction algebra and the image-column
selection, and those are exact.
"""
import pytest
import torch

from test_head_followups import tiny
from methods.attr_steer import build_steer_patch, head_attention, image_columns
from methods.common.sites import RESIDUAL_SITE

ATTRS = ['capital', 'currency']
H = 32


def fixture(rows_spec, positions):
    rows = [{'row_index': i, 'base_attribute': b, 'donor_attribute': d}
            for i, (b, d) in enumerate(rows_spec)]
    return rows, {'rows': rows, 'steer_positions': positions}


def apply(patch, x):
    """Run the hook the way a forward pass would: one multi-token tensor."""
    _, _, fn = patch[0]
    return fn(x)


def test_add_is_exactly_the_mean_difference_per_lane():
    torch.manual_seed(0)
    means = {3: {'capital': torch.randn(H), 'currency': torch.randn(H)}}
    rows, batch = fixture([('capital', 'currency'), ('currency', 'capital')],
                          torch.tensor([[2], [2]]))
    x = torch.zeros(2, 4, H)
    out = apply(build_steer_patch(means, batch, 3, ATTRS, 'add', 1.0, 'cpu'), x)
    # lane 0 gets +(currency - capital); lane 1 gets the exact negation.
    torch.testing.assert_close(out[0, 2], means[3]['currency'] - means[3]['capital'])
    torch.testing.assert_close(out[1, 2], means[3]['capital'] - means[3]['currency'])
    # and only at the steered column
    assert not out[:, [0, 1, 3]].any()


def test_alpha_scales_and_zero_is_a_no_op():
    torch.manual_seed(1)
    means = {2: {'capital': torch.randn(H), 'currency': torch.randn(H)}}
    rows, batch = fixture([('capital', 'currency')], torch.tensor([[1]]))
    x = torch.randn(1, 3, H)
    torch.testing.assert_close(apply(build_steer_patch(means, batch, 2, ATTRS, 'add', 0.0, 'cpu'), x), x)
    a1 = apply(build_steer_patch(means, batch, 2, ATTRS, 'add', 1.0, 'cpu'), x)
    a2 = apply(build_steer_patch(means, batch, 2, ATTRS, 'add', 2.0, 'cpu'), x)
    torch.testing.assert_close(a2[0, 1] - x[0, 1], 2 * (a1[0, 1] - x[0, 1]))


def test_project_installs_the_donor_coordinate_and_discards_the_row_s_own():
    """The defining property: whatever the row had along the axis is gone, and
    the coordinate that remains is the donor mean's -- regardless of the input."""
    torch.manual_seed(2)
    means = {1: {'capital': torch.randn(H), 'currency': torch.randn(H)}}
    rows, batch = fixture([('capital', 'currency')], torch.tensor([[0]]))
    d = means[1]['currency'] - means[1]['capital']
    u = d / d.norm()
    grand = (means[1]['capital'] + means[1]['currency']) / 2
    target = (means[1]['currency'] - grand) @ u
    for _ in range(3):
        x = torch.randn(1, 2, H) * 10
        out = apply(build_steer_patch(means, batch, 1, ATTRS, 'project', 1.0, 'cpu'), x)
        torch.testing.assert_close(out[0, 0] @ u, target, rtol=1e-4, atol=1e-4)
        # the component ORTHOGONAL to the axis must be untouched
        torch.testing.assert_close(out[0, 0] - (out[0, 0] @ u) * u,
                                   x[0, 0] - (x[0, 0] @ u) * u, rtol=1e-4, atol=1e-4)
    # project is idempotent; add is not
    once = apply(build_steer_patch(means, batch, 1, ATTRS, 'project', 1.0, 'cpu'), x)
    twice = apply(build_steer_patch(means, batch, 1, ATTRS, 'project', 1.0, 'cpu'), once)
    torch.testing.assert_close(once, twice, rtol=1e-4, atol=1e-4)


def test_random_direction_overrides_every_lane_identically():
    """The null must be the SAME vector for every lane -- a per-lane random
    direction would be a different (easier) control."""
    torch.manual_seed(3)
    means = {2: {'capital': torch.randn(H), 'currency': torch.randn(H)}}
    rows, batch = fixture([('capital', 'currency'), ('currency', 'capital')], torch.tensor([[1], [1]]))
    v = torch.randn(H)
    x = torch.zeros(2, 3, H)
    out = apply(build_steer_patch(means, batch, 2, ATTRS, 'add', 1.0, 'cpu', random_direction=v), x)
    torch.testing.assert_close(out[0, 1], v)
    torch.testing.assert_close(out[1, 1], v)


def test_image_columns_found_and_disagreement_rejected():
    ids = torch.tensor([[1, 2, 2, 3, 9], [1, 2, 2, 3, 9]])
    batch = {'base_input_ids': ids}
    assert image_columns(batch, 2).tolist() == [1, 2]
    bad = torch.tensor([[1, 2, 2, 3, 9], [2, 2, 2, 3, 9]])
    with pytest.raises(AssertionError, match='image span'):
        image_columns({'base_input_ids': bad}, 2)
    with pytest.raises(AssertionError, match='no image tokens'):
        image_columns({'base_input_ids': torch.tensor([[1, 3, 9]])}, 2)


def test_head_attention_reads_a_distribution_and_the_patch_moves_it(tiny):
    runner, batch = tiny
    adapter, model = runner.adapter, runner.model
    ids = batch['base_input_ids']
    b = {'base_input_ids': ids, 'attention_mask': torch.ones_like(ids),
         'base_extra': batch['base_extra'],
         'rows': [{'row_index': 0, 'base_attribute': 'capital', 'donor_attribute': 'currency'}],
         'steer_positions': torch.tensor([[ids.shape[1] - 1]])}
    cols = image_columns(b, 2)
    heads = [(2, 0), (3, 1)]
    clean = head_attention(adapter, model, b, heads, cols)
    for hk in heads:
        assert clean[hk].shape == (1, cols.numel())
        assert (clean[hk] >= 0).all() and clean[hk].sum() <= 1.0 + 1e-4   # part of a softmax row
    means = {1: {'capital': torch.zeros(runner.hidden), 'currency': torch.randn(runner.hidden) * 5}}
    steered = head_attention(adapter, model, b, heads, cols,
                             patches=build_steer_patch(means, b, 1, ATTRS, 'add', 1.0, model.device))
    assert any((steered[hk] - clean[hk]).abs().max() > 1e-6 for hk in heads), \
        'a large residual edit below these blocks left every image-head attention pattern identical'
