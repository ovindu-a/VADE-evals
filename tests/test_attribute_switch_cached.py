"""Cached/batched interventions must match the full-prefix reference VLM."""
import pytest
import torch

from test_head_followups import tiny
from test_attribute_switch_sweep import setup
from methods.attribute_switch_sweep import alignment
from methods.attribute_switch_cached import CachedPrefill


def spatial_setup(tiny):
    runner, batch = setup(tiny)
    ids = torch.tensor([[1, 2, 2, 2, 2, 3, 12, 13, 14]])
    batch['base_input_ids'] = ids
    batch['source_input_ids'] = ids.clone()
    batch['source_input_ids'][0, 7] = 15
    batch['paraphrase_ids'] = ids.clone()
    batch['paraphrase_ids'][0, 6] = 16
    batch['earlier_positions'] = alignment(ids, batch['source_input_ids'], 2, [1, 3])
    batch['base_extra'] = {'pixel_values': torch.randn(16, 24),
                           'image_grid_thw': torch.tensor([[1, 4, 4]])}
    return runner, batch


def arm(site, layer=3, control='switch', mode='prefill'):
    return (f'{site}/L{layer}/{mode}/{control}', site, layer, mode, control)


@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_every_site_scope_and_control_matches_reference_per_token(tiny, batch_size):
    runner, batch = spatial_setup(tiny)
    arms = [arm(f'{scope}/{site}', 4, control)
            for scope in ['earlier_text', 'last_token', 'all_text']
            for site in ['residual', 'attention', 'mlp', 'joint', 'blocks:3', 'attention_blocks:3']
            for control in ['switch', 'self', 'paraphrase']]
    # Include heterogeneous layers within the same chunk, not only donor kinds.
    arms += [arm('all_text/residual', 1), arm('last_token/attention_blocks:2', 2),
             arm('earlier_text/joint', 3)]
    cached = CachedPrefill(runner, batch, arms)
    for start in range(0, len(arms), batch_size):
        chunk = arms[start:start + batch_size]
        outputs = cached.generate(chunk, verify=True)
        assert len(outputs) == len(chunk)
        assert all(len(tokens) == runner.args.max_new_tokens for tokens in outputs)
    assert all(value.device.type == 'cpu' for value in cached.donors.values())
    assert all(not module._forward_hooks for module in runner.model.modules())


def test_donors_reused_and_only_the_shared_prefix_encodes_images(tiny):
    runner, batch = spatial_setup(tiny)
    arms = [arm('all_text/attention', 2, c) for c in ['switch', 'self', 'paraphrase']]
    cached = CachedPrefill(runner, batch, arms)
    assert cached.prefix_length == 6            # base/source differ at 7, base/paraphrase at 6
    calls, vision_calls = [], []
    def observe(module, args, kwargs):
        calls.append((tuple(kwargs['input_ids'].shape), 'pixel_values' in kwargs,
                      kwargs.get('past_key_values') is not None))
    h = runner.model.register_forward_pre_hook(observe, with_kwargs=True)
    v = runner.model.model.visual.register_forward_hook(lambda *a: vision_calls.append(True))
    try:
        first = cached.generate(arms)
        second = cached.generate(arms)
    finally:
        h.remove()
        v.remove()
    assert first == second
    # The image is inside the shared prefix, so the vision tower runs ONCE for the whole row --
    # not once per donor and once per recipient chunk, which is what it cost before.
    assert len(vision_calls) == 1
    assert sum(has_pixels for _, has_pixels, _ in calls) == 1
    prefix = [c for c in calls if c[1]]
    assert prefix[0] == ((3, cached.prefix_length), True, False)
    # Every later forward is a suffix or a decode step: no pixels, and a cache underneath it.
    assert all(has_cache for _, has_pixels, has_cache in calls if not has_pixels)
    suffix_len = batch['base_input_ids'].shape[1] - cached.prefix_length
    tails = [shape for shape, pixels, _ in calls if not pixels and shape[1] == suffix_len]
    assert len(tails) == 5          # 3 donors (once, reused) + 2 recipient prefills
    incremental = [shape for shape, pixels, _ in calls if not pixels and shape[1] == 1]
    assert len(incremental) == 2 * (runner.args.max_new_tokens - 1)
    assert all(shape == (3, 1) for shape in incremental)


def test_share_prefix_can_be_disabled_and_matches(tiny):
    """The opt-out path must stay live: it is the A/B reference for the shared one."""
    runner, batch = spatial_setup(tiny)
    arms = [arm('all_text/attention', 2, c) for c in ['switch', 'self', 'paraphrase']]
    shared = CachedPrefill(runner, batch, arms).generate(arms)
    plain = CachedPrefill(runner, batch, arms, share_prefix=False)
    assert plain.prefix_length == 0 and plain.positions_dropped_inert == 0
    assert plain.generate(arms) == shared


def test_inert_positions_are_dropped_not_patched(tiny):
    """Positions below the divergence hold identical values in donor and recipient,
    so they are dropped from the plan. The count is recorded, and the arm must
    still agree token-for-token with the reference that does patch them."""
    runner, batch = spatial_setup(tiny)
    # Move the paraphrase divergence later so that column 6 sits BELOW the shared prefix and is
    # therefore droppable; spatial_setup's default puts both divergences at/after every patch column.
    batch['paraphrase_ids'] = batch['base_input_ids'].clone()
    batch['paraphrase_ids'][0, 8] = 16
    arms = [arm('earlier_text/residual', 4, 'switch')]
    cached = CachedPrefill(runner, batch, arms)
    assert cached.prefix_length == 7
    kept = cached.plan[arms[0][0]][1]
    assert cached.positions_dropped_inert == 1 and kept == [7]
    cached.generate(arms, verify=True)


def test_batch_lanes_stop_at_their_own_eos(tiny, monkeypatch):
    runner, batch = spatial_setup(tiny)
    arms = [arm('all_text/attention', 2), arm('last_token/mlp', 3)]
    cached = CachedPrefill(runner, batch, arms)
    cached.prepare(len(arms))
    runner.model.generation_config.eos_token_id = [5]
    original = runner.model.forward
    calls = []
    def forward(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits.fill_(-100)
        output.logits[0, -1, 5] = 100
        output.logits[1, -1, 6 if not calls else 5] = 100
        calls.append(True)
        return output
    monkeypatch.setattr(runner.model, 'forward', forward)
    assert cached.generate(arms) == [[5], [6, 5]]
    assert len(calls) == 2


def test_continuous_is_rejected_and_hooks_removed_on_error(tiny, monkeypatch):
    runner, batch = spatial_setup(tiny)
    with pytest.raises(ValueError, match='only prefill'):
        CachedPrefill(runner, batch, [arm('last_token/residual', mode='continuous')])
    arms = [arm('all_text/attention_blocks:3')]
    cached = CachedPrefill(runner, batch, arms)
    cached.prepare()
    def fail(*args, **kwargs):
        raise RuntimeError('simulated OOM')
    monkeypatch.setattr(runner.model, 'forward', fail)
    with pytest.raises(RuntimeError, match='simulated OOM'):
        cached.generate(arms)
    assert all(not module._forward_hooks for module in runner.model.modules())


def test_donor_geometry_matches_recipient_and_retains_each_lane(tiny):
    runner, batch = spatial_setup(tiny)
    arms = [arm('earlier_text/residual', 2, 'self'), arm('all_text/attention', 3, 'self')]
    # Simulate numerical effects depending on batch shape/lane. Comparing with
    # a batch-one donor or broadcasting lane zero must not pass this regression.
    def perturb(module, inputs, output):
        z = output[0] if isinstance(output, tuple) else output
        lane = torch.arange(1, z.shape[0] + 1, device=z.device).reshape(-1, 1, 1)
        direction = torch.linspace(-1, 1, z.shape[-1], device=z.device)
        shifted = z + .1 * z.shape[0] * lane * direction
        return (shifted,) + output[1:] if isinstance(output, tuple) else shifted
    hook = runner.layers[0].register_forward_hook(perturb)
    try:
        cached = CachedPrefill(runner, batch, arms)
        cached.generate(arms)
        assert cached.donor_batch_size == 2
        assert all(z.shape[0] == 2 for z in cached.donors.values())
        assert any(not torch.equal(z[0], z[1]) for z in cached.donors.values())
        cached.generate(arms[:1])
        assert cached.donor_batch_size == 1
        assert all(z.shape[0] == 1 for z in cached.donors.values())
    finally:
        hook.remove()


def test_identity_failure_reports_arm_precision_error_and_top_tokens(tiny):
    runner, batch = spatial_setup(tiny)
    cached = CachedPrefill(runner, batch, [])
    with pytest.raises(AssertionError, match='test-arm; model_dtype=.*max_abs=.*actual_top=.*expected_top='):
        cached._assert_logits(torch.tensor([[1., 2.]]), torch.tensor([[2., 1.]]), 'test-arm')


def test_bfloat16_model_with_upcast_logits_matches_reference(tiny):
    runner, batch = spatial_setup(tiny)
    runner.model.to(dtype=torch.bfloat16)
    # Reproduce versions that expose FP32 logits despite lower-precision matmuls.
    hook = runner.model.lm_head.register_forward_hook(lambda module, inputs, output: output.float())
    arms = [arm('all_text/residual', 4, 'self'), arm('last_token/attention_blocks:3', 3),
            arm('earlier_text/joint', 2, 'paraphrase')]
    try:
        cached = CachedPrefill(runner, batch, arms)
        cached.generate(arms, verify=True)
        assert cached.donor_logits['self'].dtype == torch.float32
        cached._assert_logits(torch.tensor([[0.01, 1.]]), torch.tensor([[0., 1.]]), 'BF16 rounding')
        with pytest.raises(AssertionError, match='max_abs='):
            cached._assert_logits(torch.tensor([[0.5, 1.]]), torch.tensor([[0., 1.]]), 'large error')
    finally:
        hook.remove()
