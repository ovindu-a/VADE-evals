"""Batched prefill interventions with reusable CPU donors and per-arm KV caches.

Only prompt positions are patched. Causality makes their donor activations
independent of generated suffixes. Recipient caches are built WITH the patches
active and never shared between arms. Continuous interventions use the reference
runner instead. This module currently targets the image-only Qwen switch task.

SHARED PROMPT PREFIX (the `share_prefix` path). The base, donor and paraphrase
prompts of a row are token-identical up to their first divergence -- on the flag
task, 205 of 219 tokens, with the whole image inside that span. Under causal
attention a position's activations depend only on tokens at or before it, so
across the three prompts every activation below the divergence is the same value
at every layer and every site. Two consequences, and the second is the speedup:

  * A patch position below the divergence writes a value onto itself. It is
    inert by construction here, not merely by argument, because donor and
    recipient read the SAME cache entries -- so those positions are dropped from
    both the capture and the patch, and `positions_dropped_inert` records how
    many.
  * The prefix's KV cache can be built ONCE per row and per batch size and
    reused by every donor and every recipient arm of that row. Each prefill then
    computes only the divergent tail (14 tokens instead of 219) and, because the
    image lies entirely inside the prefix, runs no vision tower at all.

The cache is deep-copied per prefill rather than cropped: a forward APPENDS to
whatever cache it is handed, and `copy.deepcopy` is the one way to take a private
copy that does not reach into a transformers-version-specific layout.

Numerically this is the same computation, not an approximation -- but it is the
same computation in a different batch/kernel geometry, which in bf16 is the same
class of difference as changing `--batch_size` (the reason `prepare` rebuilds the
donor bank when that changes). `--verify_cached` compares every token against the
full-prefix serial reference and is the check that settles it.
"""
import copy
from collections import defaultdict

import torch

from methods.attribute_switch_sweep import intervention, multimodal_position_kwargs, parts, patch_positions
from methods.common.hooks import extra_to_device


def shared_prefix_length(batch):
    """Longest token prefix every prompt variant of this row agrees on.

    Returns 0 when there is nothing worth sharing, which makes the caller fall
    back to the full-prefill path rather than special-casing an empty prefix.
    """
    base = batch['base_input_ids'][0]
    others = [batch[k][0] for k in ('source_input_ids', 'paraphrase_ids') if k in batch]
    if not others:
        return 0
    limit = min(int(base.shape[0]), *(int(o.shape[0]) for o in others))
    length = limit
    for other in others:
        same = base[:limit] == other[:limit]
        mismatch = (~same).nonzero()
        if mismatch.numel():
            length = min(length, int(mismatch[0]))
    return int(length)


class CachedPrefill:
    def __init__(self, runner, batch, arms, share_prefix=True):
        self.runner, self.batch = runner, batch
        self.prompt_length = batch['base_input_ids'].shape[1]
        self.donors = {}
        self.donor_logits = {}
        self.donor_batch_size = None
        self.plan = {}
        self.capture_positions = defaultdict(set)
        self.prefix_length = shared_prefix_length(batch) if share_prefix else 0
        self.prefix_cache = None
        self._position_ids = None
        self._deltas = None
        self.positions_dropped_inert = 0
        if self.prefix_length:
            # The suffix forward is run WITHOUT pixel_values, so every image token must lie inside
            # the shared prefix. Asserted rather than assumed: an image token in the tail would be
            # embedded from its placeholder id and the run would still produce fluent output.
            # getattr: the adapter reads the id off model.config when it is there and only
            # consults the processor otherwise, so a runner built without one still works.
            image_id = runner.adapter.image_token_id(runner.model, getattr(runner, 'processor', None))
            tail = batch['base_input_ids'][0, self.prefix_length:]
            if bool((tail == image_id).any()):
                raise ValueError('Prompt variants diverge before the end of the image span; '
                                 'the shared-prefix path cannot skip the vision tower here')
        for arm, name, layer, mode, control in arms:
            if mode != 'prefill':
                raise ValueError('CachedPrefill supports only prefill interventions')
            positions = patch_positions(batch, intervention(name)[0], self.prompt_length, mode)
            kept = [p for p in positions if p >= self.prefix_length]
            self.positions_dropped_inert += len(positions) - len(kept)
            self.plan[arm] = (control, kept, parts(name, layer))
            for site, at in self.plan[arm][2]:
                self.capture_positions[control, site.name, at].update(kept)
        self.capture_positions = {key: sorted(value) for key, value in self.capture_positions.items()}
        self.position_offsets = {key: {p: i for i, p in enumerate(value)}
                                 for key, value in self.capture_positions.items()}

    # ---- prefix ------------------------------------------------------------

    def _rope(self, n):
        """Multimodal rotary positions for the FULL prompt at batch n, computed
        once. All three variants share length, image grid and image span, so
        they share these positions; the suffix forward slices them."""
        if self._position_ids is None or self._position_ids.shape[1] != n:
            model = self.runner.model
            ids = self.batch['base_input_ids'].to(model.device).repeat(n, 1)
            mask = torch.ones_like(ids)
            extra = self._extra(n)
            self._position_ids, deltas = model.model.get_rope_index(
                ids, attention_mask=mask, image_grid_thw=extra['image_grid_thw'],
                **multimodal_position_kwargs(model, ids))
            self._deltas = deltas.to(model.device)
        return self._position_ids, self._deltas

    def _extra(self, n):
        extras = self.batch['base_extra']
        if set(extras) != {'pixel_values', 'image_grid_thw'}:
            raise ValueError('Cached switch execution currently supports image-only pixel/grid inputs')
        model = self.runner.model
        return extra_to_device({k: v.repeat(n, *([1] * (v.ndim - 1))) for k, v in extras.items()},
                               model.device, model.dtype)

    def _build_prefix(self, n):
        """The one forward that pays for the image, per row and batch size."""
        model = self.runner.model
        position_ids, _ = self._rope(n)
        ids = self.batch['base_input_ids'][:, :self.prefix_length].to(model.device).repeat(n, 1)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    position_ids=position_ids[:, :, :self.prefix_length], **self._extra(n),
                    use_cache=True)
        cache = out.past_key_values
        if cache is None:
            raise RuntimeError('The model did not return the requested prefix KV cache')
        del out
        return cache

    def _prefill(self, prompt, n, hooks=()):
        """Same batch shape, positions and cache settings for donor/recipient.

        With a shared prefix, `hooks` see only the divergent tail, so their
        columns are prompt-absolute minus `prefix_length`; `_local` does that
        conversion in the one place it can be got wrong.
        """
        runner, model = self.runner, self.runner.model
        position_ids, deltas = self._rope(n)
        if not self.prefix_length:
            ids = prompt.to(model.device).repeat(n, 1)
            kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                          position_ids=position_ids, **self._extra(n))
        else:
            if self.prefix_cache is None:
                self.prefix_cache = self._build_prefix(n)
            ids = prompt[:, self.prefix_length:].to(model.device).repeat(n, 1)
            kwargs = dict(input_ids=ids,
                          attention_mask=torch.ones(n, self.prompt_length, dtype=torch.long,
                                                    device=model.device),
                          position_ids=position_ids[:, :, self.prefix_length:],
                          past_key_values=copy.deepcopy(self.prefix_cache))
        handles = []
        try:
            for site, layer, fn in hooks:
                handles.extend(site.register(runner.adapter, model, runner.layers, layer, fn))
            out = model(use_cache=True, logits_to_keep=1, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        return out, deltas

    def _local(self, positions):
        """Prompt-absolute columns -> columns of the tensor the hooks actually see."""
        return [p - self.prefix_length for p in positions]

    # ---- donors ------------------------------------------------------------

    @torch.no_grad()
    def prepare(self, batch_size=1):
        """Reuse donors captured with the recipient's batch shape/cache settings.

        Batched BF16 kernels need not match batch-size-one activations. Preserve
        each lane rather than broadcasting lane zero. Only one batch geometry's
        CPU bank is retained; a short final chunk rebuilds it at its actual size.
        """
        from methods.common.sites import resolve_site
        if self.donor_batch_size != batch_size:
            self.donors.clear()
            self.donor_logits.clear()
            self.donor_batch_size = batch_size
            self.prefix_cache = None       # built at one batch geometry, like the donor bank
            self._position_ids = None
        prompt_keys = {'switch': 'source_input_ids', 'self': 'base_input_ids',
                       'paraphrase': 'paraphrase_ids'}
        for control in sorted({key[0] for key in self.capture_positions}):
            if control in self.donor_logits:
                continue
            hooks = []
            for key, positions in self.capture_positions.items():
                if key[0] != control:
                    continue
                local = self._local(positions)
                def capture(z, key=key, local=local):
                    # Keep the donor bank off VRAM; upload only this batch's patches.
                    self.donors[key] = z[:, local].detach().to('cpu', copy=True)
                    return z
                hooks.append((resolve_site(key[1]), key[2], capture))
            out, _ = self._prefill(self.batch[prompt_keys[control]], batch_size, hooks)
            self.donor_logits[control] = out.logits[:, -1].detach().to('cpu', copy=True)
            del out  # Donor KV caches are never reused by recipients.

    def _hooks(self, arms):
        by_site = defaultdict(list)
        for index, (arm, _, _, _, _) in enumerate(arms):
            control, positions, site_parts = self.plan[arm]
            if not positions:
                continue      # every position of this arm was inert; the arm is a no-op by design
            for site, at in site_parts:
                key = control, site.name, at
                offsets = [self.position_offsets[key][p] for p in positions]
                values = self.donors[key][index, offsets].to(self.runner.model.device)
                by_site[site.name, at].append((index, self._local(positions), values))
        from methods.common.sites import resolve_site
        hooks = []
        for (name, at), replacements in by_site.items():
            def patch(z, replacements=replacements):
                out = z.clone()
                for index, positions, values in replacements:
                    out[index, positions] = values.to(dtype=z.dtype)
                return out
            hooks.append((resolve_site(name), at, patch))
        return hooks

    # ---- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, arms, verify=False):
        """Generate independent arms together, stopping each at its own EOS.

        Explicit rotary positions isolate cached decoding from model-global
        rope_deltas, which donor/reference forwards may overwrite. Finished
        lanes remain inert until all lanes finish; they never add output tokens.
        """
        if not arms:
            return []
        runner, model = self.runner, self.runner.model
        device = model.device
        n = len(arms)
        self.prepare(n)
        ids = self.batch['base_input_ids'].to(device).repeat(n, 1)
        mask = torch.ones_like(ids)
        out, deltas = self._prefill(self.batch['base_input_ids'], n, self._hooks(arms))
        logits, cache = out.logits[:, -1], out.past_key_values
        if cache is None:
            raise RuntimeError('The model did not return the requested KV cache')
        del out
        for index, (_, name, layer, _, control) in enumerate(arms):
            kind = intervention(name)[1]
            positions = self.plan[arms[index][0]][1]
            if control == 'self' or (kind == 'residual' and layer == len(runner.layers)
                                    and self.prompt_length - 1 in positions):
                self._assert_logits(logits[index:index + 1],
                                    self.donor_logits[control][index:index + 1].to(device),
                                    f'Cached prefill identity check failed: {arms[index][0]}, '
                                    f'lane={index}, batch_size={n}')
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        fill = model.generation_config.pad_token_id
        if fill is None:
            fill = next(iter(eos), 0)
        generated = [[] for _ in arms]
        finished = [False] * n
        references = ([runner.step(self.batch, name, layer, mode, control)
                       for _, name, layer, mode, control in arms] if verify else None)
        for step in range(runner.args.max_new_tokens):
            tokens = logits.argmax(-1).tolist()
            for index, token in enumerate(tokens):
                if finished[index]:
                    tokens[index] = fill
                    continue
                if references is not None:
                    prefix = torch.cat((self.batch['base_input_ids'].to(device),
                                        ids.new_tensor([generated[index]])), dim=1)
                    expected, _ = references[index](prefix)
                    self._assert_logits(logits[index:index + 1], expected,
                                        f'Cached/reference mismatch: {arms[index][0]}, token {step}')
                generated[index].append(token)
                finished[index] = token in eos
            if all(finished) or step + 1 == runner.args.max_new_tokens:
                break
            # Each lane owns its cache slice; no donor cache is ever installed.
            next_ids = ids.new_tensor(tokens).unsqueeze(1)
            length = self.prompt_length + step + 1
            next_positions = (deltas + length - 1).unsqueeze(0).expand(3, -1, -1)
            out = model(input_ids=next_ids,
                        attention_mask=torch.ones(n, length, dtype=mask.dtype, device=device),
                        position_ids=next_positions, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            logits, cache = out.logits[:, -1], out.past_key_values
            del out
        return generated

    def _assert_logits(self, actual, expected, message):
        # Some Transformers versions upcast logits after BF16 model computation.
        tolerance = 1e-5 if self.runner.model.dtype == torch.float32 else .02
        same_token = torch.equal(actual.argmax(-1), expected.argmax(-1))
        if not same_token or not torch.allclose(actual.float(), expected.float(), atol=tolerance, rtol=tolerance):
            delta = (actual.float() - expected.float()).abs()
            raise AssertionError(
                f'{message}; model_dtype={self.runner.model.dtype}, logits_dtype={actual.dtype}, '
                f'max_abs={float(delta.max()):.6g}, mean_abs={float(delta.mean()):.6g}, '
                f'actual_top={actual.argmax(-1).tolist()}, expected_top={expected.argmax(-1).tolist()}, '
                f'atol=rtol={tolerance}')
