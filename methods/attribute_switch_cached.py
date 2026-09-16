"""Batched prefill interventions with reusable CPU donors and per-arm KV caches.

Only prompt positions are patched. Causality makes their donor activations
independent of generated suffixes. Recipient caches are built WITH the patches
active and never shared between arms. Continuous interventions use the reference
runner instead. This module currently targets the image-only Qwen switch task.
"""
from collections import defaultdict

import torch

from methods.attribute_switch_sweep import intervention, multimodal_position_kwargs, parts, patch_positions
from methods.common.hooks import extra_to_device


class CachedPrefill:
    def __init__(self, runner, batch, arms):
        self.runner, self.batch = runner, batch
        self.prompt_length = batch['base_input_ids'].shape[1]
        self.donors = {}
        self.donor_logits = {}
        self.plan = {}
        self.capture_positions = defaultdict(set)
        for arm, name, layer, mode, control in arms:
            if mode != 'prefill':
                raise ValueError('CachedPrefill supports only prefill interventions')
            positions = patch_positions(batch, intervention(name)[0], self.prompt_length, mode)
            self.plan[arm] = (control, positions, parts(name, layer))
            for site, at in self.plan[arm][2]:
                self.capture_positions[control, site.name, at].update(positions)
        self.capture_positions = {key: sorted(value) for key, value in self.capture_positions.items()}
        self.position_offsets = {key: {p: i for i, p in enumerate(value)}
                                 for key, value in self.capture_positions.items()}

    def prepare(self):
        """One unpatched donor forward per required question variant, per row."""
        from methods.common.sites import resolve_site
        prompt_keys = {'switch': 'source_input_ids', 'self': 'base_input_ids',
                       'paraphrase': 'paraphrase_ids'}
        for control in sorted({key[0] for key in self.capture_positions}):
            if control in self.donor_logits:
                continue
            hooks = []
            for key, positions in self.capture_positions.items():
                if key[0] != control:
                    continue
                def capture(z, key=key, positions=positions):
                    # Keep the donor bank off VRAM; upload only this batch's patches.
                    self.donors[key] = z[0, positions].detach().to('cpu', copy=True)
                    return z
                hooks.append((resolve_site(key[1]), key[2], capture))
            self.donor_logits[control] = self.runner.run(
                self.batch[prompt_keys[control]], self.batch, hooks).cpu()

    def _hooks(self, arms):
        by_site = defaultdict(list)
        for index, (arm, _, _, _, _) in enumerate(arms):
            control, positions, site_parts = self.plan[arm]
            for site, at in site_parts:
                key = control, site.name, at
                offsets = [self.position_offsets[key][p] for p in positions]
                values = self.donors[key][offsets].to(self.runner.model.device)
                by_site[site.name, at].append((index, positions, values))
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

    @torch.no_grad()
    def generate(self, arms, verify=False):
        """Generate independent arms together, stopping each at its own EOS.

        Explicit rotary positions isolate cached decoding from model-global
        rope_deltas, which donor/reference forwards may overwrite. Finished
        lanes remain inert until all lanes finish; they never add output tokens.
        """
        if not arms:
            return []
        self.prepare()
        runner, model = self.runner, self.runner.model
        device = model.device
        n = len(arms)
        ids = self.batch['base_input_ids'].to(device).repeat(n, 1)
        extras = self.batch['base_extra']
        if set(extras) != {'pixel_values', 'image_grid_thw'}:
            raise ValueError('Cached switch execution currently supports image-only pixel/grid inputs')
        extra = extra_to_device({k: v.repeat(n, *([1] * (v.ndim - 1)))
                                 for k, v in extras.items()}, device, model.dtype)
        mask = torch.ones_like(ids)
        position_ids, deltas = model.model.get_rope_index(
            ids, attention_mask=mask, image_grid_thw=extra['image_grid_thw'],
            **multimodal_position_kwargs(model, ids))
        deltas = deltas.to(device)
        handles = []
        try:
            for site, layer, fn in self._hooks(arms):
                handles.extend(site.register(runner.adapter, model, runner.layers, layer, fn))
            out = model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                        **extra, use_cache=True, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        # Hooks and device patch tensors are released before incremental decode.
        del handles, extra, fn
        logits, cache = out.logits[:, -1], out.past_key_values
        if cache is None:
            raise RuntimeError('The model did not return the requested KV cache')
        del out
        for index, (_, name, layer, _, control) in enumerate(arms):
            kind = intervention(name)[1]
            positions = self.plan[arms[index][0]][1]
            if control == 'self' or (kind == 'residual' and layer == len(runner.layers)
                                    and self.prompt_length - 1 in positions):
                self._assert_logits(logits[index:index + 1], self.donor_logits[control].to(device),
                                    'Cached prefill identity check failed')
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
                    if int(expected.argmax(-1).item()) != token:
                        raise AssertionError(f'Cached/reference argmax differs: {arms[index][0]}, token {step}')
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

    @staticmethod
    def _assert_logits(actual, expected, message):
        tolerance = 1e-5 if actual.dtype == torch.float32 else .02
        if not torch.allclose(actual.float(), expected.float(), atol=tolerance, rtol=tolerance):
            raise AssertionError(message)
