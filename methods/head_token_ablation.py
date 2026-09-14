"""Observe and remove selected attention edges without replacing HF attention.

Under eager attention, capture A, V, and the concatenated head output z.
Zeroing post-softmax edges gives z' = z - sum_removed A[q,k] V[k].
Apply W_O(z' - z) to the attention module's output, BEFORE the residual add.
This is equivalent to zeroing those weights, with no probability redistribution.
An optional renormalization arm redistributes the remaining mass explicitly.
Reconstruction of A@V against o_proj's actual input is checked on every pass.
"""
import torch


def remove_edges(weights, values, keys, renormalize=False):
    """Pure reference operation for one head: [Q,K], [K,D] -> [Q,D], [Q,K]."""
    modified = weights.clone()
    modified[:, keys] = 0
    if renormalize:
        mass = modified.sum(-1, keepdim=True)
        modified = torch.where(mass > 0, modified / mass.clamp_min(1e-30), torch.zeros_like(modified))
    return modified @ values, modified


class AttentionEdges:
    def __init__(self, runner, heads, prompt_length, removals=None, query_scope='prefill',
                 renormalize=False, collect=False):
        self.runner, self.heads, self.start = runner, heads, prompt_length - 1
        self.removals = removals or {}
        self.query_scope, self.renormalize, self.collect = query_scope, renormalize, collect
        self.handles, self.values, self.z, self.snapshot = [], {}, {}, {}
        self.max_reconstruction_error = 0.0

    def __enter__(self):
        adapter, model = self.runner.adapter, self.runner.model
        try:
            for b in sorted({b for b, _ in self.heads}):
                attn = adapter.get_attn_block(model, b)
                if not hasattr(attn, 'v_proj'):
                    raise TypeError('Attention edge probe currently requires Qwen-style v_proj/o_proj')
                def value_hook(module, inputs, output, block=b):
                    self.values[block] = output.detach()
                def z_hook(module, inputs, block=b):
                    self.z[block] = inputs[0].detach()
                def attn_hook(module, inputs, output, block=b):
                    return self.modify(block, output)
                self.handles.append(attn.v_proj.register_forward_hook(value_hook))
                self.handles.append(adapter.get_attn_head_output_module(model, b).register_forward_pre_hook(z_hook))
                self.handles.append(attn.register_forward_hook(attn_hook))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def modify(self, block, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError('Attention weights unavailable: use eager attention and output_attentions=True')
        out, a = output[:2]
        if a.shape[0] != 1 or out.shape[0] != 1:
            raise AssertionError('Token edge experiments require one unpadded row per forward')
        raw_v, z = self.values[block], self.z[block]
        n_heads, dim = self.runner.n_heads, self.runner.head_dim
        kv_heads = raw_v.shape[-1] // dim
        if raw_v.shape[-1] % dim or n_heads % kv_heads or a.shape[1] != n_heads:
            raise AssertionError('Unexpected GQA dimensions')
        if raw_v.shape[1] != a.shape[-1] or z.shape[1] != a.shape[-2]:
            raise AssertionError('This probe requires a full-prefix forward without KV cache')
        v = raw_v[0].reshape(raw_v.shape[1], kv_heads, dim).float()
        projection = self.runner.adapter.get_attn_head_output_module(self.runner.model, block)
        modified_out = out.clone()
        modified_a = a.clone() if self.removals else a
        queries = (list(range(self.start, z.shape[1])) if self.query_scope == 'continuous' else [self.start])
        # Current query is also checked/recorded during decode even for prefill-only ablation.
        check_queries = sorted(set(queries + [z.shape[1] - 1]))
        for b, h in self.heads:
            if b != block:
                continue
            weights = a[0, h].float()
            vh = v[:, h // (n_heads // kv_heads)]
            sl = slice(h * dim, (h + 1) * dim)
            predicted = weights[check_queries] @ vh
            actual = z[0, check_queries, sl].float()
            rel = float((predicted - actual).norm() / actual.norm().clamp_min(1e-8))
            self.max_reconstruction_error = max(self.max_reconstruction_error, rel)
            tolerance = 1e-4 if z.dtype == torch.float32 else 0.02
            if rel > tolerance:
                raise AssertionError(f'A@V != o_proj input at {block}.{h}: {rel:.3%}; wrong tensor/GQA layout')
            if self.collect:
                q = z.shape[1] - 1
                # Per-key size of this head's direct residual contribution, distinct from attention mass.
                projected_v = vh @ projection.weight[:, sl].float().T
                contribution = weights[q] * projected_v.norm(dim=-1)
                self.snapshot[f'{block}.{h}'] = {
                    'query_position': q, 'attention': weights[q].cpu().tolist(),
                    'weighted_value_residual_norm': contribution.cpu().tolist()}
            keys = self.removals.get((block, h), [])
            if not keys:
                continue
            if len(set(keys)) != len(keys) or any(k < 0 or k > self.start for k in keys):
                raise ValueError('Knockout keys must be unique prompt-token positions')
            before = weights[queries]
            after_z, after_a = remove_edges(before, vh, keys, self.renormalize)
            delta_z = after_z - before @ vh
            delta_out = delta_z @ projection.weight[:, sl].float().T
            modified_out[0, queries] = (modified_out[0, queries].float() + delta_out).to(out.dtype)
            modified_a[0, h, queries] = after_a.to(a.dtype)
        return (modified_out, modified_a) + output[2:]


def describe_tokens(runner, ids):
    tok = runner.processor.tokenizer
    image_id = runner.adapter.image_token_id(runner.model, runner.processor)
    loc = runner.assets.object_location
    grid = loc['vlm_token_grid']
    # The smallest benchmark geometry set is the object's footprint, not its surrounding ring.
    footprint = set(min(loc['object_token_indices'].values(), key=lambda v: len(v['flat']))['flat'])
    special = set(tok.all_special_ids)
    image_index = 0
    result = []
    for pos, token in enumerate(ids):
        rec = {'position': pos, 'token_id': token, 'token': tok.convert_ids_to_tokens(token),
               'text': tok.decode([token]), 'group': 'special' if token in special else 'text'}
        if token == image_id:
            rec.update(image_index=image_index, grid_row=image_index // grid['grid_cols'],
                       grid_col=image_index % grid['grid_cols'],
                       group='object' if image_index in footprint else 'image_background')
            image_index += 1
        result.append(rec)
    if image_index != grid['n_tokens']:
        raise AssertionError('Image-token count differs from benchmark geometry')
    return result


def candidate_positions(tokens, scope):
    groups = {'all': {'text', 'special', 'object', 'image_background'},
              'image': {'object', 'image_background'}, 'object': {'object'},
              'background': {'image_background'}, 'text': {'text', 'special'}}
    return [t['position'] for t in tokens if t['group'] in groups[scope]]


def annotate_snapshot(snapshot, tokens, candidates, rank_by):
    """Rank each head independently; the knockout phase consumes these exact lists."""
    score_key = 'attention' if rank_by == 'attention' else 'weighted_value_residual_norm'
    result = {}
    for head, rec in snapshot.items():
        ranking = sorted(candidates, key=lambda k: (-rec[score_key][k], k))
        groups = {}
        for t in tokens:
            groups[t['group']] = groups.get(t['group'], 0.0) + rec['attention'][t['position']]
        result[head] = {**rec, 'group_attention': groups, 'ranked_prompt_keys': ranking,
                        'top_tokens': [{**tokens[k], 'attention': rec['attention'][k],
                                        'weighted_value_residual_norm': rec['weighted_value_residual_norm'][k]}
                                       for k in ranking[:20]]}
    return result
