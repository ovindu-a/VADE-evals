"""Question-independent coordinate masks inside a frozen set of attention heads."""
import torch
from torch import nn

from methods.common.hooks import extra_to_device


class HeadMask(nn.Module):
    def __init__(self, heads, head_dim, initial_probability=0.5):
        super().__init__()
        self.heads = [tuple(h) for h in heads]
        self.head_dim = head_dim
        self.logits = nn.Parameter(torch.full((len(heads), head_dim),
                                             torch.logit(torch.tensor(initial_probability)).item()))

    def gates(self, temperature=1., hard=False):
        soft = (self.logits / temperature).sigmoid()
        return (soft >= .5).to(soft) if hard else soft

    def patch(self, z, donor, block, start, temperature=1., hard=False):
        if donor.shape != z[:, start:].shape:
            raise ValueError('Donor and recipient prefixes must align')
        result = z.clone()
        gates = self.gates(temperature, hard).to(z)
        for i, (b, h) in enumerate(self.heads):
            if b == block:
                sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                base = z[:, start:, sl]
                result[:, start:, sl] = base + gates[i] * (donor[:, :, sl].to(z) - base)
        return result


def masked_logits(runner, batch, source, ids, mask, temperature=1., hard=False):
    """Differentiable recipient; detached donor sees exactly the same prefix.

    Returns every answer-position logit (not just the first/last token). No
    queried-attribute argument exists here: the mask cannot route by question.
    """
    blocks = sorted({b for b, _ in mask.heads})
    _, donor = runner.forward(ids, batch, source, capture_blocks=blocks)
    start = batch['base_input_ids'].shape[1] - 1
    handles = []
    try:
        for block in blocks:
            def pre(module, inputs, b=block):
                return (mask.patch(inputs[0], donor[b], b, start, temperature, hard),) + inputs[1:]
            handles.append(runner.adapter.get_attn_head_output_module(
                runner.model, block).register_forward_pre_hook(pre))
        out = runner.model(input_ids=ids.to(runner.model.device),
                           attention_mask=torch.ones_like(ids, device=runner.model.device),
                           **extra_to_device(batch['base_extra'], runner.model.device, runner.model.dtype),
                           use_cache=False, logits_to_keep=ids.shape[1] - start)
        return out.logits
    finally:
        for handle in handles:
            handle.remove()


def answer_loss(runner, batch, source, gold, mask, temperature):
    """Full-answer teacher forcing for optimization only; evaluation is free decode."""
    if not gold:
        raise ValueError('Empty answer')
    ids = batch['base_input_ids'].to(runner.model.device)
    targets = ids.new_tensor([gold])
    ids = torch.cat((ids, targets[:, :-1]), dim=1)
    logits = masked_logits(runner, batch, source, ids, mask, temperature)
    return torch.nn.functional.cross_entropy(logits[0].float(), targets[0])
