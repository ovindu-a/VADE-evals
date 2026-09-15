"""Per-layer orthogonal DAS projections in selected pre-W_O head outputs."""
import torch
from torch import nn


class HeadSubspace(nn.Module):
    def __init__(self, heads, head_dim, rank, seed=0, mode='continuous'):
        super().__init__()
        self.heads = [tuple(h) for h in heads]
        self.head_dim, self.rank, self.mode = head_dim, rank, mode
        self.groups = {}
        for b, h in self.heads:
            self.groups.setdefault(b, []).extend(range(h * head_dim, (h + 1) * head_dim))
        if len(set(self.heads)) != len(self.heads) or rank < 1:
            raise ValueError('Unique heads and a positive rank are required')
        gen = torch.Generator().manual_seed(seed)
        self.raw = nn.ParameterDict()
        for b, columns in sorted(self.groups.items()):
            if rank > len(columns):
                raise ValueError(f'Rank {rank} exceeds selected width {len(columns)} in block {b}')
            self.raw[str(b)] = nn.Parameter(torch.randn(len(columns), rank, generator=gen))

    def basis(self, block):
        return torch.linalg.qr(self.raw[str(block)].float(), mode='reduced').Q

    def patch(self, z, donor, block, start, temperature=1., hard=False, control=None):
        if donor.shape != z[:, start:].shape:
            raise ValueError('Donor and recipient prefixes must align')
        columns = self.groups[block]
        stop = start + 1 if self.mode == 'prefill' else z.shape[1]
        base = z[:, start:stop, columns]
        delta = donor[:, :stop-start, columns].to(base).float() - base.float()
        u = self.basis(block)
        edit = (delta @ u) @ u.T
        if control == 'matched_blend':
            # Match the learned edit's L2 magnitude separately at each layer/position.
            strength = edit.norm(dim=-1, keepdim=True) / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            edit = strength * delta
        result = z.clone()
        result[:, start:stop, columns] = (base.float() + edit).to(z)
        return result


class MatchedBlend:
    def __init__(self, subspace):
        self.subspace, self.heads = subspace, subspace.heads

    def patch(self, *args, **kwargs):
        return self.subspace.patch(*args, **kwargs, control='matched_blend')
