"""Toy model of see -> move -> recall: one attention layer + one MLP, trained on a lookup table.

CPU, seconds. An illustration of the mechanism X7 tests in Qwen, not evidence about Qwen.

TASK. E entities, R relations, each relation with its own small value set (so values are SHARED
across entities, like flags' languages). A sequence is

    [entity e] [pad] [pad] [relation r] [query]

and the model must output f[e, r] at the query position. The entity comes first and the
relation later, like the image before the question. Nothing at the query position knows e or r,
so attention must MOVE both there, and since f is an arbitrary table, the answer is a
conjunction of e and r that no linear readout of (e-vector + r-vector) can produce: the MLP has
to do the lookup.

MODEL. Token + position embeddings, ONE causal attention layer with --n_heads heads, ONE MLP
(ReLU), unembedding; residual connections, no LayerNorm, so the residual entering the MLP is
exactly x_query + sum_h head_h.

WHAT `analyze` MEASURES (the same probes as X7 / the Qwen experiments)
    heads           attention mass at the query on the entity vs the relation position, per
                    head: movers (-> entity) and routers (-> relation)
    shares          the two-way (entity x relation) variance decomposition of the residual
                    entering the MLP, of the MLP's hidden neurons and of what the MLP writes:
                    additive in, conjunctive out
    neuron_classes  how many of the hidden neurons are entity / relation / interaction neurons
    answer_paths    accuracy from the residual entering the MLP alone (a linear readout of the
                    moved vectors) vs the MLP's write alone: which path holds the fact
    compose         install the leave-cell-out ADDITIVE estimate rowmean(B) + colmean(A) - mean
                    into the MLP's input: does the MLP answer f[B, A]?
    cross_question  install the mover head's output from a donor (B, asked X) into (C, asked A):
                    does the model answer B's A (the head carries the entity) or B's X?
"""
from collections import Counter

import numpy as np

from methods.mlp_lookup import additive_loo, shares, two_way


def make_task(n_entities=64, n_relations=4, n_values=8, seed=0):
    rng = np.random.default_rng(seed)
    table = rng.integers(0, n_values, size=(n_entities, n_relations))
    return {"E": n_entities, "R": n_relations, "V": n_values, "table": table}


def vocab(task):
    E, R = task["E"], task["R"]
    return {"pad": E + R, "query": E + R + 1, "size": E + R + 2, "rel0": E}


def batch(task):
    """All E x R sequences, entity-major. -> (tokens [E*R, 5], targets [E*R]) as int64 arrays.
    Targets index a shared output space: relation r's values live at r*V .. r*V+V-1."""
    E, R, V = task["E"], task["R"], task["V"]
    v = vocab(task)
    toks, tgt = [], []
    for e in range(E):
        for r in range(R):
            toks.append([e, v["pad"], v["pad"], v["rel0"] + r, v["query"]])
            tgt.append(r * V + task["table"][e, r])
    return np.array(toks), np.array(tgt)


def build_model(task, d_model=128, n_heads=2, d_mlp=512, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads, self.d_head = n_heads, d_model // n_heads
            self.tok = nn.Embedding(vocab(task)["size"], d_model)
            self.pos = nn.Embedding(5, d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.o = nn.Linear(d_model, d_model, bias=False)
            self.mlp_in = nn.Linear(d_model, d_mlp)
            self.mlp_out = nn.Linear(d_mlp, d_model)
            self.unembed = nn.Linear(d_model, task["R"] * task["V"], bias=False)

        def forward(self, toks, head_override=None, mlp_input_override=None):
            """head_override: {head: [B, d_head]} replaces that head's output at the query.
            mlp_input_override: [B, d_model] replaces the residual entering the MLP at the query.
            Returns logits at the query and a dict of internals."""
            B, T = toks.shape
            x = self.tok(toks) + self.pos(torch.arange(T, device=toks.device))
            q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
            sh = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            q, k, v = sh(q), sh(k), sh(v)
            att = (q @ k.transpose(-1, -2)) / self.d_head ** 0.5
            att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=toks.device), 1), -1e9)
            att = att.softmax(-1)
            z = att @ v                                                     # [B, H, T, d_head]
            zq = z[:, :, -1].clone()                                        # query column
            for h, val in (head_override or {}).items():
                zq[:, h] = val
            W = self.o.weight.view(x.shape[-1], self.n_heads, self.d_head)  # [d, H, dh]
            head_writes = torch.einsum("bhe,dhe->bhd", zq, W)               # each head's write
            resid_mid = x[:, -1] + head_writes.sum(1)
            if mlp_input_override is not None:
                resid_mid = mlp_input_override
            hidden = torch.relu(self.mlp_in(resid_mid))
            mlp_write = self.mlp_out(hidden)
            return self.unembed(resid_mid + mlp_write), {"att": att[:, :, -1], "z": zq, "resid_mid": resid_mid,
                                                         "hidden": hidden, "mlp_out": mlp_write}

    return Toy()


def train(task, steps=3000, lr=3e-3, weight_decay=0.1, seed=0, log_every=0, **model_kw):
    import torch
    model = build_model(task, seed=seed, **model_kw)
    toks, tgt = (torch.tensor(a) for a in batch(task))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = []
    for step in range(steps):
        logits, _ = model(toks)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            acc = (logits.argmax(-1) == tgt).float().mean().item()
            hist.append({"step": step, "loss": loss.item(), "acc": acc})
    model.eval()
    return model, hist


def accuracy(task, model):
    import torch
    toks, tgt = (torch.tensor(a) for a in batch(task))
    with torch.no_grad():
        logits, _ = model(toks)
    return float((logits.argmax(-1) == tgt).float().mean())


def analyze(task, model, n_pairs=512, seed=0):
    import torch
    E, R, V = task["E"], task["R"], task["V"]
    toks, tgt = (torch.tensor(a) for a in batch(task))
    with torch.no_grad():
        logits, inside = model(toks)
    att = inside["att"].numpy()                                  # [E*R, H, T]
    heads = []
    for h in range(att.shape[1]):
        ent, rel = float(att[:, h, 0].mean()), float(att[:, h, 3].mean())
        heads.append({"head": h, "to_entity": ent, "to_relation": rel,
                      "role": "mover" if ent > rel else "router"})

    resid = inside["resid_mid"].numpy().reshape(E, R, -1)
    hidden = inside["hidden"].numpy().reshape(E, R, -1)
    ss_h = two_way(hidden)
    live = ss_h["total"] > 1e-9
    parts = np.stack([ss_h["item"], ss_h["attr"], ss_h["inter"]])[:, live]
    classes = dict(zip(("entity", "relation", "interaction"),
                       np.bincount(parts.argmax(0), minlength=3).tolist()))
    neuron_shares = (parts / parts.sum(0, keepdims=True)).T      # [n_live, 3]

    rng = np.random.default_rng(seed)
    pairs = [(int(rng.integers(E)), int(rng.integers(R))) for _ in range(n_pairs)]
    est = np.stack([additive_loo(resid, b, a) for b, a in pairs]).astype(np.float32)
    with torch.no_grad():
        lg, _ = model(toks[:len(pairs)], mlp_input_override=torch.tensor(est))
    pred = lg.argmax(-1).numpy()
    want = np.array([a * V + task["table"][b, a] for b, a in pairs])
    compose_acc = float((pred == want).mean())

    movers = [h["head"] for h in heads if h["role"] == "mover"]
    cross = None
    if movers:
        rows = []
        for _ in range(n_pairs):
            c, b = rng.choice(E, 2, replace=False)
            a, x = rng.choice(R, 2, replace=False) if R > 1 else (0, 0)
            rows.append((int(c), int(a), int(b), int(x)))
        base = torch.tensor([[c, vocab(task)["pad"], vocab(task)["pad"], vocab(task)["rel0"] + a,
                              vocab(task)["query"]] for c, a, _, _ in rows])
        donor = torch.tensor([[b, vocab(task)["pad"], vocab(task)["pad"], vocab(task)["rel0"] + x,
                               vocab(task)["query"]] for _, _, b, x in rows])
        with torch.no_grad():
            _, dinside = model(donor)
            lg, _ = model(base, head_override={h: dinside["z"][:, h] for h in movers})
        pred = lg.argmax(-1).numpy()
        kinds = Counter()
        for (c, a, b, x), p in zip(rows, pred):
            hits = {"B's asked attribute (B.A)": a * V + task["table"][b, a],
                    "base answer (C.A)": a * V + task["table"][c, a],
                    "donor's own answer (B.X)": x * V + task["table"][b, x]}
            # B.A and C.A can coincide when values are shared; count the first match in this order.
            kinds.update([next((k for k, v in hits.items() if v == p), "other")])
        cross = {k: v / len(rows) for k, v in sorted(kinds.items())}

    # Which path writes the answer: the residual entering the MLP (the moved entity + relation,
    # read linearly through the unembedding) vs the MLP's own write, each scored alone within the
    # asked relation's value slot. Chance is 1 / V.
    with torch.no_grad():
        direct = model.unembed(inside["resid_mid"]).numpy()
        via_mlp = model.unembed(inside["mlp_out"]).numpy()
    rel = np.arange(E * R) % R                                  # batch() is entity-major

    def slot_acc(L):
        pred = np.array([rel[i] * V + L[i, rel[i] * V:(rel[i] + 1) * V].argmax() for i in range(E * R)])
        return float((pred == tgt.numpy()).mean())
    answer_paths = {"chance": 1 / V, "residual_only (no MLP)": slot_acc(direct), "mlp_write_only": slot_acc(via_mlp),
                    "both": slot_acc(direct + via_mlp)}

    return {"accuracy": float((logits.argmax(-1) == tgt).float().mean()), "heads": heads,
            "answer_paths": answer_paths,
            "resid_shares": shares(two_way(resid)), "hidden_shares": shares(ss_h),
            "mlp_out_shares": shares(two_way(inside["mlp_out"].numpy().reshape(E, R, -1))),
            "neuron_classes": classes, "neuron_shares": neuron_shares, "compose_additive_acc": compose_acc,
            "cross_question": cross}

