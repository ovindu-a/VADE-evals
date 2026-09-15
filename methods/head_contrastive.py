"""Paired same-item mean differences for switching the requested attribute.

No donor-country value is injected at evaluation. Fixed directions are estimated
on training identities and added to selected head outputs on unseen identities.
"""
import itertools
from pathlib import Path
import random

from methods.head_attribute_edit import atomic_json
from methods.head_followup_common import Results

TEMPLATES = {
    'v1': 'Report the {field} of the {entity} shown.',
    'v2': 'What is the {field} of this {entity}?',
    'v3': 'Give the {field} for the pictured {entity}.',
    'v4': 'Identify this {entity}\'s {field}.',
    'v5': 'State the {field} of the {entity} in the image.',
    'v6': 'For this {entity}, provide its {field}.',
}
FIELD_NAMES = {'calling_code': 'international calling code', 'hq_country': 'headquarters country',
               'founded_year': 'founding year', 'language': 'official language'}


def question(entity, attribute, template):
    noun = {'flags': 'country', 'brands': 'brand', 'animals': 'animal'}.get(entity, 'entity')
    # These instructions are identical across attributes for a given entity.
    formats = (' For currency, use the three-letter code. For calling codes, give digits without a plus sign.'
               ' For language, give one official language.' if entity == 'flags' else '')
    return ('Give only the requested value.' + formats + ' ' + TEMPLATES[template].format(
        field=FIELD_NAMES.get(attribute, attribute.replace('_', ' ')), entity=noun))


def batch_for(runner, image, entity, attribute, template, prefill):
    built = runner.adapter.build_inputs(runner.processor, image, question(entity, attribute, template), prefill)
    return dict(base_input_ids=built['input_ids'].unsqueeze(0), base_extra=built['extra'])


def direction(means, base, target, heads, head_dim):
    import torch
    result = {}
    for b, h in heads:
        if b not in result:
            result[b] = torch.zeros_like(means[target][b])
        sl = slice(h*head_dim, (h+1)*head_dim)
        result[b][sl] = means[target][b][sl] - means[base][b][sl]
    return result


def random_direction(vector, heads, head_dim, seed):
    import torch
    gen = torch.Generator().manual_seed(seed)
    result = {}
    for b in sorted(vector):
        columns = [i for block, h in heads if block == b for i in range(h*head_dim, (h+1)*head_dim)]
        v = torch.zeros_like(vector[b])
        noise = torch.randn(len(columns), generator=gen)
        v[columns] = noise * (vector[b].norm() / noise.norm().clamp_min(1e-12))
        result[b] = v
    return result


def additive_step(runner, batch, vector, strength, mode):
    """Add a fixed learned difference; no gold tokens or target-question forward."""
    import torch
    from methods.common.hooks import extra_to_device
    start = batch['base_input_ids'].shape[1] - 1
    def step(ids):
        handles = []
        try:
            for block, delta in vector.items():
                def pre(mod, inputs, d=delta):
                    z = inputs[0].clone()
                    stop = start + 1 if mode == 'prefill' else z.shape[1]
                    z[:, start:stop] += strength * d.to(z)
                    return (z,) + inputs[1:]
                handles.append(runner.adapter.get_attn_head_output_module(runner.model, block).register_forward_pre_hook(pre))
            with torch.no_grad():
                logits = runner.model(input_ids=ids.to(runner.model.device),
                    attention_mask=torch.ones_like(ids, device=runner.model.device),
                    **extra_to_device(batch['base_extra'], runner.model.device, runner.model.dtype),
                    use_cache=False, logits_to_keep=1).logits[:, -1].detach()
            return logits, {}
        finally:
            for h in handles:
                h.remove()
    return step


def summarize(results):
    results.finish()
    baselines = {(r['row_index'], r['arm']): r for r in results.records}
    groups = {}
    for row in results.records:
        groups.setdefault((row['arm'], row['base_attribute'], row['donor_attribute']), []).append(row)
    cells = []
    for (arm, base, target), rows in sorted(groups.items()):
        eligible = [r for r in rows if baselines[(r['row_index'], 'clean')]['base_score']['full_match']
                    and baselines[(r['row_index'], 'target_clean')]['source_score']['full_match']]
        def rates(rr):
            return dict(n=len(rr), target_full=sum(r['source_score']['full_match'] for r in rr)/len(rr),
                original_full=sum(r['base_score']['full_match'] for r in rr)/len(rr),
                neither=sum(not (r['source_score']['full_match'] or r['base_score']['full_match']) for r in rr)/len(rr)) if rr else None
        cells.append(dict(arm=arm, base_attribute=base, target_attribute=target, all=rates(rows),
                          both_clean_correct=rates(eligible)))
    atomic_json(results.path / 'switch_summary.json', cells)


def run_contrastive(args, trace, sets, attributes, directory, items, train_items, test_items, config):
    from methods.head_directions import selected_heads
    if not set(args.train_templates + args.test_templates) <= set(TEMPLATES):
        raise ValueError(f'Contrastive templates must be among {list(TEMPLATES)}')
    train_items = random.Random(args.seed).sample(train_items, min(len(train_items), args.contrast_train_limit))
    test_items = random.Random(args.seed+1).sample(test_items, min(len(test_items), args.contrast_test_limit))
    pairs = [(a, b) for a, b in itertools.permutations(attributes, 2)
             if b in (args.targets or attributes) and a in (args.from_attributes or attributes)]
    if not pairs:
        raise ValueError('No non-identity attribute directions selected')
    config = dict(config, actual_train_items=train_items, actual_test_items=test_items,
        questions={a: {t: question(trace['entity'], a, t) for t in args.train_templates+args.test_templates}
                   for a in attributes}, head_sets=sets)
    print(f'Contrastive: {len(train_items)} fitting items, {len(test_items)} held-out items, '
          f'{len(pairs)} directed switches; mean differences use {len(args.train_templates)} training prompts', flush=True)
    if args.dry_run:
        return
    import torch
    from PIL import Image
    from methods.head_followup_common import Runner
    from methods.common.targets import derive_gold_token_ids
    path = Path(args.out_dir)/trace['entity']/'contrastive'
    # JSON normalization makes tuples stable across resumes.
    import json
    config = json.loads(json.dumps(config))
    results = Results(path, config)
    runner = Runner(args, trace)
    results.check_runtime(runner)
    if runner.head_dim != trace['head_dim'] or runner.n_heads != trace['n_heads']:
        raise ValueError('Trace dimensions differ from loaded model')
    blocks = sorted({b for hs in sets.values() for b, h in hs})
    checkpoint = path / 'directions.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError('Direction checkpoint configuration mismatch')
        means = saved['means']
    else:
        means = {a: {b: torch.zeros(runner.hidden) for b in blocks} for a in attributes}
        n = len(train_items)*len(args.train_templates)
        for i, item in enumerate(train_items):
            with Image.open(directory/items[item]['image']) as f:
                image = f.convert('RGB')
            for template in args.train_templates:
                extras = None
                for attribute in attributes:
                    batch = batch_for(runner, image, trace['entity'], attribute, template, args.prefill)
                    if extras is not None and any(not torch.equal(v, batch['base_extra'][k]) for k,v in extras.items()):
                        raise ValueError('Same-item prompts changed image inputs')
                    extras = batch['base_extra']
                    _, activations = runner.forward(batch['base_input_ids'], batch, capture_blocks=blocks)
                    for b in blocks:
                        means[attribute][b] += activations[b][0, 0].float().cpu()/n
            print(f'Contrastive means: item {i+1}/{len(train_items)}', flush=True)
        temp = checkpoint.with_suffix('.tmp')
        torch.save(dict(config=config, means=means), temp)
        temp.replace(checkpoint)
    rows = []
    for item, template, (base, target) in itertools.product(test_items, args.test_templates, pairs):
        rows.append(dict(row_index=len(rows), base=item, source=item, template_id=template,
            base_attribute=base, donor_attribute=target, base_label=items[item][base], source_label=items[item][target]))
    vectors = {}
    metadata = []
    for base, target in pairs:
        for k in args.head_ks:
            heads = selected_heads(sets, attributes, target, k, args.head_selection)
            v = direction(means, base, target, heads, runner.head_dim)
            rand = random_direction(v, heads, runner.head_dim, args.seed+10000)
            vectors[base, target, k] = (v, rand)
            metadata.append(dict(base=base, target=target, k=k, heads=heads,
                                 norms={b: d.norm().item() for b,d in v.items()}))
    atomic_json(path/'direction_metadata.json', metadata)
    for row in rows:
        specs = [(f'{kind}/top{k}/alpha{alpha:g}', k, alpha, kind) for k in args.head_ks
                 for alpha in args.strengths for kind in ['direction', 'reverse', 'random']]
        if all(results.has(row, arm) for arm in ['clean', 'target_clean']+[s[0] for s in specs]):
            continue
        with Image.open(directory/items[row['base']]['image']) as f:
            image = f.convert('RGB')
        batch = batch_for(runner, image, trace['entity'], row['base_attribute'], row['template_id'], args.prefill)
        target_batch = batch_for(runner, image, trace['entity'], row['donor_attribute'], row['template_id'], args.prefill)
        gold = {role: derive_gold_token_ids(runner.processor.tokenizer, args.prefill, row[f'{role}_label'])
                for role in ['base', 'source']}
        if any(not g or len(g) > args.max_new_tokens for g in gold.values()):
            raise ValueError('Increase max_new_tokens to cover complete gold answers')
        def save(arm, current, step):
            if results.has(row, arm):
                return
            generated, _ = runner.generate(current, step, gold)
            results.add(runner.record(row, arm, generated, gold, base_attribute=row['base_attribute'],
                                      donor_attribute=row['donor_attribute']))
        save('clean', batch, lambda ids: (runner.forward(ids, batch)[0], {}))
        save('target_clean', target_batch, lambda ids: (runner.forward(ids, target_batch)[0], {}))
        for arm, k, alpha, kind in specs:
            v, rand = vectors[row['base_attribute'], row['donor_attribute'], k]
            save(arm, batch, additive_step(runner, batch, rand if kind == 'random' else v,
                                          -alpha if kind == 'reverse' else alpha, args.mode))
        summarize(results)
    summarize(results)
