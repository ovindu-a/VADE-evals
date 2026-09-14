"""Same-image, different-question full-swap sweep. No masks are trained here."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.head_followup_common import ROOT, Results, Runner

ATTRIBUTES = ['capital', 'currency', 'language', 'calling_code']
FIELDS = dict(capital='capital', currency='currency', language='language', calling_code='calling')
SITES = ['earlier_text', 'last_residual', 'last_attention', 'last_mlp', 'last_joint']
PREFILL = 'Answer:'
INSTRUCTION = ('For the country shown, capital means its capital city; currency means its '
               'three-letter currency code; language means one official language; calling '
               'means its international calling code, digits only without a plus sign. '
               'Give only the requested value. ')


def question(attribute, paraphrase=False):
    return INSTRUCTION + f'{"Return" if paraphrase else "Report"} the {FIELDS[attribute]} of this country.'


def alignment(base, donor, image_id, special_ids=()):
    """No padding, resampling or silent position shifting is permitted."""
    import torch
    if base.shape != donor.shape or int(base[0, -1]) != int(donor[0, -1]):
        raise ValueError('Prompts must have equal token lengths and the same final token; adjust the controlled wording')
    image = base[0] == image_id
    if not image.any() or not torch.equal(image, donor[0] == image_id):
        raise ValueError('Image positions must match')
    last_image = int(image.nonzero()[-1])
    if not torch.equal(base[:, :last_image + 1], donor[:, :last_image + 1]):
        raise ValueError('All prompt differences must occur after the image')
    positions = [i for i, token in enumerate(base[0].tolist()[:-1])
                 if token != image_id and token not in special_ids]
    if not positions:
        raise ValueError('No earlier text positions')
    return positions


def parts(name, layer):
    from methods.common.sites import resolve_site
    mapped = {'earlier_text': 'residual', 'last_residual': 'residual',
              'last_attention': 'attn_output', 'last_mlp': 'mlp_output',
              'last_joint': 'attn_output+mlp_output'}
    site = resolve_site(mapped.get(name, name))
    if layer < site.min_layer():
        raise ValueError(f'{name} cannot end at layer {layer}')
    return [(p.site, p.layer_idx(layer)) for p in site.parts] if site.is_joint else [(site, layer)]


class SwitchRunner(Runner):
    def __init__(self, args):
        import torch
        from methods.adapters.registry import get_adapter
        self.args = args
        self.adapter = get_adapter(args.model_id)
        self.model, self.processor = self.adapter.load(device=args.device,
            dtype=torch.float32 if args.device == 'cpu' else torch.bfloat16, attn_implementation='eager')
        self.model.eval().requires_grad_(False)
        self.layers = self.adapter.get_decoder_layers(self.model)

    def batch(self, image, base_attribute, donor_attribute):
        import torch
        base = self.adapter.build_inputs(self.processor, image, question(base_attribute), PREFILL)
        donor = self.adapter.build_inputs(self.processor, image, question(donor_attribute), PREFILL)
        para = self.adapter.build_inputs(self.processor, image, question(base_attribute, True), PREFILL)
        ids = [x['input_ids'].unsqueeze(0) for x in (base, donor, para)]
        image_id = self.adapter.image_token_id(self.model, self.processor)
        special = self.processor.tokenizer.all_special_ids
        positions = alignment(ids[0], ids[1], image_id, special)
        alignment(ids[0], ids[2], image_id, special)
        for other in (donor, para):
            if set(base['extra']) != set(other['extra']) or any(
                    not torch.equal(v, other['extra'][k]) for k, v in base['extra'].items()):
                raise ValueError('Same-image runs produced different vision inputs')
        return dict(base_input_ids=ids[0], source_input_ids=ids[1], paraphrase_ids=ids[2],
                    base_extra=base['extra'], earlier_positions=positions)

    def run(self, ids, batch, hooks=()):
        import torch
        from methods.common.hooks import extra_to_device
        handles = []
        try:
            for site, layer, fn in hooks:
                handles.extend(site.register(self.adapter, self.model, self.layers, layer, fn))
            with torch.no_grad():
                return self.model(input_ids=ids.to(self.model.device),
                    attention_mask=torch.ones_like(ids, device=self.model.device),
                    **extra_to_device(batch['base_extra'], self.model.device, self.model.dtype),
                    use_cache=False, logits_to_keep=1).logits[:, -1].detach()
        finally:
            for h in handles:
                h.remove()

    def step(self, batch, name, layer, mode, donor_kind='switch'):
        import torch
        prompt_length = batch['base_input_ids'].shape[1]
        donor_prompt = batch[{'switch': 'source_input_ids', 'self': 'base_input_ids',
                             'paraphrase': 'paraphrase_ids'}[donor_kind]]
        site_parts = parts(name, layer)
        def step(ids):
            # Donor question plus recipient's free-generated suffix, never a gold answer.
            donor_ids = torch.cat((donor_prompt.to(ids.device), ids[:, prompt_length:]), dim=1)
            if name == 'earlier_text':
                # Scope stays earlier prompt text in both modes: never silently add readout positions.
                positions = batch['earlier_positions']
            else:
                positions = list(range(prompt_length - 1,
                                       ids.shape[1] if mode == 'continuous' else prompt_length))
            captured = {}
            capture_hooks = []
            for index, (site, at) in enumerate(site_parts):
                def capture(z, i=index):
                    captured[i] = z[:, positions].detach().clone()
                    return z
                capture_hooks.append((site, at, capture))
            donor_logits = self.run(donor_ids, batch, capture_hooks)
            patch_hooks = []
            for index, (site, at) in enumerate(site_parts):
                def patch(z, i=index):
                    out = z.clone()
                    out[:, positions] = captured[i].to(z)
                    return out
                patch_hooks.append((site, at, patch))
            logits = self.run(ids, batch, patch_hooks)
            # Self-patching must be a no-op. Replacing the final residual at
            # the current readout must reproduce the donor's next-token logits.
            if donor_kind == 'self' or (name == 'last_residual' and layer == len(self.layers)
                                       and ids.shape[1] - 1 in positions):
                tolerance = 1e-5 if logits.dtype == torch.float32 else .02
                if not torch.allclose(logits.float(), donor_logits.float(), atol=tolerance, rtol=tolerance):
                    raise AssertionError('Full-swap identity check failed')
            return logits, {}
        return step


def summarize_switch(results):
    clean = {r['row_index']: r for r in results.records if r['arm'] == 'clean'}
    donor = {r['row_index']: r for r in results.records if r['arm'] == 'donor_clean'}
    groups = {}
    for r in results.records:
        groups.setdefault((r['arm'], r['base_attribute'], r['donor_attribute']), []).append(r)
    cells = []
    for (arm, base, source), rr in sorted(groups.items()):
        eligible = [r for r in rr if clean[r['row_index']]['base_score']['full_match']
                    and donor[r['row_index']]['source_score']['full_match']]
        def scores(items):
            if not items:
                return None
            return dict(n=len(items), country_count=len({r['base'] for r in items}),
                donor_full=sum(r['source_score']['full_match'] for r in items) / len(items),
                original_full=sum(r['base_score']['full_match'] for r in items) / len(items),
                neither_full=sum(not (r['source_score']['full_match'] or r['base_score']['full_match'])
                                 for r in items) / len(items),
                donor_first=sum(r['source_score']['first_token'] for r in items) / len(items))
        cells.append(dict(arm=arm, base_attribute=base, donor_attribute=source,
                          all=scores(rr), both_clean_correct=scores(eligible)))
    results.finish()
    temp = results.path / 'switch_summary.json.tmp'
    temp.write_text(json.dumps(cells, indent=2) + '\n')
    temp.replace(results.path / 'switch_summary.json')


def sweep_arms(args):
    if getattr(args, 'baselines_only', False):
        return []
    requested = list(dict.fromkeys(args.sites + ['last_joint' if n == 1 else f'blocks:{n}'
                                                 for n in args.block_spans]))
    arms = []
    for layer in args.layers:
        for site in requested:
            if site.startswith('blocks:') and int(site.split(':')[1]) > layer:
                continue
            for mode in (['prefill'] if site == 'earlier_text' else args.modes):
                for control in ['switch'] + args.controls:
                    arms.append((f'{site}/L{layer}/{mode}/{control}', site, layer, mode, control))
    return arms


def execute(runner, args, rows, items, entity_dir, results):
    from PIL import Image
    import torch
    from methods.common.targets import derive_gold_token_ids
    if any(not 1 <= layer <= len(runner.layers) for layer in args.layers):
        raise ValueError(f'Layer must be in 1..{len(runner.layers)}')
    arms = sweep_arms(args)
    for row in rows:
        if all(results.has(row, a) for a in ['clean', 'donor_clean'] + [x[0] for x in arms]):
            continue
        with Image.open(entity_dir / items[row['base']]['image']) as img:
            batch = runner.batch(img.convert('RGB'), row['base_attribute'], row['donor_attribute'])
        gold = {role: derive_gold_token_ids(runner.processor.tokenizer, PREFILL, row[f'{role}_label'])
                for role in ('base', 'source')}
        if any(not g or len(g) > args.max_new_tokens for g in gold.values()):
            raise ValueError('Empty answer or insufficient max_new_tokens for full gold suffix')
        p = batch['base_input_ids'].shape[1]
        def save(arm, step, **meta):
            if results.has(row, arm):
                return
            generated, _ = runner.generate(batch, step, gold)
            results.add(runner.record(row, arm, generated, gold,
                base_attribute=row['base_attribute'], donor_attribute=row['donor_attribute'], **meta))
        save('clean', lambda ids: (runner.run(ids, batch), {}),
             base_prompt_ids=batch['base_input_ids'][0].tolist(),
             donor_prompt_ids=batch['source_input_ids'][0].tolist(),
             paraphrase_prompt_ids=batch['paraphrase_ids'][0].tolist(),
             earlier_positions=batch['earlier_positions'])
        save('donor_clean', lambda ids: (runner.run(torch.cat((batch['source_input_ids'].to(ids.device),
                                                            ids[:, p:]), dim=1), batch), {}))
        for arm, site, layer, mode, control in arms:
            save(arm, runner.step(batch, site, layer, mode, control), site=site, layer=layer,
                 mode=mode, control=control)
        summarize_switch(results)
    # Also rebuild a missing summary after an interruption following the last row write.
    summarize_switch(results)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model_id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--vade_root', default=os.environ.get('VADE_ROOT', str(ROOT.parent / 'VADE')))
    ap.add_argument('--out_dir', default='results/attribute_switch/flags')
    ap.add_argument('--countries', nargs='+', help='Explicit VADE country IDs; otherwise deterministic sample')
    ap.add_argument('--n_countries', type=int, default=8)
    ap.add_argument('--attributes', nargs='+', choices=ATTRIBUTES, default=ATTRIBUTES)
    ap.add_argument('--layers', nargs='+', type=int, default=list(range(1, 29)))
    ap.add_argument('--sites', nargs='+', choices=SITES, default=SITES)
    ap.add_argument('--block_spans', nargs='*', type=int, default=[2, 4, 8])
    ap.add_argument('--modes', nargs='+', choices=['prefill', 'continuous'], default=['prefill'])
    ap.add_argument('--controls', nargs='*', choices=['self', 'paraphrase'], default=['self', 'paraphrase'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max_new_tokens', type=int, default=12)
    ap.add_argument('--baselines_only', action='store_true', help='Check clean controlled-prompt accuracy before sweeping')
    ap.add_argument('--dry_run', action='store_true', help='Validate metadata only; runtime checks token alignment')
    args = ap.parse_args(argv)
    if (len(set(args.attributes)) < 2 or len(set(args.attributes)) != len(args.attributes)
            or args.n_countries < 1 or args.max_new_tokens < 1 or any(n < 1 for n in args.layers)
            or any(n < 1 for n in args.block_spans)):
        ap.error('Need at least two attributes and positive counts/spans')
    entity_dir = Path(args.vade_root) / 'data/flags'
    raw = (entity_dir / 'ground_truth.json').read_bytes()
    items = json.loads(raw)['countries']
    countries = args.countries or random.Random(args.seed).sample(sorted(items), min(args.n_countries, len(items)))
    if len(set(countries)) != len(countries):
        ap.error('Country IDs must be unique')
    rows = []
    for country in countries:
        item = items[country]
        if not (entity_dir / item['image']).is_file():
            raise FileNotFoundError(entity_dir / item['image'])
        for base, donor in itertools.permutations(args.attributes, 2):
            rows.append(dict(row_index=len(rows), base=country, source=country, base_attribute=base,
                donor_attribute=donor, base_label=item[base], source_label=item[donor],
                template_id='controlled_report_v1'))
    config = {k: v for k, v in vars(args).items() if k not in ('dry_run', 'out_dir')}
    config.update(rows=rows, prompts={a: [question(a), question(a, True)] for a in args.attributes},
        prefill=PREFILL, ground_truth_sha256=hashlib.sha256(raw).hexdigest(),
        image_sha256={c: hashlib.sha256((entity_dir / items[c]['image']).read_bytes()).hexdigest()
                      for c in countries},
        implementation_sha256=hashlib.sha256(b''.join((ROOT / 'methods' / p).read_bytes() for p in
            ['attribute_switch_sweep.py', 'head_followup_common.py', 'common/sites.py', 'common/hooks.py',
             'adapters/qwen2_5_vl.py'])).hexdigest())
    print(f'{len(countries)} countries, {len(rows)} directed question pairs, layers={args.layers}; '
          f'{len(rows) * (2 + len(sweep_arms(args)))} generation runs', flush=True)
    if args.dry_run:
        print('Metadata valid. Token alignment and clean answer accuracy require a model run.')
        return
    results = Results(args.out_dir, config)
    runner = SwitchRunner(args)
    results.check_runtime(runner)
    execute(runner, args, rows, items, entity_dir, results)


if __name__ == '__main__':
    main()
