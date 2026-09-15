"""Shared, full-prefix machinery for the two head-trace follow-up experiments.

Every decode step recomputes the entire prefix with use_cache=False. This is
slower than generate(), but keeps image/head/attention interventions at previous
positions in force without manually constructing a multimodal KV cache. A donor
sees the RECIPIENT's generated prefix, never gold tokens or its own rollout.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]


def parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--trace', required=True, help='Existing head_trace JSON; freezes its head ranking.')
    ap.add_argument('--model_id', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--vade_root', default=os.environ.get('VADE_ROOT', str(ROOT.parent / 'VADE')))
    ap.add_argument('--split', choices=['train', 'test'], default='test')
    ap.add_argument('--allow_unpruned', action='store_true')
    ap.add_argument('--sample_seed', type=int, default=None,
                    help='Default reproduces trace sample; change this for a fresh evaluation sample.')
    ap.add_argument('--limit', type=int, default=None, help='Take first N rows AFTER sampling trace.n_rows.')
    ap.add_argument('--max_new_tokens', type=int, default=12)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_dir', default=None, help='New or matching resume directory; config checked.')
    ap.add_argument('--dry_run', action='store_true', help='Validate trace, tuples and images without torch/model.')
    return ap


def prepare(args, experiment):
    """Read metadata/rows without importing torch so dry runs work on the laptop."""
    raw = Path(args.trace).read_bytes()
    trace = json.loads(raw)
    required = ['entity', 'attribute', 'patch_layer', 'positions', 'blocks', 'n_rows', 'seed', 'phase1']
    if any(k not in trace for k in required):
        raise ValueError(f'Trace must contain {required}')
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('max_new_tokens and limit must be positive')
    rank_by = trace.get('rank_by', 'delta_resid')
    ranked = [(int(r['block']), int(r['head'])) for r in
              sorted(trace['phase1'], key=lambda r: -abs(r[rank_by]))]
    if not ranked or len(set(ranked)) != len(ranked):
        raise ValueError('Trace has empty or duplicated head ranking')
    if trace['patch_layer'] < 1 or any(b < trace['patch_layer'] for b, _ in ranked):
        raise ValueError('Use a trace window starting at/after the image patch layer (>=1)')
    entity_dir = Path(args.vade_root) / 'data' / trace['entity']
    root = (entity_dir / 'tuples' if args.allow_unpruned else
            Path(args.vade_root) / 'models' / args.model_id.split('/')[-1] / trace['entity'] / 'tuples')
    tuples = root / trace['attribute'] / f'{args.split}.jsonl'
    rows = [json.loads(line) for line in tuples.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['rule'] == 'match_source']
    seed = trace['seed'] if args.sample_seed is None else args.sample_seed
    rows = random.Random(seed).sample(rows, min(trace['n_rows'], len(rows)))
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No cause rows selected')
    gt = json.loads((entity_dir / 'ground_truth.json').read_text())
    key = {'flags': 'countries', 'animals': 'species', 'brands': 'brands'}.get(trace['entity'])
    if key not in gt:
        key = next(k for k, v in gt.items() if isinstance(v, dict) and k != 'coverage')
    items = gt[key]
    for row in rows:
        row.setdefault('target_attribute', trace['attribute'])
        for role in ('base', 'source'):
            if not (entity_dir / items[row[role]]['image']).is_file():
                raise FileNotFoundError(entity_dir / items[row[role]]['image'])
    config = {k: v for k, v in vars(args).items() if k not in ('dry_run', 'out_dir', 'trace')}
    config.update(experiment=experiment, schema_version=1, trace_sha256=hashlib.sha256(raw).hexdigest(),
                  entity=trace['entity'], attribute=trace['attribute'], patch_layer=trace['patch_layer'],
                  positions=trace['positions'], blocks=trace['blocks'], rank_by=rank_by,
                  ranking=ranked, row_indices=[r['row_index'] for r in rows],
                  rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                  effective_seed=seed, prefix_policy='recipient_generated; full-prefix recomputation')
    implementation = [ROOT / 'methods' / f for f in ('head_followup_common.py', 'head_decode_trace.py',
                       'head_token_trace.py', 'head_token_ablation.py')]
    config['implementation_sha256'] = hashlib.sha256(b''.join(p.read_bytes() for p in implementation)).hexdigest()
    # JSON round-trip makes tuples and lists identical on resume.
    config = json.loads(json.dumps(config))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / 'results' / 'head_followups' / trace['entity'] / trace['attribute'] / f'{experiment}_{digest}')
    print(f'{experiment}: {trace["entity"]}/{trace["attribute"]}, {len(rows)} rows, '
          f'blocks={trace["blocks"]}, output={out_dir}', flush=True)
    print(f'Frozen ranking: {ranked[:8]}; sample seed={seed}', flush=True)
    return trace, ranked, rows, config, out_dir


def score_tokens(generated, gold):
    """Full, uncapped gold suffix. A short generation counts as missing/wrong."""
    correct = [i < len(generated) and generated[i] == t for i, t in enumerate(gold)]
    return {'gold_length': len(gold), 'first_token': bool(correct and correct[0]),
            'prefix3_match': bool(correct and all(correct[:3])),
            'full_match': bool(correct and all(correct)), 'per_token': correct}


def summarize(records):
    grouped = {}
    for rec in records:
        grouped.setdefault(rec['arm'], []).append(rec)
    result = {}
    for arm, rows in grouped.items():
        def metrics(rr):
            n = len(rr)
            scores = [r['source_score'] for r in rr]
            distinct = [r for r in rr if not r.get('first_token_collision', False)]
            positions = range(max(s['gold_length'] for s in scores))
            return {'n': n, **{k: sum(s[k] for s in scores) / n
                              for k in ('first_token', 'prefix3_match', 'full_match')},
                    'base_full_match': sum(r['base_score']['full_match'] for r in rr) / n,
                    'distinct_first_token_n': len(distinct),
                    'first_token_accuracy_distinct_gold': (sum(r['source_score']['first_token'] for r in distinct)
                                                           / len(distinct) if distinct else None),
                    'per_token': [{'position': i + 1,
                                   'n': sum(s['gold_length'] > i for s in scores),
                                   'accuracy': sum(s['per_token'][i] for s in scores if s['gold_length'] > i)
                                   / sum(s['gold_length'] > i for s in scores)} for i in positions],
                    'generation_budget_shorter_than_gold': sum(r['budget_shorter_than_gold'] for r in rr)}
        result[arm] = metrics(rows)
        lengths = sorted({r['source_score']['gold_length'] for r in rows})
        result[arm]['by_gold_length'] = {str(length): metrics([
            r for r in rows if r['source_score']['gold_length'] == length]) for length in lengths}
    # Paired differences are on the same rows; useful when an interrupted run is partial.
    baseline_name = 'image_patch' if 'image_patch' in grouped else 'no_knockout'
    baseline = {r['row_index']: r for r in grouped.get(baseline_name, [])}
    for arm, rows in grouped.items():
        paired = [r for r in rows if r['row_index'] in baseline]
        if paired:
            result[arm]['paired_baseline'] = baseline_name
            result[arm]['paired_baseline_full_match_drop'] = sum(
                int(baseline[r['row_index']]['source_score']['full_match'])
                - int(r['source_score']['full_match']) for r in paired) / len(paired)
            if all(r.get('token_scores') and baseline[r['row_index']].get('token_scores') for r in paired):
                result[arm]['paired_first_source_logprob_drop'] = sum(
                    baseline[r['row_index']]['token_scores'][0]['source']['logprob']
                    - r['token_scores'][0]['source']['logprob'] for r in paired) / len(paired)
    return result


class Results:
    """Append per arm; resume only the identical config; keep raw rows and summaries."""
    def __init__(self, directory, config):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        meta = self.path / 'config.json'
        if meta.exists() and json.loads(meta.read_text()) != config:
            raise ValueError(f'{self.path} belongs to a different configuration; use another --out_dir')
        meta.write_text(json.dumps(config, indent=2) + '\n')
        self.records = []
        dest = self.path / 'rows.jsonl'
        if dest.exists():
            # Discard only an incomplete final write after interruption, never a corrupt interior row.
            lines = dest.read_text().splitlines(keepends=True)
            for i, line in enumerate(lines):
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    if i != len(lines) - 1 or line.endswith('\n'):
                        raise
                    dest.write_text(''.join(lines[:i]))
            if self.records and dest.read_text() and not dest.read_text().endswith('\n'):
                with dest.open('a') as f:
                    f.write('\n')
        self.done = {(r['row_index'], r['arm']) for r in self.records}

    def has(self, row, arm):
        return (row['row_index'], arm) in self.done

    def check_runtime(self, runner):
        import torch
        import transformers
        runtime = {'torch': torch.__version__, 'transformers': transformers.__version__,
                   'model_config': runner.model.config.to_dict(), 'dtype': str(runner.model.dtype),
                   'attention_backend': 'eager'}
        runtime = json.loads(json.dumps(runtime))
        path = self.path / 'runtime.json'
        if path.exists() and json.loads(path.read_text()) != runtime:
            raise ValueError('Runtime/model differs from this output directory; use a new --out_dir')
        path.write_text(json.dumps(runtime, indent=2) + '\n')

    def add(self, record):
        key = record['row_index'], record['arm']
        if key in self.done:
            return
        with (self.path / 'rows.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
            f.flush()
        self.records.append(record)
        self.done.add(key)
        scores = record['source_score']
        print(f'  row={record["row_index"]} {record["arm"]}: '
              f'first={int(scores["first_token"])} full={int(scores["full_match"])} '
              f'base_full={int(record["base_score"]["full_match"])} '
              f'answer={record["generated_text"]!r}', flush=True)

    def finish(self):
        summary = summarize(self.records)
        tmp = self.path / 'summary.json.tmp'
        tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        tmp.replace(self.path / 'summary.json')
        return summary


class Runner:
    def __init__(self, args, trace):
        import torch
        from methods.adapters.registry import get_adapter
        from methods.common.entities import BuildBatchCache, load_entity_assets
        self.args, self.trace = args, trace
        self.adapter = get_adapter(args.model_id)
        self.model, self.processor = self.adapter.load(
            device=args.device, dtype=torch.float32 if args.device == 'cpu' else torch.bfloat16,
            attn_implementation='eager')
        self.assets = load_entity_assets(args.vade_root, trace['entity'])
        self.cache = BuildBatchCache()
        self.n_heads = self.adapter.n_attention_heads(self.model)
        self.hidden = self.adapter.hidden_size(self.model)
        self.head_dim = self.hidden // self.n_heads
        self.n_layers = len(self.adapter.get_decoder_layers(self.model))
        if self.hidden % self.n_heads or not 1 <= trace['patch_layer'] < self.n_layers:
            raise ValueError('Invalid model dimensions or patch layer')
        for r in trace['phase1']:
            if not 0 <= r['block'] < self.n_layers or not 0 <= r['head'] < self.n_heads:
                raise ValueError('Trace head indices do not fit the loaded model')

    def batch(self, row):
        from methods.common.position_sets import build_batch_at
        from methods.common.targets import derive_gold_token_ids
        batch = build_batch_at(self.trace['positions'], [row], self.assets, self.adapter,
                               self.model, self.processor, batch_cache=self.cache)
        p = batch['base_input_ids'].shape[1]
        assert p > 1 and int(batch['positions'].max()) < p - 1
        assert batch['attention_mask'].all(), 'Follow-ups deliberately run one unpadded row at a time'
        template = self.assets.template_lookup[row['queried']][row['template_id']]
        gold = {role: derive_gold_token_ids(self.processor.tokenizer, template['prefill'], row[f'{role}_label'])
                for role in ('base', 'source')}
        return batch, gold

    def source(self, batch):
        from methods.common.sites import RESIDUAL_SITE
        return RESIDUAL_SITE.capture(self.adapter, self.model, self.trace['patch_layer'],
                                     batch['source_input_ids'], batch['attention_mask'],
                                     batch['source_extra'], batch['positions'])

    @contextlib.contextmanager
    def image_patch(self, batch, source):
        from methods.common.hooks import make_cache_aware_patch_hook
        from methods.common.sites import RESIDUAL_SITE
        fn = make_cache_aware_patch_hook(batch['positions'], lambda _: source)
        handles = RESIDUAL_SITE.register(self.adapter, self.model,
                                        self.adapter.get_decoder_layers(self.model),
                                        self.trace['patch_layer'], fn)
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def forward(self, ids, batch, source=None, capture_blocks=(), patches=None, observer=None):
        """Captures o_proj input at ALL readout positions, after any head patch."""
        import torch
        from methods.common.hooks import extra_to_device
        start = batch['base_input_ids'].shape[1] - 1
        captured, handles = {}, []
        with contextlib.ExitStack() as stack:
            if source is not None:
                stack.enter_context(self.image_patch(batch, source))
            if observer is not None:
                stack.enter_context(observer)
            try:
                for b in sorted(set(capture_blocks) | set(patches or {})):
                    module = self.adapter.get_attn_head_output_module(self.model, b)
                    def pre(mod, inputs, block=b):
                        z = inputs[0]
                        if patches and block in patches:
                            donor, heads, mode = patches[block]
                            z = replace_heads(z, donor, heads, self.head_dim, start, mode)
                        if block in capture_blocks:
                            captured[block] = z[:, start:].detach().clone()
                        return (z,) + inputs[1:]
                    handles.append(module.register_forward_pre_hook(pre))
                with torch.no_grad():
                    out = self.model(input_ids=ids.to(self.model.device),
                                     attention_mask=torch.ones_like(ids, device=self.model.device),
                                     **extra_to_device(batch['base_extra'], self.model.device, self.model.dtype),
                                     use_cache=False, output_attentions=observer is not None, logits_to_keep=1)
                return out.logits[:, -1].detach(), captured
            finally:
                for h in handles:
                    h.remove()

    def generate(self, batch, step, gold=None):
        """step(ids) -> logits, telemetry. Optional gold is ONLY scored after forward.

        The next input is always argmax(logits), never a gold token. Logprobs at
        later steps are conditional on this arm's generated prefix, not teacher forcing.
        """
        import torch
        ids = batch['base_input_ids'].to(self.model.device)
        generated, telemetry = [], []
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for index in range(self.args.max_new_tokens):
            logits, info = step(ids)
            if gold is not None:
                logprobs = logits[0].float().log_softmax(-1)
                info['gold_token_scores'] = {role: {'token_id': tokens[index],
                                                   'logit': float(logits[0, tokens[index]]),
                                                   'logprob': float(logprobs[tokens[index]])}
                                            for role, tokens in gold.items() if index < len(tokens)}
            token = int(logits[0].argmax())
            generated.append(token)
            telemetry.append(info)
            if token in eos:
                break
            ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)
        return generated, telemetry

    def record(self, row, arm, generated, gold, **extra):
        tok = self.processor.tokenizer
        return {'row_index': row['row_index'], 'base': row['base'], 'source': row['source'],
                'template_id': row['template_id'], 'arm': arm, 'generated_ids': generated,
                'generated_text': tok.decode(generated, skip_special_tokens=True),
                'generated_tokens': tok.convert_ids_to_tokens(generated), 'gold_ids': gold,
                'source_score': score_tokens(generated, gold['source']),
                'base_score': score_tokens(generated, gold['base']),
                'first_token_collision': bool(gold['source'] and gold['base'] and
                                              gold['source'][0] == gold['base'][0]),
                'budget_shorter_than_gold': self.args.max_new_tokens < len(gold['source']), **extra}


def replace_heads(z, donor, heads, head_dim, start, mode):
    """Tensor-only primitive. Donor is [B, all readout positions, hidden]."""
    if mode not in ('prefill', 'continuous'):
        raise ValueError(mode)
    if donor.shape != z[:, start:].shape:
        raise ValueError('Donor and recipient must have the same generated prefix length')
    result = z.clone()
    stop = start + 1 if mode == 'prefill' else z.shape[1]
    for h in heads:
        sl = slice(h * head_dim, (h + 1) * head_dim)
        result[:, start:stop, sl] = donor[:, :stop - start, sl].to(z)
    return result


def grouped_heads(heads):
    grouped = {}
    for block, head in heads:
        grouped.setdefault(block, []).append(head)
    return grouped
