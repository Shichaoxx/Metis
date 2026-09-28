#!/usr/bin/env python3
"""Local-weight numerical control; no downloads and no performance claims."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import traceback

from benchmark_layouts import compare, compile_stats, digest, reference_readout_check, save, sync


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--backend', choices=['sdpa', 'eager'], default='sdpa')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--max-tree-tokens', type=int, default=2048)
    parser.add_argument('--pair-batch-size', type=int, default=4)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--queries', type=int, default=1)
    parser.add_argument('--split', default='test')
    parser.add_argument('--query-ids', nargs='+', default=None)
    parser.add_argument('--reference-candidates', type=int, default=3)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--repo', type=Path, default=None)
    parser.add_argument('--allow-tiny-for-smoke', action='store_true')
    return parser.parse_args()


def comparison(ids, left, right):
    result = compare(ids, left, right)
    a, b = result['pairs_ranked_ids'], result['tree_ranked_ids']
    result['left_ranked_ids'], result['right_ranked_ids'] = a, b
    del result['pairs_ranked_ids'], result['tree_ranked_ids']
    result['top3_overlap'] = len(set(a[:3]) & set(b[:3])) / min(3, len(ids))
    result['top5_overlap'] = len(set(a[:5]) & set(b[:5])) / min(5, len(ids))
    return result


def main():
    args = parse_args()
    report = {'status': 'starting', 'started_at': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Finite-sample numerical and candidate-permutation control; no latency or quality claim',
              'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'queries': [], 'failures': []}
    output = args.output.resolve()
    save(output, report)
    try:
        if args.queries < 1 or args.candidates < 1 or args.reference_candidates < 1:
            raise ValueError('queries, candidates and reference-candidates must be positive')
        if not args.model.is_dir() or not (args.model / 'config.json').is_file():
            raise ValueError('A downloaded local --model directory is required; no downloader is called')
        if args.repo:
            sys.path.insert(0, str(args.repo.resolve() / 'src'))
        import torch
        from cometa.model import QwenReranker
        from cometa.schema import load_manifest, read_jsonl
        torch.set_num_threads(4)
        device = torch.device(args.device)
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable in this process')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in this process')
        if device.type not in ('mps', 'cuda', 'cpu'):
            raise ValueError('Supported devices: mps, cuda, cpu')
        manifest = load_manifest(args.manifest)
        split = manifest['splits'][args.split]
        samples = list(read_jsonl(split['path'], require_labels=False))
        if args.query_ids:
            if len(set(args.query_ids)) != len(args.query_ids):
                raise ValueError('Duplicate query IDs')
            by_id = {sample['id']: sample for sample in samples}
            samples = [by_id[qid] for qid in args.query_ids]
        else:
            samples = samples[:args.queries]
            if len(samples) != args.queries:
                raise ValueError('Too few queries in the split')
        selected = []
        for source in samples:
            if len(source['candidates']) < args.candidates:
                raise ValueError(f"Query {source['id']} has fewer than {args.candidates} candidates")
            sample = copy.deepcopy(source)
            sample['candidates'] = sample['candidates'][:args.candidates]
            sample.pop('supervision', None)
            selected.append(sample)
        report['protocol'] = {'manifest_sha256': manifest['_manifest_sha256'],
            'split_sha256': split['sha256'], 'model_config_sha256': digest(args.model / 'config.json'),
            'selected_query_ids': [sample['id'] for sample in selected],
            'selected_candidate_ids': {s['id']: [c['id'] for c in s['candidates']] for s in selected},
            'selection': 'First K frozen candidates; no labels passed to the model; no positive injection',
            'same_model_instance_and_weights': True, 'dropout': 'disabled with eval()', 'autograd': 'no_grad',
            'dtype': args.dtype, 'backend': args.backend, 'device': str(device),
            'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
            'runner_sha256': digest(__file__)}
        report['status'] = 'loading_model'
        save(output, report)
        print(json.dumps({'event': 'loading_model', 'device': str(device), 'dtype': args.dtype}), flush=True)
        model = QwenReranker(str(args.model.resolve()), layout='pairs', backend=args.backend,
            device=str(device), dtype=args.dtype, revision=args.revision,
            max_length=args.max_length, max_tree_tokens=args.max_tree_tokens,
            pair_batch_size=args.pair_batch_size)
        model.eval()
        sync(torch, device)
        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count < 100_000_000 and not args.allow_tiny_for_smoke:
            raise ValueError('Tiny checkpoint requires --allow-tiny-for-smoke; not real-model validation')
        report['model'] = {'parameter_count': parameter_count, 'actual_dtype': str(next(model.parameters()).dtype),
            'actual_device': str(model.device), 'resolved_revision': model.revision,
            'run_kind': 'tiny_software_smoke' if parameter_count < 100_000_000 else 'public_checkpoint_numerical_control',
            'yes_token_id': model.yes_id, 'no_token_id': model.no_id, 'input_spec': model.compiler.spec()}
        report['reference_readout'] = reference_readout_check(torch, model, selected[0], args.reference_candidates)
        for sample in selected:
            reverse = copy.deepcopy(sample)
            reverse['candidates'].reverse()
            ids = [candidate['id'] for candidate in sample['candidates']]
            original_compiled = model.compiler.compile(sample)
            reversed_compiled = model.compiler.compile(reverse)
            reverse_tokens = dict(zip(reversed_compiled.candidate_ids, reversed_compiled.pairs))
            same_tokens = all(tokens == reverse_tokens[cid] for cid, tokens in zip(original_compiled.candidate_ids,
                                                                                   original_compiled.pairs))
            record = {'query_id': sample['id'], 'canonical_candidate_ids': ids,
                'reverse_input_candidate_ids': [c['id'] for c in reverse['candidates']],
                'per_candidate_tokens_identical_after_permutation': same_tokens,
                'candidate_token_ids_sha256': {cid: hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                    for cid, tokens in zip(original_compiled.candidate_ids, original_compiled.pairs)},
                'compiler_original': compile_stats(model, sample), 'compiler_reversed': compile_stats(model, reverse),
                'readout_positions': {cid: {'pair_last_token_index': len(tokens) - 1,
                                           'position_id': len(tokens) - 1, 'token_id': tokens[-1]}
                    for cid, tokens in zip(original_compiled.candidate_ids, original_compiled.pairs)},
                'scores_aligned_to_canonical_ids': {}, 'input_metadata': {}}
            report['queries'].append(record)
            if not same_tokens:
                raise ValueError('Compiler changed candidate token sequences under permutation')
            for layout, order, request in [('pairs', 'original', sample), ('pairs', 'reversed', reverse),
                                            ('tree', 'original', sample), ('tree', 'reversed', reverse)]:
                label = f'{layout}_{order}'
                report['status'] = 'checking_invariance'
                report['progress'] = {'query_id': sample['id'], 'pass': label}
                save(output, report)
                print(json.dumps({'event': 'scoring', **report['progress']}), flush=True)
                model.layout = layout
                sync(torch, device)
                raw = model.score([request])[0]
                sync(torch, device)
                if len(raw) != len(ids) or not all(math.isfinite(x) for x in raw):
                    raise ValueError('Incomplete/nonfinite scores')
                indexed = dict(zip([c['id'] for c in request['candidates']], raw))
                record['scores_aligned_to_canonical_ids'][label] = [indexed[cid] for cid in ids]
                record['input_metadata'][label] = copy.deepcopy(model.last_input_metadata[0])
                save(output, report)
            values = record['scores_aligned_to_canonical_ids']
            record['comparisons'] = {
                'pairs_vs_tree_original': comparison(ids, values['pairs_original'], values['tree_original']),
                'pairs_vs_tree_reversed': comparison(ids, values['pairs_reversed'], values['tree_reversed']),
                'pairs_original_vs_reversed': comparison(ids, values['pairs_original'], values['pairs_reversed']),
                'tree_original_vs_reversed': comparison(ids, values['tree_original'], values['tree_reversed'])}
            save(output, report)
        keys = report['queries'][0]['comparisons'].keys()
        report['summary'] = {key: {
            'max_abs_score_diff': max(row['comparisons'][key]['max_abs_score_diff'] for row in report['queries']),
            'mean_abs_score_diff_across_equal_size_queries': sum(row['comparisons'][key]['mean_abs_score_diff']
                for row in report['queries']) / len(report['queries']),
            'exact_ranking_agreement_queries': sum(row['comparisons'][key]['exact_full_ranking_agreement']
                for row in report['queries']), 'query_count': len(report['queries'])}
            for key in keys}
        report['status'] = 'completed'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report.pop('progress', None)
        save(output, report)
        print(json.dumps({'event': 'completed', 'output': str(output), 'summary': report['summary']}), flush=True)
    except Exception as exc:
        report['status'] = 'failed'
        report['failures'].append({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        save(output, report)
        raise


if __name__ == '__main__':
    main()
