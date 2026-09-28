#!/usr/bin/env python3
"""Compare real Qwen reranker pairs/tree with one shared set of loaded weights.

No downloader: --model must be an existing local model directory.
Example (after the operator downloads the public checkpoint):
  PYTHONPATH=/path/to/cometa/src python benchmark_layouts.py \
    --model /path/to/Qwen3-Reranker-0.6B --manifest /path/to/manifest.json \
    --device mps --dtype float16 --queries 3 --candidates 50 \
    --max-length 2048 --max-tree-tokens 4096 --pair-batch-size 4 \
    --warmup 1 --repeats 2 --output /tmp/layout-benchmark.json

A separate --dtype float32 --queries 1 --candidates 3 run is a useful rounding
control. Do not interpret this architecture microbenchmark as retrieval quality.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sync(torch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def memory_start(torch, device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        return {'allocated_before_bytes': torch.cuda.memory_allocated(device),
                'reserved_before_bytes': torch.cuda.memory_reserved(device)}
    if device.type == 'mps':
        return {'current_allocated_before_bytes': torch.mps.current_allocated_memory(),
                'driver_allocated_before_bytes': torch.mps.driver_allocated_memory()}
    return {}


def memory_end(torch, device, before):
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device)
        return before | {'kind': 'CUDA allocator peak during this measurement; includes model weights',
            'allocated_after_bytes': torch.cuda.memory_allocated(device),
            'peak_allocated_bytes': peak,
            'peak_extra_allocated_over_start_bytes': peak - before['allocated_before_bytes'],
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    if device.type == 'mps':
        return before | {'kind': 'MPS point-in-time snapshots, NOT peak memory',
            'current_allocated_after_bytes': torch.mps.current_allocated_memory(),
            'driver_allocated_after_bytes': torch.mps.driver_allocated_memory(),
            'peak_allocated_bytes': None,
            'peak_unavailable_reason': 'PyTorch MPS has no resettable peak-allocation counter used here; '
                                       'driver cache may persist from the preceding layout.'}
    return {'kind': 'not measured', 'peak_allocated_bytes': None,
            'peak_unavailable_reason': 'Process lifetime RSS is not a resettable per-layout tensor peak.'}


def ranking(ids, values):
    return [cid for cid, _ in sorted(zip(ids, values), key=lambda item: (-item[1], item[0]))]


def compare(ids, pairs, tree):
    diffs = [abs(a - b) for a, b in zip(pairs, tree)]
    pair_rank, tree_rank = ranking(ids, pairs), ranking(ids, tree)
    agreements = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sign_pair = (pairs[i] > pairs[j]) - (pairs[i] < pairs[j])
            sign_tree = (tree[i] > tree[j]) - (tree[i] < tree[j])
            agreements.append(sign_pair == sign_tree)
    k = min(10, len(ids))
    return {'max_abs_score_diff': max(diffs, default=0),
            'mean_abs_score_diff': statistics.mean(diffs) if diffs else 0,
            'exact_full_ranking_agreement': pair_rank == tree_rank,
            'same_rank_position_fraction': sum(a == b for a, b in zip(pair_rank, tree_rank)) / len(ids),
            'top_k': k, 'top_k_set_overlap_fraction': len(set(pair_rank[:k]) & set(tree_rank[:k])) / k,
            'strict_pairwise_order_agreement': statistics.mean(agreements) if agreements else 1.0,
            'pairwise_tie_rule': 'Exact score ties count as equal only if tied in both layouts.',
            'pairs_ranked_ids': pair_rank, 'tree_ranked_ids': tree_rank}


def compile_stats(model, sample):
    compiled = model.compiler.compile(sample)
    lengths = [len(tokens) for tokens in compiled.pairs]
    prefix = len(compiled.prefix_ids)
    chunks, current = [], prefix
    for branch in compiled.branch_ids:
        if current > prefix and current + len(branch) > model.max_tree_tokens:
            chunks.append(current)
            current = prefix
        current += len(branch)
    if current > prefix:
        chunks.append(current)
    padded_shapes = []
    for start in range(0, len(lengths), model.pair_batch_size):
        part = lengths[start:start + model.pair_batch_size]
        padded_shapes.append([len(part), max(part)])
    return {'shared_prefix_tokens': prefix,
            'pair_unpadded_lengths': dict(zip(compiled.candidate_ids, lengths)),
            'pairs_padded_batch_shapes': padded_shapes,
            'tree_packed_chunk_lengths': chunks,
            'candidate_truncated': dict(zip(compiled.candidate_ids, compiled.truncated)),
            'logical_pair_token_count': sum(lengths),
            'physical_tree_token_count_with_repeated_chunk_prefixes': sum(chunks),
            'pair_attention_shape_cells_per_head': sum(batch * length * length for batch, length in padded_shapes),
            'tree_dense_attention_shape_cells_per_head': sum(length * length for length in chunks),
            'attention_shape_note': 'Shape/work proxies only; fused kernels need not materialize these matrices.',
            'compiled_pair_token_ids_sha256': hashlib.sha256(json.dumps(compiled.pairs).encode()).hexdigest()}


def reference_readout_check(torch, model, sample, candidate_count):
    """Actual HF CausalLM forward versus optimized two-row readout, unpadded."""
    compiled = model.compiler.compile(sample)
    records = []
    with torch.no_grad():
        for cid, tokens in list(zip(compiled.candidate_ids, compiled.pairs))[:candidate_count]:
            ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
            mask = torch.ones_like(ids)
            position = torch.arange(len(tokens), device=model.device).unsqueeze(0)
            # Qwen3ForCausalLM computes its full vocabulary head on one token.
            full = model.backbone(input_ids=ids, attention_mask=mask, position_ids=position,
                                  use_cache=False, logits_to_keep=1, return_dict=True).logits[0, -1]
            expected = float((full[model.yes_id] - full[model.no_id]).float().cpu())
            actual = float(model._read_scores(ids, mask, position,
                (torch.tensor([0], device=model.device),
                 torch.tensor([len(tokens) - 1], device=model.device)))[0].cpu())
            records.append({'candidate_id': cid, 'tokens': len(tokens),
                            'hf_full_vocab_last_token_score': expected,
                            'optimized_two_row_score': actual, 'abs_diff': abs(expected - actual)})
    return {'query_id': sample['id'], 'candidate_count': len(records),
            'method': 'Same unpadded token IDs, mask and positions; HF backbone forward(logits_to_keep=1) '
                      'versus optimized two-row head; eval/no_grad; same loaded weights.',
            'max_abs_diff': max(row['abs_diff'] for row in records), 'records': records}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--source-model-id', default='Qwen/Qwen3-Reranker-0.6B')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--query-ids', nargs='+', default=None)
    parser.add_argument('--queries', type=int, default=3)
    parser.add_argument('--candidates', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float16')
    parser.add_argument('--backend', choices=['sdpa', 'eager'], default='sdpa')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--max-tree-tokens', type=int, default=4096)
    parser.add_argument('--pair-batch-size', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--reference-candidates', type=int, default=3)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--repo', type=Path, default=None)
    parser.add_argument('--allow-tiny-for-smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()
    report = {'status': 'starting', 'started_at': datetime.now(timezone.utc).isoformat(),
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'queries': [], 'failures': []}
    save(output, report)
    try:
        for key in ('queries', 'candidates', 'warmup', 'repeats', 'reference_candidates', 'cpu_threads'):
            if getattr(args, key) < 1:
                raise ValueError(f'--{key.replace("_", "-")} must be positive')
        if not args.model.is_dir() or not (args.model / 'config.json').is_file():
            raise ValueError('--model must be a downloaded local model directory; this runner never downloads')
        if args.repo:
            sys.path.insert(0, str(args.repo.resolve() / 'src'))
        import torch
        from cometa.model import QwenReranker
        from cometa.schema import load_manifest, read_jsonl
        torch.set_num_threads(args.cpu_threads)
        device = torch.device(args.device)
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is not available in this Python process')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available in this Python process')
        if device.type not in ('cpu', 'mps', 'cuda'):
            raise ValueError('Device must be cpu, mps or cuda')
        manifest = load_manifest(args.manifest)
        split = manifest['splits'][args.split]
        samples = list(read_jsonl(split['path'], require_labels=False))
        if args.query_ids:
            if len(set(args.query_ids)) != len(args.query_ids):
                raise ValueError('Duplicate --query-ids')
            mapping = {s['id']: s for s in samples}
            samples = [mapping[qid] for qid in args.query_ids]
        else:
            samples = samples[:args.queries]
            if len(samples) != args.queries:
                raise ValueError('Split has fewer queries than requested')
        selected = []
        for original in samples:
            if len(original['candidates']) < args.candidates:
                raise ValueError(f"Query {original['id']} has fewer than {args.candidates} candidates")
            sample = copy.deepcopy(original)
            sample['candidates'] = sample['candidates'][:args.candidates]
            sample.pop('supervision', None)
            selected.append(sample)
        report['protocol'] = {
            'purpose': 'Architecture correctness and warmed inference latency, not retrieval quality',
            'source_model_id': args.source_model_id,
            'model_local_config_sha256': digest(args.model / 'config.json'),
            'manifest_sha256': manifest['_manifest_sha256'], 'split_sha256': split['sha256'],
            'selection': 'Explicit query IDs or first N in frozen manifest order; first K frozen candidates; no gold injection',
            'selected_query_ids': [s['id'] for s in selected],
            'selected_candidate_ids': {s['id']: [c['id'] for c in s['candidates']] for s in selected},
            'same_model_instance_and_weights': True,
            'dropout': 'model.eval()', 'autograd': 'disabled via public score()',
            'latency_scope': 'One query score(), including tokenization, tensor construction, transfer and scores to CPU; '
                             'device synchronized before and after; excludes model load and warmup',
            'warmup_runs_per_query_per_layout': args.warmup,
            'measured_repetitions_per_query_per_layout': args.repeats,
            'measurement_order': 'pairs/tree then tree/pairs on alternating repetitions',
            'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
            'platform': platform.platform(), 'device': str(device), 'dtype': args.dtype,
            'backend': args.backend,
            'model_module_sha256': digest(inspect.getfile(QwenReranker)),
            'runner_sha256': digest(__file__),
        }
        report['status'] = 'loading_model'
        save(output, report)
        print(json.dumps({'event': 'loading_model', 'model': str(args.model), 'device': str(device)}), flush=True)
        start = time.perf_counter()
        model = QwenReranker(str(args.model.resolve()), layout='pairs', backend=args.backend,
            device=str(device), dtype=args.dtype, revision=args.revision, max_length=args.max_length,
            max_tree_tokens=args.max_tree_tokens, pair_batch_size=args.pair_batch_size)
        model.eval()
        sync(torch, device)
        report['model_load_seconds'] = time.perf_counter() - start
        params = sum(p.numel() for p in model.parameters())
        if params < 100_000_000 and not args.allow_tiny_for_smoke:
            raise ValueError('This is a tiny model; use --allow-tiny-for-smoke and do not report it as real-model validation')
        report['model'] = {'parameter_count': params, 'resolved_revision': model.revision,
            'actual_dtype': str(next(model.parameters()).dtype), 'actual_device': str(model.device),
            'hidden_size': model.backbone.config.hidden_size,
            'num_hidden_layers': model.backbone.config.num_hidden_layers,
            'vocab_size': model.backbone.config.vocab_size,
            'run_kind': 'tiny_software_smoke' if params < 100_000_000 else 'public_checkpoint_architecture_test',
            'compiler_spec': model.compiler.spec()}
        report['reference_readout'] = reference_readout_check(torch, model, selected[0], args.reference_candidates)
        report['status'] = 'benchmarking'
        save(output, report)
        for sample in selected:
            ids = [candidate['id'] for candidate in sample['candidates']]
            record = {'query_id': sample['id'], 'candidate_ids': ids,
                      'compiler': compile_stats(model, sample),
                      'measurements': {'pairs': [], 'tree': []}, 'scores': {}}
            report['queries'].append(record)
            for layout in ('pairs', 'tree'):
                model.layout = layout
                for repeat in range(args.warmup):
                    report['progress'] = {'query_id': sample['id'], 'layout': layout, 'stage': 'warmup', 'repeat': repeat}
                    save(output, report)
                    print(json.dumps({'event': 'warmup', **report['progress']}), flush=True)
                    model.score([sample])
                    sync(torch, device)
            for repeat in range(args.repeats):
                order = ('pairs', 'tree') if repeat % 2 == 0 else ('tree', 'pairs')
                for layout in order:
                    model.layout = layout
                    sync(torch, device)
                    before = memory_start(torch, device)
                    start = time.perf_counter()
                    scores = model.score([sample])[0]
                    sync(torch, device)
                    elapsed = 1000 * (time.perf_counter() - start)
                    if len(scores) != len(ids) or not all(math.isfinite(s) for s in scores):
                        raise ValueError('Model returned incomplete or nonfinite candidate scores')
                    measurement = {'repeat': repeat, 'latency_ms': elapsed,
                                   'memory': memory_end(torch, device, before),
                                   'input_metadata': copy.deepcopy(model.last_input_metadata[0])}
                    record['measurements'][layout].append(measurement)
                    if layout not in record['scores']:
                        record['scores'][layout] = dict(zip(ids, scores))
                    measurement['max_abs_score_drift_vs_first_repeat'] = max(
                        abs(record['scores'][layout][cid] - value) for cid, value in zip(ids, scores))
                    report['progress'] = {'query_id': sample['id'], 'layout': layout, 'stage': 'measured', 'repeat': repeat}
                    save(output, report)
                    print(json.dumps({'event': 'measured', **report['progress'], 'latency_ms': elapsed}), flush=True)
            record['comparison'] = compare(ids, [record['scores']['pairs'][cid] for cid in ids],
                                           [record['scores']['tree'][cid] for cid in ids])
            save(output, report)
        all_diffs = [abs(row['scores']['pairs'][cid] - row['scores']['tree'][cid])
                     for row in report['queries'] for cid in row['candidate_ids']]
        report['summary'] = {'query_count': len(selected), 'candidate_count': len(all_diffs),
            'max_abs_score_diff': max(all_diffs), 'mean_abs_score_diff': statistics.mean(all_diffs),
            'exact_ranking_agreement_queries': sum(r['comparison']['exact_full_ranking_agreement'] for r in report['queries']),
            'latency_ms': {layout: {'mean_per_query': statistics.mean(
                m['latency_ms'] for row in report['queries'] for m in row['measurements'][layout]),
                'median_per_query': statistics.median(
                m['latency_ms'] for row in report['queries'] for m in row['measurements'][layout])}
                for layout in ('pairs', 'tree')}}
        means = report['summary']['latency_ms']
        report['summary']['pairs_mean_latency_divided_by_tree_mean_latency'] = means['pairs']['mean_per_query'] / means['tree']['mean_per_query']
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
