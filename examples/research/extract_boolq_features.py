"""Freeze Qwen3 and cache identical BoolQ representations for head ablations.

This research entrypoint does not modify the v1 model or its saved artifacts.
Requires the train extra plus pyarrow; dataset and weights stay outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

from metis.compiler import ScoreHeadCompiler
from metis.schema import sha256

DATASET_REVISION = '35b264d03638db9f4ce671b711558bf7ff0f80d5'
INSTRUCTION = ('Given the question and the candidate passage, estimate whether '
               'the affirmative answer to the question is supported by the passage.')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def download_dataset(root):
    import pyarrow.parquet as pq
    root.mkdir(parents=True, exist_ok=True)
    rows, identity = {}, {}
    for split in ('train', 'validation'):
        filename = f'{split}-00000-of-00001.parquet'
        url = f'https://huggingface.co/datasets/google/boolq/resolve/{DATASET_REVISION}/data/{filename}'
        path = root / filename
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        rows[split] = pq.read_table(path).to_pylist()
        identity[split] = {'url': url, 'sha256': sha256(path), 'rows': len(rows[split])}
    return rows, identity


def fixed_splits(rows, sizes, seed):
    # Keep passage groups disjoint across train/dev/evaluation. Selection uses
    # input hashes, never labels. Reserve every official validation passage.
    reserved = {digest(x['passage']) for x in rows['validation']}
    unique = {}
    for index, row in enumerate(rows['train']):
        key = digest(row['passage'])
        if key not in reserved and key not in unique:
            unique[key] = (index, row)
    ordered = sorted(unique.values(), key=lambda item: digest(f'{seed}:train:{item[0]}'))
    ntrain, ndev, neval = sizes
    if len(ordered) < ntrain + ndev or len(rows['validation']) < neval:
        raise ValueError('Requested pilot exceeds available disjoint data')
    evaluation = sorted(enumerate(rows['validation']),
                        key=lambda item: digest(f'{seed}:validation:{item[0]}'))[:neval]
    selected = {'train': ordered[:ntrain], 'dev': ordered[ntrain:ntrain + ndev],
                'evaluation': evaluation}
    output = {}
    for split, items in selected.items():
        source = 'validation' if split == 'evaluation' else 'train'
        output[split] = [{'id': f'boolq-{source}-{index}', 'source_index': index,
                          'question': row['question'], 'passage': row['passage'],
                          'label': int(row['answer'])} for index, row in items]
    groups = {name: {digest(x['passage']) for x in items} for name, items in output.items()}
    if any(groups[a] & groups[b] for a, b in [('train', 'dev'), ('train', 'evaluation'),
                                             ('dev', 'evaluation')]):
        raise ValueError('Passage leakage across research splits')
    return output


def extract(args):
    from transformers import AutoModel, AutoTokenizer
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows, raw_identity = download_dataset(Path(args.data))
    splits = fixed_splits(rows, (args.train_size, args.dev_size, args.evaluation_size), args.seed)
    (output / 'selected_records.json').write_text(json.dumps(splits, ensure_ascii=False) + '\n')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backbone = AutoModel.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype,
                                         attn_implementation='sdpa').to(device).eval()
    backbone.requires_grad_(False)
    compiler = ScoreHeadCompiler(tokenizer, max_length=args.max_length, instruction=INSTRUCTION)
    source = Path(args.model)
    model_identity = {p.name: sha256(p) for p in sorted(source.iterdir())
                      if p.is_file() and (p.name == 'config.json' or p.suffix == '.safetensors')}
    protocol = {'schema_version': '1', 'experiment': 'frozen_base_readout_pilot',
                'dataset': {'name': 'google/boolq', 'revision': DATASET_REVISION,
                            'raw_files': raw_identity, 'license': 'CC-BY-SA-3.0'},
                'split_seed': args.seed, 'split_policy': 'input_hash_order_passage_disjoint',
                'evaluation': 'held_out_subset_of_official_validation_not_official_test',
                'base': {'name': 'Qwen/Qwen3-0.6B-Base', 'file_sha256': model_identity,
                         'frozen': True, 'mode': 'eval', 'dtype': str(dtype),
                         'hidden_size': backbone.config.hidden_size},
                'compiler': compiler.spec(), 'layout': 'pairs', 'backend': 'sdpa',
                'readout': 'last_relevance_suffix_hidden_state',
                'label_semantics': 'binary affirmative answer correctness',
                'batch_size': args.batch_size, 'splits': {}}
    # Freeze all input identities before any head training or held-out metrics.
    for split, items in splits.items():
        protocol['splits'][split] = {'count': len(items), 'ids': [x['id'] for x in items],
                                    'records_sha256': digest(json.dumps(items, sort_keys=True))}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    cached, input_stats = {}, {}
    start = time.monotonic()
    with torch.inference_mode():
        for split, items in splits.items():
            features, lengths, truncations = [], [], []
            for offset in range(0, len(items), args.batch_size):
                batch = items[offset:offset + args.batch_size]
                compiled = [compiler.compile({'id': x['id'], 'input': {'query': x['question']},
                                              'candidates': [{'id': 'passage', 'text': x['passage']}]})
                            for x in batch]
                pairs = [x.pairs[0] for x in compiled]
                n = max(map(len, pairs))
                ids = torch.full((len(pairs), n), tokenizer.pad_token_id, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for i, tokens in enumerate(pairs):
                    ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
                    mask[i, :len(tokens)] = 1
                positions = torch.arange(n, device=device).expand(len(pairs), -1)
                hidden = backbone(input_ids=ids, attention_mask=mask, position_ids=positions,
                                  use_cache=False, return_dict=True).last_hidden_state
                ends = torch.tensor([len(tokens) - 1 for tokens in pairs], device=device)
                vectors = hidden[torch.arange(len(pairs), device=device), ends].float().cpu().clone()
                if not torch.isfinite(vectors).all():
                    raise ValueError('Nonfinite frozen representations')
                features.append(vectors)
                lengths.extend(map(len, pairs))
                truncations.extend(x.truncated[0] for x in compiled)
                del hidden, vectors
                if (offset // args.batch_size) % 16 == 0:
                    print(json.dumps({'split': split, 'done': min(offset + args.batch_size, len(items)),
                                      'total': len(items), 'elapsed_seconds': round(time.monotonic() - start, 1)}), flush=True)
            cached[f'{split}_features'] = torch.cat(features).contiguous()
            cached[f'{split}_labels'] = torch.tensor([x['label'] for x in items], dtype=torch.float32)
            input_stats[split] = {'truncated': sum(truncations), 'mean_tokens': sum(lengths) / len(lengths),
                                 'max_tokens': max(lengths)}
    save_file(cached, str(output / 'features.safetensors'))
    protocol['feature_sha256'] = sha256(output / 'features.safetensors')
    protocol['input_stats'] = input_stats
    protocol['extraction'] = {'seconds': time.monotonic() - start, 'device': str(device),
                              'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device)
                              if device.type == 'cuda' else None}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'features': protocol['feature_sha256'],
                      'seconds': protocol['extraction']['seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.20)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--train-size', type=int, default=384)
    parser.add_argument('--dev-size', type=int, default=96)
    parser.add_argument('--evaluation-size', type=int, default=192)
    parser.add_argument('--seed', type=int, default=20261006)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.train_size, args.dev_size, args.evaluation_size) < 1:
        parser.error('Sizes and threads must be positive')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('GPU memory fraction must be in (0,1]')
    extract(args)


if __name__ == '__main__':
    main()
