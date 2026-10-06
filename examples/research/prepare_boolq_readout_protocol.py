"""Register the second exploratory BoolQ readout study before training.

Sampling uses question/passage hashes, never labels. Training and development
records are saved separately from the fresh official-validation subset. The
previous pilot's development passages never enter training, and its evaluation
passages never enter the new evaluation. This is not the official BoolQ test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from metis.compiler import ScoreHeadCompiler
from metis.schema import sha256

_source = Path(__file__).with_name('extract_boolq_features.py')
_spec = importlib.util.spec_from_file_location('metis_boolq_feature_protocol', _source)
_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_features)
download_dataset = _features.download_dataset
digest = _features.digest
DATASET_REVISION = _features.DATASET_REVISION
INSTRUCTION = _features.INSTRUCTION

DEFAULT_READOUTS = ('last_relevance', 'decision_marker', 'candidate_attention')
TRAINING_SEEDS = (42, 43, 44)


def input_digest(question, passage):
    return digest(json.dumps([question, passage], ensure_ascii=False, separators=(',', ':')))


def _record(row, index, source):
    question, passage = row.get('question'), row.get('passage')
    if not isinstance(question, str) or not isinstance(passage, str) or not question or not passage:
        raise ValueError('Every BoolQ row requires nonempty question and passage strings')
    answer = row.get('answer')
    if type(answer) not in (bool, int) or answer not in (0, 1):
        raise ValueError('Every BoolQ answer must be binary')
    return {'id': f'boolq-{source}-{index}', 'source_split': source, 'source_index': index,
            'question': question, 'passage': passage, 'label': int(answer),
            'input_sha256': input_digest(question, passage), 'passage_sha256': digest(passage)}


def _old_records(previous, rows, name):
    source = 'validation' if name == 'evaluation' else 'train'
    items = previous.get(name)
    if not isinstance(items, list):
        raise ValueError(f'Previous pilot requires a {name} record list')
    output = []
    seen_ids = set()
    for item in items:
        index = item.get('source_index')
        if type(index) is not int or not 0 <= index < len(rows[source]):
            raise ValueError(f'Previous {name} source index is invalid')
        record = _record(rows[source][index], index, source)
        # Label equality is deliberately excluded: changing labels must not
        # affect sampling, inclusion, or the passage exclusion policy.
        if (item.get('id') != record['id'] or item.get('question') != record['question']
                or item.get('passage') != record['passage']):
            raise ValueError(f'Previous {name} input identity does not match the raw dataset')
        if record['id'] in seen_ids:
            raise ValueError(f'Duplicate previous {name} source record')
        seen_ids.add(record['id'])
        output.append(record)
    return output


def _ordered_unique(items, seed, namespace, excluded=()):
    seen = set(excluded)
    ordered = sorted(items, key=lambda item: (digest(f'{seed}:{namespace}:{item["input_sha256"]}'),
                                             item['input_sha256'], item['id']))
    result = []
    for item in ordered:
        group = item['passage_sha256']
        if group not in seen:
            result.append(item)
            seen.add(group)
    return result


def select_records(rows, previous_records, *, train_size=1536, dev_size=192,
                   evaluation_size=384, seed=20261007):
    """Select label-blind records with passage-group isolation and prior-run exclusions."""
    counts = (train_size, dev_size, evaluation_size)
    if any(type(value) is not int or value < 1 for value in counts):
        raise ValueError('Split counts must be positive integers')
    if type(seed) is not int:
        raise ValueError('Split seed must be an integer')
    if not isinstance(rows, dict) or any(not isinstance(rows.get(k), list)
                                         for k in ('train', 'validation')):
        raise ValueError('Raw BoolQ requires train and validation lists')
    if not isinstance(previous_records, dict):
        raise ValueError('Previous pilot records must be an object')
    train_pool = [_record(row, i, 'train') for i, row in enumerate(rows['train'])]
    evaluation_pool = [_record(row, i, 'validation') for i, row in enumerate(rows['validation'])]
    old_dev = _old_records(previous_records, rows, 'dev')
    old_eval = _old_records(previous_records, rows, 'evaluation')
    reserved = {item['passage_sha256'] for item in evaluation_pool}
    old_dev_groups = {item['passage_sha256'] for item in old_dev}
    old_eval_groups = {item['passage_sha256'] for item in old_eval}
    if reserved & old_dev_groups:
        raise ValueError('Previous development passages overlap official validation')
    retained_dev = _ordered_unique(old_dev, seed, 'retained_dev')
    if len(retained_dev) > dev_size:
        raise ValueError('Development count cannot discard previous development passages')
    extra_dev = _ordered_unique(train_pool, seed, 'dev', reserved | old_dev_groups)
    dev = retained_dev + extra_dev[:dev_size - len(retained_dev)]
    dev_groups = {item['passage_sha256'] for item in dev}
    train = _ordered_unique(train_pool, seed, 'train', reserved | old_dev_groups | dev_groups)[:train_size]
    evaluation = _ordered_unique(evaluation_pool, seed, 'evaluation', old_eval_groups)[:evaluation_size]
    result = {'train': train, 'dev': dev, 'evaluation': evaluation}
    for name, count in zip(result, counts):
        if len(result[name]) != count:
            raise ValueError(f'Requested {name} count exceeds available disjoint passages')
    groups = {name: {item['passage_sha256'] for item in items} for name, items in result.items()}
    if any(groups[a] & groups[b] for a, b in (('train', 'dev'), ('train', 'evaluation'),
                                             ('dev', 'evaluation'))):
        raise ValueError('Passage leakage across research splits')
    return result


def load_previous_pilot(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    previous = json.loads((root / 'selected_records.json').read_text())
    if protocol.get('experiment') != 'frozen_base_readout_pilot':
        raise ValueError('Expected the previous frozen-base BoolQ pilot')
    if protocol.get('dataset', {}).get('revision') != DATASET_REVISION:
        raise ValueError('Previous BoolQ dataset revision differs')
    for name in ('dev', 'evaluation'):
        items = previous.get(name)
        identity = protocol.get('splits', {}).get(name, {})
        if (not isinstance(items, list) or identity.get('count') != len(items)
                or identity.get('ids') != [item['id'] for item in items]
                or identity.get('records_sha256') != digest(json.dumps(items, sort_keys=True))):
            raise ValueError(f'Previous {name} records do not match their protocol')
    return previous, protocol


def model_identity(root):
    root = Path(root)
    if not root.is_dir() or not (root / 'config.json').is_file():
        raise ValueError('A local model directory with config.json is required')
    weights = [path for path in root.iterdir()
               if path.is_file() and path.suffix in ('.safetensors', '.bin')]
    tokenizers = [path for path in root.iterdir() if path.is_file()
                  and ('token' in path.name or path.name in ('vocab.json', 'merges.txt', 'vocab.txt'))]
    if not weights or not tokenizers:
        raise ValueError('Local model weights and tokenizer files are required')
    files = {path.name: sha256(path) for path in sorted(root.iterdir())
             if path.is_file() and (path.suffix in ('.json', '.safetensors', '.bin')
                                    or path in tokenizers)}
    config = json.loads((root / 'config.json').read_text())
    return {'name': config.get('_name_or_path', 'Qwen/Qwen3-0.6B-Base'),
            'file_sha256': files, 'hidden_size': config.get('hidden_size'),
            'base_weights_frozen': True, 'adaptation': 'LoRA', 'master_dtype': 'float32',
            'compute_dtype': 'bfloat16'}


def training_plan(readouts=DEFAULT_READOUTS):
    if len(readouts) != 3 or len(set(readouts)) != 3:
        raise ValueError('Exactly three distinct readout arms are required')
    configuration = {'epochs': 3, 'effective_batch_size': 16, 'batch_size': 2,
                     'gradient_accumulation_steps': 8, 'lora_learning_rate': 2e-5,
                     'head_learning_rate': 2e-4, 'weight_decay': 0.01,
                     'warmup_fraction': 0.10, 'scheduler': 'cosine', 'gradient_clip': 1.0,
                     'master_dtype': 'float32', 'compute_dtype': 'bfloat16', 'precision': 'bf16', 'max_length': 256,
                     'mlp_width': 256, 'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0.0,
                     'loss': 'binary_cross_entropy_with_logits',
                     'selection': 'minimum_development_NLL_among_trained_epoch_checkpoints',
                     'calibration': 'none; fixed sigmoid threshold 0.5'}
    runs = [{'id': f'{kind}-seed-{seed}', 'readout': kind, 'seed': seed,
             'training': dict(configuration)} for kind in readouts for seed in TRAINING_SEEDS]
    return configuration, runs


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def prepare(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output directory already exists; protocols are immutable')
    rows, raw_identity = download_dataset(Path(args.data))
    previous, old_protocol = load_previous_pilot(args.previous_pilot)
    for name, identity in raw_identity.items():
        previous_identity = old_protocol.get('dataset', {}).get('raw_files', {}).get(name, {})
        if previous_identity.get('sha256') != identity['sha256']:
            raise ValueError(f'Raw {name} file differs from the previous pilot')
    records = select_records(rows, previous, train_size=args.train_size, dev_size=args.dev_size,
                             evaluation_size=args.evaluation_size, seed=args.seed)
    base = model_identity(args.model)
    configuration, runs = training_plan()
    indices = {name: [{key: item[key] for key in ('id', 'source_split', 'source_index',
                                                 'input_sha256', 'passage_sha256')}
                      for item in items] for name, items in records.items()}
    protocol = {'schema_version': '1', 'experiment': 'boolq_readout_round2',
                'stage': 'registered_before_training',
                'scope': 'previous_configurations_observed_second_exploratory_round',
                'dataset': {'name': 'google/boolq', 'revision': DATASET_REVISION,
                            'license': 'CC-BY-SA-3.0', 'raw_files': raw_identity},
                'previous_pilot': {'protocol_sha256': sha256(Path(args.previous_pilot) / 'protocol.json'),
                                   'records_sha256': sha256(Path(args.previous_pilot) / 'selected_records.json')},
                'split_seed': args.seed,
                'sampling': {'ranking': 'sha256(seed:split:sha256(JSON[question,passage]))',
                             'labels_used_for_sampling': False, 'one_record_per_passage_per_split': True,
                             'official_validation_passages_reserved_from_train_and_dev': True,
                             'previous_dev_passages_never_train': True,
                             'previous_dev_retained_then_input_hash_filled': True,
                             'previous_evaluation_passages_excluded_from_new_evaluation': True,
                             'previous_train_reuse_allowed': True},
                'evaluation': 'fresh_reserved_subset_of_official_validation_not_official_test',
                'evaluation_gate': 'read_only_after_all_nine_checkpoint_selections_frozen',
                'base': base, 'layout': 'pairs', 'backend': 'sdpa',
                'compiler': {'version': ScoreHeadCompiler.version, 'instruction': INSTRUCTION,
                             'max_length': 256, 'position_policy': 'prefix_then_independent_branch',
                             'truncation': 'document_tail'},
                'label_semantics': 'P(affirmative_answer | question, passage)',
                'instruction': INSTRUCTION, 'configuration': configuration,
                'train_positive_prior': sum(item['label'] for item in records['train']) / len(records['train']),
                'readouts': list(DEFAULT_READOUTS), 'seeds': list(TRAINING_SEEDS),
                'registered_runs': runs,
                'splits': {name: {'count': len(items), 'ids': [item['id'] for item in items],
                                  'source_indices': [item['source_index'] for item in items],
                                  'input_sha256': [item['input_sha256'] for item in items],
                                  'records_sha256': digest(json.dumps(items, sort_keys=True))}
                           for name, items in records.items()},
                'preparation_source_sha256': sha256(Path(__file__)),
                'feature_protocol_source_sha256': sha256(_source), 'files': {}, 'record_files': {}}
    output.mkdir(parents=True, exist_ok=False)
    files = {'train_dev_records.json': {'train': records['train'], 'dev': records['dev']},
             'sealed_evaluation.json': {'evaluation': records['evaluation']},
             'selected_records.json': indices, 'registered_runs.json': runs}
    for name, value in files.items():
        _write_json(output / name, value)
        protocol['files'][name] = {'sha256': sha256(output / name)}
        if name in ('train_dev_records.json', 'sealed_evaluation.json'):
            protocol['record_files'][name] = sha256(output / name)
    _write_json(output / 'protocol.json', protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('data', 'model', 'previous-pilot', 'output'):
        parser.add_argument(f'--{option}', required=True)
    parser.add_argument('--train-size', type=int, default=1536)
    parser.add_argument('--dev-size', type=int, default=192)
    parser.add_argument('--evaluation-size', type=int, default=384)
    parser.add_argument('--seed', type=int, default=20261007)
    args = parser.parse_args()
    protocol = prepare(args)
    print(json.dumps({'status': 'registered', 'runs': len(protocol['registered_runs']),
                      'counts': {name: entry['count'] for name, entry in protocol['splits'].items()}}))


if __name__ == '__main__':
    main()
