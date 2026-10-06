"""Summarize completed, selection-locked BoolQ readout experiments on CPU.

All nine evaluation reports must exist before the sealed evaluation records
are opened. Saved research artifacts and their selection lock are checked;
this program does not load a backbone, run inference, or select a checkpoint.
The readout candidate is identified by mean selected development NLL, never
by the highest held-out accuracy. Existing summary files are not overwritten.

    python examples/research/summarize_boolq_readout.py --protocol /path/to/round2
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F


READOUTS = ('last_relevance', 'decision_marker', 'candidate_attention')
METRICS = ('accuracy', 'balanced_accuracy', 'nll', 'brier', 'ece')
PAIRED_METRICS = ('accuracy', 'nll', 'brier')
_MODULES = {}


def sibling(name):
    if name not in _MODULES:
        path = Path(__file__).with_name(name + '.py')
        spec = importlib.util.spec_from_file_location('metis_summary_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_registration(protocol: dict) -> list[str]:
    require(protocol.get('experiment') == 'boolq_readout_round2', 'Expected a round-two BoolQ protocol')
    require(set(protocol.get('readouts', [])) == set(READOUTS) and len(protocol['readouts']) == 3,
            'Exactly the three registered readout arms are required')
    seeds = protocol.get('seeds', [])
    require(len(seeds) == 3 and len(set(seeds)) == 3 and all(type(seed) is int for seed in seeds),
            'Exactly three distinct training seeds are required')
    keys = [f'{readout}/seed-{seed}' for readout in protocol['readouts'] for seed in seeds]
    registered = protocol.get('registered_runs', [])
    require(len(registered) == 9 and
            {f"{item['readout']}/seed-{item['seed']}" for item in registered} == set(keys),
            'Protocol registration must contain all nine runs')
    require(all(item['training'] == protocol['configuration'] for item in registered),
            'Registered run configuration differs from the common protocol')
    return keys


def check_complete(root: Path, keys: list[str]) -> None:
    required = ('run_spec.json', 'training_audit.json', 'input_audit.json', 'selection.json',
                'evaluation.json', 'fresh-process-reload.json',
                'best/standalone_spec.json', 'best/files.sha256.json')
    missing = [f'models/{key}/{name}' for key in keys for name in required
               if not (root / 'models' / key / name).is_file()]
    if not (root / 'selection-lock.json').is_file():
        missing.append('selection-lock.json')
    require(not missing, 'Incomplete experiment; sealed evaluation remains unopened: ' + ', '.join(missing))


def validate_record_files(root: Path, protocol: dict) -> None:
    expected = dict(protocol['record_files'])
    for name, identity in protocol.get('files', {}).items():
        if name in expected:
            require(expected[name] == identity['sha256'], 'Conflicting record file identity')
        expected[name] = identity['sha256']
    for name, identity in expected.items():
        require(digest(root / name) == identity, f'Fixed data file checksum mismatch: {name}')
    if (root / 'registered_runs.json').exists():
        require(read_json(root / 'registered_runs.json') == protocol['registered_runs'],
                'Registered run file differs from the protocol')


def validate_records(records: dict, protocol: dict) -> None:
    identifiers, passages = set(), set()
    for split in ('train', 'dev', 'evaluation'):
        rows, identity = records[split], protocol['splits'][split]
        require(len(rows) == identity['count'] and len(rows) > 0, f'Split size differs: {split}')
        require([row['id'] for row in rows] == identity['ids'], f'Split IDs differ: {split}')
        require(json_digest(rows) == identity['records_sha256'], f'Split records differ: {split}')
        if 'input_sha256' in identity:
            require([row['input_sha256'] for row in rows] == identity['input_sha256'],
                    f'Split input identities differ: {split}')
        for row in rows:
            require(type(row['label']) is int and row['label'] in (0, 1), 'Labels must be binary integers')
            require(row['id'] not in identifiers, 'Duplicate sample ID across fixed splits')
            identifiers.add(row['id'])
            passage = hashlib.sha256(row['passage'].encode()).hexdigest()
            require(row['passage_sha256'] == passage, 'Passage identity mismatch')
            require(passage not in passages, 'Bootstrap requires independent, unique passage groups across splits')
            passages.add(passage)
            input_hash = hashlib.sha256(json.dumps([row['question'], row['passage']], ensure_ascii=False,
                                                   separators=(',', ':')).encode()).hexdigest()
            require(row['input_sha256'] == input_hash, 'Question/passage input identity mismatch')
    prior = sum(row['label'] for row in records['train']) / len(records['train'])
    require(math.isclose(prior, protocol['train_positive_prior'], abs_tol=1e-12, rel_tol=0),
            'Training prior differs from the fixed labels')


def validate_training(root: Path, protocol: dict, lock: dict, keys: list[str]) -> list[dict]:
    """Read dev-only training evidence before parsing held-out predictions."""
    configuration = protocol['configuration']
    expected_steps = configuration['epochs'] * math.ceil(
        protocol['splits']['train']['count'] / configuration['effective_batch_size'])
    protocol_hash = digest(root / 'protocol.json')
    records, common_inputs = [], {}
    dimension = protocol['base']['hidden_size']
    expected_head_count = (dimension + 2) * configuration['mlp_width'] + 1
    for key in keys:
        directory = root / 'models' / key
        readout, seed_text = key.split('/seed-')
        seed = int(seed_text)
        audit = read_json(directory / 'training_audit.json')
        selected = read_json(directory / 'selection.json')
        spec = read_json(directory / 'best/standalone_spec.json')
        run_spec = read_json(directory / 'run_spec.json')
        for value in (spec, run_spec):
            require(value['protocol_sha256'] == protocol_hash and value['protocol'] == protocol and
                    value['readout'] == readout and value['seed'] == seed,
                    f'Run identity differs from fixed protocol: {key}')
        require(spec['format'] == 'metis-research-boolq-readout-v2' and
                'not a v1 Predictor' in spec['artifact_scope'], f'Incorrect research artifact scope: {key}')
        require(audit['evaluation_opened'] is False and
                selected['evaluation_used_for_selection'] is False, f'Evaluation used for selection: {key}')
        require(audit['steps'] == expected_steps, f'Optimizer call budget mismatch: {key}')
        require(audit['head_updated'] and audit['lora_updated'] and audit['frozen_base_unchanged'] and
                audit['initial']['head'] != audit['final']['head'] and
                audit['initial']['lora'] != audit['final']['lora'] and
                audit['initial']['frozen_base'] == audit['final']['frozen_base'],
                f'Actual weight-update audit failed: {key}')
        require(spec['selected_parameter_sha256'] == audit['selected_parameter_sha256'],
                f'Selected parameter identity mismatch: {key}')
        for name in ('head', 'lora'):
            require(audit['selected_parameter_sha256'][name] != audit['initial'][name],
                    f'Selected {name} did not update: {key}')
        require(audit['pool_updated'] == (readout == 'candidate_attention'), f'Pool update audit failed: {key}')
        if readout == 'candidate_attention':
            require(audit['initial']['pool'] != audit['final']['pool'] and
                    audit['selected_parameter_sha256']['pool'] != audit['initial']['pool'],
                    f'Selected attention query did not update: {key}')
        history = audit['history']
        require(len(history) == configuration['epochs'], f'Incomplete training history: {key}')
        chosen = min(history, key=lambda row: row['dev']['nll'])
        require(chosen['epoch'] == audit['best_epoch'] == spec['epoch'] and
                chosen['dev']['nll'] == audit['best_dev_nll'] == selected['best_dev_nll'] and
                selected['checkpoint'] == f"checkpoints/epoch-{chosen['epoch']}",
                f'Checkpoint differs from minimum development NLL: {key}')
        require(all(math.isfinite(row['dev']['nll']) and row['dev']['nll'] >= 0 for row in history),
                f'Invalid development NLL: {key}')
        require({name: audit['initial'][name] for name in ('head', 'lora', 'frozen_base')} ==
                lock['common_initial_parameters_by_seed'][str(seed)], f'Common initialization mismatch: {key}')
        require([row['epoch_order_sha256'] for row in history] ==
                lock['common_epoch_order_hashes_by_seed'][str(seed)], f'Sample order mismatch: {key}')
        require(audit['head_parameters'] == expected_head_count and
                audit['pool_parameters'] == (dimension if readout == 'candidate_attention' else 0),
                f'Unexpected parameter count: {key}')
        inputs = read_json(directory / 'input_audit.json')
        require(set(inputs) == {'train', 'dev'}, f'Unexpected input audit splits: {key}')
        for split, value in inputs.items():
            require(value['candidate_body_ids_sha256'] == lock['common_candidate_body_hashes'][split],
                    f'Candidate body identity mismatch: {key}/{split}')
            count = protocol['splits'][split]['count']
            tokens = value['mean_tokens'] * count
            require(math.isfinite(tokens) and math.isclose(tokens, round(tokens), abs_tol=1e-6) and
                    0 <= value['truncated'] <= count and
                    0 < value['mean_tokens'] <= value['max_tokens'] <= configuration['max_length'],
                    f'Invalid token/truncation audit: {key}/{split}')
            value.update(count=count, total_tokens=round(tokens))
        if readout in common_inputs:
            require(inputs == common_inputs[readout], f'Input audit differs between seeds: {key}')
        else:
            common_inputs[readout] = inputs
        records.append({'key': key, 'readout': readout, 'seed': seed, 'optimizer_calls': audit['steps'],
                        'best_epoch': audit['best_epoch'], 'best_dev_nll': audit['best_dev_nll'],
                        'training_seconds': audit['training_seconds'],
                        'gpu_peak_allocated_bytes': audit['gpu_peak_allocated_bytes'],
                        'head_parameters': audit['head_parameters'], 'pool_parameters': audit['pool_parameters'],
                        'lora_parameters': audit['lora_parameters'],
                        'trainable_parameters': audit['head_parameters'] + audit['pool_parameters'] +
                                                audit['lora_parameters'],
                        'updates': {'head': True, 'lora': True, 'pool': audit['pool_updated'], 'base_frozen': True},
                        'selected_parameter_sha256': audit['selected_parameter_sha256'], 'inputs': inputs})
    require(len({record['lora_parameters'] for record in records}) == 1, 'LoRA budget differs across runs')
    for split in ('train', 'dev'):
        require(len({value[split]['truncated'] for value in common_inputs.values()}) == 1,
                'Common candidate truncation differs across readouts')
    return records


def mean_std(values) -> dict:
    values = list(values)
    return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}


def per_example_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    logits, labels = logits.double(), labels.double()
    probabilities = logits.sigmoid()
    return {'accuracy': ((probabilities >= 0.5) == labels.bool()).double(),
            'nll': F.binary_cross_entropy_with_logits(logits, labels, reduction='none'),
            'brier': (probabilities - labels).square()}


def paired_bootstrap(candidate: dict, reference: dict, *, samples: int = 10000,
                     seed: int = 20261007) -> dict:
    """Pair passages; average example-level metric differences over fixed seeds."""
    require(samples > 0, 'Bootstrap sample count must be positive')
    delta = {metric: (candidate[metric] - reference[metric]).mean(dim=0).double()
             for metric in PAIRED_METRICS}
    size = len(delta['accuracy'])
    require(size > 0 and all(value.shape == (size,) for value in delta.values()),
            'Paired bootstrap requires aligned, nonempty evaluation samples')
    generator = torch.Generator().manual_seed(seed)
    estimates = {metric: torch.empty(samples, dtype=torch.float64) for metric in PAIRED_METRICS}
    for offset in range(0, samples, 128):
        count = min(128, samples - offset)
        indices = torch.randint(size, (count, size), generator=generator)
        for metric in PAIRED_METRICS:
            estimates[metric][offset:offset + count] = delta[metric][indices].mean(dim=1)
    return {'metrics': {metric: {'candidate_minus_last_relevance': delta[metric].mean().item(),
                                 'percentile_95_interval': torch.quantile(
                                     estimates[metric], torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()}
                        for metric in PAIRED_METRICS},
            'resamples': samples, 'seed': seed,
            'unit': 'unique evaluation passage/sample, paired between readouts',
            'seed_handling': 'per-example metric differences averaged over the same three fixed training seeds',
            'interpretation': 'evaluation-sampling uncertainty only; fixed training seeds; exploratory; no multiple-comparison correction'}


def load_evaluations(root, protocol, records, labels, rows, lock_hash):
    expected_ids = [row['id'] for row in rows]
    arrays = {readout: {metric: [] for metric in PAIRED_METRICS} for readout in READOUTS}
    metrics_helper = sibling('train_readout_ablation').binary_metrics
    for record in records:
        directory = root / 'models' / record['key']
        evaluation = read_json(directory / 'evaluation.json')
        require(evaluation['selection_lock_sha256'] == lock_hash, f'Evaluation lock hash mismatch: {record["key"]}')
        require(evaluation['evaluation_records_sha256'] == protocol['record_files']['sealed_evaluation.json'],
                f'Evaluation data hash mismatch: {record["key"]}')
        predictions = evaluation['predictions']
        require([row['id'] for row in predictions] == expected_ids, f'Evaluation prediction IDs differ: {record["key"]}')
        require([row['label'] for row in predictions] == labels.long().tolist(),
                f'Evaluation prediction labels differ: {record["key"]}')
        logits = torch.tensor([row['logit'] for row in predictions], dtype=torch.float64)
        require(torch.isfinite(logits).all().item(), 'Evaluation logits must be finite')
        for row, probability in zip(predictions, logits.sigmoid()):
            require(math.isfinite(row['probability_yes']) and 0 <= row['probability_yes'] <= 1 and
                    math.isclose(row['probability_yes'], probability.item(), abs_tol=1e-7, rel_tol=1e-6),
                    'Saved probability differs from its logit')
        recomputed = metrics_helper(logits, labels)
        for metric, value in recomputed.items():
            require(metric in evaluation['metrics'] and
                    math.isclose(value, evaluation['metrics'][metric], abs_tol=1e-10, rel_tol=1e-10),
                    f'Saved evaluation metric differs from predictions: {record["key"]}/{metric}')
        record['evaluation'] = recomputed
        reload = read_json(directory / 'fresh-process-reload.json')
        require(reload['status'] == 'passed' and reload['samples'] == min(3, protocol['splits']['dev']['count']) and
                math.isfinite(reload['maximum_absolute_difference']) and
                0 <= reload['maximum_absolute_difference'] <= 1e-5,
                f'Fresh-process research reload failed: {record["key"]}')
        record['reload'] = reload
        for metric, values in per_example_metrics(logits, labels).items():
            arrays[record['readout']][metric].append(values)
    return {readout: {metric: torch.stack(values) for metric, values in metrics.items()}
            for readout, metrics in arrays.items()}


def markdown(summary: dict) -> str:
    lines = ['# BoolQ readout experiment: round two', '',
             'Exploratory task-specific training with a fixed Qwen base, identical initial MLP/LoRA parameters, '
             'and three training seeds. Evaluation uses a fresh reserved subset of official validation; '
             'these are standalone research artifacts, not production v1 Predictor exports.', '',
             f"Development-selected readout candidate: **{summary['development_selection']['readout']}**. "
             'Selection uses mean development NLL of the independently selected checkpoints. '
             'Held-out metrics below do not determine the selected readout.', '',
             '| Readout | Dev NLL | Accuracy (mean ± sample SD) | Balanced accuracy | NLL | Binary Brier |',
             '|---|---:|---:|---:|---:|---:|']
    baseline = summary['train_prior_baseline']['evaluation']
    lines.append(f"| Training-prior / majority reference | — | {baseline['accuracy'] * 100:.2f}% | "
                 f"{baseline['balanced_accuracy'] * 100:.2f}% | {baseline['nll']:.4f} | {baseline['brier']:.4f} |")
    for readout, value in summary['by_readout'].items():
        metrics = value['evaluation']
        lines.append(f"| {readout} | {value['selected_dev_nll']['mean']:.4f} | "
                     f"{metrics['accuracy']['mean'] * 100:.2f}% ± {metrics['accuracy']['sample_std'] * 100:.2f} pp | "
                     f"{metrics['balanced_accuracy']['mean'] * 100:.2f}% | {metrics['nll']['mean']:.4f} | "
                     f"{metrics['brier']['mean']:.4f} |")
    lines += ['', '## Paired differences from last_relevance', '',
              'Accuracy differences and intervals are percentage points. NLL/Brier differences use their '
              'original scales. Positive accuracy is favorable; negative NLL/Brier is favorable.', '',
              '| Candidate | Accuracy Δ [95% interval] | NLL Δ [95% interval] | Brier Δ [95% interval] |',
              '|---|---:|---:|---:|']
    for readout, value in summary['paired_bootstrap'].items():
        cells = []
        for metric in PAIRED_METRICS:
            factor = 100 if metric == 'accuracy' else 1
            result = value['metrics'][metric]
            low, high = result['percentile_95_interval']
            cells.append(f"{result['candidate_minus_last_relevance'] * factor:+.4f} "
                         f"[{low * factor:+.4f}, {high * factor:+.4f}]")
        lines.append('| ' + readout + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', f"{summary['bootstrap_configuration']['resamples']:,} paired resamples, seed "
              f"{summary['bootstrap_configuration']['seed']}. Per-example metrics are first averaged over "
              'the three fixed training seeds. Intervals describe evaluation-sampling uncertainty only, '
              'not variability from retraining. They are exploratory and have no multiple-comparison correction.', '',
              '## Training and input audits', '',
              '| Readout | Seed | Optimizer calls | Best epoch | Head / pool / LoRA parameters | Peak GPU GiB |',
              '|---|---:|---:|---:|---:|---:|']
    for record in summary['runs']:
        peak = record['gpu_peak_allocated_bytes']
        lines.append(f"| {record['readout']} | {record['seed']} | {record['optimizer_calls']} | {record['best_epoch']} | "
                     f"{record['head_parameters']:,} / {record['pool_parameters']:,} / {record['lora_parameters']:,} | "
                     + (f'{peak / 2**30:.3f}' if peak is not None else '—') + ' |')
    lines += ['', 'Optimizer calls include the scheduled final call with learning rate zero; '
              'the weight-update audit verifies changed selected head/LoRA weights, an updated attention '
              'query where applicable, and unchanged frozen base weights. Fresh-process reload checks '
              'use three fixed development references, not another held-out evaluation.', '',
              '| Readout | Split | Samples | Truncated | Total input tokens | Mean / max tokens |',
              '|---|---|---:|---:|---:|---:|']
    for readout, splits in summary['input_audits'].items():
        for split, value in splits.items():
            lines.append(f"| {readout} | {split} | {value['count']} | {value['truncated']} | {value['total_tokens']:,} | "
                         f"{value['mean_tokens']:.2f} / {value['max_tokens']} |")
    lines += ['', 'Input audits above cover training and development. No held-out token counts are inferred. '
              'Candidate token identities and truncation counts agree across readouts; final text-marker '
              'lengths may differ within the shared body budget.', '',
              'Binary Brier is `(p_yes - y)^2`. NLL is binary cross-entropy. The JSON also includes '
              'confidence-based 10-bin ECE; a small ECE alone does not establish useful discrimination.', '',
              f"Protocol SHA256: `{summary['identity']['protocol_sha256']}`", '',
              f"Selection-lock SHA256: `{summary['identity']['selection_lock_sha256']}`", '']
    return '\n'.join(lines)


def summarize(root: Path, *, bootstrap_samples: int = 10000, bootstrap_seed: int = 20261007,
              write: bool = True) -> dict:
    root = Path(root)
    if write and any((root / name).exists() for name in ('summary.json', 'summary.md')):
        raise FileExistsError('Summary files already exist; refusing to overwrite')
    protocol = read_json(root / 'protocol.json')
    keys = validate_registration(protocol)
    check_complete(root, keys)  # No sealed file is opened until all nine evaluations exist.
    lock = sibling('train_boolq_readout').assert_locked(root, protocol)
    require(lock['evaluation_opened'] is False, 'Selection lock must precede evaluation')
    records = validate_training(root, protocol, lock, keys)
    by_readout = {}
    for readout in protocol['readouts']:
        selected = [record for record in records if record['readout'] == readout]
        by_readout[readout] = {'seeds': [record['seed'] for record in selected],
                              'selected_dev_nll': mean_std(record['best_dev_nll'] for record in selected)}
    # Development-only candidate choice is fixed before parsing evaluation labels/predictions.
    chosen = min(protocol['readouts'], key=lambda readout: by_readout[readout]['selected_dev_nll']['mean'])
    validate_record_files(root, protocol)
    data = read_json(root / 'train_dev_records.json')
    evaluation_rows = read_json(root / 'sealed_evaluation.json')['evaluation']
    require(set(data) == {'train', 'dev'}, 'Training data must contain only train/dev')
    validate_records({**data, 'evaluation': evaluation_rows}, protocol)
    labels = torch.tensor([row['label'] for row in evaluation_rows], dtype=torch.float64)
    lock_hash = digest(root / 'selection-lock.json')
    arrays = load_evaluations(root, protocol, records, labels, evaluation_rows, lock_hash)
    for readout in protocol['readouts']:
        selected = [record for record in records if record['readout'] == readout]
        by_readout[readout]['evaluation'] = {
            metric: mean_std(record['evaluation'][metric] for record in selected) for metric in METRICS}
    prior = protocol['train_positive_prior']
    clipped = min(max(prior, 1e-7), 1 - 1e-7)
    baseline = sibling('train_readout_ablation').binary_metrics(
        torch.full_like(labels, math.log(clipped / (1 - clipped))), labels)
    summary = {'scope': protocol.get('scope', 'second exploratory BoolQ readout round'),
               'evaluation_scope': protocol['evaluation'],
               'identity': {'protocol_sha256': digest(root / 'protocol.json'),
                            'selection_lock_sha256': lock_hash,
                            'evaluation_records_sha256': protocol['record_files']['sealed_evaluation.json']},
               'configuration': protocol['configuration'],
               'split_sizes': {split: value['count'] for split, value in protocol['splits'].items()},
               'development_selection': {'readout': chosen, 'criterion': 'minimum mean selected development NLL',
                                         'tie_break': 'registered readout order', 'evaluation_used': False},
               'train_prior_baseline': {'positive_probability': prior, 'majority_label': int(prior >= 0.5),
                                        'evaluation': baseline},
               'by_readout': by_readout,
               'bootstrap_configuration': {'resamples': bootstrap_samples, 'seed': bootstrap_seed,
                                           'evaluation_passages_unique': True},
               'paired_bootstrap': {readout: paired_bootstrap(arrays[readout], arrays['last_relevance'],
                                                           samples=bootstrap_samples, seed=bootstrap_seed)
                                    for readout in protocol['readouts'] if readout != 'last_relevance'},
               'input_audits': {readout: next(record['inputs'] for record in records if record['readout'] == readout)
                                for readout in protocol['readouts']},
               'runs': records, 'artifact_scope': 'standalone research weights; not v1 Predictor exports'}
    if write:
        with (root / 'summary.json').open('x') as output:
            output.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        with (root / 'summary.md').open('x') as output:
            output.write(markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261007)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('--bootstrap-samples must be positive')
    torch.set_num_threads(1)
    summary = summarize(args.protocol, bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
    print(json.dumps({'summary': str(args.protocol / 'summary.json'),
                      'development_selection': summary['development_selection'],
                      'by_readout': summary['by_readout']}, indent=2))


if __name__ == '__main__':
    main()
