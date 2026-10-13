"""Diagnose whether a BoolQ LoRA + scalar readout can memorize 32 training inputs.

This is an optimization diagnostic, not a generalization benchmark. Preparation
uses only the train member of an existing round-two training-record file. The
JSON container also holds dev, which is never selected, scored, or used. The
sealed evaluation file is never opened.
The reference and stronger arms have identical initialization and sample order;
both LoRA and head learning rates change together in the stronger arm.

    python examples/research/diagnose_boolq_training.py prepare \
        --round2 /path/to/round-two-run --model /path/to/Qwen3-0.6B-Base \
        --output /path/to/new-diagnostic
    python examples/research/diagnose_boolq_training.py train \
        --protocol-dir /path/to/new-diagnostic --arm reference --device cuda:0
    python examples/research/diagnose_boolq_training.py train \
        --protocol-dir /path/to/new-diagnostic --arm stronger --device cuda:0
    python examples/research/diagnose_boolq_training.py verify \
        --export /path/to/new-diagnostic/stronger/final --device cuda:0

Run verification in a separate process. Exports reuse the standalone research
format; they do not change the Metis v1 Predictor contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn

from metis.schema import sha256


EXPERIMENT = 'boolq_training_memorization_diagnostic'
SELECTION_SEED = 20261009
READOUT = 'last_relevance'
ARM_RATES = {'reference': (2e-5, 2e-4), 'stronger': (2e-4, 1e-3)}
CONFIGURATION = {
    'max_length': 256, 'mlp_width': 256, 'lora_rank': 8, 'lora_alpha': 16,
    'lora_dropout': 0.0, 'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'batch_size': 4, 'effective_batch_size': 8, 'gradient_accumulation_steps': 2,
    'maximum_updates': 240, 'evaluate_every_updates': 20, 'warmup_fraction': 0.1,
    'scheduler': 'linear_warmup_then_constant', 'weight_decay': 0.0,
    'gradient_clip': 1.0, 'master_dtype': 'float32', 'precision': 'bf16',
    'compute_dtype': 'bfloat16', 'loss': 'binary_cross_entropy_with_logits',
    'calibration': 'none; fixed sigmoid threshold 0.5',
}
STOPPING = {'minimum_correct': 31, 'minimum_correct_per_class': 15,
            'maximum_nll': 0.15, 'consecutive_evaluations': 2}
_TRAINER = None


def trainer():
    global _TRAINER
    if _TRAINER is None:
        path = Path(__file__).with_name('train_boolq_readout.py')
        spec = importlib.util.spec_from_file_location('metis_boolq_diagnostic_helpers', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TRAINER = module
    return _TRAINER


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    trainer().write_json(path, value)


def source_identity():
    """Freeze executable research helpers and the installed project source."""
    root = Path(__file__).resolve().parents[2]
    helpers = ('diagnose_boolq_training.py', 'train_boolq_readout.py',
               'boolq_readout_model.py', 'train_boolq_lora_ablation.py',
               'train_readout_ablation.py')
    paths = [root / 'examples/research' / name for name in helpers]
    paths.extend(sorted((root / 'src/metis').rglob('*.py')))
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def checked_record(record):
    if any(not isinstance(record.get(name), str) or not record[name].strip()
           for name in ('question', 'passage')):
        raise ValueError('Question and passage must be nonempty text')
    trainer().sibling('train_boolq_lora_ablation').to_sample(record)
    if not isinstance(record['id'], str) or not record['id']:
        raise ValueError('Every training record needs a nonempty ID')
    input_hash = hashlib.sha256(json.dumps(
        [record['question'], record['passage']], ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()
    passage_hash = hashlib.sha256(record['passage'].encode()).hexdigest()
    if record.get('input_sha256') != input_hash or record.get('passage_sha256') != passage_hash:
        raise ValueError('Training input identity does not match its text')
    if record.get('source_split') != 'train':
        raise ValueError('A diagnostic may only use official training records')
    return input_hash


def select_training_records(records, compiler, *, per_class=16):
    """Exclude truncated inputs, then use a fixed hash order within each label.

    Label stratification is deliberate for this balanced memorization check.
    Hash tie-breaking depends on input identity, never a model score.
    """
    if type(per_class) is not int or per_class < 1:
        raise ValueError('A positive per-class count is required')
    pools, metadata, identifiers, passages = {0: [], 1: []}, {}, set(), set()
    for record in records:
        identity = checked_record(record)
        if record['id'] in identifiers or record['passage_sha256'] in passages:
            raise ValueError('The training pool must have unique IDs and passages')
        identifiers.add(record['id'])
        passages.add(record['passage_sha256'])
        sample = trainer().sibling('train_boolq_lora_ablation').to_sample(record)
        compiled = compiler.compile(sample)
        rank = digest([SELECTION_SEED, identity])
        metadata[record['id']] = {'input_sha256': identity, 'selection_rank': rank,
                                 'truncated': compiled.truncated[0],
                                 'input_tokens': len(compiled.pairs[0]),
                                 'input_ids_sha256': digest(compiled.pairs[0])}
        if not compiled.truncated[0]:
            pools[record['label']].append(record)
    selected = []
    for label, rows in pools.items():
        rows.sort(key=lambda row: (metadata[row['id']]['selection_rank'], row['id']))
        if len(rows) < per_class:
            raise ValueError(f'Not enough untruncated training records for class {label}')
        selected.extend(rows[:per_class])
    selected.sort(key=lambda row: (metadata[row['id']]['selection_rank'], row['id']))
    return selected, {row['id']: metadata[row['id']] for row in selected}


def update_order(identifiers, seed, *, maximum_updates=240, effective_batch_size=8):
    if len(identifiers) != 32 or len(set(identifiers)) != 32 or effective_batch_size != 8:
        raise ValueError('This diagnostic fixes 32 distinct samples and effective batch 8')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    groups = []
    while len(groups) < maximum_updates:
        permutation = torch.randperm(len(identifiers), generator=generator).tolist()
        for offset in range(0, len(permutation), effective_batch_size):
            groups.append([identifiers[index] for index in
                           permutation[offset:offset + effective_batch_size]])
            if len(groups) == maximum_updates:
                break
    return groups


def prepare(args):
    from transformers import AutoTokenizer
    output, round2, source = Path(args.output), Path(args.round2), Path(args.model)
    if output.exists():
        raise FileExistsError('Prepared diagnostics are immutable; choose a new output directory')
    old = trainer().load_protocol(round2)
    records_path = round2 / 'train_dev_records.json'
    if sha256(records_path) != old['record_files']['train_dev_records.json']:
        raise ValueError('Round-two training records differ from their frozen identity')
    # The existing serialized container includes dev; only its train member is
    # inspected or transformed. The sealed evaluation file is never opened.
    records = json.loads(records_path.read_text())['train']
    if not isinstance(records, list) or not records:
        raise ValueError('An existing, nonempty training split is required')
    trainer().sibling('train_boolq_lora_ablation').verify_base_identity(
        source, old['base']['file_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    compiler = trainer().sibling('boolq_readout_model').CommonBudgetCompiler(
        tokenizer, max_length=256, instruction=old['instruction'], readout=READOUT)
    selected, input_audit = select_training_records(records, compiler)
    ids = [row['id'] for row in selected]
    groups = update_order(ids, args.seed)
    output.mkdir(parents=True)
    write_json(output / 'training_records.json', {'train': selected})
    write_json(output / 'update_order.json', groups)
    protocol = {
        'experiment': EXPERIMENT, 'scope': '32-input training memorization diagnostic; no generalization claim',
        'base': old['base'], 'instruction': old['instruction'], 'backend': 'sdpa',
        'configuration': dict(CONFIGURATION), 'stopping': dict(STOPPING),
        'readout': READOUT, 'seed': args.seed, 'selection_seed': SELECTION_SEED,
        'source_model': str(source.resolve()), 'source_sha256': source_identity(),
        'origin': {'protocol_path': str((round2 / 'protocol.json').resolve()),
                   'protocol_sha256': sha256(round2 / 'protocol.json'),
                   'training_records_path': str(records_path.resolve()),
                   'training_records_sha256': sha256(records_path),
                   'experiment': old['experiment'], 'dataset': old['dataset']},
        'record_files': {name: sha256(output / name)
                         for name in ('training_records.json', 'update_order.json')},
        'train_positive_prior': 0.5, 'training_counts': {'negative': 16, 'positive': 16},
        'sample_ids': ids, 'compiler': compiler.spec(), 'input_audit': input_audit,
        'selection_policy': 'training only; 16 per class; exclude truncated, then fixed input-hash order',
        'initialization': 'same seed and zero output weight; logit bias equals balanced training prior (0)',
        'arms': {name: {'lora_learning_rate': rates[0], 'head_learning_rate': rates[1]}
                 for name, rates in ARM_RATES.items()},
        'comparison_scope': 'joint optimization-strength diagnostic; both learning rates change together',
        'development_used': False, 'sealed_evaluation_opened': False,
        'maximum_update_order_sha256': digest(groups),
    }
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'prepared.sha256.json', {
        name: sha256(output / name)
        for name in ('protocol.json', 'training_records.json', 'update_order.json')})
    print(json.dumps({'status': 'prepared', 'samples': 32,
                      'truncated': sum(row['truncated'] for row in input_audit.values()),
                      'protocol_sha256': sha256(output / 'protocol.json')}), flush=True)


def load_prepared(root):
    root = Path(root)
    identity = json.loads((root / 'prepared.sha256.json').read_text())
    if set(identity) != {'protocol.json', 'training_records.json', 'update_order.json'}:
        raise ValueError('Unexpected diagnostic preparation manifest')
    if any(sha256(root / name) != expected for name, expected in identity.items()):
        raise ValueError('Diagnostic preparation changed')
    protocol = json.loads((root / 'protocol.json').read_text())
    if (protocol.get('experiment') != EXPERIMENT or protocol['configuration'] != CONFIGURATION
            or protocol['stopping'] != STOPPING or protocol['readout'] != READOUT
            or protocol['train_positive_prior'] != 0.5):
        raise ValueError('Unexpected diagnostic contract')
    if protocol['source_sha256'] != source_identity():
        raise ValueError('Executable source changed after preparation')
    origin = protocol['origin']
    if (sha256(origin['protocol_path']) != origin['protocol_sha256']
            or sha256(origin['training_records_path']) != origin['training_records_sha256']):
        raise ValueError('Origin protocol or training-record identity changed')
    container = json.loads((root / 'training_records.json').read_text())
    if set(container) != {'train'}:
        raise ValueError('Only training records belong in this diagnostic')
    records = container['train']
    if (len(records) != 32 or [row['id'] for row in records] != protocol['sample_ids']
            or len({row['id'] for row in records}) != 32
            or len({row['passage_sha256'] for row in records}) != 32
            or sum(row['label'] for row in records) != 16):
        raise ValueError('The balanced 32-input identity changed')
    for row in records:
        checked_record(row)
    groups = json.loads((root / 'update_order.json').read_text())
    if (groups != update_order(protocol['sample_ids'], protocol['seed'])
            or digest(groups) != protocol['maximum_update_order_sha256']):
        raise ValueError('Training update order changed')
    for name, rates in ARM_RATES.items():
        if protocol['arms'][name] != {'lora_learning_rate': rates[0], 'head_learning_rate': rates[1]}:
            raise ValueError('Diagnostic learning rates changed')
    helper = trainer().sibling('train_boolq_lora_ablation')
    return protocol, [helper.to_sample(row) for row in records], groups


def distribution(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim != 1 or not len(values) or not torch.isfinite(values).all():
        raise ValueError('Distributions require nonempty finite vectors')
    return {'minimum': float(values.min()), 'maximum': float(values.max()),
            'mean': float(values.mean()), 'standard_deviation': float(values.std(unbiased=False)),
            'quantiles': {str(q): float(values.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}


def training_metrics(predictions):
    logits = torch.tensor([row['logit'] for row in predictions], dtype=torch.float64)
    labels = torch.tensor([row['label'] for row in predictions], dtype=torch.float64)
    metrics = trainer().sibling('train_readout_ablation').binary_metrics(logits, labels)
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError('Both classes are required for this diagnostic')
    predicted = logits.sigmoid() >= 0.5
    correct = predicted == labels.bool()
    positive, negative = logits[labels == 1], logits[labels == 0]
    difference = positive[:, None] - negative[None, :]
    # Pairwise Mann-Whitney AUROC gives half credit to equal scores.
    metrics['auroc'] = float(((difference > 0).double() + 0.5 * (difference == 0)).mean())
    metrics['correct_count'] = int(correct.sum())
    metrics['predicted_positive_count'] = int(predicted.sum())
    metrics['confusion_matrix'] = {
        'rows': ['actual_no', 'actual_yes'], 'columns': ['predicted_no', 'predicted_yes'],
        'counts': [[int(((labels == label) & (predicted == output)).sum())
                    for output in (False, True)] for label in (0, 1)]}
    metrics['class_recall'] = {
        str(label): {'correct': int(correct[labels == label].sum()),
                     'count': int((labels == label).sum()),
                     'recall': float(correct[labels == label].double().mean())}
        for label in (0, 1)}
    metrics['logits'] = distribution(logits)
    metrics['probabilities'] = distribution(logits.sigmoid())
    metrics['class_distributions'] = {
        str(label): {'logits': distribution(logits[labels == label]),
                     'probabilities': distribution(logits[labels == label].sigmoid())}
        for label in (0, 1)}
    return metrics


def meets_stopping(metrics):
    return (metrics['correct_count'] >= STOPPING['minimum_correct']
            and metrics['nll'] <= STOPPING['maximum_nll']
            and all(metrics['class_recall'][str(label)]['correct'] >=
                    STOPPING['minimum_correct_per_class'] for label in (0, 1)))


def warmup_scale(step):
    warmup = math.ceil(CONFIGURATION['maximum_updates'] * CONFIGURATION['warmup_fraction'])
    return min(step / warmup, 1.0)


def gradient_statistics(parameters):
    squared, maximum, present, zero = 0.0, 0.0, 0, 0
    for value in parameters:
        if value.grad is None:
            continue
        gradient = value.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite training gradient')
        norm = float(torch.linalg.vector_norm(gradient.float()))
        squared += norm * norm
        maximum = max(maximum, norm)
        present += 1
        zero += int(norm == 0)
    return {'pre_clip_l2_norm': math.sqrt(squared), 'maximum_tensor_l2_norm': maximum,
            'gradient_tensors': present, 'zero_gradient_tensors': zero,
            'missing_gradient_tensors': len(parameters) - present}


def require_finite_parameters(parameters):
    if any(not torch.isfinite(value).all() for value in parameters):
        raise FloatingPointError('Nonfinite trained parameter')


def common_initialization(root, value):
    """Atomically establish a common initialization even for concurrent arms."""
    root = Path(root)
    target = root / 'common_initialization.json'
    temporary = root / f'.common_initialization.{os.getpid()}.json'
    write_json(temporary, value)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if json.loads(target.read_text()) != value:
                raise ValueError('The two diagnostic arms have different initialization or inputs')
    finally:
        temporary.unlink(missing_ok=True)


def run_train(args):
    root = Path(args.protocol_dir)
    protocol, samples, groups = load_prepared(root)
    if torch.device(args.device).type != 'cuda':
        raise ValueError('Diagnostic training requires BF16 CUDA with FP32 masters')
    destination = root / args.arm
    destination.mkdir(exist_ok=False)
    source = args.model or protocol['source_model']
    rates = protocol['arms'][args.arm]
    spec = {'experiment': EXPERIMENT, 'arm': args.arm, 'source': str(Path(source).resolve()),
            'training_pid': os.getpid(),
            'protocol': protocol, 'protocol_sha256': sha256(root / 'protocol.json'),
            'readout': READOUT, 'seed': protocol['seed'], 'backend': 'sdpa',
            'precision': 'bf16', 'batch_size': 4, 'learning_rates': rates,
            'scope': 'balanced 32-input training memorization diagnostic, not generalization',
            'development_used': False, 'sealed_evaluation_opened': False}
    write_json(destination / 'run_spec.json', spec)
    try:
        model = trainer().make_model(source, protocol, READOUT, protocol['seed'], args.device)
        if any(value.is_floating_point() and value.dtype != torch.float32
               for value in model.parameters()):
            raise ValueError('All floating-point parameter masters must remain FP32')
        if model.compiler.spec() != protocol['compiler']:
            raise ValueError('The compiler differs from CPU preparation')
        for sample in samples:
            compiled = model.compiler.compile(sample)
            expected = protocol['input_audit'][sample['id']]
            if (digest(compiled.pairs[0]) != expected['input_ids_sha256']
                    or compiled.truncated[0] != expected['truncated']):
                raise ValueError('Prepared and training token inputs differ')
        lora = [value for name, value in model.backbone.named_parameters() if value.requires_grad]
        if any('lora_' not in name for name, value in model.backbone.named_parameters()
               if value.requires_grad):
            raise ValueError('Only LoRA backbone parameters may train')
        head = list(model.score_head.parameters())
        trainable = lora + head
        initial = trainer().fingerprints(model)
        common_initialization(root, {'parameter_sha256': initial, 'seed': protocol['seed'],
                                    'protocol_sha256': spec['protocol_sha256'],
                                    'maximum_update_order_sha256': digest(groups)})
        optimizer = torch.optim.AdamW([
            {'params': lora, 'lr': rates['lora_learning_rate']},
            {'params': head, 'lr': rates['head_learning_rate']}], weight_decay=0.0)
        mapping = {sample['id']: sample for sample in samples}
        history, consecutive, steps = [], 0, 0
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(model.device)

        def record_evaluation(step):
            nonlocal consecutive
            _, predictions = trainer().evaluate(model, samples, precision='bf16', batch_size=4)
            metrics = training_metrics(predictions)
            satisfied = meets_stopping(metrics)
            consecutive = consecutive + 1 if satisfied else 0
            row = {'updates': step, 'training': metrics, 'stopping_threshold_satisfied': satisfied,
                   'consecutive_threshold_evaluations': consecutive,
                   'elapsed_seconds': time.monotonic() - started}
            history.append(row)
            write_json(destination / 'training_history.json', history)
            print(json.dumps({'arm': args.arm, 'updates': step, 'train_nll': metrics['nll'],
                              'train_accuracy': metrics['accuracy'], 'train_auroc': metrics['auroc'],
                              'consecutive_threshold_evaluations': consecutive}), flush=True)
            return predictions

        final_predictions = record_evaluation(0)
        model.train()
        with (destination / 'gradient_audit.jsonl').open('x') as gradient_log:
            for step, identifiers in enumerate(groups, 1):
                batch_group = [mapping[name] for name in identifiers]
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for offset in range(0, len(batch_group), 4):
                    batch = batch_group[offset:offset + 4]
                    with trainer().precision_context(model.device, 'bf16'):
                        logits = model(batch)
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            logits, trainer().labels_for(batch, model.device))
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training logit or loss')
                    (loss * (len(batch) / len(batch_group))).backward()
                    group_loss += float(loss.detach()) * len(batch) / len(batch_group)
                lora_gradient, head_gradient = gradient_statistics(lora), gradient_statistics(head)
                total_norm = nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                scale = warmup_scale(step)
                for optimizer_group, rate in zip(optimizer.param_groups,
                                                (rates['lora_learning_rate'], rates['head_learning_rate'])):
                    optimizer_group['lr'] = rate * scale
                optimizer.step()
                require_finite_parameters(trainable)
                gradient_row = {'updates': step, 'sample_ids': identifiers, 'microbatch_count': 2,
                                'training_batch_nll': group_loss, 'lora': lora_gradient,
                                'head': head_gradient, 'total_pre_clip_l2_norm': float(total_norm),
                                'clip_coefficient': min(1.0, 1.0 / (float(total_norm) + 1e-6)),
                                'lora_learning_rate': optimizer.param_groups[0]['lr'],
                                'head_learning_rate': optimizer.param_groups[1]['lr']}
                gradient_log.write(json.dumps(gradient_row, allow_nan=False) + '\n')
                gradient_log.flush()
                steps = step
                if step % CONFIGURATION['evaluate_every_updates'] == 0:
                    final_predictions = record_evaluation(step)
                    if consecutive >= STOPPING['consecutive_evaluations']:
                        break
        final = trainer().fingerprints(model)
        if final['frozen_base'] != initial['frozen_base']:
            raise ValueError('Frozen backbone parameters changed')
        if final['lora'] == initial['lora'] or final['head'] == initial['head']:
            raise ValueError('Both LoRA and head must actually update')
        write_json(destination / 'training_predictions.json', final_predictions)
        reference_ids = [next(row['id'] for row in final_predictions if row['label'] == label)
                         for label in (0, 1)]
        reference_ids.append(next(row['id'] for row in final_predictions if row['id'] not in reference_ids))
        references = [mapping[name] for name in reference_ids]
        _, reference_predictions = trainer().evaluate(model, references, precision='bf16', batch_size=4)
        training_metrics(reference_predictions)
        trainer().export(model, destination / 'final', dict(spec, updates=steps), references,
                         [row['logit'] for row in reference_predictions])
        audit = {'status': 'complete', 'scope': spec['scope'], 'initial': initial, 'final': final,
                 'frozen_base_unchanged': True, 'lora_updated': True, 'head_updated': True,
                 'optimizer_calls': steps, 'nonzero_learning_rate_calls': steps,
                 'early_stopped': steps < CONFIGURATION['maximum_updates'],
                 'stopping_confirmed': consecutive >= STOPPING['consecutive_evaluations'],
                 'training_seconds': time.monotonic() - started,
                 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(model.device),
                 'lora_parameters': sum(value.numel() for value in lora),
                 'head_parameters': sum(value.numel() for value in head),
                 'initial_zero_lora_gradient_is_expected': True,
                 'training_count': 32, 'reference_training_ids': reference_ids,
                 'development_used': False, 'sealed_evaluation_opened': False,
                 'history': history, 'joint_learning_rate_change': True,
                 'actual_update_order_sha256': digest(groups[:steps])}
        write_json(destination / 'training_audit.json', audit)
        print(json.dumps({'arm': args.arm, 'status': 'complete', 'updates': steps,
                          'stopping_confirmed': audit['stopping_confirmed']}), flush=True)
    except Exception as error:
        write_json(destination / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                                 'error': str(error), 'scope': spec['scope']})
        raise


def verify(args):
    destination = Path(args.export)
    model, spec = trainer().load_export(destination, device=args.device, source=args.model)
    if spec.get('experiment') != EXPERIMENT or spec.get('readout') != READOUT:
        raise ValueError('Expected a BoolQ training-memorization artifact')
    if spec.get('training_pid') == os.getpid():
        raise ValueError('Reload verification must run in a separate process')
    references = json.loads((destination / 'verification_inputs.json').read_text())
    expected = torch.tensor(json.loads((destination / 'reference_logits.json').read_text()))
    if len(references) != 3 or any(row['id'] not in spec['protocol']['sample_ids'] for row in references):
        raise ValueError('Verification references must be three fixed training inputs')
    _, predictions = trainer().evaluate(model, references, precision=spec['precision'], batch_size=4)
    training_metrics(predictions)
    actual = torch.tensor([row['logit'] for row in predictions])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    loaded = trainer().fingerprints(model)
    selected = spec['selected_parameter_sha256']
    if any(loaded[name] != selected[name] for name in ('head', 'lora', 'pool')):
        raise ValueError('Reloaded research parameters differ from the export')
    result = {'status': 'passed', 'samples': 3, 'parameter_sha256': loaded,
              'training_pid': spec['training_pid'], 'verification_pid': os.getpid(),
              'maximum_absolute_difference': float((actual - expected).abs().max()),
              'scope': 'three fixed training references in a fresh process; no generalization claim'}
    write_json(destination.parent / 'reload_verification.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare', help='Freeze a balanced training-only 32-input protocol on CPU')
    preparation.add_argument('--round2', required=True)
    preparation.add_argument('--model', required=True)
    preparation.add_argument('--output', required=True)
    preparation.add_argument('--seed', type=int, default=42)
    preparation.set_defaults(function=prepare)
    training = commands.add_parser('train', help='Run one of the two predeclared CUDA diagnostic arms')
    training.add_argument('--protocol-dir', required=True)
    training.add_argument('--arm', choices=tuple(ARM_RATES), required=True)
    training.add_argument('--model')
    training.add_argument('--device', default='cuda:0')
    training.set_defaults(function=run_train)
    verification = commands.add_parser('verify', help='Reload three training references in a separate process')
    verification.add_argument('--export', required=True)
    verification.add_argument('--model')
    verification.add_argument('--device', default='cuda:0')
    verification.set_defaults(function=verify)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == '__main__':
    main()
