"""Train isolated BoolQ readout experiments; freeze all selections before evaluation.

The protocol preparer fixes the data, nine training runs, and optimizer settings.
Training workers never open the sealed evaluation records. Finalization requires
every registered run; evaluation loads only the selected research artifacts.
These artifacts do not change the Metis v1 Predictor format.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import shutil
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from metis.model import QwenScoreHead
from metis.schema import sha256


FORMAT = 'metis-research-boolq-readout-v2'
READOUTS = ('last_relevance', 'decision_marker', 'candidate_attention')
_MODULES = {}


def sibling(name):
    if name not in _MODULES:
        path = Path(__file__).with_name(name + '.py')
        spec = importlib.util.spec_from_file_location('metis_research_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def file_identity(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob('*'))
            if path.is_file() and path.name != 'files.sha256.json'}


def verify_files(root):
    root = Path(root)
    identity = json.loads((root / 'files.sha256.json').read_text())
    if identity != file_identity(root):
        raise ValueError('Research artifact checksum mismatch')


def load_protocol(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['experiment'] != 'boolq_readout_round2':
        raise ValueError('Expected a round-two BoolQ protocol')
    return protocol


def load_training_samples(root, protocol):
    # No evaluation file is opened by this function or a training worker.
    if sha256(Path(root) / 'train_dev_records.json') != protocol['record_files']['train_dev_records.json']:
        raise ValueError('Training records differ from fixed identity')
    records = json.loads((Path(root) / 'train_dev_records.json').read_text())
    if set(records) != {'train', 'dev'}:
        raise ValueError('Training requires train/dev records only')
    helper = sibling('train_boolq_lora_ablation')
    return {split: [helper.to_sample(item) for item in rows] for split, rows in records.items()}


def precision_context(device, precision):
    if precision == 'float32':
        return nullcontext()
    if precision != 'bf16' or torch.device(device).type != 'cuda':
        raise ValueError('BF16 execution requires CUDA')
    return torch.autocast('cuda', dtype=torch.bfloat16)


def labels_for(samples, device):
    return torch.tensor([sample['supervision']['labels']['passage'] for sample in samples],
                        device=device, dtype=torch.float32)


def fingerprints(model):
    digest = sibling('train_boolq_lora_ablation').parameter_hash
    return {
        'lora': digest((name, value) for name, value in model.backbone.named_parameters() if 'lora_' in name),
        'frozen_base': digest((name, value) for name, value in model.backbone.named_parameters() if 'lora_' not in name),
        'head': digest(model.score_head.named_parameters()),
        'pool': digest([('pool_query', model.pool_query)] if model.pool_query is not None else []),
    }


def make_model(source, protocol, readout, seed, device, *, training=True, backend='sdpa'):
    helper = sibling('train_boolq_lora_ablation')
    helper.verify_base_identity(source, protocol['base']['file_sha256'])
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    scorer = QwenScoreHead(source, device=device, dtype='float32', backend=backend,
                          max_length=protocol['configuration']['max_length'],
                          instruction=protocol['instruction'], layout='pairs')
    module = sibling('boolq_readout_model')
    model = module.BoolQReadoutModel(scorer, readout=readout,
                                    mlp_width=protocol['configuration']['mlp_width'])
    if training:
        module.configure_lora(model, rank=protocol['configuration']['lora_rank'],
                              alpha=protocol['configuration']['lora_alpha'],
                              gradient_checkpointing=True)
        # Identical constant initial prediction for all readout variants. This
        # removes random output scale as a readout comparison confound.
        model.initialize_readout(protocol['train_positive_prior'], seed=seed)
    return model


def evaluate(model, samples, *, precision, batch_size=2):
    previous = model.training
    model.eval()
    logits = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(samples), batch_size):
                with precision_context(model.device, precision):
                    values = model(samples[offset:offset + batch_size])
                logits.append(values.detach().float().cpu())
        logits = torch.cat(logits)
        labels = labels_for(samples, 'cpu')
        metrics = sibling('train_readout_ablation').binary_metrics(logits, labels)
        predictions = [{'id': sample['id'], 'label': int(label), 'logit': float(logit),
                        'probability_yes': float(logit.sigmoid())}
                       for sample, label, logit in zip(samples, labels, logits)]
        return metrics, predictions
    finally:
        model.train(previous)


def export(model, destination, spec, references, reference_logits):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model.backbone.save_pretrained(destination / 'lora', safe_serialization=True)
    model.tokenizer.save_pretrained(destination / 'tokenizer')
    readout_state = {'head.' + name: value.detach().cpu().contiguous()
                     for name, value in model.score_head.state_dict().items()}
    if model.pool_query is not None:
        readout_state['pool_query'] = model.pool_query.detach().cpu().contiguous()
    save_file(readout_state, str(destination / 'readout.safetensors'))
    digest = sibling('train_boolq_lora_ablation').parameter_hash
    selected_parameters = {'head': digest(model.score_head.named_parameters()),
                           'lora': digest((name, value) for name, value in model.backbone.named_parameters()
                                          if 'lora_' in name),
                           'pool': digest([('pool_query', model.pool_query)] if model.pool_query is not None else [])}
    write_json(destination / 'standalone_spec.json', dict(spec, format=FORMAT,
               readout_spec=model.readout_spec(),
               selected_parameter_sha256=selected_parameters,
               artifact_scope='research LoRA + scalar readout; not a v1 Predictor artifact'))
    write_json(destination / 'verification_inputs.json', references)
    write_json(destination / 'reference_logits.json', reference_logits)
    write_json(destination / 'files.sha256.json', file_identity(destination))


def load_export(destination, *, device='cpu', source=None):
    from peft import PeftModel
    destination = Path(destination)
    verify_files(destination)
    spec = json.loads((destination / 'standalone_spec.json').read_text())
    if spec.get('format') != FORMAT or spec.get('readout') not in READOUTS:
        raise ValueError('Unsupported research artifact')
    model = make_model(source or spec['source'], spec['protocol'], spec['readout'],
                       spec['seed'], device, training=False, backend=spec['backend'])
    model.backbone = PeftModel.from_pretrained(model.backbone, destination / 'lora', is_trainable=False)
    state = load_file(str(destination / 'readout.safetensors'))
    model.score_head.load_state_dict({name.removeprefix('head.'): value for name, value in state.items()
                                     if name.startswith('head.')}, strict=True)
    allowed = {'head.' + name for name in model.score_head.state_dict()}
    if model.pool_query is not None:
        allowed.add('pool_query')
        with torch.no_grad():
            model.pool_query.copy_(state['pool_query'].to(model.device))
    if set(state) != allowed or model.readout_spec() != spec['readout_spec']:
        raise ValueError('Loaded readout differs from the trained contract')
    return model.eval(), spec


def verify_loaded(model, destination, spec):
    root = Path(destination)
    samples = json.loads((root / 'verification_inputs.json').read_text())
    expected = torch.tensor(json.loads((root / 'reference_logits.json').read_text()))
    _, predictions = evaluate(model, samples, precision=spec['precision'], batch_size=spec['batch_size'])
    actual = torch.tensor([row['logit'] for row in predictions])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    return {'status': 'passed', 'samples': len(samples),
            'maximum_absolute_difference': float((actual - expected).abs().max()),
            'scope': 'selected research weights; three fixed dev references, identical batching'}


def learning_rate_scale(step, total, warmup_fraction):
    warmup = max(1, math.ceil(total * warmup_fraction))
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_one(model, samples, destination, spec, configuration):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    batch_size = configuration['batch_size']
    group_size = configuration['effective_batch_size']
    if group_size % batch_size or batch_size < 1:
        raise ValueError('Effective batch size must be a positive multiple of microbatch')
    steps_per_epoch = math.ceil(len(samples['train']) / group_size)
    total_steps = configuration['epochs'] * steps_per_epoch
    backbone_parameters = [value for value in model.backbone.parameters() if value.requires_grad]
    readout_parameters = list(model.score_head.parameters())
    if model.pool_query is not None:
        readout_parameters.append(model.pool_query)
    trainable = backbone_parameters + readout_parameters
    optimizer = torch.optim.AdamW([
        {'params': backbone_parameters, 'lr': configuration['lora_learning_rate']},
        {'params': readout_parameters, 'lr': configuration['head_learning_rate']}],
        weight_decay=configuration['weight_decay'])
    initial = fingerprints(model)
    ordering = torch.Generator().manual_seed(spec['seed'])
    history, steps, best_nll, selected = [], 0, math.inf, None
    start = time.monotonic()
    if model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
    references = samples['dev'][:3]
    write_json(destination / 'run_spec.json', spec)
    input_stats = {}
    for split, items in samples.items():
        lengths, truncated, candidates = [], 0, []
        for offset in range(0, len(items), batch_size):
            batch = model.prepare_batch(items[offset:offset + batch_size])
            lengths.extend(batch['lengths'])
            truncated += sum(batch['truncated'])
            for ids, mask in zip(batch['input_ids'], batch['candidate_mask']):
                candidates.append(ids[mask].cpu().tolist())
        input_stats[split] = {'truncated': truncated, 'mean_tokens': sum(lengths) / len(lengths),
                              'max_tokens': max(lengths),
                              'candidate_body_ids_sha256': hashlib.sha256(
                                  json.dumps(candidates, sort_keys=True).encode()).hexdigest()}
    write_json(destination / 'input_audit.json', input_stats)
    for epoch in range(1, configuration['epochs'] + 1):
        model.train()
        permutation = torch.randperm(len(samples['train']), generator=ordering).tolist()
        losses, count, gradient_norms = 0.0, 0, []
        optimizer.zero_grad(set_to_none=True)
        for group_offset in range(0, len(permutation), group_size):
            group = permutation[group_offset:group_offset + group_size]
            for offset in range(0, len(group), batch_size):
                batch = [samples['train'][index] for index in group[offset:offset + batch_size]]
                with precision_context(model.device, spec['precision']):
                    logits = model(batch)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_for(batch, model.device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss * (len(batch) / len(group))).backward()
                losses += float(loss.detach()) * len(batch)
                count += len(batch)
            norm = nn.utils.clip_grad_norm_(trainable, configuration['gradient_clip'])
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient norm')
            gradient_norms.append(float(norm))
            steps += 1
            scale = learning_rate_scale(steps, total_steps, configuration['warmup_fraction'])
            for opt_group, rate in zip(optimizer.param_groups,
                                       (configuration['lora_learning_rate'], configuration['head_learning_rate'])):
                opt_group['lr'] = rate * scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if steps % 16 == 0:
                print(json.dumps({'readout': spec['readout'], 'seed': spec['seed'], 'epoch': epoch,
                                  'steps': steps, 'total_steps': total_steps,
                                  'elapsed_seconds': round(time.monotonic() - start, 1)}), flush=True)
        dev, _ = evaluate(model, samples['dev'], precision=spec['precision'], batch_size=batch_size)
        row = {'epoch': epoch, 'steps': steps, 'train_nll': losses / count, 'dev': dev,
               'gradient_norm_mean': sum(gradient_norms) / len(gradient_norms),
               'gradient_norm_max': max(gradient_norms),
               'epoch_order_sha256': hashlib.sha256(json.dumps(
                   [samples['train'][index]['id'] for index in permutation]).encode()).hexdigest()}
        history.append(row)
        checkpoint = destination / 'checkpoints' / f'epoch-{epoch}'
        _, reference_rows = evaluate(model, references, precision=spec['precision'], batch_size=batch_size)
        export(model, checkpoint, dict(spec, epoch=epoch, batch_size=batch_size), references,
               [item['logit'] for item in reference_rows])
        if dev['nll'] < best_nll:
            best_nll, selected = dev['nll'], checkpoint
        write_json(destination / 'training_history.json', history)
        write_json(destination / 'selection.json', {'criterion': 'minimum dev NLL',
                   'checkpoint': str(selected.relative_to(destination)), 'best_dev_nll': best_nll,
                   'evaluation_used_for_selection': False})
        print(json.dumps({'readout': spec['readout'], 'seed': spec['seed'], **row}), flush=True)
    final = fingerprints(model)
    if final['frozen_base'] != initial['frozen_base']:
        raise ValueError('Frozen base changed during training')
    if initial['head'] == final['head'] or initial['lora'] == final['lora']:
        raise ValueError('Readout and LoRA must both actually update')
    if model.pool_query is not None and initial['pool'] == final['pool']:
        raise ValueError('Attention pooling query did not update')
    shutil.copytree(selected, destination / 'best')
    selected_parameters = json.loads((selected / 'standalone_spec.json').read_text())['selected_parameter_sha256']
    if (selected_parameters['head'] == initial['head'] or selected_parameters['lora'] == initial['lora']
            or (model.pool_query is not None and selected_parameters['pool'] == initial['pool'])):
        raise ValueError('Selected readout and adapter must actually update')
    audit = {'initial': initial, 'final': final, 'frozen_base_unchanged': True,
             'head_updated': True, 'lora_updated': True,
             'pool_updated': model.pool_query is not None and initial['pool'] != final['pool'],
             'selected_parameter_sha256': selected_parameters,
             'steps': steps, 'best_epoch': int(selected.name.split('-')[-1]), 'best_dev_nll': best_nll,
             'training_seconds': time.monotonic() - start,
             'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(model.device)
                  if model.device.type == 'cuda' else None,
             'head_parameters': sum(value.numel() for value in model.score_head.parameters()),
             'pool_parameters': model.pool_query.numel() if model.pool_query is not None else 0,
             'lora_parameters': sum(value.numel() for value in backbone_parameters),
             'evaluation_opened': False, 'history': history}
    write_json(destination / 'training_audit.json', audit)
    return audit


def finalize(root):
    root = Path(root)
    protocol = load_protocol(root)
    destination = root / 'selection-lock.json'
    if destination.exists():
        raise FileExistsError('Selections are already frozen')
    runs, common_initial, common_inputs, common_orders = {}, {}, None, {}
    for readout in protocol['readouts']:
        for seed in protocol['seeds']:
            key = f'{readout}/seed-{seed}'
            directory = root / 'models' / key
            audit = json.loads((directory / 'training_audit.json').read_text())
            if not (audit['head_updated'] and audit['lora_updated'] and audit['frozen_base_unchanged']):
                raise ValueError(f'Training contract failed: {key}')
            verify_files(directory / 'best')
            saved = json.loads((directory / 'best/standalone_spec.json').read_text())
            run_spec = json.loads((directory / 'run_spec.json').read_text())
            for spec in (saved, run_spec):
                if (spec['protocol_sha256'] != sha256(root / 'protocol.json')
                        or spec['protocol'] != protocol or spec['readout'] != readout or spec['seed'] != seed
                        or spec['precision'] != protocol['configuration'].get('precision', spec['precision'])):
                    raise ValueError(f'Training run differs from the frozen protocol: {key}')
            if audit['steps'] != protocol['configuration']['epochs'] * math.ceil(
                    len(json.loads((root / 'train_dev_records.json').read_text())['train']) /
                    protocol['configuration']['effective_batch_size']):
                raise ValueError(f'Optimizer budget differs from the protocol: {key}')
            initial = {name: audit['initial'][name] for name in ('head', 'lora', 'frozen_base')}
            if seed in common_initial and common_initial[seed] != initial:
                raise ValueError(f'Common parameters have different initial values: {key}')
            common_initial[seed] = initial
            orders = [item['epoch_order_sha256'] for item in audit['history']]
            if seed in common_orders and common_orders[seed] != orders:
                raise ValueError(f'Training sample order differs across variants: {key}')
            common_orders[seed] = orders
            inputs = json.loads((directory / 'input_audit.json').read_text())
            body_hashes = {split: value['candidate_body_ids_sha256'] for split, value in inputs.items()}
            if common_inputs is not None and common_inputs != body_hashes:
                raise ValueError(f'Candidate bodies differ across readout variants: {key}')
            common_inputs = body_hashes
            runs[key] = {'selection_sha256': sha256(directory / 'selection.json'),
                         'audit_sha256': sha256(directory / 'training_audit.json'),
                         'artifact_manifest_sha256': sha256(directory / 'best/files.sha256.json')}
    lock = {'protocol_sha256': sha256(root / 'protocol.json'), 'runs': runs,
            'scope': 'all planned models selected by dev only before any round-two evaluation',
            'evaluation_opened': False, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'common_initial_parameters_by_seed': {str(seed): value for seed, value in common_initial.items()},
            'common_epoch_order_hashes_by_seed': {str(seed): value for seed, value in common_orders.items()},
            'common_candidate_body_hashes': common_inputs}
    write_json(destination, lock)
    return lock


def assert_locked(root, protocol):
    root = Path(root)
    lock = json.loads((root / 'selection-lock.json').read_text())
    if lock['protocol_sha256'] != sha256(root / 'protocol.json'):
        raise ValueError('Protocol changed after selection freeze')
    expected_keys = {f'{readout}/seed-{seed}' for readout in protocol['readouts'] for seed in protocol['seeds']}
    if set(lock['runs']) != expected_keys:
        raise ValueError('Selection freeze omits registered runs')
    for key, identity in lock['runs'].items():
        directory = root / 'models' / key
        actual = {'selection_sha256': sha256(directory / 'selection.json'),
                  'audit_sha256': sha256(directory / 'training_audit.json'),
                  'artifact_manifest_sha256': sha256(directory / 'best/files.sha256.json')}
        if identity != actual:
            raise ValueError(f'Selection changed after freeze: {key}')
        verify_files(directory / 'best')
    return lock


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    root = Path(args.protocol)
    protocol = load_protocol(root)
    if args.action == 'finalize':
        print(json.dumps(finalize(root)), flush=True)
        return
    if args.readout not in protocol['readouts']:
        raise ValueError('Choose a registered readout')
    configuration = protocol['configuration']
    if args.action == 'train':
        if (root / 'selection-lock.json').exists():
            raise ValueError('Training is closed after selection freeze')
        samples = load_training_samples(root, protocol)
        if args.precision != configuration.get('precision', args.precision):
            raise ValueError('Training precision differs from the registered protocol')
        for seed in protocol['seeds']:
            model = make_model(args.model, protocol, args.readout, seed, args.device)
            spec = {'source': str(Path(args.model).resolve()), 'protocol': protocol,
                    'readout': args.readout, 'seed': seed, 'backend': 'sdpa',
                    'precision': args.precision, 'protocol_sha256': sha256(root / 'protocol.json')}
            train_one(model, samples, root / 'models' / args.readout / f'seed-{seed}', spec, configuration)
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    elif args.action == 'evaluate':
        lock = assert_locked(root, protocol)
        helper = sibling('train_boolq_lora_ablation')
        if sha256(root / 'sealed_evaluation.json') != protocol['record_files']['sealed_evaluation.json']:
            raise ValueError('Evaluation records differ from the frozen protocol')
        records = json.loads((root / 'sealed_evaluation.json').read_text())['evaluation']
        samples = [helper.to_sample(record) for record in records]
        for seed in protocol['seeds']:
            directory = root / 'models' / args.readout / f'seed-{seed}'
            if (directory / 'evaluation.json').exists():
                raise FileExistsError(f'Evaluation already exists: {directory}')
            model, spec = load_export(directory / 'best', device=args.device, source=args.model)
            report = verify_loaded(model, directory / 'best', spec)
            write_json(directory / 'fresh-process-reload.json', report)
            metrics, predictions = evaluate(model, samples, precision=spec['precision'],
                                             batch_size=configuration['batch_size'])
            write_json(directory / 'evaluation.json', {
                'scope': protocol['evaluation'], 'metrics': metrics, 'predictions': predictions,
                'selection_lock_sha256': sha256(root / 'selection-lock.json'),
                'evaluation_records_sha256': protocol['record_files']['sealed_evaluation.json']})
            print(json.dumps({'readout': args.readout, 'seed': seed, 'evaluation': metrics, 'reload': report}), flush=True)
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'finalize', 'evaluate'])
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--model')
    parser.add_argument('--readout', choices=READOUTS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['float32', 'bf16'], default='bf16')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.20)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_fraction <= 1 or args.threads < 1:
        parser.error('Invalid memory fraction or thread count')
    if args.action == 'train' and (not args.model or not args.readout):
        parser.error('Training requires model and readout')
    if args.action == 'train' and args.precision == 'bf16' and torch.device(args.device).type != 'cuda':
        parser.error('Use float32 for CPU training')
    run(args)


if __name__ == '__main__':
    main()
