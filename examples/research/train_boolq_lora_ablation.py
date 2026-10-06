"""Jointly train LoRA and scalar heads on a fixed BoolQ research pilot.

This entrypoint consumes selected_records.json and protocol.json from
extract_boolq_features.py. It uses the identical inputs but recomputes backbone
representations during training. Its exports are standalone research artifacts,
not v1 Metis Predictor artifacts. No cached feature is used for optimization.

    python examples/research/train_boolq_lora_ablation.py \
        --pilot /path/to/frozen-feature-run --model /path/to/Qwen3-0.6B-Base \
        --output /path/to/new-lora-run --device cuda:0

Reload selected weights in a separate process:

    python examples/research/train_boolq_lora_ablation.py \
        --verify /path/to/new-lora-run/mlp/seed-42/best --device cuda:0
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
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from metis.compiler import ScoreHeadCompiler
from metis.model import QwenScoreHead
from metis.schema import sha256, validate_sample
from metis.tasks.objectives import supervised_loss


FORMAT = 'metis-research-boolq-lora-readout-v1'
HEAD_NAMES = ('linear', 'mlp', 'swiglu', 'residual_swiglu')
_READOUT = None


def readout_module():
    """Load the sibling research implementation without changing package APIs."""
    global _READOUT
    if _READOUT is None:
        path = Path(__file__).with_name('train_readout_ablation.py')
        spec = importlib.util.spec_from_file_location('metis_research_scalar_heads', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _READOUT = module
    return _READOUT


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def records_hash(records):
    # Match the extractor's records_sha256, including its default ASCII escaping.
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def to_sample(record):
    label = record['label']
    if type(label) is not int or label not in (0, 1):
        raise ValueError('BoolQ research labels must be integer 0/1')
    sample = {'schema_version': '1.0', 'id': record['id'], 'task_id': 'boolq-affirmative',
              'input': {'query': record['question'], 'context': ''},
              'candidates': [{'id': 'passage', 'text': record['passage']}],
              'supervision': {'kind': 'candidate_labels', 'labels': {'passage': label},
                              'unjudged_policy': 'error'}}
    return validate_sample(sample, task_kind='multi_label')


def load_pilot(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    records = json.loads((root / 'selected_records.json').read_text())
    if protocol.get('experiment') != 'frozen_base_readout_pilot':
        raise ValueError('Expected the frozen BoolQ pilot protocol')
    if protocol.get('compiler', {}).get('version') != ScoreHeadCompiler.version:
        raise ValueError('Pilot input compiler differs from the current readout compiler')
    if set(records) != {'train', 'dev', 'evaluation'}:
        raise ValueError('Pilot requires train/dev/evaluation splits')
    samples, groups, identifiers = {}, {}, set()
    for split, items in records.items():
        identity = protocol['splits'][split]
        if (not items or identity['count'] != len(items)
                or identity['records_sha256'] != records_hash(items)
                or identity['ids'] != [item['id'] for item in items]):
            raise ValueError(f'Pilot records differ from frozen identity: {split}')
        samples[split] = [to_sample(item) for item in items]
        ids = {item['id'] for item in items}
        if len(ids) != len(items) or ids & identifiers:
            raise ValueError('Pilot sample IDs must be unique across splits')
        identifiers |= ids
        groups[split] = {hashlib.sha256(item['passage'].encode()).hexdigest() for item in items}
    names = list(groups)
    if any(groups[left] & groups[right] for i, left in enumerate(names) for right in names[i + 1:]):
        raise ValueError('Pilot passage groups overlap across splits')
    return protocol, samples


def verify_base_identity(source, expected):
    if not expected:
        raise ValueError('The pilot does not identify its local backbone weights')
    for name, digest in expected.items():
        path = Path(source) / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Backbone differs from the frozen pilot: {name}')


class KeepScalarDimension(nn.Module):
    """Adapt research [N] heads to QwenScoreHead's [N, 1] readout contract."""
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hidden):
        return self.head(hidden).unsqueeze(-1)


def configure_lora_head(model, kind, *, mlp_width=256, rank=8, alpha=16,
                        gradient_checkpointing=True):
    from peft import LoraConfig, get_peft_model
    model.backbone.requires_grad_(False)
    model.backbone = get_peft_model(model.backbone, LoraConfig(
        task_type=model.peft_task_type, r=rank, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none'))
    dimension = model.backbone.config.hidden_size
    model.score_head = KeepScalarDimension(
        readout_module().build_head(kind, dimension, mlp_width)).to(model.device, dtype=torch.float32)
    if gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        model.backbone.enable_input_require_grads()
    assert_trainable_contract(model)
    return model


def assert_trainable_contract(model):
    backbone = [(name, value) for name, value in model.backbone.named_parameters() if value.requires_grad]
    head = list(model.score_head.parameters())
    if not backbone or any('lora_' not in name for name, _ in backbone):
        raise ValueError('Only LoRA backbone parameters may be trainable')
    if not head or not all(value.requires_grad for value in head):
        raise ValueError('Every independent head parameter must be trainable')


def parameter_hash(named_parameters):
    digest = hashlib.sha256()
    for name, value in sorted(named_parameters, key=lambda item: item[0]):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_fingerprints(model):
    return {'head': parameter_hash(model.score_head.named_parameters()),
            'lora': parameter_hash((name, value) for name, value in model.backbone.named_parameters()
                                   if 'lora_' in name),
            'frozen_base': parameter_hash((name, value) for name, value in model.backbone.named_parameters()
                                          if 'lora_' not in name)}


def precision_context(model, precision):
    if precision == 'float32':
        return nullcontext()
    if precision != 'bf16' or model.device.type != 'cuda':
        raise ValueError('BF16 research execution requires CUDA')
    return torch.autocast('cuda', dtype=torch.bfloat16)


def batch_loss(model, samples):
    scores = model.score_tensors(samples)
    return torch.stack([supervised_loss(score, sample, task_kind='multi_label',
                                       bce_weight=1.0, pairwise_weight=0.0)
                        for score, sample in zip(scores, samples)]).mean()


def evaluate_model(model, samples, *, precision='float32', batch_size=2):
    previous = model.training
    model.eval()
    values = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(samples), batch_size):
                with precision_context(model, precision):
                    values.extend(score[0].detach().float().cpu() for score in
                                  model.score_tensors(samples[offset:offset + batch_size]))
        logits = torch.stack(values)
        labels = torch.tensor([sample['supervision']['labels']['passage'] for sample in samples])
        metrics = readout_module().binary_metrics(logits, labels)
        predictions = [{'id': sample['id'], 'label': int(label), 'logit': float(logit),
                        'probability_yes': float(logit.sigmoid())}
                       for sample, label, logit in zip(samples, labels, logits)]
        return metrics, predictions
    finally:
        model.train(previous)


def export_research(model, destination, spec, verification_samples, reference_logits):
    """Save PEFT and head weights together, explicitly outside v1 serialization."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model.backbone.save_pretrained(destination / 'lora', safe_serialization=True)
    model.tokenizer.save_pretrained(destination / 'tokenizer')
    save_file({name: value.detach().cpu().contiguous()
               for name, value in model.score_head.head.state_dict().items()},
              str(destination / 'head.safetensors'))
    artifact_spec = dict(spec, format=FORMAT,
                         artifact_scope='joint LoRA + research scalar head; not a v1 Predictor export',
                         source=str(Path(model.source).resolve()),
                         compiler=model.compiler.spec(),
                         head_parameters=sum(value.numel() for value in model.score_head.parameters()),
                         lora_parameters=sum(value.numel() for name, value in model.backbone.named_parameters()
                                             if value.requires_grad),
                         master_dtype='float32', probability_semantics='P(affirmative answer | question, passage)')
    write_json(destination / 'standalone_spec.json', artifact_spec)
    write_json(destination / 'verification_inputs.json', verification_samples)
    write_json(destination / 'reference_logits.json', reference_logits)
    identity = {str(path.relative_to(destination)): sha256(path)
                for path in destination.rglob('*') if path.is_file()}
    write_json(destination / 'files.sha256.json', identity)


def load_research(destination, *, device='cpu', source=None):
    from peft import PeftModel
    destination = Path(destination)
    spec = json.loads((destination / 'standalone_spec.json').read_text())
    if spec.get('format') != FORMAT or spec.get('head') not in HEAD_NAMES:
        raise ValueError('Unsupported standalone research artifact')
    identity = json.loads((destination / 'files.sha256.json').read_text())
    for name, digest in identity.items():
        if not (destination / name).is_file() or sha256(destination / name) != digest:
            raise ValueError(f'Research artifact checksum mismatch: {name}')
    source = source or spec['source']
    verify_base_identity(source, spec['base_file_sha256'])
    model = QwenScoreHead(source, device=device, dtype='float32', backend=spec['backend'],
                          layout='pairs', max_length=spec['compiler']['max_length'],
                          instruction=spec['compiler']['instruction'], pair_batch_size=2)
    model.backbone = PeftModel.from_pretrained(model.backbone, destination / 'lora', is_trainable=False)
    model.score_head = KeepScalarDimension(readout_module().build_head(
        spec['head'], model.backbone.config.hidden_size, spec['mlp_width'])).to(model.device)
    model.score_head.head.load_state_dict(load_file(str(destination / 'head.safetensors')), strict=True)
    if model.compiler.spec() != spec['compiler']:
        raise ValueError('Reloaded compiler differs from the trained input contract')
    return model.eval(), spec


def verify_export(destination, *, device='cpu', source=None, atol=1e-5):
    model, spec = load_research(destination, device=device, source=source)
    samples = json.loads((Path(destination) / 'verification_inputs.json').read_text())
    expected = torch.tensor(json.loads((Path(destination) / 'reference_logits.json').read_text()))
    _, predictions = evaluate_model(model, samples, precision=spec['precision'])
    actual = torch.tensor([row['logit'] for row in predictions])
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
    report = {'status': 'passed', 'samples': len(samples), 'atol': atol, 'rtol': 0,
              'maximum_absolute_difference': float((actual - expected).abs().max()),
              'head': spec['head'], 'precision': spec['precision'], 'device': device,
              'scope': 'new model/tokenizer/head objects; use --verify in a fresh process for process isolation'}
    del model
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return report


def train_head(model, samples, destination, spec, *, epochs=3, batch_size=2,
               accumulation_steps=8, lora_learning_rate=2e-5, head_learning_rate=1e-3,
               weight_decay=0.01, seed=42, precision='float32'):
    """Train only on train; dev NLL selects; held-out evaluation runs once at end."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if min(epochs, batch_size, accumulation_steps) < 1:
        raise ValueError('Training sizes must be positive')
    assert_trainable_contract(model)
    optimizer = torch.optim.AdamW([
        {'params': [value for value in model.backbone.parameters() if value.requires_grad],
         'lr': lora_learning_rate},
        {'params': list(model.score_head.parameters()), 'lr': head_learning_rate}],
        weight_decay=weight_decay)
    ordering = torch.Generator().manual_seed(seed)
    initial = parameter_fingerprints(model)
    history, optimizer_steps, best_nll, selected = [], 0, math.inf, None
    start = time.monotonic()
    device = model.device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    verification_samples = samples['dev'][:3]
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(samples['train']), generator=ordering).tolist()
        losses, sample_count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        # Normalize by the actual number of examples per accumulation group,
        # including a short final group, rather than silently shrinking its step.
        group_size = batch_size * accumulation_steps
        for group_offset in range(0, len(permutation), group_size):
            group = permutation[group_offset:group_offset + group_size]
            for offset in range(0, len(group), batch_size):
                indices = group[offset:offset + batch_size]
                batch = [samples['train'][index] for index in indices]
                with precision_context(model, precision):
                    loss = batch_loss(model, batch)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss * (len(batch) / len(group))).backward()
                losses += float(loss.detach()) * len(batch)
                sample_count += len(batch)
            norm = nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], 1.0)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite trainable parameter gradients')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if optimizer_steps % 8 == 0:
                print(json.dumps({'head': spec['head'], 'epoch': epoch, 'optimizer_steps': optimizer_steps,
                                  'elapsed_seconds': round(time.monotonic() - start, 1)}), flush=True)
        dev_metrics, _ = evaluate_model(model, samples['dev'], precision=precision, batch_size=batch_size)
        record = {'epoch': epoch, 'train_nll': losses / sample_count,
                  'dev': dev_metrics, 'optimizer_steps': optimizer_steps,
                  'epoch_order_sha256': records_hash([samples['train'][index]['id'] for index in permutation])}
        history.append(record)
        improved = dev_metrics['nll'] < best_nll
        checkpoint = destination / 'checkpoints' / f'epoch-{epoch}'
        _, references = evaluate_model(model, verification_samples, precision=precision, batch_size=batch_size)
        export_research(model, checkpoint, dict(spec, epoch=epoch, seed=seed, precision=precision),
                        verification_samples, [row['logit'] for row in references])
        if improved:
            best_nll, selected = dev_metrics['nll'], checkpoint
        write_json(destination / 'selection.json', {'criterion': 'minimum dev NLL', 'best_dev_nll': best_nll,
                                                   'checkpoint': str(selected.relative_to(destination)),
                                                   'evaluation_used_for_selection': False})
        write_json(destination / 'training_history.json', history)
        print(json.dumps({'head': spec['head'], **record, 'improved': improved}), flush=True)
    final = parameter_fingerprints(model)
    if initial['frozen_base'] != final['frozen_base']:
        raise ValueError('Frozen backbone parameters changed during LoRA training')
    if initial['head'] == final['head'] or initial['lora'] == final['lora']:
        raise ValueError('Both the head and LoRA must actually update')
    shutil.copytree(selected, destination / 'best')
    # Release the final model before constructing selected weights again.
    training_seconds = time.monotonic() - start
    memory = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    audit = {'initial_parameter_sha256': initial, 'final_parameter_sha256': final,
             'head_updated': initial['head'] != final['head'], 'lora_updated': initial['lora'] != final['lora'],
             'frozen_base_unchanged': initial['frozen_base'] == final['frozen_base'],
             'optimizer_steps': optimizer_steps, 'training_seconds': training_seconds,
             'gpu_peak_allocated_bytes': memory, 'precision': precision,
             'training_scope': 'joint LoRA + head; no feature-cache training',
             'best_epoch': json.loads((selected / 'standalone_spec.json').read_text())['epoch'],
             'best_dev_nll': best_nll, 'history': history}
    write_json(destination / 'training_audit.json', audit)
    return audit


def run(args):
    torch.set_num_threads(args.threads)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(args.device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, args.device)
    if args.verify:
        report = verify_export(args.verify, device=args.device, source=args.model,
                               atol=args.verification_atol)
        print(json.dumps(report), flush=True)
        return report
    if not args.pilot or not args.model or not args.output:
        raise ValueError('Training requires --pilot, --model and --output')
    protocol, samples = load_pilot(args.pilot)
    verify_base_identity(args.model, protocol['base']['file_sha256'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'training_protocol.json', {
        'experiment': 'joint_lora_readout_pilot', 'parent_protocol': protocol,
        'parent_protocol_sha256': sha256(Path(args.pilot) / 'protocol.json'),
        'selected_records_sha256': sha256(Path(args.pilot) / 'selected_records.json'),
        'configuration': vars(args), 'selection': 'minimum dev NLL, each epoch',
        'evaluation': 'each head best evaluated once; no held-out tuning',
        'architecture': 'research scalar heads; production v1 source unchanged'})
    results = {}
    for kind in args.heads:
        torch.manual_seed(args.seed)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        model = QwenScoreHead(args.model, dtype='float32', device=args.device, backend='sdpa', layout='pairs',
                              max_length=protocol['compiler']['max_length'],
                              instruction=protocol['compiler']['instruction'], pair_batch_size=args.batch_size)
        if model.compiler.spec() != protocol['compiler']:
            raise ValueError('Joint training input compiler must exactly match the frozen pilot')
        configure_lora_head(model, kind, mlp_width=args.mlp_width, rank=args.lora_rank,
                            alpha=args.lora_alpha, gradient_checkpointing=True)
        destination = output / kind / f'seed-{args.seed}'
        spec = {'head': kind, 'mlp_width': args.mlp_width, 'backend': 'sdpa',
                'base_file_sha256': protocol['base']['file_sha256'], 'parent_protocol': protocol,
                'lora_rank': args.lora_rank, 'lora_alpha': args.lora_alpha, 'lora_dropout': 0.0}
        audit = train_head(model, samples, destination, spec, epochs=args.epochs, batch_size=args.batch_size,
                           accumulation_steps=args.accumulation_steps, lora_learning_rate=args.lora_learning_rate,
                           head_learning_rate=args.head_learning_rate, weight_decay=args.weight_decay,
                           seed=args.seed, precision=args.precision)
        del model
        gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
        reloaded, selected_spec = load_research(destination / 'best', device=args.device)
        references = json.loads((destination / 'best' / 'verification_inputs.json').read_text())
        expected = torch.tensor(json.loads((destination / 'best' / 'reference_logits.json').read_text()))
        _, actual_rows = evaluate_model(reloaded, references, precision=args.precision, batch_size=args.batch_size)
        actual = torch.tensor([row['logit'] for row in actual_rows])
        torch.testing.assert_close(actual, expected, atol=args.verification_atol, rtol=0)
        metrics, predictions = evaluate_model(reloaded, samples['evaluation'], precision=args.precision,
                                              batch_size=args.batch_size)
        write_json(destination / 'evaluation.json', {'scope': protocol['evaluation'], 'metrics': metrics,
                                                    'predictions': predictions})
        write_json(destination / 'reload.json', {'status': 'passed', 'samples': len(references),
                                                'maximum_absolute_difference': float((actual - expected).abs().max()),
                                                'process': 'new objects in training process; separate --verify recommended'})
        results[kind] = {'metrics': metrics, 'head_parameters': selected_spec['head_parameters'],
                         'lora_parameters': selected_spec['lora_parameters'], **audit}
        write_json(output / 'results.json', results)
        print(json.dumps({'head': kind, 'status': 'completed', 'evaluation': metrics}), flush=True)
        del reloaded
        gc.collect()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot')
    parser.add_argument('--model')
    parser.add_argument('--output')
    parser.add_argument('--verify')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['float32', 'bf16'], default='bf16')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.20)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--heads', nargs='+', choices=HEAD_NAMES, default=['mlp', 'swiglu', 'residual_swiglu'])
    parser.add_argument('--mlp-width', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--lora-learning-rate', type=float, default=2e-5)
    parser.add_argument('--head-learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--verification-atol', type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.threads, args.epochs, args.batch_size, args.accumulation_steps,
           args.mlp_width, args.lora_rank, args.lora_alpha) < 1:
        parser.error('Training sizes must be positive')
    if len(set(args.heads)) != len(args.heads):
        parser.error('Heads must be unique')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('GPU memory fraction must be in (0,1]')
    if min(args.lora_learning_rate, args.head_learning_rate, args.verification_atol) <= 0 or args.weight_decay < 0:
        parser.error('Invalid optimizer or verification values')
    if not args.verify and args.precision == 'bf16' and torch.device(args.device).type != 'cuda':
        parser.error('Use --precision float32 for CPU training')
    run(args)


if __name__ == '__main__':
    main()
