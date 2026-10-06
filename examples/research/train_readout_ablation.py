"""Compare small scalar heads on a fixed, frozen backbone feature cache.

This is an exploratory binary-classification experiment, not an end-to-end
backbone/LoRA comparison or a Metis Predictor export. Every head sees identical
features and sample orders. Only development NLL selects a checkpoint; the
evaluation split is read after training and selection are complete.

Required safetensors keys: {train,dev,evaluation}_{features,labels}. Features
have shape [N, D], labels [N] contain 0/1. The accompanying protocol.json must
describe the dataset, split selection, backbone, compiler, and readout used to
extract these features. Example:

    python examples/research/train_readout_ablation.py \
        --features /path/to/features.safetensors \
        --protocol /path/to/protocol.json --output /path/to/new-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


HEAD_NAMES = ('linear', 'mlp', 'swiglu', 'residual_swiglu')
METRIC_NAMES = ('accuracy', 'balanced_accuracy', 'nll', 'brier', 'ece')


class ScalarHead(nn.Module):
    """Unnormalized scalar logits; biases match the v1 MLP convention."""

    def __init__(self, kind: str, dimension: int, width: int | None = None):
        super().__init__()
        if kind not in HEAD_NAMES or dimension < 1:
            raise ValueError('Unknown head or invalid feature dimension')
        if kind != 'linear' and (width is None or width < 1):
            raise ValueError('Nonlinear heads require a positive width')
        self.kind, self.dimension, self.width = kind, dimension, width
        if kind == 'linear':
            self.skip = nn.Linear(dimension, 1)
        else:
            self.value = nn.Linear(dimension, width)
            self.output = nn.Linear(width, 1)
            if kind in ('swiglu', 'residual_swiglu'):
                self.gate = nn.Linear(dimension, width)
            if kind == 'residual_swiglu':
                self.skip = nn.Linear(dimension, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.kind == 'linear':
            return self.skip(features).squeeze(-1)
        hidden = self.value(features)
        if self.kind == 'mlp':
            hidden = F.gelu(hidden)
        else:
            hidden = hidden * F.silu(self.gate(features))
        scores = self.output(hidden)
        if self.kind == 'residual_swiglu':
            scores = scores + self.skip(features)
        return scores.squeeze(-1)


def parameter_count(kind: str, dimension: int, width: int | None = None) -> int:
    if kind == 'linear':
        return dimension + 1
    if width is None or width < 1:
        raise ValueError('Nonlinear heads require a positive width')
    if kind == 'mlp':
        return width * (dimension + 2) + 1
    if kind in ('swiglu', 'residual_swiglu'):
        return width * (2 * dimension + 3) + 1 + (
            dimension + 1 if kind == 'residual_swiglu' else 0)
    raise ValueError('Unknown head')


def head_specs(dimension: int, mlp_width: int = 256) -> list[dict]:
    """Nearest integer widths minimize the difference from the MLP budget."""
    if dimension < 1 or mlp_width < 1:
        raise ValueError('Feature dimension and MLP width must be positive')
    budget = parameter_count('mlp', dimension, mlp_width)
    specs = []
    for kind in HEAD_NAMES:
        if kind == 'linear':
            width = None
        elif kind == 'mlp':
            width = mlp_width
        else:
            offset = 1 + (dimension + 1 if kind == 'residual_swiglu' else 0)
            real_width = (budget - offset) / (2 * dimension + 3)
            candidates = {max(1, math.floor(real_width)), max(1, math.ceil(real_width))}
            width = min(candidates, key=lambda w: (abs(parameter_count(kind, dimension, w) - budget), w))
        count = parameter_count(kind, dimension, width)
        specs.append({'kind': kind, 'dimension': dimension, 'width': width,
                      'parameters': count, 'mlp_budget': budget,
                      'budget_difference_fraction': (count - budget) / budget})
    return specs


def build_head(kind: str, input_size: int, hidden_size: int = 256) -> ScalarHead:
    """Build a head, matching nonlinear widths to an MLP hidden-size budget."""
    spec = next((value for value in head_specs(input_size, hidden_size) if value['kind'] == kind), None)
    if spec is None:
        raise ValueError('Unknown head')
    return ScalarHead(kind, input_size, spec['width'])


def state_hash(head: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> dict:
    """Binary Brier has one probability term; ECE bins predicted confidence."""
    logits, labels = logits.detach().double(), labels.detach().double()
    if logits.ndim != 1 or labels.shape != logits.shape or len(labels) == 0 or bins < 1:
        raise ValueError('Metrics require nonempty equal one-dimensional tensors')
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError('Metric inputs must be finite')
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Labels must be binary')
    probabilities = logits.sigmoid()
    predicted = probabilities >= 0.5
    correct = predicted == labels.bool()
    recalls = [correct[labels == target].double().mean().item()
               for target in (0, 1) if (labels == target).any()]
    confidence = torch.maximum(probabilities, 1 - probabilities)
    bin_ids = torch.floor(confidence * bins).long().clamp(max=bins - 1)
    ece = 0.0
    for index in range(bins):
        members = bin_ids == index
        if members.any():
            ece += members.double().mean().item() * abs(
                correct[members].double().mean().item() - confidence[members].mean().item())
    return {'accuracy': correct.double().mean().item(),
            'balanced_accuracy': statistics.mean(recalls),
            'nll': F.binary_cross_entropy_with_logits(logits, labels).item(),
            'brier': ((probabilities - labels) ** 2).mean().item(), 'ece': ece,
            'count': len(labels), 'positive_count': int(labels.sum().item())}


def fit_head(spec: dict, train_features: torch.Tensor, train_labels: torch.Tensor,
             dev_features: torch.Tensor, dev_labels: torch.Tensor, *, seed: int,
             epochs: int = 30, batch_size: int = 32, learning_rate: float = 1e-3,
             weight_decay: float = 0.01) -> tuple[ScalarHead, dict]:
    """Select by development NLL. Evaluation data is deliberately not an input."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError('Invalid training configuration')
    torch.manual_seed(seed)
    head = ScalarHead(spec['kind'], spec['dimension'], spec['width']).float()
    initial_hash = state_hash(head)
    with torch.no_grad():
        initial_train_nll = F.binary_cross_entropy_with_logits(head(train_features), train_labels).item()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # This generator is independent of parameter initialization and head width.
    ordering = torch.Generator().manual_seed(seed)
    best_nll, best_epoch, best_state = math.inf, None, None
    history, steps = [], 0
    for epoch in range(1, epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_labels), generator=ordering)
        for offset in range(0, len(permutation), batch_size):
            batch = permutation[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(train_features[batch]), train_labels[batch])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            steps += 1
        head.eval()
        with torch.no_grad():
            train_nll = F.binary_cross_entropy_with_logits(head(train_features), train_labels).item()
            dev_nll = F.binary_cross_entropy_with_logits(head(dev_features), dev_labels).item()
        if not math.isfinite(dev_nll) or not math.isfinite(train_nll):
            raise ValueError('Nonfinite checkpoint loss')
        history.append({'epoch': epoch, 'train_nll': train_nll, 'dev_nll': dev_nll})
        if dev_nll < best_nll:
            best_nll, best_epoch = dev_nll, epoch
            best_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
    final_hash = state_hash(head)
    head.load_state_dict(best_state)
    return head, {'seed': seed, 'initial_state_sha256': initial_hash,
                  'final_trained_state_sha256': final_hash,
                  'selected_state_sha256': state_hash(head), 'optimizer_steps': steps,
                  'best_epoch': best_epoch, 'best_dev_nll': best_nll,
                  'initial_train_nll': initial_train_nll, 'epochs': history,
                  'sample_order': 'independent torch.Generator seeded per head/run; same seed gives same epoch orders'}


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_head(head: ScalarHead, spec: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'head_spec.json', {
        **spec, 'format': 'metis-research-frozen-readout-v1', 'normalization': 'none',
        'artifact_scope': 'standalone scalar head; requires matching frozen features; not a Predictor export'})
    save_file({key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
              str(destination / 'head.safetensors'))


def load_head(destination: Path) -> ScalarHead:
    spec = json.loads((destination / 'head_spec.json').read_text())
    if spec.get('format') != 'metis-research-frozen-readout-v1' or spec.get('normalization') != 'none':
        raise ValueError('Unsupported frozen readout artifact')
    head = ScalarHead(spec['kind'], spec['dimension'], spec['width']).float()
    head.load_state_dict(load_file(str(destination / 'head.safetensors'), device='cpu'), strict=True)
    return head.eval()


def head_latency(head: ScalarHead, features: torch.Tensor, warmups: int = 10,
                 repeats: int = 100) -> dict:
    """Warm CPU readout latency, excluding backbone, feature I/O and tokenization."""
    head.eval()
    with torch.inference_mode():
        for _ in range(warmups):
            head(features)
        values = []
        for _ in range(repeats):
            start = time.perf_counter()
            head(features)
            values.append((time.perf_counter() - start) * 1000)
    values.sort()
    return {'scope': 'warm CPU head only; not end-to-end inference', 'batch_size': len(features),
            'threads': torch.get_num_threads(), 'warmups': warmups, 'repeats': repeats,
            'median_ms': statistics.median(values),
            'p95_ms': values[math.ceil(0.95 * repeats) - 1]}


def paired_accuracy_bootstrap(candidate_probabilities: torch.Tensor,
                              reference_probabilities: torch.Tensor,
                              labels: torch.Tensor, *, samples: int = 2000,
                              seed: int = 20261006) -> dict:
    """Resample evaluation examples, averaging paired correctness over fixed seeds.

    This conditional, exploratory interval does not estimate variability from
    training new seeds, changing feature extraction, or selecting architectures.
    """
    if candidate_probabilities.ndim != 2 or candidate_probabilities.shape != reference_probabilities.shape:
        raise ValueError('Paired predictions must have shape [seeds, examples]')
    if candidate_probabilities.shape[1] != len(labels) or len(labels) == 0 or samples < 1:
        raise ValueError('Invalid paired bootstrap inputs')
    candidate = (candidate_probabilities >= 0.5) == labels.bool().unsqueeze(0)
    reference = (reference_probabilities >= 0.5) == labels.bool().unsqueeze(0)
    delta = (candidate.double() - reference.double()).mean(dim=0)
    generator = torch.Generator().manual_seed(seed)
    estimates = torch.empty(samples, dtype=torch.float64)
    for offset in range(0, samples, 128):
        size = min(128, samples - offset)
        indices = torch.randint(len(labels), (size, len(labels)), generator=generator)
        estimates[offset:offset + size] = delta[indices].mean(dim=1)
    lower, upper = torch.quantile(estimates, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {'accuracy_delta': delta.mean().item(), 'percentile_95_interval': [lower, upper],
            'samples': samples, 'bootstrap_seed': seed,
            'unit': 'evaluation example, paired across heads; correctness averaged over fixed training seeds',
            'interpretation': 'exploratory and conditional on this cache and fixed seeds; no multiple-comparison adjustment'}


def validate_cache(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    validated, dimension = {}, None
    for split in ('train', 'dev', 'evaluation'):
        x_key, y_key = f'{split}_features', f'{split}_labels'
        if x_key not in tensors or y_key not in tensors:
            raise ValueError(f'Missing cache keys {x_key}/{y_key}')
        features, labels = tensors[x_key].float(), tensors[y_key].float()
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels) or len(labels) == 0:
            raise ValueError(f'Invalid {split} feature/label shapes')
        if features.shape[1] < 1 or (dimension is not None and features.shape[1] != dimension):
            raise ValueError('All splits require the same positive feature dimension')
        if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
            raise ValueError(f'Nonfinite {split} cache values')
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError(f'{split} labels must be binary')
        dimension = features.shape[1]
        validated[x_key], validated[y_key] = features.contiguous(), labels.contiguous()
    return validated


def summarize(records: list[dict], predictions: dict[str, list[torch.Tensor]],
              labels: torch.Tensor, bootstrap_samples: int) -> dict:
    by_head = {}
    for kind in HEAD_NAMES:
        runs = [record for record in records if record['head']['kind'] == kind]
        by_head[kind] = {
            'parameters': runs[0]['head']['parameters'], 'seeds': [r['training']['seed'] for r in runs],
            'metrics': {metric: {'mean': statistics.mean(r['evaluation'][metric] for r in runs),
                                 'sample_std': statistics.stdev(r['evaluation'][metric] for r in runs)
                                 if len(runs) > 1 else None} for metric in METRIC_NAMES}}
    reference = {r['training']['seed']: r for r in records if r['head']['kind'] == 'mlp'}
    deltas = []
    for record in records:
        seed = record['training']['seed']
        deltas.append({'kind': record['head']['kind'], 'seed': seed,
                       'candidate_minus_mlp': {metric: record['evaluation'][metric] -
                                              reference[seed]['evaluation'][metric]
                                              for metric in METRIC_NAMES}})
    bootstrap = {kind: paired_accuracy_bootstrap(torch.stack(predictions[kind]),
                                                torch.stack(predictions['mlp']), labels,
                                                samples=bootstrap_samples)
                 for kind in HEAD_NAMES if kind != 'mlp'}
    return {'by_head': by_head, 'paired_seed_deltas': deltas, 'paired_accuracy_bootstrap': bootstrap}


def run(features_path: Path, protocol_path: Path, destination: Path, *,
        seeds: list[int] | tuple[int, ...] = (42, 43, 44), epochs: int = 30,
        batch_size: int = 32, learning_rate: float = 1e-3, weight_decay: float = 0.01,
        mlp_width: int = 256, bootstrap_samples: int = 2000) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or bootstrap_samples < 1:
        raise ValueError('Use distinct seeds and a positive bootstrap sample count')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Output directory must be new or empty')
    protocol = json.loads(protocol_path.read_text())
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError('Extraction protocol must be a nonempty JSON object')
    cache_sha256 = file_hash(features_path)
    if protocol.get('feature_sha256') and protocol['feature_sha256'] != cache_sha256:
        raise ValueError('Feature cache SHA256 does not match its extraction protocol')
    tensors = validate_cache(load_file(str(features_path), device='cpu'))
    specs = head_specs(tensors['train_features'].shape[1], mlp_width)
    for spec in specs:
        if spec['kind'] != 'linear' and abs(spec['budget_difference_fraction']) > 0.01:
            raise ValueError('Integer widths cannot match the MLP parameter budget within 1%; increase --mlp-width')
    destination.mkdir(parents=True, exist_ok=True)
    identity = {'feature_cache_sha256': cache_sha256, 'protocol_sha256': file_hash(protocol_path)}
    configuration = {'seeds': list(seeds), 'epochs': epochs, 'batch_size': batch_size,
                     'learning_rate': learning_rate, 'weight_decay': weight_decay,
                     'optimizer': 'AdamW', 'loss': 'binary_cross_entropy_with_logits',
                     'precision': 'CPU FP32', 'feature_normalization': 'none',
                     'selection': 'minimum development NLL after an epoch; first checkpoint wins exact ties',
                     'evaluation': 'one evaluation per selected head/seed, after all training and selection',
                     'ece': '10 equal-width bins on predicted-class confidence; binary Brier is (p-y)^2',
                     'torch_version': torch.__version__, 'cpu_threads': torch.get_num_threads()}
    write_json(destination / 'protocol.json', {'experiment': configuration, 'feature_identity': identity,
                                             'extraction_protocol': protocol, 'heads': specs})
    selected = []
    for seed in seeds:
        for spec in specs:
            head, training = fit_head(spec, tensors['train_features'], tensors['train_labels'],
                                     tensors['dev_features'], tensors['dev_labels'], seed=seed,
                                     epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                     weight_decay=weight_decay)
            output = destination / f"{spec['kind']}-seed-{seed}"
            save_head(head, {**spec, **identity, 'seed': seed, 'best_epoch': training['best_epoch']}, output)
            reloaded = load_head(output)
            with torch.no_grad():
                before, after = head(tensors['train_features']), reloaded(tensors['train_features'])
            if not torch.equal(before, after) or state_hash(head) != state_hash(reloaded):
                raise AssertionError('Freshly loaded head changed scores or weights')
            selected.append((reloaded, spec, training, output))
    # Evaluation is inaccessible to checkpoint selection and begins only now.
    labels = tensors['evaluation_labels']
    records, predictions = [], {kind: [] for kind in HEAD_NAMES}
    for head, spec, training, output in selected:
        with torch.inference_mode():
            logits = head(tensors['evaluation_features'])
            probabilities = logits.sigmoid()
        evaluation = binary_metrics(logits, labels)
        record = {'head': spec, 'training': training, 'evaluation': evaluation,
                  'reload': {'weights_equal': True, 'train_scores_equal': True},
                  'head_latency': head_latency(head, tensors['train_features'][:batch_size])}
        write_json(output / 'result.json', record)
        write_json(output / 'evaluation_predictions.json', [
            {'row': row, 'label': int(label), 'probability': float(probability), 'logit': float(logit)}
            for row, (label, probability, logit) in enumerate(zip(labels, probabilities, logits))])
        records.append(record)
        predictions[spec['kind']].append(probabilities)
    prior = float(tensors['train_labels'].mean())
    # Keep the baseline finite even when a tiny training split contains one class.
    prior_logit = math.log(max(prior, 1e-7) / max(1 - prior, 1e-7))
    summary = {'scope': 'exploratory frozen-backbone readout ablation; not end-to-end tuning',
               'feature_identity': identity, 'configuration': configuration,
               'split_sizes': {split: len(tensors[f'{split}_labels'])
                               for split in ('train', 'dev', 'evaluation')},
               'train_prior_baseline': {'positive_probability': prior,
                                        'evaluation': binary_metrics(torch.full_like(labels, prior_logit), labels)},
               **summarize(records, predictions, labels, bootstrap_samples)}
    write_json(destination / 'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--mlp-width', type=int, default=256)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    summary = run(args.features, args.protocol, args.output, seeds=args.seeds,
                  epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                  weight_decay=args.weight_decay, mlp_width=args.mlp_width,
                  bootstrap_samples=args.bootstrap_samples)
    print(json.dumps({'summary': str(args.output / 'summary.json'), 'by_head': summary['by_head']}, indent=2))


if __name__ == '__main__':
    main()
