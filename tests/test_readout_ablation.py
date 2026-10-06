"""Offline behavior of the independent frozen-feature research experiment."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
safetensors = pytest.importorskip('safetensors.torch')

spec = importlib.util.spec_from_file_location(
    'readout_ablation', Path(__file__).parents[1] / 'examples/research/train_readout_ablation.py')
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def features():
    generator = torch.Generator().manual_seed(917)
    x = torch.randn(96, 8, generator=generator)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).float()
    return x, y


def test_parameter_matching_and_real_gradient_training():
    torch.set_num_threads(1)
    for dimension in (8, 1024):
        for head_spec in ablation.head_specs(dimension):
            head = ablation.ScalarHead(head_spec['kind'], dimension, head_spec['width'])
            assert sum(p.numel() for p in head.parameters()) == head_spec['parameters']
            if head_spec['kind'] != 'linear':
                assert abs(head_spec['budget_difference_fraction']) < 0.01
    x, y = features()
    head_spec = ablation.head_specs(8)[1]
    trained, evidence = ablation.fit_head(head_spec, x[:64], y[:64], x[64:], y[64:],
                                         seed=42, epochs=15, batch_size=16, learning_rate=0.01)
    assert evidence['optimizer_steps'] == 60
    assert evidence['initial_state_sha256'] != evidence['final_trained_state_sha256']
    assert min(epoch['train_nll'] for epoch in evidence['epochs']) < evidence['initial_train_nll'] * 0.5
    assert ablation.binary_metrics(trained(x[64:]), y[64:])['accuracy'] > 0.8


@pytest.mark.parametrize('kind', ablation.HEAD_NAMES)
def test_standalone_reload_restores_all_trained_parameters(tmp_path, kind):
    torch.set_num_threads(1)
    x, y = features()
    head_spec = next(s for s in ablation.head_specs(8) if s['kind'] == kind)
    head, _ = ablation.fit_head(head_spec, x[:64], y[:64], x[64:], y[64:],
                                seed=43, epochs=2, batch_size=16)
    ablation.save_head(head, head_spec, tmp_path)
    loaded = ablation.load_head(tmp_path)
    assert torch.equal(head(x), loaded(x))
    assert ablation.state_hash(head) == ablation.state_hash(loaded)
    assert 'not a Predictor export' in json.loads((tmp_path / 'head_spec.json').read_text())['artifact_scope']


def test_metrics_and_bootstrap_preserve_pairing():
    # Four balanced examples; confidence is .8 and all classifications correct.
    labels = torch.tensor([0., 1., 0., 1.])
    p = torch.tensor([0.2, 0.8, 0.2, 0.8], dtype=torch.float64)
    result = ablation.binary_metrics(torch.logit(p), labels)
    assert result['accuracy'] == result['balanced_accuracy'] == 1
    assert result['nll'] == pytest.approx(-torch.log(torch.tensor(0.8)).item())
    assert result['brier'] == pytest.approx(0.04)
    assert result['ece'] == pytest.approx(0.2)
    paired = torch.stack([p, 1 - p])
    identical = ablation.paired_accuracy_bootstrap(paired, paired, labels, samples=50)
    assert identical['accuracy_delta'] == 0
    assert identical['percentile_95_interval'] == [0., 0.]
    better = ablation.paired_accuracy_bootstrap(torch.stack([p, p]),
                                               torch.stack([1 - p, 1 - p]), labels, samples=50)
    assert better['accuracy_delta'] == 1
    assert better['percentile_95_interval'] == [1., 1.]


def test_selection_uses_dev_nll_and_is_independent_of_evaluation(tmp_path):
    torch.set_num_threads(1)
    x, y = features()
    cache = {'train_features': x[:64], 'train_labels': y[:64],
             'dev_features': x[64:80], 'dev_labels': y[64:80],
             'evaluation_features': x[80:], 'evaluation_labels': y[80:]}
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'dataset': 'synthetic offline fixture', 'backbone': 'fixed features'}))
    first = tmp_path / 'first.safetensors'
    safetensors.save_file(cache, str(first))
    ablation.run(first, protocol, tmp_path / 'first-run', seeds=[42], epochs=4, bootstrap_samples=10)
    # Reverse evaluation labels: selection, weights and training evidence must stay identical.
    cache['evaluation_labels'] = 1 - cache['evaluation_labels']
    second = tmp_path / 'second.safetensors'
    safetensors.save_file(cache, str(second))
    ablation.run(second, protocol, tmp_path / 'second-run', seeds=[42], epochs=4, bootstrap_samples=10)
    for kind in ablation.HEAD_NAMES:
        left = tmp_path / 'first-run' / f'{kind}-seed-42'
        right = tmp_path / 'second-run' / f'{kind}-seed-42'
        a = json.loads((left / 'result.json').read_text())
        b = json.loads((right / 'result.json').read_text())
        assert a['training'] == b['training']
        assert a['training']['best_epoch'] == min(a['training']['epochs'], key=lambda e: e['dev_nll'])['epoch']
        assert ablation.state_hash(ablation.load_head(left)) == ablation.state_hash(ablation.load_head(right))
        assert a['evaluation']['accuracy'] + b['evaluation']['accuracy'] == pytest.approx(1)
    with pytest.raises(ValueError, match='new or empty'):
        ablation.run(first, protocol, tmp_path / 'first-run', seeds=[42], epochs=1)


def test_cache_rejects_shape_and_label_corruption():
    x, y = features()
    cache = {f'{split}_{name}': value for split in ('train', 'dev', 'evaluation')
             for name, value in (('features', x), ('labels', y))}
    cache['dev_labels'] = y[:-1]
    with pytest.raises(ValueError, match='shapes'):
        ablation.validate_cache(cache)
    cache['dev_labels'] = y.clone()
    cache['dev_labels'][0] = 2
    with pytest.raises(ValueError, match='binary'):
        ablation.validate_cache(cache)
