"""Research training/selection/export contracts, independent of task quality."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')
pytest.importorskip('transformers')

from metis.schema import sha256

ROOT = Path(__file__).parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'examples' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


research = load_script('readout_training', 'research/train_boolq_readout.py')
fixture = load_script('readout_tiny_fixture', 'smoke_train.py')


def tiny_protocol(tmp_path):
    torch.set_num_threads(1)
    base = tmp_path / 'base'
    backbone, tokenizer = fixture.tiny_components()
    backbone.save_pretrained(base)
    tokenizer.save_pretrained(base)
    root = tmp_path / 'pilot'
    root.mkdir()
    records = {
        'train': [{'id': f't{i}', 'question': 'apple fruit',
                   'passage': 'red apple fruit' if i % 2 else 'blue sky water', 'label': i % 2}
                  for i in range(4)],
        'dev': [{'id': f'd{i}', 'question': 'orange fruit',
                 'passage': 'orange fruit green' if i else 'cold sky', 'label': i} for i in range(2)]}
    research.write_json(root / 'train_dev_records.json', records)
    research.write_json(root / 'sealed_evaluation.json', [{'id': 'e1', 'label': 1}])
    protocol = {'experiment': 'boolq_readout_round2', 'readouts': list(research.READOUTS), 'seeds': [7],
                'base': {'file_sha256': {p.name: sha256(p) for p in base.iterdir()
                                        if p.suffix in ('.json', '.safetensors')}},
                'instruction': 'Judge whether the affirmative answer is supported',
                'configuration': {'max_length': 128, 'mlp_width': 256, 'lora_rank': 2, 'lora_alpha': 4,
                                  'batch_size': 2, 'effective_batch_size': 2, 'epochs': 2,
                                  'lora_learning_rate': 0.01, 'head_learning_rate': 0.01,
                                  'weight_decay': 0.01, 'warmup_fraction': 0.1, 'gradient_clip': 1},
                'train_positive_prior': 0.5, 'evaluation': 'synthetic software fixture only',
                'record_files': {name: sha256(root / name) for name in
                                 ['train_dev_records.json', 'sealed_evaluation.json']}}
    research.write_json(root / 'protocol.json', protocol)
    samples = research.load_training_samples(root, protocol)
    return base, root, protocol, samples


def test_training_records_load_without_opening_sealed_evaluation(tmp_path):
    _, root, protocol, _ = tiny_protocol(tmp_path)
    (root / 'sealed_evaluation.json').unlink()
    assert research.load_protocol(root) == protocol
    assert len(research.load_training_samples(root, protocol)['train']) == 4
    (root / 'train_dev_records.json').write_text('{}')
    with pytest.raises(ValueError, match='fixed identity'):
        research.load_training_samples(root, protocol)


def test_all_readouts_train_select_reload_and_freeze_before_evaluation(tmp_path):
    base, root, protocol, samples = tiny_protocol(tmp_path)
    initial = []
    with pytest.raises(FileNotFoundError):
        research.finalize(root)
    for readout in research.READOUTS:
        model = research.make_model(base, protocol, readout, 7, 'cpu', backend='eager')
        initial.append(research.fingerprints(model))
        destination = root / 'models' / readout / 'seed-7'
        spec = {'source': str(base), 'protocol': protocol, 'readout': readout,
                'seed': 7, 'backend': 'eager', 'precision': 'float32',
                'protocol_sha256': sha256(root / 'protocol.json')}
        audit = research.train_one(model, samples, destination, spec, protocol['configuration'])
        assert audit['steps'] == 4 and not audit['evaluation_opened']
        assert audit['head_updated'] and audit['lora_updated'] and audit['frozen_base_unchanged']
        assert audit['best_dev_nll'] == min(row['dev']['nll'] for row in audit['history'])
        assert audit['pool_updated'] == (readout == 'candidate_attention')
        reloaded, saved = research.load_export(destination / 'best')
        assert research.verify_loaded(reloaded, destination / 'best', saved)['status'] == 'passed'
        assert not (destination / 'best' / 'model_spec.json').exists()
    assert len({item['head'] for item in initial}) == 1
    assert len({item['lora'] for item in initial}) == 1
    lock = research.finalize(root)
    assert len(lock['runs']) == 3 and not lock['evaluation_opened']
    assert research.assert_locked(root, protocol) == lock
    with pytest.raises(FileExistsError):
        research.finalize(root)
    selection = root / 'models' / 'last_relevance' / 'seed-7' / 'selection.json'
    selection.write_text('{}')
    with pytest.raises(ValueError, match='changed after freeze'):
        research.assert_locked(root, protocol)


def test_cosine_schedule_warms_up_and_ends_at_zero():
    scales = [research.learning_rate_scale(i, 10, 0.2) for i in range(1, 11)]
    assert scales[:2] == [0.5, 1]
    assert scales[-1] == 0 and all(0 <= value <= 1 for value in scales)
    assert scales[2:] == sorted(scales[2:], reverse=True)
