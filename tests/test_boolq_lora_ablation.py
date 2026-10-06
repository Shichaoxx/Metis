"""Offline contracts for joint research LoRA/readout training, not quality tests."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')
pytest.importorskip('safetensors')

from metis.model import QwenScoreHead
from metis.schema import sha256


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[1]
research = import_file('boolq_lora_research', ROOT / 'examples/research/train_boolq_lora_ablation.py')
fixture = import_file('boolq_lora_fixture', ROOT / 'examples/smoke_train.py')


def samples():
    return {
        'train': [research.to_sample({'id': 't1', 'question': 'apple fruit',
                                     'passage': 'red apple fruit', 'label': 1}),
                  research.to_sample({'id': 't2', 'question': 'apple fruit',
                                      'passage': 'blue sky water', 'label': 0}),
                  research.to_sample({'id': 't3', 'question': 'cold water',
                                      'passage': 'cold blue water', 'label': 1})],
        'dev': [research.to_sample({'id': 'd1', 'question': 'orange fruit',
                                   'passage': 'green orange fruit', 'label': 1}),
                research.to_sample({'id': 'd2', 'question': 'warm water',
                                    'passage': 'cold sky', 'label': 0})],
        'evaluation': [research.to_sample({'id': 'e1', 'question': 'blue sky',
                                          'passage': 'blue green sky', 'label': 1})]}


def model_and_spec(tmp_path, kind='mlp', checkpointing=True):
    torch.set_num_threads(1)
    lm, tokenizer = fixture.tiny_components()
    base = tmp_path / 'base'
    base.mkdir()
    lm.save_pretrained(base)
    tokenizer.save_pretrained(base)
    model = QwenScoreHead.from_components(lm.model, tokenizer, source=str(base),
                                          max_length=128, backend='eager',
                                          instruction='Judge whether the affirmative answer is supported')
    research.configure_lora_head(model, kind, mlp_width=8, rank=2, alpha=4,
                                 gradient_checkpointing=checkpointing)
    spec = {'head': kind, 'mlp_width': 8, 'backend': 'eager',
            'base_file_sha256': {path.name: sha256(path) for path in base.iterdir()
                                if path.name == 'config.json' or path.suffix == '.safetensors'},
            'lora_rank': 2, 'lora_alpha': 4, 'lora_dropout': 0.0}
    return model, spec


@pytest.mark.parametrize('kind', research.HEAD_NAMES)
def test_all_heads_have_finite_gradients_and_update_only_lora_and_readout(tmp_path, kind):
    model, _ = model_and_spec(tmp_path, kind)
    before = research.parameter_fingerprints(model)
    model.train()
    loss = research.batch_loss(model, samples()['train'][:2])
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.backbone.named_parameters() if 'lora_' in name)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.score_head.parameters())
    for name, parameter in model.backbone.named_parameters():
        if 'lora_' not in name:
            assert not parameter.requires_grad and parameter.grad is None
        elif parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    optimizer.step()
    after = research.parameter_fingerprints(model)
    assert before['frozen_base'] == after['frozen_base']
    assert before['lora'] != after['lora'] and before['head'] != after['head']
    # Regression: a one-candidate batch must remain a vector, not squeeze to 0D.
    assert model.score_tensors(samples()['train'][:1])[0].shape == (1,)


def test_joint_training_selects_exports_and_reloads_a_trained_artifact(tmp_path):
    model, spec = model_and_spec(tmp_path)
    data = samples()
    destination = tmp_path / 'run'
    audit = research.train_head(model, data, destination, spec, epochs=2, batch_size=2,
                                accumulation_steps=2, lora_learning_rate=0.01,
                                head_learning_rate=0.01, precision='float32')
    assert audit['optimizer_steps'] == 2  # One actual 3-example group per epoch.
    assert audit['head_updated'] and audit['lora_updated'] and audit['frozen_base_unchanged']
    selection = json.loads((destination / 'selection.json').read_text())
    assert not selection['evaluation_used_for_selection']
    assert audit['best_dev_nll'] == min(row['dev']['nll'] for row in audit['history'])
    report = research.verify_export(destination / 'best')
    assert report['status'] == 'passed' and report['maximum_absolute_difference'] <= 1e-5
    reloaded, saved = research.load_research(destination / 'best')
    assert saved['format'] == research.FORMAT
    assert not (destination / 'best' / 'model_spec.json').exists()
    metrics, rows = research.evaluate_model(reloaded, data['evaluation'])
    assert metrics['count'] == 1 and len(rows) == 1
    with pytest.raises(FileExistsError):
        research.train_head(model, data, destination, spec, epochs=1)
    # Integrity validation happens before silently constructing random weights.
    (destination / 'best' / 'head.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        research.load_research(destination / 'best')


def write_pilot(root, records):
    root.mkdir()
    protocol = {'experiment': 'frozen_base_readout_pilot',
                'compiler': {'version': 'qwen-score-head-v1'},
                'splits': {name: {'count': len(items), 'ids': [item['id'] for item in items],
                                  'records_sha256': research.records_hash(items)}
                           for name, items in records.items()}}
    (root / 'protocol.json').write_text(json.dumps(protocol))
    (root / 'selected_records.json').write_text(json.dumps(records))


def test_pilot_records_reject_identity_changes_and_passage_leakage(tmp_path):
    records = {name: [{'id': name, 'question': 'apple', 'passage': name + ' fruit', 'label': 1}]
               for name in ('train', 'dev', 'evaluation')}
    write_pilot(tmp_path / 'valid', records)
    _, loaded = research.load_pilot(tmp_path / 'valid')
    assert loaded['train'][0]['input']['context'] == ''
    changed = copy.deepcopy(records)
    changed['train'][0]['label'] = 0
    (tmp_path / 'valid' / 'selected_records.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='frozen identity'):
        research.load_pilot(tmp_path / 'valid')
    changed['dev'][0]['passage'] = changed['train'][0]['passage']
    write_pilot(tmp_path / 'leaking', changed)
    with pytest.raises(ValueError, match='passage groups overlap'):
        research.load_pilot(tmp_path / 'leaking')


def test_supervision_never_changes_compiled_token_inputs(tmp_path):
    model, _ = model_and_spec(tmp_path, checkpointing=False)
    sample = samples()['train'][0]
    changed = copy.deepcopy(sample)
    changed['supervision']['labels']['passage'] = 0
    assert model.compiler.compile(sample).pairs == model.compiler.compile(changed).pairs


def test_frozen_backbone_identity_rejects_missing_or_changed_weights(tmp_path):
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'reference')
    research.verify_base_identity(tmp_path, {weights.name: sha256(weights)})
    with pytest.raises(ValueError, match='differs'):
        research.verify_base_identity(tmp_path, {weights.name: '0' * 64})
    with pytest.raises(ValueError, match='identify'):
        research.verify_base_identity(tmp_path, {})
