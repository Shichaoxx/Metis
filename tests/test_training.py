import copy
import json
import importlib.util
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('accelerate')
from cometa.training import (train, supervised_loss, _sample_training_candidates,
                             _selection_data, _complete_validation_metrics)
from cometa.model import QwenReranker

fixture_spec = importlib.util.spec_from_file_location('training_fixture', Path(__file__).parents[1] / 'examples/smoke_train.py')
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)


def test_graded_labels_and_unknowns():
    sample = fixture.sample_records()[0]
    sample['candidates'].append({'id': 'unknown', 'text': 'water'})
    sample['supervision']['labels'] = {'d1': 2, 'd2': 1}
    scores = torch.tensor([0.2, -0.3, 100.0], requires_grad=True)
    loss = supervised_loss(scores, sample)
    loss.backward()
    assert scores.grad[2] == 0
    assert torch.isfinite(loss)


def test_multilabel_and_single_choice_objectives():
    sample = fixture.sample_records()[0]
    scores = torch.tensor([1.0, -1.0], requires_grad=True)
    torch.testing.assert_close(supervised_loss(scores, sample, task_kind='single_choice'),
        torch.nn.functional.cross_entropy(scores[None], torch.tensor([0])))
    torch.testing.assert_close(supervised_loss(scores, sample, task_kind='multi_label'),
        torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.tensor([1.0, 0.0])))


@pytest.mark.parametrize('tuning', ['full', 'lora'])
def test_real_trainer_export_and_exact_resume(tmp_path, tuning):
    if tuning == 'lora':
        pytest.importorskip('peft')
    torch.set_num_threads(1)
    config = fixture.make_fixture(tmp_path)
    # Multiple different examples exercise sampler progress rather than merely
    # repeating one example after reload. Dropout exercises the RNG checkpoint.
    base_config = Path(config['model']['name_or_path']) / 'config.json'
    settings = json.loads(base_config.read_text())
    settings['attention_dropout'] = 0.1
    base_config.write_text(json.dumps(settings))
    records = []
    for i, query in enumerate(('apple', 'sky', 'water')):
        item = fixture.sample_records()[0]
        item['id'], item['input']['query'] = f'q{i}', query
        records.append(item)
    Path(config['data']['train_file']).write_text(''.join(json.dumps(x) + '\n' for x in records))
    config['training']['max_steps'] = 3
    config['training']['tuning'] = tuning
    config['training']['gradient_checkpointing'] = True
    trained = train(config, tmp_path / 'run')
    checkpoint = tmp_path / 'run/checkpoints/checkpoint-1'
    for filename in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'cometa_resume.json'):
        assert (checkpoint / filename).is_file()
    logs = [json.loads(line) for line in (tmp_path / 'run/logs/events.jsonl').read_text().splitlines()]
    assert any(row['event'] == 'trainer_log' and 'loss' in row['metrics'] for row in logs)
    restored_cfg = copy.deepcopy(config)
    restored_cfg['training']['resume_from_checkpoint'] = str(checkpoint)
    resumed = train(restored_cfg, tmp_path / 'resumed')
    left, right = QwenReranker.load(trained), QwenReranker.load(resumed)
    for name, tensor in left.state_dict().items():
        torch.testing.assert_close(tensor, right.state_dict()[name], atol=1e-7, rtol=1e-6, msg=name)
    assert left.score(fixture.sample_records()) == right.score(fixture.sample_records())


def test_lora_real_step_and_export_load(tmp_path):
    pytest.importorskip('peft')
    torch.set_num_threads(1)
    config = fixture.make_fixture(tmp_path)
    config['training']['tuning'] = 'lora'
    artifact = train(config, tmp_path / 'lora-run')
    model = QwenReranker.load(artifact)
    assert (artifact / 'model/adapter_config.json').is_file()
    assert hasattr(model.backbone, 'peft_config')
    assert len(model.score(fixture.sample_records())[0]) == 2
    adapter_weights = [p for name, p in model.named_parameters() if 'lora_B' in name]
    assert adapter_weights and any(torch.count_nonzero(p) for p in adapter_weights)
    base_config = Path(config['model']['name_or_path']) / 'config.json'
    base_config.write_text(base_config.read_text() + '\n')
    with pytest.raises(ValueError, match='base model files changed'):
        QwenReranker.load(artifact)


@pytest.mark.parametrize('all_unjudged', [False, True])
def test_validation_loss_subset_preserves_explicit_counts(tmp_path, all_unjudged):
    torch.set_num_threads(1)
    config = fixture.make_fixture(tmp_path)
    records = fixture.sample_records()
    unknown = copy.deepcopy(records[0])
    unknown['id'] = 'unjudged-query'
    unknown['supervision']['labels'] = {}
    validation = tmp_path / 'validation.jsonl'
    validation.write_text(''.join(json.dumps(x) + '\n' for x in
                                  ([unknown] if all_unjudged else records + [unknown])))
    config['data']['validation_file'] = str(validation)
    train(config, tmp_path / 'subset-run')
    stats = json.loads((tmp_path / 'subset-run/training_metrics.json').read_text())['validation_loss_subset']
    assert stats['total'] == (1 if all_unjudged else 2)
    assert stats['eligible'] == (0 if all_unjudged else 1)
    assert stats['skipped_unjudged'] == 1


def test_smoke_export_is_sealed_and_predictor_ready(tmp_path):
    from cometa.artifacts import verify_artifact
    from cometa.api import Predictor
    result = fixture.run_smoke(tmp_path / 'smoke')
    manifest = verify_artifact(result['artifact'])
    assert manifest['task']['kind'] == 'ranking'
    predictions = Predictor(result['artifact']).predict(fixture.sample_records())
    assert predictions == result['predictions']
    assert predictions[0]['status'] == 'ok'
    assert len(predictions[0]['scores']) == 2


def test_candidate_sampling_is_deterministic_and_preserves_grades_and_ids():
    sample = fixture.sample_records()[0]
    sample['candidates'] = [{'id': f'p{i}', 'text': 'apple'} for i in range(3)] + [
        {'id': f'n{i}', 'text': 'sky'} for i in range(12)] + [{'id': 'unknown', 'text': 'water'}]
    sample['supervision']['labels'] = {f'p{i}': i + 1 for i in range(3)} | {f'n{i}': 0 for i in range(12)}
    original = copy.deepcopy(sample)
    policy = {'enabled': True, 'max_candidates': 8, 'max_positives': 2, 'seed': 17}
    records, summary, lineage = _sample_training_candidates([sample], policy)
    assert sample == original
    assert _sample_training_candidates([sample], policy) == (records, summary, lineage)
    assert summary['mode'] == 'fixed_per_run' and summary['scope'] == 'train_only'
    assert len(records[0]['candidates']) == 8
    assert lineage[0]['selected_positives'] == 2 and lineage[0]['selected_negatives'] == 6
    assert 'unknown' not in lineage[0]['selected_ids']
    assert all(original['supervision']['labels'][cid] == grade
               for cid, grade in records[0]['supervision']['labels'].items())
    selected_ids = lineage[0]['selected_ids']
    assert selected_ids == [x['id'] for x in original['candidates'] if x['id'] in selected_ids]
    # Reordering candidates cannot change which stable IDs are sampled.
    sample['candidates'].reverse()
    shuffled, _, _ = _sample_training_candidates([sample], policy)
    assert {x['id'] for x in shuffled[0]['candidates']} == set(selected_ids)
    # Missing positive slots are filled from explicit zero-grade negatives.
    sample['candidates'] = [x for x in sample['candidates'] if x['id'] not in {'p1', 'p2'}]
    sample['supervision']['labels'].pop('p1'); sample['supervision']['labels'].pop('p2')
    _, _, lineage = _sample_training_candidates([sample], policy)
    assert lineage[0]['selected_positives'] == 1 and lineage[0]['selected_negatives'] == 7


def selection_fixture(directory):
    config = fixture.make_fixture(directory)
    judged = copy.deepcopy(fixture.sample_records()[0])
    judged['id'] = 'dev-judged'
    judged['supervision']['labels'] = {'d1': 1, 'd2': 1}
    missed = copy.deepcopy(judged)
    missed['id'] = 'dev-no-candidate-positive'
    missed['supervision']['labels'] = {}
    dev = Path(directory) / 'dev.jsonl'
    dev.write_text(''.join(json.dumps(x) + '\n' for x in [judged, missed]))
    qrels = Path(directory) / 'dev-qrels.json'
    qrels.write_text(json.dumps({'dev-judged': {'d1': 1, 'd2': 1, 'outside': 1},
                                'dev-no-candidate-positive': {'outside': 1}}))
    config['data'].update(validation_file=str(dev), validation_qrels_path=str(qrels), validation_split='validation')
    config['training']['selection'] = {'enabled': True, 'metric': 'ndcg@10'}
    return config


def test_complete_dev_includes_no_candidate_positive_and_full_qrels(tmp_path):
    config = selection_fixture(tmp_path)
    samples, qrels, contract = _selection_data(config['data'], config['training']['selection'],
                                              'ranking', fixture.sample_records())
    class FixedScores(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def score(self, samples):
            return [[2., 1.] for _ in samples]

    summary, rows = _complete_validation_metrics(FixedScores(), samples, qrels)
    from cometa.metrics import ranking_metrics
    expected = ranking_metrics(['d1', 'd2'], [2., 1.], {'d1': 1, 'd2': 1, 'outside': 1})['ndcg@10'] / 2
    assert summary['queries'] == 2 and summary['ndcg@10'] == pytest.approx(expected)
    assert rows[1]['metrics']['ndcg@10'] == 0
    assert contract['scope'] == 'complete_validation_full_qrels'
    assert contract['validation_candidate_sampling'] is False
    assert [len(x['candidate_ids']) for x in rows] == [2, 2]
    # Held-out test selection and overlapping query identities fail before a model is loaded.
    with pytest.raises(ValueError, match='Test split'):
        _selection_data(config['data'] | {'validation_split': 'test'}, config['training']['selection'],
                        'ranking', fixture.sample_records())
    with pytest.raises(ValueError, match='disjoint'):
        _selection_data(config['data'], config['training']['selection'], 'ranking', samples)
    Path(config['data']['validation_qrels_path']).write_text(json.dumps({'dev-judged': {'d1': 1}}))
    with pytest.raises(ValueError, match='match exactly'):
        _selection_data(config['data'], config['training']['selection'], 'ranking', fixture.sample_records())


def test_score_head_lora_training_selects_full_dev_best_and_resumes(tmp_path):
    pytest.importorskip('peft')
    from cometa.model_registry import load_model
    torch.set_num_threads(1)
    config = selection_fixture(tmp_path / 'fixture')
    config['model'].update(adapter='qwen3_score_head', head_hidden_size=9)
    config['data']['train_candidate_sampling'] = {'enabled': True, 'max_candidates': 2,
                                                 'max_positives': 1, 'seed': 17}
    config['training'].update(max_steps=2, tuning='lora', head_learning_rate=0.003,
                              learning_rate=0.001, gradient_checkpointing=True)
    config['training']['selection']['evaluate_initial'] = True
    selected = train(config, tmp_path / 'run')
    assert selected == tmp_path / 'run/exports/best'
    best = json.loads((selected / 'selection.json').read_text())
    final_dir = tmp_path / 'run/exports/final'
    final = json.loads((final_dir / 'selection.json').read_text())
    # All retrieved candidates have the same grade for dev-judged; the second
    # query has no retrieved positive. Metric ties must keep the earlier model.
    assert best['global_step'] == 1 and best['kind'] == 'best_validation_metric'
    assert best['queries'] == 2 and best['scope'] == 'complete_validation_full_qrels'
    assert final['global_step'] == 2 and not final['selected_for_use']
    metrics = json.loads((tmp_path / 'run/training_metrics.json').read_text())
    assert len(metrics['validation_history']) == 2
    assert metrics['initial_validation']['metrics']['queries'] == 2
    assert metrics['initial_validation']['global_step'] == 0
    assert metrics['initial_validation']['eligible_for_best'] is False
    assert all(x['metrics']['queries'] == 2 for x in metrics['validation_history'])
    assert not metrics['validation_loss_subset']['loss_evaluation_enabled']
    audit = json.loads((tmp_path / 'run/parameter_updates.json').read_text())
    assert audit['summary']['changed_head_tensors'] > 0
    assert audit['summary']['changed_backbone_tensors'] > 0
    assert audit['summary']['frozen_tensors_with_version_changes'] == 0
    assert all(x['dtype'] == 'torch.float32' for x in audit['parameters'].values() if x['trainable'])
    groups = json.loads((tmp_path / 'run/optimizer_groups.json').read_text())
    assert {x['learning_rate'] for x in groups if x['role'] == 'score_head'} == {0.003}
    assert {x['learning_rate'] for x in groups if x['role'] == 'backbone'} == {0.001}
    assert all('lora_' in n for x in groups if x['role'] == 'backbone' for n in x['names'])
    best_model, final_model = load_model(selected), load_model(final_dir)
    assert any(not torch.equal(p, final_model.score_head.state_dict()[name])
               for name, p in best_model.score_head.state_dict().items())
    checkpoint = tmp_path / 'run/checkpoints/checkpoint-1'
    saved_selection = json.loads((checkpoint / 'cometa_selection.json').read_text())
    assert saved_selection['best']['global_step'] == 1
    restored = copy.deepcopy(config)
    restored['training']['resume_from_checkpoint'] = str(checkpoint)
    resumed = train(restored, tmp_path / 'resumed')
    resumed_final = load_model(tmp_path / 'resumed/exports/final')
    for name, value in final_model.state_dict().items():
        torch.testing.assert_close(value, resumed_final.state_dict()[name], atol=1e-7, rtol=1e-6, msg=name)
    assert json.loads((resumed / 'selection.json').read_text())['global_step'] == 1
    assert load_model(resumed).score(fixture.sample_records()) == best_model.score(fixture.sample_records())
    restored['data']['train_candidate_sampling']['seed'] += 1
    with pytest.raises(ValueError, match='Resume contract'):
        train(restored, tmp_path / 'invalid-resume')
    restored['data']['train_candidate_sampling']['seed'] -= 1
    # Same path and unchanged config are not enough to identify a LoRA base.
    # Replacing only its weights must be detected before checkpoint restoration.
    from safetensors.torch import load_file, save_file
    weights = Path(config['model']['name_or_path']) / 'model.safetensors'
    state = load_file(str(weights))
    name = next(iter(state))
    state[name] = state[name] + 0.01
    save_file(state, str(weights), metadata={'format': 'pt'})
    with pytest.raises(ValueError, match='Resume contract'):
        train(restored, tmp_path / 'changed-base-resume')


def test_bf16_autocast_keeps_trainable_master_weights_float32(tmp_path):
    torch.set_num_threads(1)
    config = fixture.make_fixture(tmp_path / 'fixture')
    config['model'].update(adapter='qwen3_score_head', dtype='float16')
    config['training'].update(precision='bf16', head_learning_rate=0.001)
    train(config, tmp_path / 'run')
    audit = json.loads((tmp_path / 'run/parameter_updates.json').read_text())
    assert audit['summary']['changed_head_tensors'] > 0
    assert all(x['dtype'] == 'torch.float32' for x in audit['parameters'].values() if x['trainable'])
    metrics = json.loads((tmp_path / 'run/training_metrics.json').read_text())
    assert metrics['compute_precision'] == 'bf16' and metrics['master_dtype'] == 'float32'
