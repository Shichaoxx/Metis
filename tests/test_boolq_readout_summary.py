"""Synthetic completed reports test evidence checks, not model quality."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('boolq_readout_summary',
                                            ROOT / 'examples/research/summarize_boolq_readout.py')
summary_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def records(split, count):
    output = []
    for index in range(count):
        question, passage = f'{split} question {index}', f'{split} unique passage {index}'
        output.append({'id': f'{split}-{index}', 'question': question, 'passage': passage, 'label': index % 2,
                       'input_sha256': hashlib.sha256(json.dumps([question, passage], ensure_ascii=False,
                                                    separators=(',', ':')).encode()).hexdigest(),
                       'passage_sha256': hashlib.sha256(passage.encode()).hexdigest()})
    return output


@pytest.fixture
def completed(tmp_path):
    torch.set_num_threads(1)
    helper = summary_module.sibling('train_boolq_readout')
    metric_helper = summary_module.sibling('train_readout_ablation').binary_metrics
    root = tmp_path / 'round2'
    root.mkdir()
    rows = {'train': records('train', 8), 'dev': records('dev', 4), 'evaluation': records('evaluation', 4)}
    write(root / 'train_dev_records.json', {key: rows[key] for key in ('train', 'dev')})
    write(root / 'sealed_evaluation.json', {'evaluation': rows['evaluation']})
    configuration = {'epochs': 3, 'effective_batch_size': 2, 'max_length': 256, 'mlp_width': 256}
    protocol = {'experiment': 'boolq_readout_round2', 'readouts': list(summary_module.READOUTS),
                'seeds': [42, 43, 44], 'configuration': configuration, 'base': {'hidden_size': 4},
                'train_positive_prior': 0.5, 'evaluation': 'synthetic reports; not quality evidence',
                'registered_runs': [{'readout': readout, 'seed': seed, 'training': configuration}
                                    for readout in summary_module.READOUTS for seed in (42, 43, 44)],
                'splits': {name: {'count': len(value), 'ids': [row['id'] for row in value],
                                  'records_sha256': summary_module.json_digest(value)}
                           for name, value in rows.items()},
                'record_files': {name: summary_module.digest(root / name) for name in
                                 ('train_dev_records.json', 'sealed_evaluation.json')}}
    write(root / 'protocol.json', protocol)
    protocol_hash = summary_module.digest(root / 'protocol.json')
    common_initial = {str(seed): {'head': f'head-{seed}', 'lora': f'lora-{seed}', 'frozen_base': 'frozen-base'}
                      for seed in protocol['seeds']}
    common_orders = {str(seed): [f'order-{seed}-{epoch}' for epoch in (1, 2, 3)] for seed in protocol['seeds']}
    body_hashes = {'train': 'same-train-bodies', 'dev': 'same-dev-bodies'}
    lock = {'protocol_sha256': protocol_hash, 'evaluation_opened': False,
            'common_initial_parameters_by_seed': common_initial,
            'common_epoch_order_hashes_by_seed': common_orders,
            'common_candidate_body_hashes': body_hashes, 'runs': {}}
    # Deliberately make dev-selected decision_marker worse on evaluation than
    # last_relevance, so the test catches accidental evaluation-based selection.
    dev_nll = {'last_relevance': 0.4, 'decision_marker': 0.2, 'candidate_attention': 0.3}
    probabilities = {'last_relevance': [0.2, 0.8, 0.2, 0.8],
                     'decision_marker': [0.8, 0.2, 0.8, 0.2],
                     'candidate_attention': [0.4, 0.6, 0.4, 0.6]}
    for readout in protocol['readouts']:
        for seed in protocol['seeds']:
            key = f'{readout}/seed-{seed}'
            directory = root / 'models' / key
            initial = dict(common_initial[str(seed)], pool='pool-initial')
            final = dict(initial, head=f'{key}-head-trained', lora=f'{key}-lora-trained',
                         pool='pool-trained' if readout == 'candidate_attention' else 'pool-initial')
            selected_parameters = {key: final[key] for key in ('head', 'lora', 'pool')}
            run_spec = {'protocol': protocol, 'protocol_sha256': protocol_hash, 'readout': readout, 'seed': seed}
            write(directory / 'run_spec.json', run_spec)
            best = directory / 'best'
            write(best / 'standalone_spec.json', dict(run_spec,
                  format='metis-research-boolq-readout-v2', epoch=2,
                  artifact_scope='synthetic research report; not a v1 Predictor artifact',
                  selected_parameter_sha256=selected_parameters))
            # No real tensor is needed: the summary verifies manifests without loading models.
            (best / 'readout.safetensors').write_bytes(b'synthetic artifact evidence')
            write(best / 'files.sha256.json', helper.file_identity(best))
            history = [{'epoch': epoch, 'dev': {'nll': dev_nll[readout] + (0 if epoch == 2 else 0.1)},
                        'epoch_order_sha256': common_orders[str(seed)][epoch - 1]} for epoch in (1, 2, 3)]
            audit = {'initial': initial, 'final': final, 'head_updated': True, 'lora_updated': True,
                     'frozen_base_unchanged': True, 'pool_updated': readout == 'candidate_attention',
                     'selected_parameter_sha256': selected_parameters, 'evaluation_opened': False,
                     'steps': 12, 'best_epoch': 2, 'best_dev_nll': dev_nll[readout], 'history': history,
                     'training_seconds': 1.0, 'gpu_peak_allocated_bytes': None,
                     'head_parameters': 1537, 'pool_parameters': 4 if readout == 'candidate_attention' else 0,
                     'lora_parameters': 64}
            write(directory / 'training_audit.json', audit)
            write(directory / 'selection.json', {'checkpoint': 'checkpoints/epoch-2',
                                                'best_dev_nll': dev_nll[readout],
                                                'evaluation_used_for_selection': False})
            write(directory / 'input_audit.json', {name: {'candidate_body_ids_sha256': body_hashes[name],
                  'truncated': 1, 'mean_tokens': 20., 'max_tokens': 25} for name in ('train', 'dev')})
            write(directory / 'fresh-process-reload.json', {'status': 'passed', 'samples': 3,
                  'maximum_absolute_difference': 0.0, 'scope': 'synthetic software fixture'})
            lock['runs'][key] = {'selection_sha256': summary_module.digest(directory / 'selection.json'),
                                'audit_sha256': summary_module.digest(directory / 'training_audit.json'),
                                'artifact_manifest_sha256': summary_module.digest(best / 'files.sha256.json')}
    write(root / 'selection-lock.json', lock)
    for readout in protocol['readouts']:
        for seed in protocol['seeds']:
            p = torch.tensor(probabilities[readout], dtype=torch.float64)
            logits, labels = torch.logit(p), torch.tensor([0., 1., 0., 1.], dtype=torch.float64)
            write(root / 'models' / readout / f'seed-{seed}' / 'evaluation.json', {
                'selection_lock_sha256': summary_module.digest(root / 'selection-lock.json'),
                'evaluation_records_sha256': protocol['record_files']['sealed_evaluation.json'],
                'metrics': metric_helper(logits, labels),
                'predictions': [{'id': row['id'], 'label': row['label'], 'probability_yes': probability,
                                 'logit': float(logit)} for row, probability, logit in
                                zip(rows['evaluation'], probabilities[readout], logits)]})
    return root


def test_known_results_and_development_selection_do_not_follow_evaluation(completed):
    result = summary_module.summarize(completed, bootstrap_samples=30)
    assert result['development_selection']['readout'] == 'decision_marker'
    assert result['development_selection']['evaluation_used'] is False
    assert result['by_readout']['last_relevance']['evaluation']['accuracy'] == {'mean': 1., 'sample_std': 0.}
    assert result['by_readout']['decision_marker']['evaluation']['accuracy']['mean'] == 0
    assert result['by_readout']['last_relevance']['evaluation']['nll']['mean'] == pytest.approx(-math.log(0.8))
    assert result['by_readout']['last_relevance']['evaluation']['brier']['mean'] == pytest.approx(0.04)
    assert result['train_prior_baseline']['evaluation']['accuracy'] == 0.5
    assert result['train_prior_baseline']['evaluation']['nll'] == pytest.approx(math.log(2))
    paired = result['paired_bootstrap']['decision_marker']['metrics']
    assert paired['accuracy']['candidate_minus_last_relevance'] == -1
    assert paired['accuracy']['percentile_95_interval'] == [-1., -1.]
    assert paired['brier']['candidate_minus_last_relevance'] == pytest.approx(0.6)
    assert paired['nll']['candidate_minus_last_relevance'] == pytest.approx(math.log(4))
    assert result['input_audits']['last_relevance']['train']['total_tokens'] == 160
    assert (completed / 'summary.md').exists()
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        summary_module.summarize(completed, bootstrap_samples=30)


def test_incomplete_guard_does_not_open_sealed_records(completed):
    (completed / 'models/last_relevance/seed-42/evaluation.json').unlink()
    (completed / 'sealed_evaluation.json').unlink()
    with pytest.raises(ValueError, match='Incomplete experiment; sealed evaluation remains unopened'):
        summary_module.summarize(completed, bootstrap_samples=10)
    assert not (completed / 'summary.json').exists()


@pytest.mark.parametrize('field,value,message', [('id', 'wrong-sample', 'prediction IDs differ'),
                                               ('label', 1, 'prediction labels differ'),
                                               ('logit', 0.0, 'Saved probability differs')])
def test_rejects_misaligned_predictions(completed, field, value, message):
    path = completed / 'models/last_relevance/seed-42/evaluation.json'
    data = json.loads(path.read_text())
    data['predictions'][0][field] = value
    write(path, data)
    with pytest.raises(ValueError, match=message):
        summary_module.summarize(completed, bootstrap_samples=10)


def test_rejects_tampered_artifact_and_lock_hash(completed):
    artifact = completed / 'models/last_relevance/seed-42/best/readout.safetensors'
    original = artifact.read_bytes()
    artifact.write_bytes(original + b'tampered')
    with pytest.raises(ValueError, match='artifact checksum mismatch'):
        summary_module.summarize(completed, bootstrap_samples=10)
    artifact.write_bytes(original)
    lock = completed / 'selection-lock.json'
    lock.write_text(lock.read_text() + ' ')
    with pytest.raises(ValueError, match='Evaluation lock hash mismatch'):
        summary_module.summarize(completed, bootstrap_samples=10)


def test_pairing_averages_metrics_instead_of_ensembling_probabilities():
    # Every example gains one correct fixed-seed decision out of three, even
    # though an ensemble majority would classify all examples as incorrect.
    reference = {metric: torch.zeros(3, 4, dtype=torch.float64) for metric in summary_module.PAIRED_METRICS}
    candidate = {metric: torch.zeros(3, 4, dtype=torch.float64) for metric in summary_module.PAIRED_METRICS}
    candidate['accuracy'][0] = 1
    candidate['nll'][:] = 0.2
    candidate['brier'][:] = 0.1
    result = summary_module.paired_bootstrap(candidate, reference, samples=40)
    assert result['metrics']['accuracy']['candidate_minus_last_relevance'] == pytest.approx(1 / 3)
    assert result['metrics']['accuracy']['percentile_95_interval'] == pytest.approx([1 / 3, 1 / 3])
