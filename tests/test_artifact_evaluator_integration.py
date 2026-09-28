"""Actual tiny CPU ScoreHead integration; execute on the test host, no downloads."""
from argparse import Namespace
import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('tokenizers')
pytest.importorskip('safetensors')


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_scorehead_full_evaluation_resume_and_integrity(tmp_path, monkeypatch):
    """Real Predictor forward, full-qrels misses, no duplicate resume, hash refusal.

    Predictor.predict is only wrapped to count calls; all initial predictions
    load a real sealed tiny Qwen3 + MLP head and execute its CPU forward.
    """
    from cometa.api import Predictor
    from cometa.artifacts import seal_artifact
    from cometa.model import QwenScoreHead
    from cometa.schema import sha256

    root = Path(__file__).resolve().parents[1]
    scripts = root / 'examples' / 'benchmarks'
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    evaluator = load_script('tiny_artifact_evaluator', scripts / 'evaluate_artifact.py')
    fixture = load_script('tiny_artifact_fixture', root / 'examples' / 'smoke_train.py')

    torch.set_num_threads(1)
    config = fixture.make_fixture(tmp_path / 'fixture')
    scorer = QwenScoreHead(config['model']['name_or_path'], layout='pairs', backend='eager',
                           device='cpu', dtype='float32', max_length=256,
                           pair_batch_size=1, head_hidden_size=8)
    artifact = scorer.save(tmp_path / 'sealed-scorehead')
    seal_artifact(artifact, task={'kind': 'ranking'})
    # This is a full checkpoint, so evaluation does not depend on a remote
    # pretrained base or a PEFT installation.
    saved_spec = json.loads((artifact / 'model_spec.json').read_text())
    assert saved_spec['family'] == 'qwen3_score_head'
    assert saved_spec['tuning'] == 'full'

    judged = copy.deepcopy(fixture.sample_records()[0])
    judged['id'] = 'dev-with-positive'
    judged['supervision']['labels'] = {'d1': 1}
    missed = copy.deepcopy(judged)
    missed['id'] = 'dev-without-candidate-positive'
    missed['input']['query'] = 'water'
    missed['supervision']['labels'] = {}
    samples = [judged, missed]
    split = tmp_path / 'validation.jsonl'
    split.write_text(''.join(json.dumps(row) + '\n' for row in samples))
    qrels = tmp_path / 'validation-qrels.json'
    # Relevant documents outside the supplied candidates must remain in the
    # denominator; the second query must not be dropped as "unjudged".
    qrels.write_text(json.dumps({judged['id']: {'d1': 2, 'outside-1': 1},
                                missed['id']: {'outside-2': 1}}))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': '1.0', 'dataset_id': 'tests/tiny-scorehead',
        'task_id': 'rerank', 'task_kind': 'ranking', 'splits': {'validation': {
            'path': split.name, 'sha256': sha256(split), 'count': 2,
            'qrels_path': qrels.name, 'qrels_sha256': sha256(qrels)}}}))
    output = tmp_path / 'evaluation'
    args = Namespace(manifest=str(manifest), artifact=str(artifact), output=str(output),
                     split='validation', device='cpu', dtype='float32', precision='none',
                     threads=1, warmup_queries=1, limit=None, resume=False)

    actual_predict = Predictor.predict
    predicted_ids = []

    def tracked_predict(self, batch, **policy):
        predicted_ids.extend(row['id'] for row in batch)
        return actual_predict(self, batch, **policy)

    monkeypatch.setattr(Predictor, 'predict', tracked_predict)
    assert evaluator.run(args) == 0
    protocol = json.loads((output / 'evaluation.json').read_text())
    metrics = json.loads((output / 'metrics.json').read_text())
    predictions = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
    assert protocol['status'] == 'completed'
    assert protocol['scope'] == metrics['scope'] == 'full_split'
    assert protocol['completed'] == metrics['queries'] == protocol['full_split_queries'] == 2
    assert metrics['candidates'] == 4
    assert metrics['candidate_recall'] == pytest.approx(.25)
    assert [row['id'] for row in predictions] == [row['id'] for row in samples]
    assert predictions[0]['metrics']['judged_positive_count'] == 2
    assert predictions[1]['metrics']['ndcg@10'] == 0
    assert predictions[1]['metrics']['candidate_recall'] == 0
    assert all(row['score_semantics'] == 'raw_relevance_logit' for row in predictions)
    assert predicted_ids == [judged['id'], judged['id'], missed['id']]  # one warmup, two scored queries

    original_predictions = (output / 'predictions.jsonl').read_bytes()
    original_sessions = (output / 'sessions.jsonl').read_bytes()
    original_warmups = (output / 'warmups.jsonl').read_bytes()
    calls_before_resume = list(predicted_ids)
    args.resume = True
    assert evaluator.run(args) == 0
    assert predicted_ids == calls_before_resume
    assert (output / 'predictions.jsonl').read_bytes() == original_predictions
    assert (output / 'sessions.jsonl').read_bytes() == original_sessions
    assert (output / 'warmups.jsonl').read_bytes() == original_warmups

    # A complete newline-terminated edit is not an interrupted-write tail and
    # must be rejected, even if all candidate IDs still look valid.
    altered = copy.deepcopy(predictions)
    altered[0]['scores'][0]['score'] += 1.0
    (output / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in altered))
    with pytest.raises(ValueError, match='Completed predictions hash mismatch'):
        evaluator.run(args)
    assert predicted_ids == calls_before_resume
    (output / 'predictions.jsonl').write_bytes(original_predictions)

    # Modifying sealed metadata also fails before any model prediction; even a
    # semantics-preserving whitespace edit changes the artifact identity.
    sealed_input = artifact / 'input_spec.json'
    sealed_input.write_text(sealed_input.read_text() + '\n')
    with pytest.raises(ValueError, match='Artifact integrity error'):
        evaluator.run(args)
    assert predicted_ids == calls_before_resume
