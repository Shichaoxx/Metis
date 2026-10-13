"""Training-only selection and stopping contracts for the memorization probe."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('safetensors')

_path = Path(__file__).resolve().parents[1] / 'examples/research/diagnose_boolq_training.py'
_spec = importlib.util.spec_from_file_location('boolq_training_diagnostic_test', _path)
diagnostic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnostic)


def record(index, label, *, truncated=False):
    question = f'Question {index}'
    passage = ('TRUNCATED ' if truncated else '') + f'Passage {index}'
    return {'id': f'train-{index}', 'question': question, 'passage': passage,
            'label': label, 'source_split': 'train', 'source_index': index,
            'input_sha256': hashlib.sha256(json.dumps(
                [question, passage], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
            'passage_sha256': hashlib.sha256(passage.encode()).hexdigest()}


class FakeCompiler:
    def compile(self, sample):
        truncated = sample['candidates'][0]['text'].startswith('TRUNCATED')
        return SimpleNamespace(truncated=[truncated], pairs=[[1, 2, 3]])


def test_selection_balanced_untruncated_and_independent_of_pool_order():
    rows = [record(i, i % 2, truncated=i < 4) for i in range(40)]
    selected, audit = diagnostic.select_training_records(rows, FakeCompiler())
    reversed_selection, _ = diagnostic.select_training_records(rows[::-1], FakeCompiler())
    assert selected == reversed_selection
    assert len(selected) == 32
    assert sum(row['label'] for row in selected) == 16
    assert len({row['passage_sha256'] for row in selected}) == 32
    assert not any(item['truncated'] for item in audit.values())


def test_selection_refuses_to_fill_missing_class_with_truncated_inputs():
    rows = [record(i, i % 2, truncated=i == 0) for i in range(32)]
    with pytest.raises(ValueError):
        diagnostic.select_training_records(rows, FakeCompiler())


def test_selection_rejects_nontraining_sources_and_changed_text():
    row = record(0, 0)
    with pytest.raises(ValueError):
        diagnostic.checked_record(row | {'source_split': 'validation'})
    with pytest.raises(ValueError):
        diagnostic.checked_record(row | {'passage': 'changed input'})


def test_both_arms_receive_same_complete_training_passes():
    identifiers = [f'train-{i}' for i in range(32)]
    groups = diagnostic.update_order(identifiers, 42)
    assert groups == diagnostic.update_order(identifiers, 42)
    assert groups != diagnostic.update_order(identifiers, 43)
    assert len(groups) == 240
    for offset in range(0, len(groups), 4):
        assert sorted(item for group in groups[offset:offset + 4] for item in group) == sorted(identifiers)


def test_constant_predictions_do_not_pass_and_auc_handles_ties():
    predictions = [{'id': str(i), 'label': i % 2, 'logit': 0.0} for i in range(32)]
    metrics = diagnostic.training_metrics(predictions)
    assert metrics['auroc'] == 0.5
    assert metrics['correct_count'] == 16
    assert not diagnostic.meets_stopping(metrics)


def test_stopping_requires_probability_quality_as_well_as_classification():
    weak = [{'id': str(i), 'label': i % 2, 'logit': 0.1 if i % 2 else -0.1}
            for i in range(32)]
    strong = [row | {'logit': 5.0 if row['label'] else -5.0} for row in weak]
    assert diagnostic.training_metrics(weak)['correct_count'] == 32
    assert not diagnostic.meets_stopping(diagnostic.training_metrics(weak))
    assert diagnostic.meets_stopping(diagnostic.training_metrics(strong))
