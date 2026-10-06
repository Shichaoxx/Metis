"""Dataset isolation and label blindness of the research feature protocol."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('safetensors')
spec = importlib.util.spec_from_file_location(
    'boolq_features', Path(__file__).parents[1] / 'examples/research/extract_boolq_features.py')
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def fixture_rows():
    train = [{'question': f'question-{i}', 'passage': f'evidence-{i}', 'answer': bool(i % 2)}
             for i in range(18)]
    # Duplicated passages must stay in one research split, including cases
    # where the official train and validation populations share a passage.
    train += [{'question': 'other question', 'passage': 'evidence-2', 'answer': True}]
    validation = [{'question': 'held out', 'passage': 'evidence-2', 'answer': False},
                  {'question': 'held out 2', 'passage': 'external', 'answer': True}]
    return {'train': train, 'validation': validation}


def test_fixed_sampling_is_label_blind_and_passage_disjoint():
    rows = fixture_rows()
    first = features.fixed_splits(rows, (8, 4, 2), 71)
    flipped = {name: [dict(row, answer=not row['answer']) for row in values]
               for name, values in rows.items()}
    second = features.fixed_splits(flipped, (8, 4, 2), 71)
    for name in first:
        assert [r['id'] for r in first[name]] == [r['id'] for r in second[name]]
        assert [r['label'] for r in first[name]] == [1 - r['label'] for r in second[name]]
    groups = {name: {r['passage'] for r in values} for name, values in first.items()}
    assert not groups['train'] & groups['dev']
    assert not groups['train'] & groups['evaluation']
    assert not groups['dev'] & groups['evaluation']


def test_split_budget_excludes_reserved_and_duplicate_passages():
    with pytest.raises(ValueError, match='available disjoint data'):
        features.fixed_splits(fixture_rows(), (17, 1, 2), 71)
