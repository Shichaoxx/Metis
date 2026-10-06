"""Offline split isolation and immutable registration for the second BoolQ study."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('safetensors')

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    'boolq_readout_protocol', ROOT / 'examples/research/prepare_boolq_readout_protocol.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def fixture_data():
    train = [{'question': f'question-{i}', 'passage': f'passage-{i}', 'answer': bool(i % 2)}
             for i in range(28)]
    train += [{'question': 'duplicate question', 'passage': 'passage-2', 'answer': True},
              {'question': 'cross split question', 'passage': 'validation-1', 'answer': False}]
    validation = [{'question': f'validation-question-{i}', 'passage': f'validation-{i}',
                   'answer': bool(i % 2)} for i in range(12)]
    validation.append({'question': 'same evaluation passage', 'passage': 'validation-0', 'answer': True})
    rows = {'train': train, 'validation': validation}
    old = {'dev': [protocol._record(train[i], i, 'train') for i in (2, 3)],
           'evaluation': [protocol._record(validation[i], i, 'validation') for i in (0, 1)]}
    return rows, old


def test_sampling_is_label_blind_and_excludes_prior_evaluation_passages():
    rows, old = fixture_data()
    selected = protocol.select_records(rows, old, train_size=12, dev_size=5, evaluation_size=5)
    flipped = copy.deepcopy(rows)
    for items in flipped.values():
        for item in items:
            item['answer'] = not item['answer']
    other = protocol.select_records(flipped, old, train_size=12, dev_size=5, evaluation_size=5)
    for name, items in selected.items():
        assert [item['id'] for item in items] == [item['id'] for item in other[name]]
        assert [item['input_sha256'] for item in items] == [item['input_sha256'] for item in other[name]]
        assert [item['label'] for item in items] == [1 - item['label'] for item in other[name]]
        assert len({item['passage_sha256'] for item in items}) == len(items)
    groups = {name: {item['passage'] for item in items} for name, items in selected.items()}
    reserved = {item['passage'] for item in rows['validation']}
    assert not groups['train'] & reserved
    assert not groups['dev'] & reserved
    assert not groups['train'] & groups['dev']
    assert {'passage-2', 'passage-3'} <= groups['dev']
    assert not groups['train'] & {'passage-2', 'passage-3'}
    assert not groups['evaluation'] & {'validation-0', 'validation-1'}


@pytest.mark.parametrize('counts', [(0, 5, 5), (True, 5, 5), (12, -1, 5),
                                  (12, 5, 0), (40, 5, 5), (12, 5, 11), (12, 1, 5)])
def test_invalid_or_unavailable_split_counts_fail(counts):
    rows, old = fixture_data()
    with pytest.raises(ValueError):
        protocol.select_records(rows, old, train_size=counts[0], dev_size=counts[1],
                                evaluation_size=counts[2])


def test_previous_source_identity_mismatch_is_rejected():
    rows, old = fixture_data()
    old['dev'][0]['question'] = 'modified old development input'
    with pytest.raises(ValueError, match='input identity'):
        protocol.select_records(rows, old, train_size=12, dev_size=5, evaluation_size=5)


def test_registration_separates_evaluation_and_hashes_tokenizer_and_weight_files(tmp_path, monkeypatch):
    rows, old = fixture_data()
    previous = tmp_path / 'previous'
    previous.mkdir()
    previous_records = dict(old, train=[])
    previous_protocol = {'experiment': 'frozen_base_readout_pilot',
                         'dataset': {'revision': protocol.DATASET_REVISION,
                                     'raw_files': {name: {'sha256': name} for name in rows}},
                         'splits': {name: {'count': len(items), 'ids': [item['id'] for item in items],
                                           'records_sha256': protocol.digest(json.dumps(items, sort_keys=True))}
                                    for name, items in previous_records.items()}}
    (previous / 'selected_records.json').write_text(json.dumps(previous_records))
    (previous / 'protocol.json').write_text(json.dumps(previous_protocol))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'hidden_size': 1024}))
    (model / 'tokenizer.json').write_text('{}')
    (model / 'tokenizer_config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'fixture weights')
    monkeypatch.setattr(protocol, 'download_dataset',
                        lambda root: (rows, {name: {'sha256': name} for name in rows}))
    args = argparse.Namespace(data=tmp_path / 'raw', model=model, previous_pilot=previous,
                              output=tmp_path / 'registered', train_size=12, dev_size=5,
                              evaluation_size=5, seed=20261007)
    registered = protocol.prepare(args)
    train_dev = json.loads((args.output / 'train_dev_records.json').read_text())
    sealed = json.loads((args.output / 'sealed_evaluation.json').read_text())
    indices = json.loads((args.output / 'selected_records.json').read_text())
    assert set(train_dev) == {'train', 'dev'}
    assert set(sealed) == {'evaluation'}
    assert all('label' not in item and 'passage' not in item and 'question' not in item
               for items in indices.values() for item in items)
    assert len(registered['registered_runs']) == 9
    assert len({run['id'] for run in registered['registered_runs']}) == 9
    assert {run['seed'] for run in registered['registered_runs']} == {42, 43, 44}
    assert {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors'} <= set(
        registered['base']['file_sha256'])
    for name, entry in registered['files'].items():
        assert entry['sha256'] == protocol.sha256(args.output / name)
    with pytest.raises(ValueError, match='already exists'):
        protocol.prepare(args)
    previous_records['dev'][0]['label'] = 1 - previous_records['dev'][0]['label']
    (previous / 'selected_records.json').write_text(json.dumps(previous_records))
    with pytest.raises(ValueError, match='do not match their protocol'):
        protocol.load_previous_pilot(previous)
