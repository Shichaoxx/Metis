import copy
import importlib.util
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from cometa.model import QwenReranker

fixture_spec = importlib.util.spec_from_file_location('smoke_fixture', Path(__file__).parents[1] / 'examples/smoke_train.py')
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)


def models(backend='eager'):
    torch.set_num_threads(1)
    backbone, tokenizer = fixture.tiny_components()
    pair = QwenReranker.from_components(backbone, tokenizer, layout='pairs', backend=backend, max_length=256)
    tree = QwenReranker.from_components(copy.deepcopy(backbone), tokenizer, layout='tree', backend=backend,
                                       max_length=256, max_tree_tokens=512)
    return pair, tree


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_pair_tree_scores_and_gradients_match(backend):
    pair, tree = models(backend)
    sample = fixture.sample_records()
    left = pair.score_tensors(sample)[0]
    right = tree.score_tensors(sample)[0]
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    weight = torch.tensor([1.0, -0.7])
    (left * weight).sum().backward()
    (right * weight).sum().backward()
    for (name, parameter), (other_name, other) in zip(pair.named_parameters(), tree.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, atol=4e-6, rtol=3e-4, msg=name)


def test_tree_candidate_permutation():
    _, tree = models()
    sample = fixture.sample_records()[0]
    original = tree.score([sample])[0]
    permuted = copy.deepcopy(sample)
    permuted['candidates'].reverse()
    torch.testing.assert_close(torch.tensor(original), torch.tensor(tree.score([permuted])[0][::-1]),
                               atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_bounded_pair_batches_preserve_scores_and_gradients(backend):
    pair, _ = models(backend)
    bounded = copy.deepcopy(pair)
    bounded.pair_batch_size = 2
    sample = fixture.sample_records()[0]
    sample['candidates'] = [{'id': f'd{i}', 'text': 'red apple fruit ' * (i + 1)} for i in range(5)]
    left = pair.score_tensors([sample])[0]
    right = bounded.score_tensors([sample])[0]
    assert bounded.last_input_metadata[0]['physical_chunks'] == 3
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    weights = torch.tensor([1., -0.7, 0.4, -0.2, 0.1])
    (left * weights).sum().backward()
    (right * weights).sum().backward()
    for (name, parameter), (_, other) in zip(pair.named_parameters(), bounded.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, atol=4e-6, rtol=3e-4, msg=name)


def test_tree_chunking_and_document_tail_truncation():
    _, tree = models()
    sample = fixture.sample_records()[0]
    sample['candidates'][0]['text'] = 'red apple fruit ' * 100
    sample['candidates'][1]['text'] = 'blue sky ' * 100
    compiled = tree.compiler.compile(sample)
    assert all(compiled.truncated)
    assert len(compiled.prefix_ids) + sum(map(len, compiled.branch_ids)) > tree.max_length
    assert all(len(pair) == tree.max_length for pair in compiled.pairs)
    original = tree.score([sample])[0]
    tree.max_tree_tokens = tree.max_length
    torch.testing.assert_close(torch.tensor(original), torch.tensor(tree.score([sample])[0]),
                               atol=2e-6, rtol=2e-5)
    assert tree.last_input_metadata[0]['physical_chunks'] == 2
    assert tree.last_input_metadata[0]['candidate_truncated'] == {'d1': True, 'd2': True}


def test_readout_equals_full_lm_logits_and_export(tmp_path):
    pair, _ = models()
    sample = fixture.sample_records()
    compiled = pair.compiler.compile(sample[0])
    expected = []
    pair.eval()
    with torch.no_grad():
        for ids in compiled.pairs:
            logits = pair.backbone(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
            expected.append((logits[pair.yes_id] - logits[pair.no_id]).item())
    torch.testing.assert_close(torch.tensor(pair.score(sample)[0]), torch.tensor(expected), atol=2e-6, rtol=2e-5)
    pair.save(tmp_path / 'export')
    loaded = QwenReranker.load(tmp_path / 'export')
    assert loaded.score(sample) == pair.score(sample)


def test_compiler_never_uses_labels_or_candidate_ids():
    pair, _ = models()
    sample = fixture.sample_records()[0]
    changed = copy.deepcopy(sample)
    changed.pop('supervision')
    changed['candidates'][0]['id'] = 'renamed'
    assert pair.compiler.compile(sample).pairs == pair.compiler.compile(changed).pairs


def test_existing_lm_head_rows_receive_gradients():
    pair, _ = models()
    # Untie the fixture's head to observe its readout gradient independently
    # of input embeddings. The implementation uses this existing head directly.
    pair.backbone.lm_head.weight = torch.nn.Parameter(pair.backbone.lm_head.weight.detach().clone())
    pair.score_tensors(fixture.sample_records())[0].sum().backward()
    gradient = pair.backbone.lm_head.weight.grad
    assert gradient[pair.yes_id].abs().sum() > 0
    torch.testing.assert_close(gradient[pair.no_id], -gradient[pair.yes_id])
    unrelated = [i for i in range(len(pair.tokenizer)) if i not in (pair.yes_id, pair.no_id)]
    assert torch.count_nonzero(gradient[unrelated]) == 0


def test_compiled_pairs_match_model_card_template():
    from cometa.compiler import SYSTEM_PREFIX, ASSISTANT_SUFFIX, DEFAULT_INSTRUCTION
    pair, _ = models()
    sample = fixture.sample_records()[0]
    expected = []
    encode = lambda text: pair.tokenizer.encode(text, add_special_tokens=False)
    for candidate in sample['candidates']:
        body = f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: {sample['input']['query']}\n<Document>: {candidate['text']}"
        expected.append(encode(SYSTEM_PREFIX) + encode(body) + encode(ASSISTANT_SUFFIX))
    assert pair.compiler.compile(sample).pairs == expected


def test_backend_rejection():
    with pytest.raises(NotImplementedError, match='flex'):
        QwenReranker('must-not-download', backend='flex')


@pytest.mark.parametrize('batch_size', [0, -1, 1.5, True])
def test_invalid_pair_batch_size(batch_size):
    with pytest.raises(ValueError, match='pair_batch_size'):
        QwenReranker('must-not-download', pair_batch_size=batch_size)


def test_registry_loads_legacy_yesno_artifacts(tmp_path):
    import json
    from cometa.model_registry import load_model
    pair, _ = models()
    destination = pair.save(tmp_path / 'legacy-export')
    path = destination / 'model_spec.json'
    spec = json.loads(path.read_text())
    # These fields did not exist in format 1.0's original yes/no exports.
    for key in ('architecture_version', 'score_semantics', 'readout'):
        spec.pop(key)
    path.write_text(json.dumps(spec))
    loaded = load_model(destination)
    assert isinstance(loaded, QwenReranker)
    assert loaded.score(fixture.sample_records()) == pair.score(fixture.sample_records())
