"""Behavioral contracts for the independently trained scalar readout.

These tiny offline tests verify software behavior, not pretrained quality.
Run on the training host along with the existing model/trainer suite.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from cometa.compiler import SCORE_HEAD_COMPILER_VERSION
from cometa.model import QwenReranker, QwenScoreHead
from cometa.model_registry import MODEL_REGISTRY, build_model, load_model


fixture_spec = importlib.util.spec_from_file_location(
    'score_head_fixture', Path(__file__).parents[1] / 'examples/smoke_train.py')
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)


def score_head(backend='eager', **kwargs):
    torch.set_num_threads(1)
    language_model, tokenizer = fixture.tiny_components()
    return QwenScoreHead.from_components(language_model.model, tokenizer,
        backend=backend, max_length=128, max_tree_tokens=512, head_hidden_size=11, **kwargs)


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_pairs_tree_match_scores_and_all_parameter_gradients(backend):
    pair = score_head(backend)
    tree = copy.deepcopy(pair)
    tree.layout = 'tree'
    samples = fixture.sample_records()
    left, right = pair.score_tensors(samples)[0], tree.score_tensors(samples)[0]
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    weights = torch.tensor([1.0, -0.7])
    (left * weights).sum().backward()
    (right * weights).sum().backward()
    for (name, parameter), (other_name, other) in zip(pair.named_parameters(), tree.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, atol=4e-6, rtol=3e-4, msg=name)
    assert sum(p.grad.abs().sum() for p in pair.score_head.parameters()) > 0
    assert pair.backbone.get_input_embeddings().weight.grad.abs().sum() > 0


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_tree_isolates_other_candidates_and_preserves_permuted_ids(backend):
    model = score_head(backend, layout='tree')
    sample = fixture.sample_records()[0]
    expected = model.score([sample])[0]
    permuted = copy.deepcopy(sample)
    permuted['candidates'].reverse()
    actual = model.score([permuted])[0]
    assert [c['id'] for c in permuted['candidates']] == ['d2', 'd1']
    torch.testing.assert_close(torch.tensor(actual[::-1]), torch.tensor(expected), atol=2e-6, rtol=2e-5)
    changed = copy.deepcopy(sample)
    changed['candidates'][1]['text'] = 'green orange water ' * 6
    actual = model.score([changed])[0]
    assert actual[0] == pytest.approx(expected[0], abs=2e-6)
    # The test checks isolation, not an accidentally constant scoring function.
    assert abs(actual[1] - expected[1]) > 1e-6


def test_tree_chunking_and_truncation_preserve_head_scores():
    model = score_head(layout='tree')
    sample = fixture.sample_records()[0]
    sample['candidates'] = [
        {'id': f'c{i}', 'text': ('red apple fruit ' if i % 2 else 'blue sky ') * 100}
        for i in range(4)]
    expected = model.score([sample])[0]
    assert all(model.last_input_metadata[0]['candidate_truncated'].values())
    model.max_tree_tokens = model.max_length
    actual = model.score([sample])[0]
    assert model.last_input_metadata[0]['physical_chunks'] == 4
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=2e-6, rtol=2e-5)


def test_compiler_keeps_supervision_out_and_has_no_generation_template():
    model = score_head()
    sample = fixture.sample_records()[0]
    changed = copy.deepcopy(sample)
    changed['supervision']['labels'] = {'private-gold': 3}
    changed['candidates'][0]['id'] = 'private-id'
    changed['metadata'] = {'answer': 'private-gold'}
    assert model.compiler.compile(sample).pairs == model.compiler.compile(changed).pairs
    spec = model.compiler.spec()
    assert spec['version'] == SCORE_HEAD_COMPILER_VERSION
    assert spec['readout'] == 'last_relevance_suffix_token_scalar_logit'
    assert '<|im_start|>' not in spec['system_prefix'] + spec['assistant_suffix']
    assert '"yes" or "no"' not in spec['system_prefix']
    assert model.backbone.get_output_embeddings() is None
    assert not any('lm_head' in name for name, _ in model.named_parameters())


def test_factory_and_full_export_restore_trained_head_and_backbone(tmp_path):
    config = fixture.make_fixture(tmp_path / 'fixture')['model']
    config.update(adapter='qwen3_score_head', head_hidden_size=13, instruction=None)
    model = build_model(config)
    assert isinstance(model, QwenScoreHead)
    assert model.head_hidden_size == 13
    assert model.peft_task_type == 'FEATURE_EXTRACTION'
    assert model.score_semantics == 'raw_relevance_logit'
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model.score_tensors(fixture.sample_records())[0], torch.tensor([1., 0.]))
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(parameter, original[name]) for name, parameter in model.named_parameters()
               if name.startswith('score_head.'))
    assert any(not torch.equal(parameter, original[name]) for name, parameter in model.named_parameters()
               if name.startswith('backbone.'))
    destination = model.save(tmp_path / 'export')
    spec = json.loads((destination / 'model_spec.json').read_text())
    assert spec['family'] == 'qwen3_score_head'
    assert spec['readout']['hidden_size'] == 13
    assert spec['compiler_version'] == SCORE_HEAD_COMPILER_VERSION
    assert (destination / 'score_head.safetensors').is_file()
    loaded = load_model(destination)
    assert isinstance(loaded, QwenScoreHead)
    assert loaded.score(fixture.sample_records()) == model.score(fixture.sample_records())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='Unsupported model'):
        QwenReranker.load(destination)


def test_lora_updates_and_restores_adapter_and_independent_head(tmp_path):
    peft = pytest.importorskip('peft')
    config = fixture.make_fixture(tmp_path / 'fixture')['model']
    config.update(adapter='qwen3_score_head', head_hidden_size=9)
    model = build_model(config)
    model.backbone = peft.get_peft_model(model.backbone, peft.LoraConfig(
        task_type=model.peft_task_type, r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none'))
    assert all(p.requires_grad for p in model.score_head.parameters())
    before_head = {name: p.detach().clone() for name, p in model.score_head.named_parameters()}
    before_backbone = {name: p.detach().clone() for name, p in model.backbone.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model.score_tensors(fixture.sample_records())[0], torch.tensor([1., 0.]))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.backbone.named_parameters() if 'lora_' in name)
    assert all(p.grad is None for name, p in model.backbone.named_parameters() if 'lora_' not in name)
    optimizer.step()
    assert any(not torch.equal(p, before_head[name]) for name, p in model.score_head.named_parameters())
    assert any(not torch.equal(p, before_backbone[name]) for name, p in model.backbone.named_parameters()
               if 'lora_' in name)
    destination = model.save(tmp_path / 'export')
    spec = json.loads((destination / 'model_spec.json').read_text())
    assert spec['tuning'] == 'lora' and spec['base_dependency_sha256']
    loaded = load_model(destination)
    torch.testing.assert_close(torch.tensor(loaded.score(fixture.sample_records())),
                               torch.tensor(model.score(fixture.sample_records())), rtol=0, atol=0)
    for name, value in model.score_head.state_dict().items():
        torch.testing.assert_close(value, loaded.score_head.state_dict()[name], rtol=0, atol=0)
    # A missing trained head must not silently return a newly initialized head.
    (destination / 'score_head.safetensors').unlink()
    with pytest.raises(FileNotFoundError):
        load_model(destination)


def test_model_registry_and_readout_configuration_fail_before_download():
    assert MODEL_REGISTRY.names() == ['qwen3_score_head', 'qwen3_yesno']
    with pytest.raises(ValueError, match='Unknown registration'):
        build_model({'name_or_path': 'must-not-download', 'adapter': 'unknown'})
    with pytest.raises(ValueError, match='only valid'):
        build_model({'name_or_path': 'must-not-download', 'adapter': 'qwen3_yesno', 'head_hidden_size': 4})
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='head_hidden_size'):
            build_model({'name_or_path': 'must-not-download', 'adapter': 'qwen3_score_head',
                         'head_hidden_size': value})


def test_default_head_dimension_and_no_lm_backbone_injection():
    backbone, tokenizer = fixture.tiny_components()
    model = QwenScoreHead.from_components(copy.deepcopy(backbone.model), tokenizer, max_length=128)
    assert model.head_hidden_size == backbone.config.hidden_size // 4
    with pytest.raises(ValueError, match='without an LM head'):
        QwenScoreHead.from_components(backbone, tokenizer, max_length=128)
