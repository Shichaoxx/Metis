"""Offline checks for candidate pooling and common-budget readout experiments."""
import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from metis.model import QwenScoreHead


ROOT = Path(__file__).parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


research = load_script('boolq_readout_model', ROOT / 'examples/research/boolq_readout_model.py')
fixture = load_script('boolq_readout_fixture', ROOT / 'examples/smoke_train.py')


def model(kind='last_relevance', max_length=160):
    torch.set_num_threads(1)
    backbone, tokenizer = fixture.tiny_components()
    scorer = QwenScoreHead.from_components(backbone.model, tokenizer, backend='eager',
                                          max_length=max_length, instruction='Judge the evidence')
    return research.BoolQReadoutModel(scorer, readout=kind)


def samples():
    return [{'id': 'first', 'input': {'query': 'apple', 'context': 'green fruit'},
             'candidates': [{'id': 'evidence', 'text': 'red apple fruit orange'}]},
            {'id': 'second', 'input': {'query': 'sky'},
             'candidates': [{'id': 'evidence', 'text': 'blue sky'}]}]


def test_candidate_mask_excludes_request_suffix_and_padding():
    value = model('candidate_attention')
    batch = value.prepare_batch(samples())
    assert batch['candidate_mask'].sum(dim=1).tolist() == [4, 2]
    assert not (batch['candidate_mask'] & ~batch['attention_mask'].bool()).any()
    for row, sample in enumerate(samples()):
        compiled = value.compiler.compile(sample)
        prefix = len(compiled.prefix_ids)
        suffix = len(value.compiler.suffix_ids)
        length = batch['lengths'][row]
        assert not batch['candidate_mask'][row, :prefix].any()
        assert not batch['candidate_mask'][row, length - suffix:].any()
        decoded = value.tokenizer.decode(batch['input_ids'][row][batch['candidate_mask'][row]])
        assert decoded == sample['candidates'][0]['text']
    assert batch['boundary_merge_tokens'] == [0, 0]
    assert torch.equal(value.pool_query, torch.zeros_like(value.pool_query))


@pytest.mark.parametrize('kind', research.READOUT_NAMES)
def test_batched_scores_have_expected_shape_and_nonzero_gradients(kind):
    value = model(kind)
    scores = value(samples())
    assert scores.shape == (2,)
    assert scores.dtype == torch.float32 and torch.isfinite(scores).all()
    torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.tensor([1., 0.])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in value.backbone.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in value.score_head.parameters())
    if kind == 'candidate_attention':
        assert value.pool_query.grad is not None and value.pool_query.grad.abs().sum() > 0
    else:
        assert value.pool_query is None
    assert value(samples()[:1]).shape == (1,)


@pytest.mark.parametrize('kind', research.READOUT_NAMES)
def test_batch_order_and_other_sample_content_do_not_change_a_sample(kind):
    value = model(kind).eval()
    original = samples()
    with torch.no_grad():
        expected = value(original)
        reversed_scores = value(list(reversed(original)))
        changed = copy.deepcopy(original)
        changed[1]['candidates'][0]['text'] = 'warm cold water blue sky ' * 5
        actual = value(changed)
        standalone = value(original[:1])
    torch.testing.assert_close(expected, reversed_scores.flip(0), atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(expected[0], actual[0], atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(expected[:1], standalone, atol=2e-7, rtol=2e-6)
    assert abs(float(actual[1] - expected[1])) > 1e-6


def test_marker_reserves_a_common_candidate_budget_without_new_embeddings():
    baseline = model()
    original_tokenizer = baseline.tokenizer

    class LongerDecisionTokenizer:
        # Exercise unequal marker lengths while reusing only existing token IDs.
        def __getattr__(self, name):
            return getattr(original_tokenizer, name)

        def encode(self, text, **kwargs):
            tokens = original_tokenizer.encode(text, **kwargs)
            return tokens + [original_tokenizer.eos_token_id] if text == research.DECISION_SUFFIX else tokens

        def __call__(self, *args, **kwargs):
            return original_tokenizer(*args, **kwargs)

    tokenizer = LongerDecisionTokenizer()
    compiler = research.CommonBudgetCompiler(tokenizer, 70, 'Judge the evidence', 'last_relevance')
    marker = research.CommonBudgetCompiler(tokenizer, 70, 'Judge the evidence', 'decision_marker')
    attention = research.CommonBudgetCompiler(tokenizer, 70, 'Judge the evidence', 'candidate_attention')
    sample = samples()[0]
    sample['candidates'][0]['text'] = 'red apple fruit ' * 100
    compiled = [item.compile(sample) for item in (compiler, marker, attention)]
    bodies = [item.pairs[0][:-len(c.suffix_ids)] for item, c in zip(compiled, (compiler, marker, attention))]
    assert bodies[0] == bodies[1] == bodies[2]
    assert compiled[0].pairs == compiled[2].pairs
    assert all(item.truncated == (True,) for item in compiled)
    assert len(compiled[1].pairs[0]) == 70
    assert len(compiled[0].pairs[0]) == 69
    assert compiled[1].pairs[0][-len(marker.suffix_ids):] == list(tokenizer.encode(
        research.DECISION_SUFFIX, add_special_tokens=False))
    assert baseline.backbone.get_input_embeddings().num_embeddings == len(original_tokenizer)
    assert marker.spec()['reserved_suffix_tokens'] == len(marker.suffix_ids)
    with pytest.raises(ValueError, match='no document budget'):
        research.CommonBudgetCompiler(tokenizer, 1, 'Judge the evidence', 'decision_marker').compile(sample)


def test_initialization_is_shared_and_uses_only_a_supplied_training_prior():
    values = [model(kind) for kind in research.READOUT_NAMES]
    for value in values:
        random_state = torch.get_rng_state().clone()
        value.initialize_readout(0.6, seed=93)
        assert torch.equal(torch.get_rng_state(), random_state)
    for left, right in zip(values, values[1:]):
        for a, b in zip(left.score_head.parameters(), right.score_head.parameters()):
            assert torch.equal(a, b)
    for value in values:
        scores = value(samples())
        torch.testing.assert_close(scores.sigmoid(), torch.full((2,), 0.6), rtol=0, atol=1e-7)
        assert value.readout_spec()['head_parameters'] == 8705
    assert values[2].readout_spec()['pooling_parameters'] == 32
    assert all('not a production v1' in value.readout_spec()['artifact_scope'] for value in values)
    with pytest.raises(ValueError, match='finite probability'):
        values[0].initialize_readout(float('nan'))


def test_real_token_boundary_merge_is_included_and_reported():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast, Qwen3ForCausalLM
    # BPE joins the prefix's final space with the candidate's first word.
    # Separately encoding the candidate would produce different token IDs.
    characters = sorted(set(''.join(chr(i) for i in range(32, 127)) + '\n'))
    tokens = ['[PAD]', '[UNK]', '[EOS]'] + characters + [' r', ' re', ' red']
    backend = Tokenizer(models.BPE({token: i for i, token in enumerate(tokens)},
                                  [(' ', 'r'), (' r', 'e'), (' re', 'd')], unk_token='[UNK]'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       unk_token='[UNK]', eos_token='[EOS]')
    original, _ = fixture.tiny_components()
    config = copy.deepcopy(original.config)
    config.vocab_size = len(tokenizer)
    backbone = Qwen3ForCausalLM(config)
    scorer = QwenScoreHead.from_components(backbone.model, tokenizer, max_length=512,
                                          instruction='Judge evidence')
    value = research.BoolQReadoutModel(scorer, 'candidate_attention')
    batch = value.prepare_batch(samples()[:1])
    pooled = batch['input_ids'][0][batch['candidate_mask'][0]]
    assert pooled[0] == tokenizer.convert_tokens_to_ids(' red')
    assert batch['boundary_merge_tokens'] == [1]
    assert value.last_input_metadata[0]['candidate_boundary_merge_tokens'] == 1
    assert not batch['candidate_mask'][0, :len(value.compiler.compile(samples()[0]).prefix_ids)].any()
    assert value.tokenizer.decode(pooled).replace(' ', '') == 'redapplefruitorange'


def test_attention_rejects_empty_body_and_multiple_candidates():
    value = model('candidate_attention')
    sample = samples()[0]
    sample['candidates'][0]['text'] = ''
    with pytest.raises(ValueError, match='candidate-body token'):
        value([sample])
    sample['candidates'].append({'id': 'other', 'text': 'blue sky'})
    with pytest.raises(ValueError, match='exactly one candidate'):
        value([sample])


def test_lora_only_trains_adapters_and_readout_with_checkpointing():
    pytest.importorskip('peft')
    value = research.configure_lora(model('candidate_attention'), rank=2, alpha=4)
    value.train()
    scores = value(samples())
    torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.tensor([1., 0.])).backward()
    for name, parameter in value.backbone.named_parameters():
        assert parameter.requires_grad == ('lora_' in name)
        if 'lora_' not in name:
            assert parameter.grad is None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in value.backbone.named_parameters() if 'lora_' in name)
    assert value.pool_query.grad is not None and value.pool_query.grad.abs().sum() > 0
    assert all(parameter.dtype == torch.float32 for parameter in value.parameters())
