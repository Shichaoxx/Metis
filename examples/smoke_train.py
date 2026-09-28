"""Tiny random Qwen3 CPU smoke; downloads no pretrained weights or dataset.

After installing the package, run:
    python examples/smoke_train.py --output /tmp/cometa-smoke
Or from the repository: PYTHONPATH=src python examples/smoke_train.py --output /tmp/cometa-smoke
The tiny model checks software behavior, never retrieval quality.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def tiny_components():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(7)
    words = ['[PAD]', '[UNK]', '[EOS]', 'yes', 'no', 'query', 'apple', 'orange',
             'red', 'green', 'fruit', 'blue', 'sky', 'warm', 'cold', 'water',
             'Document', 'Query', 'Instruct', 'Context']
    vocabulary = {word: i for i, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       unk_token='[UNK]', eos_token='[EOS]')
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=512, attention_dropout=0.0,
        tie_word_embeddings=True, pad_token_id=0, eos_token_id=2)
    return Qwen3ForCausalLM(config), tokenizer


def sample_records():
    return [{'schema_version': '1.0', 'id': 'q1', 'task_id': 'rerank',
             'input': {'query': 'apple', 'context': ''},
             'candidates': [{'id': 'd1', 'text': 'red apple fruit'},
                            {'id': 'd2', 'text': 'blue sky'}],
             'supervision': {'kind': 'candidate_labels', 'labels': {'d1': 1, 'd2': 0},
                             'unjudged_policy': 'ignore'}}]


def make_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    backbone, tokenizer = tiny_components()
    base = directory / 'tiny-base'
    backbone.save_pretrained(base)
    tokenizer.save_pretrained(base)
    data = directory / 'train.jsonl'
    data.write_text(''.join(json.dumps(s) + '\n' for s in sample_records()))
    return {'schema_version': '1.0', 'task': {'kind': 'ranking'},
            'model': {'name_or_path': str(base), 'layout': 'tree', 'backend': 'eager',
                      'device': 'cpu', 'dtype': 'float32', 'max_length': 256, 'max_tree_tokens': 512},
            'data': {'train_file': str(data), 'validation_file': str(data)},
            'training': {'max_steps': 1, 'batch_size': 1, 'learning_rate': 1e-3,
                         'save_steps': 1, 'seed': 42, 'tuning': 'full'},
            'objective': {'bce_weight': 1.0, 'pairwise_weight': 0.5}}


def run_smoke(output):
    import torch
    from cometa.training import train
    from cometa.artifacts import seal_artifact
    from cometa.api import Predictor
    output = Path(output)
    torch.set_num_threads(1)
    config = make_fixture(output)
    artifact = train(config, output / 'run')
    seal_artifact(artifact, task=config['task'], source_run=str(output / 'run'))
    predictions = Predictor(artifact).predict(sample_records())
    return {'artifact': str(artifact),
            'scores': [[row['score'] for row in prediction['scores']] for prediction in predictions],
            'predictions': predictions,
            'note': 'Random tiny model: correctness smoke only, no quality claim'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))


if __name__ == '__main__':
    main()
