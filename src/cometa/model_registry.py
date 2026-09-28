"""Explicit model adapters used by training and inference artifact loading."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .model import QwenReranker, QwenScoreHead
from .registry import Registry


MODEL_REGISTRY = Registry()
MODEL_REGISTRY.register(QwenReranker.family, QwenReranker)
MODEL_REGISTRY.register(QwenScoreHead.family, QwenScoreHead)


def build_model(model_cfg):
    """Build a trainable adapter from the resolved ``model`` config section.

    Omitting adapter keeps legacy recipes on qwen3_yesno. A new ScoreHead
    starts randomly initialized; loading a trained head requires load_model.
    """
    if not isinstance(model_cfg, Mapping):
        raise ValueError('model config must be a mapping')
    options = dict(model_cfg)
    adapter = options.pop('adapter', 'qwen3_yesno')
    model_class = MODEL_REGISTRY.get(adapter)
    if 'name_or_path' not in options:
        raise ValueError('model.name_or_path is required')
    source = options.pop('name_or_path')
    head_size = options.pop('head_hidden_size', None)
    if adapter == 'qwen3_score_head':
        options['head_hidden_size'] = head_size
    elif head_size is not None:
        raise ValueError('head_hidden_size is only valid for qwen3_score_head')
    return model_class(source, **options)


def load_model(artifact, device='cpu', dtype='float32'):
    """Dispatch an exported model by its recorded family, never by a default.

    Predictor performs artifact integrity verification before this call;
    direct model tests may load an unsealed save() directory.
    """
    spec = json.loads((Path(artifact) / 'model_spec.json').read_text())
    model_class = MODEL_REGISTRY.get(spec.get('family'))
    return model_class.load(artifact, device=device, dtype=dtype)
