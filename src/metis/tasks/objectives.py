"""Supervised objectives for candidate ranking and selection."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def supervised_loss(scores, sample, *, task_kind='ranking', bce_weight=1.0, pairwise_weight=0.5):
    """Query-mean objective. BCE maps graded relevance >0 to positive;
    ranking pairs retain all strict grade comparisons, including 2 > 1.
    Missing judgments are ignored, never implicitly negative.
    """
    supervision = sample.get('supervision')
    if not supervision or supervision.get('kind') != 'candidate_labels':
        raise ValueError('Training requires candidate_labels supervision')
    labels = supervision['labels']
    candidate_ids = [c['id'] for c in sample['candidates']]
    unknown_ids = set(labels) - set(candidate_ids)
    if unknown_ids:
        raise ValueError('Supervision contains unknown candidate IDs')
    known = [i for i, cid in enumerate(candidate_ids) if cid in labels]
    if supervision.get('unjudged_policy') == 'error' and len(known) != len(candidate_ids):
        raise ValueError('Fully judged sample is missing candidate labels')
    if not known:
        raise ValueError('A training sample must contain at least one judged candidate')
    values = [float(labels[candidate_ids[i]]) for i in known]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Relevance labels must be finite and nonnegative')
    # Keep the reduction and pairwise margin in FP32 under BF16 autocast.
    selected = scores[known].float()
    grades = torch.tensor(values, device=scores.device, dtype=torch.float32)
    if task_kind == 'single_choice':
        if len(known) != len(candidate_ids) or int((grades > 0).sum()) != 1:
            raise ValueError('single_choice needs complete judgments and exactly one positive')
        target = (grades > 0).nonzero(as_tuple=False)[0, 0].reshape(1)
        return F.cross_entropy(selected[None], target)
    if task_kind not in ('ranking', 'multi_label'):
        raise ValueError(f'Unsupported task kind: {task_kind}')
    bce = F.binary_cross_entropy_with_logits(selected, (grades > 0).to(selected.dtype))
    if task_kind == 'multi_label':
        return bce_weight * bce
    preferred = grades[:, None] > grades[None, :]
    differences = selected[:, None] - selected[None, :]
    pairwise = F.softplus(-differences[preferred]).mean() if preferred.any() else selected.sum() * 0
    return bce_weight * bce + pairwise_weight * pairwise
