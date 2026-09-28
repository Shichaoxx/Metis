#!/usr/bin/env python3
"""Bounded real-weight correctness preflight; no benchmark quality claim."""
import argparse
import copy
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import set_seed
from cometa.artifacts import seal_artifact, write_json
from cometa.config import resolve_config
from cometa.model_registry import build_model
from cometa.schema import read_jsonl
from cometa.training import supervised_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    config, _ = resolve_config(args.config)
    set_seed(config['training']['seed'])
    model_cfg = dict(config['model'], dtype='float32', max_length=512, max_tree_tokens=2048)
    model = build_model(model_cfg)
    model.backbone = get_peft_model(model.backbone, LoraConfig(
        task_type=model.peft_task_type, r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=['q_proj','k_proj','v_proj','o_proj'], bias='none'))
    sample = copy.deepcopy(next(read_jsonl(config['data']['train_file'])))
    labels = sample['supervision']['labels']
    positive = [c for c in sample['candidates'] if labels.get(c['id'], 0) > 0][:2]
    negative = [c for c in sample['candidates'] if labels.get(c['id']) == 0][:2]
    sample['candidates'] = positive + negative
    sample['supervision']['labels'] = {c['id']: labels[c['id']] for c in sample['candidates']}
    results, gradients = {}, {}
    model.train()
    for layout in ['pairs', 'tree']:
        model.layout = layout
        model.zero_grad(set_to_none=True)
        scores = model.score_tensors([sample])[0]
        loss = supervised_loss(scores, sample, bce_weight=0.1, pairwise_weight=1.0)
        loss.backward()
        gradients[layout] = {n: p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        results[layout] = {'scores': scores.detach().cpu().tolist(), 'loss': loss.item()}
    score_diff = max(abs(a-b) for a,b in zip(results['pairs']['scores'],results['tree']['scores']))
    max_gradient_diff = 0.0
    assert gradients['pairs'].keys() == gradients['tree'].keys()
    for name in gradients['pairs']:
        a,b = gradients['pairs'][name], gradients['tree'][name]
        torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-3,msg=name)
        max_gradient_diff = max(max_gradient_diff,(a-b).abs().max().item())
    torch.testing.assert_close(torch.tensor(results['pairs']['scores']),torch.tensor(results['tree']['scores']),atol=3e-5,rtol=3e-3)
    model.eval()
    reversed_sample = copy.deepcopy(sample)
    reversed_sample['candidates'].reverse()
    reversed_scores = list(reversed(model.score([reversed_sample])[0]))
    torch.testing.assert_close(torch.tensor(reversed_scores),torch.tensor(results['tree']['scores']),atol=3e-5,rtol=3e-3)
    # This artifact has no optimizer update; it exists only for fresh-process reload checks.
    model.layout = 'pairs'
    artifact = model.save(output / 'initialization-artifact')
    seal_artifact(artifact, task={'kind':'ranking'})
    payload = {'scope':'real-weight forward/gradient/order/reload preflight; not task training or quality',
               'model':model_cfg, 'sample':sample, 'layouts':results,
               'max_score_difference':score_diff,'max_gradient_difference':max_gradient_diff,
               'tree_reverse_max_difference':max(abs(a-b) for a,b in zip(reversed_scores,results['tree']['scores'])),
               'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'head_gradient_nonzero':any(torch.count_nonzero(g).item() for n,g in gradients['pairs'].items() if n.startswith('score_head.')),
               'lora_gradient_nonzero':any(torch.count_nonzero(g).item() for n,g in gradients['pairs'].items() if 'lora_' in n),
               'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'artifact':str(artifact)}
    assert payload['head_gradient_nonzero'] and payload['lora_gradient_nonzero']
    write_json(output/'check.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('sample','model','layouts')}))


if __name__ == '__main__':
    main()
