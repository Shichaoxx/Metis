#!/usr/bin/env python3
"""Use an exported task model as a retrieval tool in a deterministic harness.

Selects the first validation query by file order, never by its relevance labels.
The downstream step assembles a cited evidence context from returned IDs. This
is a workflow integration example, not an LLM answer-quality benchmark.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

from agent_tool import DecisionTool
from cometa.artifacts import write_json
from cometa.schema import load_manifest, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--precision', choices=['none', 'bf16'], default='none')
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite a workflow trace')
    manifest = load_manifest(args.manifest)
    sample = next(read_jsonl(manifest['splits']['validation']['path'], require_labels=False))
    # Supervision, gold qrels and metadata never enter the tool's input.
    tool = DecisionTool(args.artifact, device=args.device, dtype=args.dtype)
    if args.precision == 'bf16':
        import torch
        if torch.device(args.device).type != 'cuda':
            raise ValueError('BF16 workflow autocast requires CUDA')
        compute_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        compute_context = nullcontext()
    with compute_context:
        result = tool.rank_candidates(sample['input']['query'], sample['candidates'],
                                      context=sample['input'].get('context', ''),
                                      request_id=sample['id'], top_k=args.top_k)
    documents = {item['id']: item['text'] for item in sample['candidates']}
    evidence = [{'candidate_id': cid, 'text': documents[cid]} for cid in result['selected_ids']]
    write_json(output, {
        'scope': 'trained-model tool integration; no generative answer or downstream quality claim',
        'query_selection': 'first validation query in frozen manifest order',
        'dtype': args.dtype, 'precision': args.precision,
        'query': sample['input']['query'],
        'trace': [{'step': 'rank_candidates', 'result': result},
                  {'step': 'build_evidence_context', 'documents': evidence}],
        'evidence_context': '\n\n'.join(f"[{item['candidate_id']}] {item['text']}" for item in evidence),
    })
    print(json.dumps({'output': str(output), 'selected_ids': result['selected_ids'],
                      'score_semantics': result['score_semantics']}))


if __name__ == '__main__':
    main()
