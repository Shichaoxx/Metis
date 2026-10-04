"""Pure report/protocol tests: no torch import, weights or model inference."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'examples' / 'benchmarks'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('metis_compare_task_models', SCRIPTS / 'compare_task_models.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def write_json(path, value):
    path.write_text(json.dumps(value) + '\n')


def fixture(path, *, scores=(0.2, 0.4), scope='full_split', compiler='yesno', max_length=2048,
            source_key='metis_source_sha256'):
    path.mkdir()
    query_ids = [f'q{i}' for i in range(len(scores))]
    contract = {'manifest_sha256': 'a' * 64, 'split_sha256': 'b' * 64, 'qrels_sha256': 'c' * 64,
                'sample_ids': query_ids, 'parameters': {'limit': len(scores) if scope == 'pilot' else None},
                source_key: {'metrics.py': 'd' * 64}}
    inputs = {'version': compiler, 'readout': compiler, 'max_length': max_length}
    records = []
    for ordinal, value in enumerate(scores):
        row_metrics = {'ndcg@10': value, 'map': value / 2, 'mrr': .5, 'recall@10': .4,
                       'candidate_recall': .4, 'judged_positive_count': 5}
        records.append({'id': query_ids[ordinal], 'ordinal': ordinal, 'status': 'ok', 'coverage': 1.0,
                        'scores': [{'candidate_id': 'd1', 'score': 1.0}, {'candidate_id': 'd2', 'score': 0.0}],
                        'metrics': row_metrics})
    (path / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    metrics = {key: sum(r['metrics'][key] for r in records) / len(records) for key in comparison.METRICS}
    metrics.update(queries=len(records), candidates=2 * len(records), scope=scope, full_split_queries=2)
    protocol = {'status': 'completed', 'completed': len(records), 'dataset_id': 'test/data',
                'split': 'validation', 'sample_ids': query_ids, 'scope': scope, 'full_split_queries': 2,
                'k': 10, 'gain': 'linear_trec_eval', 'candidate_policy': 'frozen_manifest_candidates_no_gold_injection',
                'manifest_sha256': 'a' * 64, 'split_sha256': 'b' * 64, 'qrels_sha256': 'c' * 64,
                'model': compiler, 'model_spec': {'compiler': inputs},
                'protocol_sha256': comparison.digest_json(contract),
                'predictions_sha256': comparison.sha256(path / 'predictions.jsonl')}
    write_json(path / 'protocol.json', contract)
    write_json(path / 'evaluation.json', protocol)
    write_json(path / 'metrics.json', metrics)


class TaskReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def pair(self, **right):
        a, b = self.root / 'baseline', self.root / 'candidate'
        fixture(a)
        fixture(b, **right)
        return a, b

    def test_different_compiler_readout_allowed_and_disclosed(self):
        a, b = self.pair(scores=(.3, .5), compiler='mlp_natural')
        result = comparison.compare(a, b)
        self.assertFalse(result['method_comparison']['input_compilers_equal'])
        self.assertAlmostEqual(result['ndcg@10_paired_bootstrap']['mean_delta'], .1)
        self.assertEqual(result['ndcg@10_paired_bootstrap']['resamples'], 10000)
        self.assertEqual(result['ndcg@10_paired_bootstrap']['seed'], 42)
        self.assertTrue(result['selection_allowed'])

    def test_legacy_source_hash_field_compatible_with_metis(self):
        a, b = self.pair(source_key='cometa_source_sha256')
        result = comparison.compare(a, b)
        self.assertAlmostEqual(result['ndcg@10_paired_bootstrap']['mean_delta'], 0.0)

    def test_both_pilots_remain_pilots(self):
        a, b = self.root / 'a', self.root / 'b'
        fixture(a, scores=(.2,), scope='pilot')
        fixture(b, scores=(.3,), scope='pilot', compiler='mlp')
        result = comparison.compare(a, b)
        self.assertEqual(result['scope'], 'pilot')
        self.assertFalse(result['selection_allowed'])
        self.assertIn('PILOT', comparison.markdown(result))

    def test_even_full_length_pilot_cannot_mix_with_full(self):
        a, b = self.pair(scope='pilot')
        with self.assertRaisesRegex(ValueError, 'scope'):
            comparison.compare(a, b)

    def test_token_budget_mismatch_rejected(self):
        a, b = self.pair(max_length=4096)
        with self.assertRaisesRegex(ValueError, 'token budgets'):
            comparison.compare(a, b)

    def test_corrupt_prediction_rejected(self):
        a, b = self.pair()
        with (b / 'predictions.jsonl').open('a') as sink:
            sink.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            comparison.compare(a, b)

    def test_candidate_order_mismatch_rejected_even_when_resealed(self):
        a, b = self.pair()
        rows = [json.loads(s) for s in (b / 'predictions.jsonl').read_text().splitlines()]
        rows[0]['scores'].reverse()
        (b / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        protocol = json.loads((b / 'evaluation.json').read_text())
        protocol['predictions_sha256'] = comparison.sha256(b / 'predictions.jsonl')
        write_json(b / 'evaluation.json', protocol)
        with self.assertRaisesRegex(ValueError, 'Candidate IDs/order'):
            comparison.compare(a, b)

    def test_metric_implementation_mismatch_rejected(self):
        a, b = self.pair()
        contract = json.loads((b / 'protocol.json').read_text())
        contract['metis_source_sha256']['metrics.py'] = 'f' * 64
        write_json(b / 'protocol.json', contract)
        protocol = json.loads((b / 'evaluation.json').read_text())
        protocol['protocol_sha256'] = comparison.digest_json(contract)
        write_json(b / 'evaluation.json', protocol)
        with self.assertRaisesRegex(ValueError, 'Metric source'):
            comparison.compare(a, b)

    def test_failed_run_rejected(self):
        a, b = self.pair()
        protocol = json.loads((b / 'evaluation.json').read_text())
        protocol.update(status='failed')
        write_json(b / 'evaluation.json', protocol)
        with self.assertRaisesRegex(ValueError, 'completed'):
            comparison.compare(a, b)


if __name__ == '__main__':
    unittest.main()
