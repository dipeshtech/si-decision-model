import copy
import json
import tempfile
import unittest
from pathlib import Path
from risk_classifier.llm_generate import run


def configuration():
    return {'provider': 'mock', 'model': 'mock', 'seed': 42, 'difficulty': 'mixed',
            'min_chars': 10, 'max_chars': 1000, 'codebook_version': 'sample-only-v0',
            'rules': [{'rule_id': 'context', 'class_counts': {'NO_RISK': 2, 'SI3': 1}}]}


def live_configuration():
    config = configuration()
    config.update(provider='openai_compatible', model='operator-selected',
                  base_url='https://example.org/v1', allowed_hosts=['example.org'], credential_env='TEST_SYNTH_KEY',
                  max_calls=2, max_input_tokens=10000, max_output_tokens=200, budget_usd=1.0,
                  input_usd_per_million=1.0, output_usd_per_million=2.0, public_synthetic_inputs_only=True)
    return config


class GenerationTests(unittest.TestCase):
    def test_default_dry_run_has_no_files_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'out.jsonl'
            result = run(configuration(), out, transport=lambda *args: self.fail('network called'))
            self.assertEqual(result['network_calls'], 0)
            self.assertFalse(out.exists())

    def test_mock_resume_is_idempotent_and_unreviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'out.jsonl'
            run(configuration(), out, execute=True)
            first = out.read_text()
            result = run(configuration(), out, execute=True)
            self.assertEqual(first, out.read_text())
            self.assertEqual(result['written_records'], 3)
            self.assertTrue(all(not json.loads(x)['reviewed'] for x in first.splitlines()))
            changed = configuration(); changed['seed'] = 99
            with self.assertRaises(ValueError): run(changed, out, execute=True)

    def test_live_requires_budget_and_safe_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            for key, value in [('budget_usd', None), ('max_calls', True), ('base_url', 'http://example.org/v1'),
                               ('public_synthetic_inputs_only', False), ('allowed_hosts', [])]:
                config = live_configuration(); config[key] = value
                with self.assertRaises(ValueError): run(config, Path(directory) / 'out', execute=True)

    def test_budget_cap_stops_before_call(self):
        config = live_configuration(); config['budget_usd'] = 0.000001
        with tempfile.TemporaryDirectory() as directory:
            result = run(config, Path(directory) / 'out', execute=True, transport=lambda *args: self.fail('called'))
            self.assertEqual(result['calls'], 0)
            self.assertFalse(result['complete'])

    def test_failed_call_reserved_and_not_retried(self):
        config = live_configuration(); config['max_calls'] = 1
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'out'
            def fail(*args): raise TimeoutError('do not log this response')
            with self.assertRaises(RuntimeError): run(config, out, execute=True, transport=fail)
            ledger = json.loads(Path(str(out) + '.ledger.json').read_text())
            self.assertEqual(ledger['calls'], 1)
            self.assertGreater(ledger['reserved_usd'], 0)
            self.assertNotIn('do not log', json.dumps(ledger))
            result = run(config, out, execute=True, transport=lambda *args: self.fail('retry'))
            self.assertEqual(result['calls'], 1)

    def test_response_schema_and_duplicate_guard(self):
        config = live_configuration()
        def response(config, prompt):
            request = json.loads(prompt.split('\n', 1)[1])
            return ({'text': 'An entirely invented example text.', 'label': request['requested_label'],
                     'rule_ids': [request['rule']['rule_id']], 'case_type': request['case_type']},
                    {'prompt_tokens': 100, 'completion_tokens': 20})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'out'
            with self.assertRaises(RuntimeError): run(config, out, execute=True, transport=response)
            self.assertEqual(len(out.read_text().splitlines()), 1)

    def test_reported_usage_over_cap_fails_closed(self):
        config = live_configuration()
        def response(*args): return {}, {'prompt_tokens': 10001, 'completion_tokens': 1}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError): run(config, Path(directory) / 'out', execute=True, transport=response)

    def test_duplicate_rules_rejected(self):
        config = configuration(); config['rules'].append(copy.deepcopy(config['rules'][0]))
        with self.assertRaises(ValueError): run(config, 'unused')
