import copy
import json
import tempfile
import unittest
from pathlib import Path
from risk_classifier.data import taxonomy, load_records, validate, split_records, assert_disjoint
from risk_classifier.evaluation import select_thresholds, evaluate, flag, validate_predictions
from risk_classifier.smoke import run
from risk_classifier.cli import main

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.spec = taxonomy(ROOT / 'configs/si_draft.json')
        self.spec['labels'] = ['SI3','SI4','SI5']
        self.spec['definitions'].pop('NO_RISK')
        self.rows = load_records(ROOT / 'examples/synthetic.jsonl', self.spec)

    def test_unknown_is_explicit(self):
        row = copy.deepcopy(self.rows[0]); del row['label']
        with self.assertRaises(ValueError): validate([row], self.spec)
        row['label'] = 'unknown'; validate([row], self.spec)

    def test_evidence_rejects_boolean_and_missing_state(self):
        row = copy.deepcopy(self.rows[0]); row['evidence']['plan'] = False
        with self.assertRaises(ValueError): validate([row], self.spec)

    def test_duplicates_fail(self):
        with self.assertRaises(ValueError): validate([self.rows[0]] * 2, self.spec)

    def test_subject_splits(self):
        splits = split_records(self.rows)
        assert_disjoint(splits)
        self.assertEqual(sum(map(len, splits.values())), 24)
        self.assertEqual(splits, split_records(self.rows))

    def test_linked_families_stay_together(self):
        self.rows[0]['template_family'] = self.rows[1]['template_family'] = 'family'
        result = split_records(self.rows)
        self.assertTrue(any(self.rows[0] in rows and self.rows[1] in rows for rows in result.values()))

    def test_identical_text_leak_fails(self):
        second = copy.deepcopy(self.rows[1]); second['text'] = self.rows[0]['text']
        with self.assertRaises(ValueError): assert_disjoint({'a': [self.rows[0]], 'b': [second]})

    def test_small_dataset_rejected(self):
        with self.assertRaises(ValueError): split_records(self.rows[:3])

    def test_smoke_updates_weights(self):
        result = run()
        self.assertLess(result['loss_after'], result['loss_before'])
        self.assertFalse(result['qwen_loaded'])

    def test_draft_always_abstains(self):
        self.assertEqual(flag([.99,.005,.005], self.spec['labels'], {'SI3': .5})['flag'], 'INSUFFICIENT_EVIDENCE')

    def test_flags_only(self):
        result = flag([.99,.005,.005], self.spec['labels'], {'SI3': .5}, approved=True)
        self.assertEqual(result, {'flag': 'SI3', 'human_review_required': True})

    def test_nan_and_non_normalized_rejected(self):
        for scores in ([float('nan'),0,1], [.8,.8,.8], [-.1,.5,.6]):
            with self.assertRaises(ValueError): flag(scores, self.spec['labels'], {})

    def test_unsupported_class_abstains(self):
        rows = self.rows[:1]
        pred = [{'id': rows[0]['id'], 'probabilities': [.8,.1,.1]}]
        thresholds = select_thresholds(rows, pred, self.spec['labels'])
        self.assertIsNone(thresholds['SI5'])

    def test_metrics_and_unknowns(self):
        rows = self.rows[:4]
        pred = [{'id': r['id'], 'probabilities': [.8,.1,.1]} for r in rows]
        result = evaluate(rows, pred, self.spec['labels'], {'SI3':.5}, approved=True)
        self.assertEqual(result['unknown_records'], 1)
        self.assertAlmostEqual(result['per_class']['SI3']['precision'], 1/3)
        self.assertEqual(result['per_class']['SI5']['recall'], 0)

    def test_prediction_ids_must_match(self):
        with self.assertRaises(ValueError): select_thresholds(self.rows[:1], [], self.spec['labels'])

    def test_cli_calibration_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            predictions = d/'predictions.jsonl'
            predictions.write_text('\n'.join(json.dumps({'id':r['id'], 'probabilities':[.1,.7,.1,.1]}) for r in self.rows))
            common = ['--taxonomy', str(ROOT/'configs/si_draft.json'), '--data', str(ROOT/'examples/synthetic.jsonl'), '--predictions', str(predictions)]
            main(['calibrate', *common, '--output', str(d/'cal.json')])
            with self.assertRaises(SystemExit) as raised:
                main(['evaluate', *common, '--calibration', str(d/'cal.json'), '--output', str(d/'eval.json')])
            self.assertEqual(raised.exception.code, 2)

if __name__ == '__main__':
    unittest.main()

class KevTests(unittest.TestCase):
    def setUp(self):
        self.spec = taxonomy(ROOT / 'configs/si_draft.json')
        self.spec['labels'] = ['SI3','SI4','SI5']
        self.spec['definitions'].pop('NO_RISK')
        self.rows = load_records(ROOT / 'examples/synthetic.jsonl', self.spec)

    def test_draft_export_blocked(self):
        from risk_classifier.kev import export_records
        with self.assertRaises(ValueError): export_records(self.rows, self.spec)

    def test_choice_export_preserves_names_and_skips_unknown(self):
        from risk_classifier.kev import export_records
        self.spec.update(status='approved', unresolved=[], no_risk_label='NONE')
        self.spec['labels'].append('NONE')
        self.spec['definitions']['NONE'] = 'Test-only no qualifying content.'
        exported, skipped = export_records(self.rows, self.spec)
        self.assertEqual(skipped, 6)
        self.assertEqual(exported[0]['questions']['severity']['type'], 'choice')
        self.assertEqual(exported[0]['questions']['severity']['label'], 'SI3')
        self.assertNotIn('subject_id', exported[0])

    def test_binary_export_uses_actual_boolean(self):
        from risk_classifier.kev import export_records
        spec = {'status':'approved','unresolved':[], 'mode':'multilabel','labels':['A'], 'definitions':{'A':'Test definition'}}
        rows = [{'id':'a','subject_id':'s','text':'abstract fixture','labels':{'A':'false'}}]
        exported, skipped = export_records(rows,spec)
        self.assertIs(exported[0]['questions']['A']['label'], False)

    def test_train_plan_does_not_execute(self):
        from risk_classifier.kev import train_plan
        self.spec.update(status='approved', unresolved=[], no_risk_label='NONE')
        self.spec['labels'].append('NONE'); self.spec['definitions']['NONE'] = 'Test'
        result = train_plan(ROOT/'examples/synthetic.jsonl', self.spec)
        self.assertFalse(result['executed'])
        self.assertEqual(result['argv_template'][:3], ['python','-m','kev.train'])
        self.assertNotIn('--init_from', result['argv_template'])

class AddedTests(unittest.TestCase):
    def test_threshold_matches_argmax_emission(self):
        labels=['A','B','C']
        rows=[{'id':'1','label':'A'},{'id':'2','label':'B'}]
        # A's only true case is NOT emitted as A; fitting all marginal scores
        # could incorrectly use it to justify A's precision threshold.
        pred=[{'id':'1','probabilities':[.4,.5,.1]},{'id':'2','probabilities':[.4,.3,.3]}]
        self.assertIsNone(select_thresholds(rows,pred,labels,minimum_precision=.5)['A'])

    def test_teacher_and_human_provenance_preserved(self):
        from risk_classifier.teacher_import import normalize
        spec=taxonomy(ROOT/'configs/si_draft.json')
        row={'example_id':'x','participant_group':'p','conversation_group':'c','text':'abstract fixture','teacher_label':'SI3','human_label':'NO_RISK'}
        result=normalize([row],spec,'testhash','v1')[0]
        self.assertEqual(result['teacher_label'],'SI3')
        self.assertEqual(result['label'],'NO_RISK')
        self.assertTrue(result['review_needed'])

    def test_teacher_confidence_is_not_soft_target(self):
        from risk_classifier.teacher_import import normalize
        spec=taxonomy(ROOT/'configs/si_draft.json')
        row={'example_id':'x','participant_group':'p','conversation_group':'c','text':'abstract fixture','teacher_label':'SI3','teacher_probabilities':[.1,.7,.1,.1]}
        with self.assertRaises(ValueError): normalize([row],spec,'hash','v1')

    def test_baseline_json_roundtrip(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest('Optional baseline dependencies not installed')
        from risk_classifier.baseline import fit, predict
        rows=[{'id':str(i),'text':phrase,'label':label,'source_type':'synthetic'} for i,(phrase,label) in enumerate([('alpha one','A'),('beta two','B'),('gamma three','C'),('delta four','D')]*3)]
        model=fit(rows,['A','B','C','D'])
        predictions=predict(json.loads(json.dumps(model)),rows)
        self.assertEqual(len(predictions),len(rows))
        self.assertAlmostEqual(sum(predictions[0]['probabilities']),1)

class StratifiedTests(unittest.TestCase):
    def test_stratified_groups_cover_all_classes(self):
        rows=[{'id':f'{c}{i}','subject_id':f'{c}{i}','text':f'unique {c} {i}','label':c,'template_family':f'{c}{i}'} for c in ['A','B','C','D'] for i in range(12)]
        splits=split_records(rows,stratify=True)
        assert_disjoint(splits)
        for records in splits.values(): self.assertEqual({r['label'] for r in records},{'A','B','C','D'})

    def test_stratified_insufficient_groups_fails(self):
        rows=[{'id':str(i),'subject_id':str(i),'text':f'unique {i}','label':'A' if i<6 else 'B'} for i in range(12)]
        with self.assertRaises(ValueError): split_records(rows,stratify=True)
