import unittest
from risk_classifier.coverage import report

class CoverageTests(unittest.TestCase):
    def test_reports_gaps_and_unknown_rules(self):
        result=report({'scope':'SAMPLE','rules':[{'rule_id':'a'},{'rule_id':'b'}]},[{'rule_ids':['a','unknown'],'case_type':'positive','template_family':'x','split':'train','reviewed':False}])
        self.assertEqual(result['unknown_rule_ids'],['unknown'])
        self.assertIn('missing_challenge_family',result['rules']['a']['gaps'])
        self.assertEqual(result['rules']['b']['example_count'],0)
        self.assertFalse(result['semantic_coverage_proven'])

    def test_detects_challenge_overlap(self):
        rows=[{'rule_ids':['a'],'case_type':'context','template_family':'same','split':s} for s in ['train','challenge']]
        result=report({'rules':[{'rule_id':'a'}]},rows)
        self.assertIn('challenge_family_overlaps_train',result['rules']['a']['gaps'])
