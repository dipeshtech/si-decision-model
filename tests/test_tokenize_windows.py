import unittest
from risk_classifier.tokenize_windows import prepare

class FakeFastTokenizer:
    is_fast=True
    model_max_length=8
    def num_special_tokens_to_add(self,pair=False):return 2
    def __call__(self,text,**kwargs):
        assert kwargs['truncation'] is False
        offsets=[(i,i+1) for i,c in enumerate(text) if not c.isspace()]
        return {'input_ids':list(range(len(offsets))),'offset_mapping':offsets}

class TokenizeTests(unittest.TestCase):
    def test_automatic_offsets_budget_and_unknown(self):
        rows={'train':[{'id':'a','subject_id':'a','text':'a b c d e f g','label':'SI5'}]}
        result=prepare(rows,FakeFastTokenizer(),'test-revision',8,1,2)
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['label']=='unknown' for r in result))
        self.assertTrue(all(r['verified_content_tokens']+2<=8 for r in result))

    def test_model_limit_and_overhead(self):
        for budget,reserve in [(9,2),(8,1)]:
            with self.assertRaises(ValueError):prepare({},FakeFastTokenizer(),'r',budget,0,reserve)
