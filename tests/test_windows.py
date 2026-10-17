import unittest
from risk_classifier.windows import window_records

class WindowTests(unittest.TestCase):
    def row(self):
        return {'id':'a','subject_id':'p','conversation_id':'c','template_family':'f','text':'a b c d e f g','label':'SI5','split':'train','tokenizer_revision':'test-character-offsets-not-real-model','token_offsets':[[i,i+1] for i in range(0,13,2)]}

    def test_budget_overlap_and_no_inherited_labels(self):
        row=self.row(); windows=window_records([row],5,1,1)
        self.assertEqual([(r['token_start'],r['token_end']) for r in windows],[(0,4),(3,7)])
        for w in windows:
            self.assertEqual(w['label'],'unknown');self.assertEqual(w['subject_id'],'p')
            self.assertEqual(w['template_family'],'f');self.assertLessEqual(w['content_tokens']+w['reserved_tokens'],5)
            self.assertEqual(w['text'],row['text'][w['char_start']:w['char_end']])

    def test_split_required(self):
        row=self.row();del row['split']
        with self.assertRaises(ValueError): window_records([row])

    def test_cross_split_and_repeated_family_rejected(self):
        a=self.row(); b=self.row();b.update(id='b',subject_id='other',conversation_id='other',split='test',text='z b c d e f g')
        with self.assertRaises(ValueError): window_records([a,b])

    def test_invalid_budget_rejected(self):
        for settings in [(5,4,1),(5,1,5),(0,0,0)]:
            with self.assertRaises(ValueError): window_records([self.row()],*settings)

    def test_silent_truncation_rejected(self):
        row=self.row();row['token_offsets']=row['token_offsets'][:-1]
        with self.assertRaises(ValueError): window_records([row])
