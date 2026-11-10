import copy
import unittest
from harness.answer_review import template,score
from test_evaluation import CORPUS,CASES

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.cases=copy.deepcopy(CASES)
        for c in self.cases:c.update(review_status='approved',reviewer='TEST FIXTURE',reviewed_at='2026-11-08',reference_answer='TEST FIXTURE ONLY')
        self.answers={c['id']:'TEST FIXTURE ANSWER' for c in self.cases if c['split']=='dev'}
        self.rows=template(CORPUS,self.cases,self.answers,'dev')
        for r in self.rows:r.update(grounded='1',complete='1',behavior_correct='1',citations_correct='1',review_status='approved',reviewer='TEST',reviewed_at='2026-11-08',notes='test fixture')
    def test_complete_reviews_score(self):
        self.assertEqual(score(CORPUS,self.cases,self.answers,self.rows,'dev')['all_criteria_pass_rate'],1)
    def test_pending_review_rejected(self):
        self.rows[0]['review_status']='pending'
        with self.assertRaises(ValueError):score(CORPUS,self.cases,self.answers,self.rows,'dev')
    def test_changed_answer_rejected(self):
        self.answers['demo-dev-1']='CHANGED'
        with self.assertRaises(ValueError):score(CORPUS,self.cases,self.answers,self.rows,'dev')
    def test_incomplete_review_rejected(self):
        with self.assertRaises(ValueError):score(CORPUS,self.cases,self.answers,self.rows[:-1],'dev')
