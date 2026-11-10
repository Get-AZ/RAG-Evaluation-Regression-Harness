import copy
import json
import math
import unittest
from pathlib import Path
from harness.evaluate import metrics, run, compare, validate

ROOT = Path(__file__).resolve().parents[1]
CORPUS = json.loads((ROOT/'data/demo/corpus.json').read_text())
CASES = json.loads((ROOT/'data/demo/cases.json').read_text())

class EvaluationTests(unittest.TestCase):
    def test_known_metrics(self):
        r = metrics(['irrelevant', 'a', 'b'], ['a', 'b'], 3)
        self.assertEqual(r['recall'], 1)
        self.assertEqual(r['mrr'], .5)
        self.assertAlmostEqual(r['ndcg'], (1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3)))
    def test_k_and_duplicates(self):
        r=metrics(['a','a','b'], ['a','b'],1)
        self.assertEqual(r['recall'],.5)
    def test_missing_information_excluded(self):
        self.assertIsNone(metrics(['a'], [], 3))
    def test_same_version_passes(self):
        a=run(CORPUS,CASES,demo=True)
        self.assertEqual(compare(a,a),[])
    def test_broken_retriever_rejected(self):
        a=run(CORPUS,CASES,demo=True)
        b=run(CORPUS,CASES,demo=True,mode='empty')
        self.assertEqual(len(compare(a,b)),3)
    def test_changed_dataset_cannot_compare(self):
        a=run(CORPUS,CASES,demo=True);b=copy.deepcopy(a);b['dataset_sha256']='changed'
        with self.assertRaises(ValueError):compare(a,b)
    def test_unreviewed_real_cases_rejected(self):
        with self.assertRaises(ValueError):validate(CORPUS,CASES)
    def test_cross_split_document_leakage_rejected(self):
        cs=copy.deepcopy(CASES);cs[-1]['document_ids']=cs[0]['document_ids']
        cs[-1]['relevant_ids']=cs[0]['relevant_ids']
        with self.assertRaises(ValueError):validate(CORPUS,cs,True)
    def test_unknown_predictions_rejected(self):
        pred={c['id']:['unknown'] for c in CASES if c['split']=='dev'}
        with self.assertRaises(ValueError):run(CORPUS,CASES,demo=True,predictions=pred)
    def test_incomplete_predictions_rejected(self):
        with self.assertRaises(ValueError):run(CORPUS,CASES,demo=True,predictions={})
    def test_nan_cannot_pass_gate(self):
        a=run(CORPUS,CASES,demo=True);b=copy.deepcopy(a);b['metrics']['recall']=float('nan')
        with self.assertRaises(ValueError):compare(a,b)
    def test_unanswerable_not_counted_as_retrieval_pass(self):
        r=run(CORPUS,CASES,demo=True)
        self.assertEqual(r['scored_cases'],2);self.assertEqual(r['unanswerable_cases'],1)

if __name__=='__main__':unittest.main()
