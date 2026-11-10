import math
import unittest
from run_retrieval_experiment import BM25,predictions,tokenize

class RetrievalExperimentTests(unittest.TestCase):
    def test_numeric_score(self):
        idx=BM25([{'id':'a','text':'alpha alpha beta'},{'id':'b','text':'beta'}])
        expected=math.log(2)*2*2.5/(2+1.5*(.25+.75*3/2))
        self.assertAlmostEqual(idx.scores('alpha')[0],expected)
        self.assertEqual(idx.scores('alpha')[1],0)
    def test_unique_query_terms(self):
        idx=BM25([{'id':'a','text':'alpha beta'}])
        self.assertEqual(idx.scores('alpha'),idx.scores('alpha alpha'))
    def test_no_matches_no_forced_results(self):
        self.assertEqual(BM25([{'id':'a','text':'alpha'}]).retrieve('missing',3),[])
    def test_empty_corpus(self):
        self.assertEqual(BM25([]).retrieve('alpha',3),[])
    def test_all_stopwords(self):
        self.assertEqual(BM25([{'id':'a','text':'the and'}],True).retrieve('the and',3),[])
    def test_stable_tie_break(self):
        idx=BM25([{'id':'z','text':'alpha'},{'id':'a','text':'alpha'}])
        self.assertEqual(idx.retrieve('alpha',1),['a'])
    def test_document_scope_is_enforced(self):
        corpus=[{'id':'a','document_id':'allowed','text':'alpha'},{'id':'b','document_id':'other','text':'alpha alpha'}]
        cases=[{'id':'q','question':'alpha','document_ids':['allowed']}]
        for mode in ['lexical','overlap_stopwords','bm25','bm25_stopwords']:
            self.assertEqual(predictions(corpus,cases,3,mode),{'q':['a']})
    def test_stopwords_do_not_remove_negation(self):
        self.assertIn('not',tokenize('is not allowed',True))
    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):BM25([],k1=0)
        with self.assertRaises(ValueError):BM25([],b=1.1)

if __name__=='__main__':unittest.main()
