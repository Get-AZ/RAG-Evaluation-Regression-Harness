import unittest
from frozen_pipeline import select_candidates,order_scores,predict

class FrozenPipelineTests(unittest.TestCase):
    def test_five_word_boundary(self):
        p={'a':{'text':'one two three four'},'b':{'text':'one two three four five'}}
        self.assertEqual(select_candidates(['a','b'],p,5),(['b'],['a']))
    def test_empty_pool(self):
        self.assertEqual(select_candidates([],{},5),([],[]))
        self.assertEqual(order_scores([],[]),[])
    def test_duplicate_and_unknown_rejected(self):
        with self.assertRaises(ValueError):select_candidates(['a','a'],{},5)
        with self.assertRaises(ValueError):select_candidates(['unknown'],{},5)
    def test_invalid_scores(self):
        for scores in [[float('nan')],[float('inf')],[]]:
            with self.assertRaises(ValueError):order_scores(['a'],scores)
    def test_stable_ties(self):
        self.assertEqual(order_scores(['b','a'],[1,1]),[('a',1.0),('b',1.0)])
    def test_reference_fields_cannot_enter_ranking(self):
        with self.assertRaises(ValueError):
            predict([], [{'id':'x','question':'q','document_ids':[],
                          'reference_answer':'hidden'}],{},None,None)
    def test_pipeline_filters_before_pair_scoring(self):
        class Tokenizer:
            def __call__(self,a,b=None,**kwargs):
                return {'input_ids':[[1]*len(t.split()) for t in a]} if isinstance(a,list) else {'input_ids':[1]*len((a+' '+(b or '')).split())}
        class Encoder:
            max_seq_length=256
            tokenizer=Tokenizer()
            def encode(self,texts,**kwargs):return [[1.0,0.0] for t in texts]
        class Reranker:
            max_seq_length=512
            tokenizer=Tokenizer()
            def predict(self,pairs,**kwargs):
                assert all(len(text.split())>=5 for question,text in pairs)
                return [1.0]*len(pairs)
        corpus=[{'id':'a','document_id':'doc','text':'will'},
                {'id':'b','document_id':'doc','text':'one two three four five'},
                {'id':'c','document_id':'other','text':'one two three four five'}]
        pred,audit,meta=predict(corpus,[{'id':'q','question':'will','document_ids':['doc']}],
            {'candidate_limit_per_arm':20,'rrf_constant':60,'minimum_words':5,'k':3},Encoder(),Reranker())
        self.assertEqual(pred,{'q':['b']})
        self.assertEqual(audit['q']['excluded_ids'],['a'])
        self.assertEqual(meta['embedding_over_limit_ids'],[])

if __name__=='__main__':unittest.main()
