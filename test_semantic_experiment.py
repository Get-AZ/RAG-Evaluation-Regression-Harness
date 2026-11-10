import math
import unittest
from run_semantic_experiment import rank_dense,rrf,unit

class SemanticHelpersTests(unittest.TestCase):
    def test_unit_norm(self):
        self.assertEqual(unit([3,4]),[.6,.8])
    def test_zero_and_nan_rejected(self):
        for x in [[0,0],[float('nan'),1],[]]:
            with self.assertRaises(ValueError):unit(x)
    def test_cosine_rank_and_score(self):
        c=[{'id':'x','document_id':'d'},{'id':'y','document_id':'d'},{'id':'z','document_id':'d'}]
        result=rank_dense(c,[[0,1],[1,1],[1,0]],[2,0],{'d'})
        self.assertEqual([x[0] for x in result],['z','y','x'])
        self.assertAlmostEqual(result[1][1],1/math.sqrt(2))
    def test_scope(self):
        c=[{'id':'a','document_id':'allowed'},{'id':'b','document_id':'blocked'}]
        self.assertEqual(rank_dense(c,[[0,1],[1,0]],[1,0],{'allowed'}),[('a',0.0)])
    def test_dimension_mismatch_rejected(self):
        with self.assertRaises(ValueError):rank_dense([{'id':'a','document_id':'d'}],[[1,0]],[1],{'d'})
    def test_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):rank_dense([],[[1]],[1],{'d'})
    def test_ties_are_stable(self):
        c=[{'id':'z','document_id':'d'},{'id':'a','document_id':'d'}]
        self.assertEqual([x[0] for x in rank_dense(c,[[1,0],[1,0]],[1,0],{'d'})],['a','z'])
    def test_rrf_known_values(self):
        result=rrf([['a','b'],['b','a']],constant=60)
        self.assertEqual([x[0] for x in result],['a','b'])
        self.assertAlmostEqual(result[0][1],1/61+1/62)
    def test_rrf_pool_cap_and_duplicates(self):
        self.assertEqual(rrf([['a','b'],['c','d']],limit=1),[('a',1/61),('c',1/61)])
        with self.assertRaises(ValueError):rrf([['a','a']])

if __name__=='__main__':unittest.main()
