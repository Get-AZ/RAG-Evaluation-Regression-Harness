"""Controlled retrieval ablation on a frozen corpus; Python standard library only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from harness.evaluate import read, run, compare, validate, fingerprint

# Fixed list selected before running this experiment. No domain answer terms.
STOPWORDS = frozenset('a an the and or of to in on for with what which how many was were is are used according do does did why where when'.split())
VARIANTS = ('lexical', 'overlap_stopwords', 'bm25', 'bm25_stopwords')


def tokenize(text, remove_stopwords=False):
    terms=re.findall(r'\w+',text.lower())
    return [t for t in terms if not remove_stopwords or t not in STOPWORDS]


class BM25:
    """Positive smoothed Robertson IDF; k1=1.5 and b=.75 fixed a priori."""
    def __init__(self, corpus, remove_stopwords=False, k1=1.5, b=.75):
        if k1<=0 or not 0<=b<=1:raise ValueError('Invalid BM25 parameters.')
        self.corpus=corpus;self.stop=remove_stopwords;self.k1=k1;self.b=b
        self.counts=[Counter(tokenize(p['text'],self.stop)) for p in corpus]
        self.lengths=[sum(c.values()) for c in self.counts]
        self.avg=sum(self.lengths)/len(corpus) if corpus else 0
        self.df=Counter(t for c in self.counts for t in c)

    def scores(self,question):
        q=sorted(set(tokenize(question,self.stop)));n=len(self.corpus)
        scores=[]
        for counts,length in zip(self.counts,self.lengths):
            score=0.0
            for term in q:
                tf=counts.get(term,0)
                if not tf:continue
                df=self.df[term]
                idf=math.log(1+(n-df+.5)/(df+.5))
                norm=self.k1*(1-self.b+self.b*length/self.avg)
                score+=idf*tf*(self.k1+1)/(tf+norm)
            scores.append(score)
        return scores

    def retrieve(self,question,k):
        pairs=sorted(zip(self.corpus,self.scores(question)),key=lambda pair:(-pair[1],pair[0]['id']))
        return [p['id'] for p,score in pairs if score>0][:k]


def predictions(corpus,cases,k,variant):
    result={}
    # Fit corpus statistics within each allowed document scope, never on answers,
    # reference passages, review notes or held-out queries.
    indexes={}
    for c in cases:
        scope=tuple(sorted(c['document_ids']))
        scoped=[p for p in corpus if p['document_id'] in scope]
        if variant.startswith('bm25'):
            if scope not in indexes:indexes[scope]=BM25(scoped,variant.endswith('stopwords'))
            ranked=indexes[scope].retrieve(c['question'],k)
        else:
            stop=variant=='overlap_stopwords';q=set(tokenize(c['question'],stop))
            scored=[(p,len(q & set(tokenize(p['text'],stop)))) for p in scoped]
            ranked=[p['id'] for p,s in sorted(scored,key=lambda pair:(-pair[1],pair[0]['id'])) if s>0][:k]
        result[c['id']]=ranked
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='private_data/transformer')
    p.add_argument('--k',type=int,default=3)
    p.add_argument('--outdir')
    a=p.parse_args()
    try:
        folder=Path(a.data);corpus=read(folder/'corpus.json');cases=read(folder/'cases.json')
        if a.k<1:raise ValueError('k must be positive.')
        validate(corpus,cases,demo=False)
        baseline=read(folder/'lexical_baseline_k3.json')
        if a.k!=baseline['k']:raise ValueError('Use the frozen baseline k for this comparison.')
        if fingerprint({'corpus':corpus,'cases':cases})!=baseline['dataset_sha256']:
            raise ValueError('Frozen dataset differs from baseline. Do not compare different corpora.')
        selected=[c for c in cases if c['split']=='dev']
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target=Path(a.outdir) if a.outdir else Path('reports/retrieval_experiment')/stamp
        if target.exists() and any(target.iterdir()):raise ValueError('Output folder is not empty; choose a new --outdir.')
        target.mkdir(parents=True,exist_ok=True)
        rows=[];reports={}
        print('Development-only retrieval ablation. Fixed corpus, labels and k=3. No LLM calls.')
        print(f"{'variant':22} {'recall':>8} {'mrr':>8} {'ndcg':>8} {'gate':>8}")
        for variant in VARIANTS:
            pred=predictions(corpus,selected,a.k,variant)
            report=run(corpus,cases,k=a.k,split='dev',demo=False,predictions=pred)
            report['configuration']=variant
            report['experiment_parameters']={'k1':1.5,'b':.75,'query_term_frequency':'unique query terms',
                'bm25_idf':'log(1 + (N-df+0.5)/(df+0.5))','stopwords':sorted(STOPWORDS) if variant.endswith('stopwords') else []}
            fails=compare(baseline,report)
            row={'variant':variant,**report['metrics'],'gate_failures':fails}
            rows.append(row);reports[variant]=report
            for name,data in [(variant,report),(variant+'_predictions',pred)]:
                (target/(name+'.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
            m=report['metrics'];print(f"{variant:22} {m['recall']:8.4f} {m['mrr']:8.4f} {m['ndcg']:8.4f} {('FAIL' if fails else 'PASS'):>8}")
        # The lexical arm must reproduce the prior score and exact candidate ordering.
        assert reports['lexical']['cases']==baseline['cases'],'Lexical reproduction mismatch.'
        summary={'dataset_sha256':baseline['dataset_sha256'],'split':'dev','k':a.k,'rows':rows,
                 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'meaning':'Reference-evidence retrieval on the same small development set; not chatbot accuracy, held-out performance or a baseline promotion.'}
        (target/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('\nTR-DEV-001 labeled evidence rank (within top 3):')
        for variant in VARIANTS:
            c=next(c for c in cases if c['id']=='TR-DEV-001')
            r=next(r for r in reports[variant]['cases'] if r['id']==c['id'])
            ranks=[i+1 for i,x in enumerate(r['ranked_ids']) if x in c['relevant_ids']]
            print(variant+': '+(', '.join(map(str,ranks)) if ranks else 'not retrieved'))
        print('\nReports:',target)
        print('Original data, harness and frozen baseline were not changed. Inspect all cases before promotion.')
        return 0
    except (OSError,ValueError,KeyError,TypeError,AssertionError) as exc:
        print('EXPERIMENT FAILED:',exc);return 2

if __name__=='__main__':raise SystemExit(main())
