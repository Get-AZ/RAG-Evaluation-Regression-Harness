"""MiniLM retrieval on frozen passage IDs; measured live results only after model load."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import time
from harness.evaluate import read, run, compare, validate, fingerprint
from run_retrieval_experiment import predictions as lexical_predictions

MODEL='sentence-transformers/all-MiniLM-L6-v2'


def unit(vector):
    values=[float(x) for x in vector]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Embeddings must be nonempty and finite.')
    norm=math.sqrt(sum(x*x for x in values))
    if norm==0:raise ValueError('Zero embedding is invalid.')
    return [x/norm for x in values]


def rank_dense(corpus, document_vectors, question_vector, allowed_documents):
    if len(corpus)!=len(document_vectors):raise ValueError('Corpus/vector length mismatch.')
    q=unit(question_vector);scored=[]
    for passage,vector in zip(corpus,document_vectors):
        if passage['document_id'] not in allowed_documents:continue
        v=unit(vector)
        if len(v)!=len(q):raise ValueError('Embedding dimension mismatch.')
        scored.append((passage['id'],sum(a*b for a,b in zip(q,v))))
    return sorted(scored,key=lambda x:(-x[1],x[0]))


def rrf(rankings,limit=20,constant=60):
    if limit<1 or constant<=0:raise ValueError('Invalid fusion settings.')
    scores={}
    for ranking in rankings:
        if len(ranking)!=len(set(ranking)):raise ValueError('Duplicate fusion candidate.')
        for rank,pid in enumerate(ranking[:limit],1):
            scores[pid]=scores.get(pid,0)+1/(constant+rank)
    return sorted(scores.items(),key=lambda x:(-x[1],x[0]))


def check_data(folder):
    corpus=read(folder/'corpus.json');cases=read(folder/'cases.json')
    validate(corpus,cases,demo=False)
    baseline=read(folder/'lexical_baseline_k3.json')
    if fingerprint({'corpus':corpus,'cases':cases})!=baseline['dataset_sha256']:
        raise ValueError('Dataset differs from frozen baseline.')
    manifest=read(folder/'manifest.json')
    for name,expected in manifest['files'].items():
        if Path(name).name!=name:raise ValueError('Invalid manifest filename.')
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Data file changed: '+name)
    return corpus,cases,baseline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='private_data/transformer')
    p.add_argument('--outdir')
    p.add_argument('--revision',help='Optional exact 40-character Hugging Face model commit SHA')
    p.add_argument('--offline',action='store_true',help='Requires --revision and cached model files')
    p.add_argument('--check',action='store_true',help='Check data and imports only; no model download')
    a=p.parse_args()
    try:
        corpus,cases,baseline=check_data(Path(a.data))
        print('Data verified:',len(corpus),'passages;',len(cases),'reviewed cases.')
        if a.revision and not re.fullmatch(r'[a-fA-F0-9]{40}',a.revision):
            raise ValueError('--revision must be an exact model commit SHA.')
        if a.offline and not a.revision:raise ValueError('--offline requires --revision.')
        try:
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ValueError('Semantic dependencies missing. Follow START_HERE.md in a separate environment. '+str(e)) from e
        versions={name:importlib.metadata.version(name) for name in
                  ('sentence-transformers','transformers','torch','numpy','huggingface-hub')}
        print('Python:',platform.python_version());print('Packages:',json.dumps(versions,sort_keys=True))
        if a.check:
            print('READY: imports and data pass. Model weights and retrieval have not been tested.')
            return 0
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target=Path(a.outdir) if a.outdir else Path('reports/semantic_experiment')/stamp
        if target.exists() and any(target.iterdir()):raise ValueError('Output folder is not empty; use a new --outdir.')
        revision=a.revision or HfApi().model_info(MODEL).sha
        if not re.fullmatch(r'[a-fA-F0-9]{40}',revision or ''):raise ValueError('Could not resolve an exact model revision.')
        print('Loading MiniLM on CPU. First use downloads public model weights.')
        model=SentenceTransformer(MODEL,revision=revision,device='cpu',
                                  trust_remote_code=False,local_files_only=a.offline)
        if int(model.get_sentence_embedding_dimension())!=384:
            raise ValueError('Unexpected embedding dimension for the selected model.')
        selected=[c for c in cases if c['split']=='dev'];k=baseline['k']
        texts=[p['text'] for p in corpus];questions=[c['question'] for c in selected]
        # Audit truncation. Inputs/IDs remain unchanged; no silent chunking rewrite.
        limit=int(model.max_seq_length)
        tokenizer=model.tokenizer
        lengths=[len(ids) for ids in tokenizer(texts,add_special_tokens=True,
                   truncation=False,padding=False)['input_ids']]
        question_lengths=[len(ids) for ids in tokenizer(questions,add_special_tokens=True,
                   truncation=False,padding=False)['input_ids']]
        truncated=[p['id'] for p,n in zip(corpus,lengths) if n>limit]
        gold={pid for c in selected for pid in c['relevant_ids']}
        truncated_gold=sorted(gold.intersection(truncated))
        print('Model sequence limit:',limit,'tokens; passages exceeding limit:',len(truncated))
        print('Labeled reference blocks exceeding limit:',len(truncated_gold))
        t0=time.perf_counter()
        vectors=model.encode(texts,batch_size=32,normalize_embeddings=True,
                             convert_to_numpy=True,show_progress_bar=True)
        document_encoding_seconds=time.perf_counter()-t0
        t0=time.perf_counter()
        qvectors=model.encode(questions,batch_size=16,normalize_embeddings=True,
                              convert_to_numpy=True,show_progress_bar=False)
        query_encoding_seconds=time.perf_counter()-t0
        if len(qvectors)!=len(selected):raise ValueError('Question/vector length mismatch.')
        lexical=lexical_predictions(corpus,selected,len(corpus),'lexical')
        overlap=lexical_predictions(corpus,selected,len(corpus),'overlap_stopwords')
        dense={};fused={};audit={}
        t0=time.perf_counter()
        for c,qvector in zip(selected,qvectors):
            scores=rank_dense(corpus,vectors,qvector,set(c['document_ids']))
            dense[c['id']]=[pid for pid,score in scores]
            fusion=rrf([dense[c['id']],overlap[c['id']]],limit=20,constant=60)
            fused[c['id']]=[pid for pid,score in fusion]
            audit[c['id']]={'dense_full_ranking':[{'id':pid,'cosine':score} for pid,score in scores],
                            'overlap_full_ranking':overlap[c['id']],
                            'fusion_full_pool':[{'id':pid,'rrf_score':score} for pid,score in fusion]}
        ranking_seconds=time.perf_counter()-t0
        target.mkdir(parents=True,exist_ok=True)
        metadata={'model':MODEL,'model_revision':revision,'python':platform.python_version(),
                  'packages':versions,'device':'cpu','embedding_dimension':384,'model_max_seq_length':limit,
                  'truncated_passage_ids':truncated,'truncated_reference_ids':truncated_gold,
                  'query_ids_exceeding_limit':[c['id'] for c,n in zip(selected,question_lengths) if n>limit],
                  'fusion':{'method':'equal-weight RRF of dense and stopword-overlap','candidate_limit_per_arm':20,'constant':60},
                  'timings_seconds':{'document_encoding_batch':document_encoding_seconds,
                      'question_encoding_batch':query_encoding_seconds,'ranking_batch':ranking_seconds},
                  'timing_note':'Batch timings; not end-to-end latency percentiles or service benchmarks.',
                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        rows=[];reports={}
        print(f"\n{'variant':22} {'recall':>8} {'mrr':>8} {'ndcg':>8} {'gate':>8}")
        for name,pred in [('lexical',lexical),('overlap_stopwords',overlap),('dense_minilm',dense),('dense_overlap_rrf',fused)]:
            top={cid:ids[:k] for cid,ids in pred.items()}
            report=run(corpus,cases,k=k,split='dev',demo=False,predictions=top)
            report['configuration']=name;report['experiment_metadata']=metadata
            if name=='lexical' and report['cases']!=baseline['cases']:
                raise ValueError('Original lexical rankings no longer reproduce the baseline.')
            fails=compare(baseline,report);m=report['metrics']
            rows.append(dict(variant=name,**m,gate_failures=fails));reports[name]=report
            (target/(name+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
            (target/(name+'_predictions.json')).write_text(json.dumps(top,indent=2),encoding='utf-8')
            print(f"{name:22} {m['recall']:8.4f} {m['mrr']:8.4f} {m['ndcg']:8.4f} {('FAIL' if fails else 'PASS'):>8}")
        print('\nFirst labeled evidence rank within top 3:')
        for cid in ['TR-DEV-001','TR-DEV-009']:
            c=next(c for c in selected if c['id']==cid)
            for name in ['dense_minilm','dense_overlap_rrf']:
                r=next(r for r in reports[name]['cases'] if r['id']==cid)
                ranks=[i+1 for i,pid in enumerate(r['ranked_ids']) if pid in c['relevant_ids']]
                print(cid,name,':',', '.join(map(str,ranks)) if ranks else 'not retrieved')
        (target/'candidate_rankings.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        (target/'summary.json').write_text(json.dumps({'rows':rows,'metadata':metadata,
             'dataset_sha256':baseline['dataset_sha256'],'k':k,'split':'dev',
             'scope':'Same small development set. No LLM generation, human answer scoring, held-out performance or automatic baseline promotion.'},indent=2),encoding='utf-8')
        print('\nReports:',target)
        print('Experiment completed; PASS/FAIL is against the original lexical baseline.')
        print('No existing data, labels, app code or baseline files were changed.')
        return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        print('SEMANTIC EXPERIMENT FAILED:',exc);return 2

if __name__=='__main__':raise SystemExit(main())
