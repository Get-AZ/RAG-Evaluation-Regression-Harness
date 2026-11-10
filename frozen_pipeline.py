"""Pinned retrieval pipeline. Held-out evaluation requires a successful development replay."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

from harness.evaluate import fingerprint, run, validate
from run_retrieval_experiment import predictions as overlap_predictions
from run_semantic_experiment import rank_dense, rrf
from verify_agents_holdout import main as verify_data

ROOT = Path(__file__).resolve().parent
PACKAGES = ('sentence-transformers','transformers','torch','numpy','huggingface-hub')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, data):
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False)

def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def versions():
    return {name:importlib.metadata.version(name) for name in PACKAGES}

def select_candidates(ids, passages, minimum_words):
    if minimum_words < 1 or len(ids)!=len(set(ids)):
        raise ValueError('Invalid threshold or duplicate candidates.')
    eligible=[];excluded=[]
    for pid in ids:
        if pid not in passages:
            raise ValueError('Unknown candidate ID.')
        (eligible if len(passages[pid]['text'].split())>=minimum_words else excluded).append(pid)
    return eligible,excluded

def order_scores(ids, scores):
    values=[float(x) for x in scores]
    if len(values)!=len(ids) or not all(map(math.isfinite,values)):
        raise ValueError('Invalid reranker scores.')
    return sorted(zip(ids,values),key=lambda pair:(-pair[1],pair[0]))

def predict(corpus, queries, config, encoder, reranker):
    # Only questions and allowed document IDs reach ranking; reference answers,
    # evidence IDs and answerability labels are not inputs to this function.
    allowed={'id','question','document_ids'}
    if any(set(q)!=allowed for q in queries):
        raise ValueError('Ranking inputs must contain only id, question, document_ids.')
    lookup={p['id']:p for p in corpus}
    texts=[p['text'] for p in corpus]
    lengths=[len(ids) for ids in encoder.tokenizer(texts,add_special_tokens=True,
        truncation=False,padding=False)['input_ids']]
    questions=[q['question'] for q in queries]
    qlengths=[len(ids) for ids in encoder.tokenizer(questions,add_special_tokens=True,
        truncation=False,padding=False)['input_ids']]
    started=time.perf_counter()
    vectors=encoder.encode(texts,batch_size=32,normalize_embeddings=True,
        convert_to_numpy=True,show_progress_bar=True)
    qvectors=encoder.encode(questions,batch_size=16,normalize_embeddings=True,
        convert_to_numpy=True,show_progress_bar=False)
    if len(qvectors)!=len(queries):
        raise ValueError('Query vector count mismatch.')
    lexical=overlap_predictions(corpus,queries,len(corpus),'overlap_stopwords')
    predictions={};audit={}
    for q,vector in zip(queries,qvectors):
        cid=q['id']
        dense=rank_dense(corpus,vectors,vector,set(q['document_ids']))
        pool=rrf([[pid for pid,score in dense],lexical[cid]],
            limit=config['candidate_limit_per_arm'],constant=config['rrf_constant'])
        ids,excluded=select_candidates([pid for pid,score in pool],lookup,config['minimum_words'])
        pairs=[[q['question'],lookup[pid]['text']] for pid in ids]
        pair_lengths=[len(reranker.tokenizer(a,b,truncation=False)['input_ids']) for a,b in pairs]
        scores=reranker.predict(pairs,batch_size=16,show_progress_bar=False) if pairs else []
        ranked=order_scores(ids,scores)
        predictions[cid]=[pid for pid,score in ranked[:config['k']]]
        audit[cid]={'dense_full_ranking':[{'id':pid,'cosine':score} for pid,score in dense],
            'overlap_full_ranking':lexical[cid],
            'fusion_full_pool':[{'id':pid,'rrf_score':score} for pid,score in pool],
            'excluded_ids':excluded,'ranking':[{'id':pid,'score':score} for pid,score in ranked],
            'over_limit_ids':[pid for pid,n in zip(ids,pair_lengths) if n>reranker.max_seq_length]}
        print(cid+': ranked',flush=True)
    return predictions,audit,{
        'embedding_over_limit_ids':[p['id'] for p,n in zip(corpus,lengths) if n>encoder.max_seq_length],
        'query_over_limit_ids':[q['id'] for q,n in zip(queries,qlengths) if n>encoder.max_seq_length],
        'batch_encoding_and_ranking_seconds':time.perf_counter()-started,
        'timing_note':'Batch timing, not service latency percentiles.'}

def identity(config):
    baseline=ROOT/'reports/filtered_reranking_experiment_v1/reranked.json'
    return {'config_sha256':fingerprint(config),'runner_sha256':filehash(Path(__file__)),
        'development_report_sha256':filehash(baseline),'packages':versions(),
        'python':platform.python_version()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','replay-dev','heldout'])
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--outdir')
    parser.add_argument('--certificate',default='reports/frozen_pipeline_replay_v1/replay_certificate.json')
    args=parser.parse_args()
    verify_data()
    config=read(ROOT/'private_data/agents_taxonomy/frozen_config.json')
    expected={'candidate_limit_per_arm':20,'rrf_constant':60,'minimum_words':5,'k':3,
        'embedding_dimension':384,'embedding_max_seq_length':256,'reranker_max_seq_length':512}
    if any(config[k]!=v for k,v in expected.items()):
        raise ValueError('Frozen configuration differs from supported pipeline.')
    metadata=read(ROOT/'reports/semantic_experiment_v1/summary.json')['metadata']
    current=identity(config)
    if current['packages']!=metadata['packages']:
        raise ValueError('Retrieval dependency versions differ from development checkpoint. Inspect before proceeding.')
    if args.action=='check':
        print('READY: data, source scripts, configuration and retrieval package versions match.')
        print('No models loaded and no retrieval or API calls performed.')
        return 0
    if args.action=='heldout':
        certificate=read(ROOT/args.certificate)
        if certificate.get('identity')!=current or certificate.get('exact_rankings_match') is not True:
            raise ValueError('A matching successful development replay is required.')
        replay_path=(ROOT/args.certificate).parent/'retrieval.json'
        if filehash(replay_path)!=certificate['replay_report_sha256']:
            raise ValueError('Replay report changed.')
        old=read(ROOT/'reports/filtered_reranking_experiment_v1/reranked.json')
        if read(replay_path)['cases']!=old['cases']:
            raise ValueError('Replay rankings do not match original development report.')
    folder=ROOT/('private_data/transformer' if args.action=='replay-dev' else 'private_data/agents_taxonomy')
    corpus,cases=read(folder/'corpus.json'),read(folder/'cases.json')
    split='dev' if args.action=='replay-dev' else 'heldout'
    validate(corpus,cases,demo=False)
    selected=[c for c in cases if c['split']==split]
    out=ROOT/(args.outdir or ('reports/frozen_pipeline_replay_v1' if split=='dev' else 'reports/agents_holdout_v1'))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output folder is not empty. Preserve earlier results and choose --outdir.')
    from sentence_transformers import SentenceTransformer,CrossEncoder
    print('Loading pinned models on CPU; no Gemini requests.',flush=True)
    encoder=SentenceTransformer(config['embedding_model'],revision=config['embedding_revision'],
        device='cpu',trust_remote_code=False,local_files_only=args.offline)
    reranker=CrossEncoder(config['reranker_model'],revision=config['reranker_revision'],
        device='cpu',trust_remote_code=False,local_files_only=args.offline)
    if encoder.get_sentence_embedding_dimension()!=384 or encoder.max_seq_length!=256 or reranker.max_seq_length!=512:
        raise ValueError('Model dimensions/sequence limits differ from recorded configuration.')
    queries=[{k:c[k] for k in ('id','question','document_ids')} for c in selected]
    predictions,audit,truncation=predict(corpus,queries,config,encoder,reranker)
    report=run(corpus,cases,k=3,split=split,predictions=predictions)
    report['frozen_pipeline']={'identity':current,'config':config,'audit':truncation,
        'scope':'retrieval only; generated-answer quality pending separate review'}
    out.mkdir(parents=True,exist_ok=True)
    save(out/'retrieval.json',report);save(out/'rankings.json',audit)
    if split=='dev':
        old=read(ROOT/'reports/filtered_reranking_experiment_v1/reranked.json')
        oldpool=read(ROOT/'reports/semantic_experiment_v1/candidate_rankings.json')
        pools_match=all(audit[cid]['fusion_full_pool']==oldpool[cid]['fusion_full_pool'] for cid in audit)
        exact=report['cases']==old['cases'] and pools_match
        if not exact:
            print('REPLAY FAILED: rankings or candidate pools differ. Held-out run remains blocked.')
            return 1
        save(out/'replay_certificate.json',{'identity':current,'exact_rankings_match':True,
            'replay_report_sha256':filehash(out/'retrieval.json')})
        print('REPLAY PASS: all development top-three rankings and fusion pools match.')
    else:
        print('HELD-OUT FIRST-RUN RESULTS (no comparison gate across different datasets):')
    print(json.dumps(report['metrics']))
    print('Reports:',out)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
