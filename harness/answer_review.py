"""Human answer review bound to exact case, corpus and generated answer versions."""
import argparse
import csv
import json
from pathlib import Path
from harness.evaluate import read, fingerprint, validate

CRITERIA = ('grounded', 'complete', 'behavior_correct', 'citations_correct')

def template(corpus, cases, answers, split):
    validate(corpus, cases, demo=False)
    selected=[c for c in cases if c['split']==split]
    if not selected or set(answers)!={c['id'] for c in selected}:
        raise ValueError('Answers must cover exactly the selected split.')
    dataset_hash=fingerprint({'corpus':corpus,'cases':cases})
    rows=[]
    for c in selected:
        answer=answers[c['id']]
        if not isinstance(answer,str) or not answer.strip():raise ValueError('Nonempty answer text required.')
        rows.append(dict(case_id=c['id'],dataset_sha256=dataset_hash,answer_sha256=fingerprint(answer),
                         grounded='',complete='',behavior_correct='',citations_correct='',
                         review_status='pending',reviewer='',reviewed_at='',notes=''))
    return rows


def score(corpus,cases,answers,reviews,split):
    expected={r['case_id']:r for r in template(corpus,cases,answers,split)}
    if len(reviews)!=len(expected) or {r['case_id'] for r in reviews}!=set(expected):
        raise ValueError('Review coverage must match the split without duplicates.')
    for r in reviews:
        e=expected[r['case_id']]
        for key in ('dataset_sha256','answer_sha256'):
            if r[key]!=e[key]:raise ValueError('Stale review: '+key)
        if r['review_status']!='approved' or not r['reviewer'] or not r['reviewed_at'] or not r['notes']:
            raise ValueError('Each review requires approval, reviewer, date and rationale.')
        if any(r[k] not in ('0','1') for k in CRITERIA):
            raise ValueError('Score each criterion 0 or 1 using the rubric.')
    return {'split':split,'reviewed_answers':len(reviews),
            'dataset_sha256':next(iter(expected.values()))['dataset_sha256'],
            'means':{k:sum(int(r[k]) for r in reviews)/len(reviews) for k in CRITERIA},
            'all_criteria_pass_rate':sum(all(r[k]=='1' for k in CRITERIA) for r in reviews)/len(reviews),
            'cases':reviews,'method':'human rubric; no automatic judge or general-accuracy claim'}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score'])
    for name in ['corpus','cases','answers','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=['dev','heldout'],default='dev');p.add_argument('--reviews')
    a=p.parse_args()
    try:
        corpus,cases,answers=read(a.corpus),read(a.cases),read(a.answers)
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
        if a.action=='prepare':
            rows=template(corpus,cases,answers,a.split)
            with out.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        else:
            if not a.reviews:raise ValueError('--reviews is required for scoring.')
            with Path(a.reviews).open(encoding='utf-8',newline='') as f:reviews=list(csv.DictReader(f))
            out.write_text(json.dumps(score(corpus,cases,answers,reviews,a.split),indent=2),encoding='utf-8')
        print('Saved '+str(out));return 0
    except (ValueError,KeyError,TypeError,OSError) as e:
        print('INVALID REVIEW: '+str(e));return 2

if __name__=='__main__':raise SystemExit(main())
