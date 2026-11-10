"""Dependency-free retrieval evaluation. Demo data is never a real benchmark."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tokens(text):
    return set(re.findall(r'\w+', text.lower()))


def validate(corpus, cases, demo=False):
    if not corpus or not cases:
        raise ValueError('Corpus and cases must be nonempty.')
    passages = {p['id']: p for p in corpus}
    if len(passages) != len(corpus):
        raise ValueError('Duplicate passage IDs.')
    for p in corpus:
        if not p.get('text') or not p.get('document_id') or not isinstance(p.get('page'), int) or p['page'] < 1:
            raise ValueError('Passages need text, document identity and a 1-based PDF page.')
    ids, questions, documents = set(), set(), {'dev': set(), 'heldout': set()}
    for c in cases:
        if c['id'] in ids or ' '.join(c['question'].lower().split()) in questions:
            raise ValueError('Duplicate case ID or question.')
        ids.add(c['id']); questions.add(' '.join(c['question'].lower().split()))
        if c['split'] not in documents or c['answerable'] not in (True, False):
            raise ValueError('Invalid split or answerability.')
        if not c.get('category') or not c.get('question', '').strip():
            raise ValueError('Question and category required.')
        if not demo and (c.get('review_status') != 'approved' or not c.get('reviewer') or not c.get('reviewed_at')):
            raise ValueError('Real cases require explicit human review approval, reviewer and date.')
        refs = c['relevant_ids']
        if len(refs) != len(set(refs)) or any(x not in passages for x in refs):
            raise ValueError('Invalid or duplicate evidence references.')
        if bool(refs) != c['answerable']:
            raise ValueError('Answerable cases need evidence; missing-information cases need none.')
        scope = set(c['document_ids'])
        if not scope or not scope <= {p['document_id'] for p in corpus}:
            raise ValueError('Invalid document scope.')
        if any(passages[x]['document_id'] not in scope for x in refs):
            raise ValueError('Evidence lies outside the case scope.')
        if not demo and not c.get('reference_answer'):
            raise ValueError('Reference answer or expected abstention required.')
        documents[c['split']].update(scope)
    if documents['dev'] & documents['heldout']:
        raise ValueError('Document overlap between development and held-out sets.')
    return passages


def retrieve(question, corpus, k, mode='lexical'):
    if mode == 'empty':
        return []
    q = tokens(question)
    ranked = sorted(corpus, key=lambda p: (-len(q & tokens(p['text'])), p['id']))
    return [p['id'] for p in ranked if q & tokens(p['text'])][:k]


def metrics(ranked, relevant, k):
    ranked = list(dict.fromkeys(ranked))[:k]
    if not relevant:
        return None  # Retrieval ranking cannot establish a correct abstention.
    gold = set(relevant)
    gains = [int(x in gold) for x in ranked]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {'recall': len(set(ranked) & gold) / len(gold),
            'mrr': next((1 / (i + 1) for i, g in enumerate(gains) if g), 0.0),
            'ndcg': dcg / ideal}


def aggregate(rows):
    valid = [r['metrics'] for r in rows if r['metrics'] is not None]
    return {name: sum(r[name] for r in valid) / len(valid) if valid else None
            for name in ('recall', 'mrr', 'ndcg')}


def run(corpus, cases, k=3, mode='lexical', split='dev', demo=False, predictions=None):
    if k < 1:
        raise ValueError('k must be positive.')
    passages = validate(corpus, cases, demo)
    selected = [c for c in cases if c['split'] == split]
    if not selected or not any(c['answerable'] for c in selected):
        raise ValueError('Split must contain answerable cases for a retrieval score.')
    if predictions is not None and set(predictions) != {c['id'] for c in selected}:
        raise ValueError('Predictions must cover exactly the selected cases.')
    rows = []
    for c in selected:
        scoped = [p for p in corpus if p['document_id'] in c['document_ids']]
        ranked = predictions[c['id']] if predictions is not None else retrieve(c['question'], scoped, k, mode)
        if not isinstance(ranked, list) or any(not isinstance(x, str) or x not in passages for x in ranked):
            raise ValueError('Predictions must be lists of known passage IDs.')
        if len(ranked) != len(set(ranked)):
            raise ValueError('Duplicate prediction IDs are not allowed.')
        if any(passages[x]['document_id'] not in c['document_ids'] for x in ranked):
            raise ValueError('Retrieved passage outside document scope.')
        rows.append({'id': c['id'], 'category': c['category'], 'ranked_ids': ranked[:k],
                     'answerable': c['answerable'], 'metrics': metrics(ranked, c['relevant_ids'], k)})
    return {'schema_version': 1, 'demo': demo, 'split': split, 'k': k,
            'dataset_sha256': fingerprint({'corpus': corpus, 'cases': cases}),
            'configuration': mode if predictions is None else 'external_predictions',
            'scored_cases': sum(c['answerable'] for c in selected),
            'unanswerable_cases': sum(not c['answerable'] for c in selected),
            'metrics': aggregate(rows),
            'by_category': {cat: aggregate([r for r in rows if r['category'] == cat])
                            for cat in sorted({r['category'] for r in rows})},
            'cases': rows, 'answer_quality': 'pending separate human review'}


def compare(base, candidate, tolerance=0.01):
    if not 0 <= tolerance <= 1:
        raise ValueError('Tolerance must be between zero and one.')
    for key in ('schema_version', 'demo', 'split', 'k', 'dataset_sha256'):
        if base[key] != candidate[key]:
            raise ValueError('Incomparable reports: ' + key)
    if [r['id'] for r in base['cases']] != [r['id'] for r in candidate['cases']]:
        raise ValueError('Case coverage changed.')
    failed = []
    for name in ('recall', 'mrr', 'ndcg'):
        a, b = base['metrics'][name], candidate['metrics'][name]
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Missing or invalid metric: ' + name)
        if a - b > tolerance + 1e-12:
            failed.append(f'{name}: {a:.4f} -> {b:.4f}')
    return failed


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('run')
    e.add_argument('--corpus', required=True); e.add_argument('--cases', required=True)
    e.add_argument('--out', required=True); e.add_argument('--k', type=int, default=3)
    e.add_argument('--split', choices=['dev', 'heldout'], default='dev')
    e.add_argument('--mode', choices=['lexical', 'empty'], default='lexical')
    e.add_argument('--demo', action='store_true'); e.add_argument('--predictions')
    c = sub.add_parser('compare')
    c.add_argument('--baseline', required=True); c.add_argument('--candidate', required=True)
    c.add_argument('--tolerance', type=float, default=.01)
    args = p.parse_args()
    try:
        if args.command == 'run':
            result = run(read(args.corpus), read(args.cases), args.k, args.mode, args.split,
                         args.demo, read(args.predictions) if args.predictions else None)
            target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(('DEMO ONLY ' if args.demo else '') + json.dumps(result['metrics']))
        else:
            failed = compare(read(args.baseline), read(args.candidate), args.tolerance)
            print('REGRESSION: ' + '; '.join(failed) if failed else 'PASS: no gated regression')
            return 1 if failed else 0
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print('INVALID EVALUATION: ' + str(exc))
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
