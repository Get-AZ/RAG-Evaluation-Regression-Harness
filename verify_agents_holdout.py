"""Validate approved held-out data and development separation; no model/API calls."""
import hashlib
import json
from pathlib import Path
from harness.evaluate import validate, fingerprint

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    folder = ROOT/'private_data/agents_taxonomy'
    manifest = read(folder/'manifest.json')
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid manifest filename.')
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed held-out file: '+name)
    pdf = ROOT/'private_data/pdfs/agents_taxonomy.pdf'
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != manifest['pdf_sha256']:
        raise ValueError('PDF hash mismatch.')
    corpus, cases = read(folder/'corpus.json'), read(folder/'cases.json')
    validate(corpus,cases,demo=False)
    if fingerprint({'corpus':corpus,'cases':cases}) != manifest['dataset_sha256']:
        raise ValueError('Held-out dataset fingerprint mismatch.')
    if len(cases)!=12 or sum(c['answerable'] for c in cases)!=9:
        raise ValueError('Unexpected case counts.')
    if any(c['split']!='heldout' or c['review_status']!='approved' for c in cases):
        raise ValueError('Expected approved held-out cases only.')
    if len(corpus)!=manifest['passage_count'] or {p['page'] for p in corpus}!=set(range(1,31)):
        raise ValueError('Unexpected passage or page coverage.')
    config = read(folder/'frozen_config.json')
    if max(len(c['relevant_ids']) for c in cases)>config['k']:
        raise ValueError('Required evidence exceeds the final passage budget.')
    dev = ROOT/'private_data/transformer'
    devcorpus,devcases = read(dev/'corpus.json'),read(dev/'cases.json')
    if fingerprint({'corpus':devcorpus,'cases':devcases})!=config['development_dataset_sha256']:
        raise ValueError('Development data differs from checkpoint.')
    validate(devcorpus+corpus,devcases+cases,demo=False)
    for name,expected in config['source_script_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Script differs from archived checkpoint: '+name)
    print('VERIFIED: source PDF SHA256 matches.')
    print(f'VERIFIED: {len(corpus)} passages across 30 PDF pages.')
    print('VERIFIED: 12 approved held-out cases: 9 answerable, 3 missing-information.')
    print('VERIFIED: development/held-out documents are separate; checkpoint scripts match.')
    print('READY FOR RUNNER PREPARATION: no retrieval, model download, or Gemini requests performed.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
