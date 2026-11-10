"""Verify the reviewed data pack with the existing Stage 1 harness; no dependencies."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from harness.evaluate import validate, fingerprint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pdf',default='private_data/pdfs/transformer.pdf')
    args=parser.parse_args()
    folder=Path(__file__).resolve().parent/'private_data/transformer'
    try:
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        for name,expected in manifest['files'].items():
            if Path(name).name!=name:raise ValueError('Invalid manifest file name.')
            if digest(folder/name)!=expected:raise ValueError('Changed data file: '+name)
        pdf=Path(args.pdf)
        if not pdf.is_file():raise ValueError('PDF not found. Copy it to private_data/pdfs/transformer.pdf, or use --pdf "full path".')
        if digest(pdf)!=manifest['pdf_sha256']:raise ValueError('PDF SHA-256 differs from the reviewed document.')
        corpus=json.loads((folder/'corpus.json').read_text(encoding='utf-8'))
        cases=json.loads((folder/'cases.json').read_text(encoding='utf-8'))
        baseline=json.loads((folder/'lexical_baseline_k3.json').read_text(encoding='utf-8'))
        validate(corpus,cases,demo=False)
        if fingerprint({'corpus':corpus,'cases':cases})!=baseline['dataset_sha256']:
            raise ValueError('Baseline dataset fingerprint mismatch.')
        if len({p['page'] for p in corpus})!=manifest['pdf_page_count']:
            raise ValueError('Extracted page coverage mismatch.')
        original=Path('data/review/transformer_development_review.csv')
        if original.is_file() and digest(original)!=manifest['files']['review_source.csv']:
            raise ValueError('Your review CSV differs from the uploaded version. Preserve changes and request regeneration; do not overwrite it.')
        print('VERIFIED: PDF hash matches the human-reviewed source.')
        print(f"VERIFIED: {len(corpus)} text passages across {manifest['pdf_page_count']} PDF pages.")
        print('VERIFIED: 10 approved development cases: 9 answerable, 1 missing-information.')
        print('VERIFIED: data files and baseline fingerprints match.')
        print('This is a frozen lexical/block baseline, not a FAISS app result or a held-out test.')
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print('VERIFICATION FAILED:',exc)
        return 2

if __name__=='__main__':raise SystemExit(main())
