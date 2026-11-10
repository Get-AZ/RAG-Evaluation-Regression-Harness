"""Offline synthetic regression gate. Does not measure real-PDF or LLM quality."""
import argparse
import json
from pathlib import Path
from harness.evaluate import read,run,compare

ROOT=Path(__file__).resolve().parent

def execute(mode, output):
    if mode not in ('lexical','empty'):
        raise ValueError('mode must be lexical or empty')
    base=ROOT/'data/demo'
    report=run(read(base/'corpus.json'),read(base/'cases.json'),k=3,
               mode=mode,split='dev',demo=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/'ci_candidate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    failures=compare(read(base/'baseline.json'),report)
    print('DEMO ONLY:',json.dumps(report['metrics']))
    if failures:
        print('REGRESSION: '+'; '.join(failures))
        return 1
    print('PASS: no gated regression')
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/ci_retriever.json')
    p.add_argument('--outdir',default='reports/ci')
    a=p.parse_args()
    try:
        config=read(ROOT/a.config)
        if set(config)!= {'mode'}:
            raise ValueError('Config must contain exactly the mode field.')
        return execute(config['mode'],ROOT/a.outdir)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print('INVALID CI CONFIGURATION:',exc)
        return 2

if __name__=='__main__':raise SystemExit(main())
