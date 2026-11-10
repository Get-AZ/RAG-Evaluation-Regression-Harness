"""Prepare frozen-context prompts; --generate explicitly enables Gemini requests."""
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from harness.evaluate import fingerprint, run
from harness.answer_review import template
from verify_agents_holdout import main as verify_data
from frozen_pipeline import identity

ROOT = Path(__file__).resolve().parent
RULES = '''Answer only the requested question.
Include relevant facts and accurate page citations.
Do not discuss prior work or unrelated metrics unless asked.
If the requested information is missing, say so briefly.
Preserve mathematical notation accurately.
Read related sentences together. A general statement applies to the models discussed immediately afterward unless the text gives an exception.
Before declaring a detail missing, check all reference passages.'''

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing file differs: {path}. Use a new --outdir.')
    else:
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False)

def prepare(corpus, cases, report):
    if report['demo'] or report['split'] != 'heldout' or report['k'] != 3:
        raise ValueError('Expected a real-data held-out report with k=3.')
    if report['dataset_sha256'] != fingerprint({'corpus':corpus,'cases':cases}):
        raise ValueError('Retrieval report dataset mismatch.')
    selected = [c for c in cases if c['split'] == 'heldout']
    rows = report['cases']
    predictions = {r['id']:r['ranked_ids'] for r in rows}
    if len(predictions) != len(rows) or any(len(v)>3 for v in predictions.values()):
        raise ValueError('Duplicate cases or excessive context.')
    checked = run(corpus, cases, k=3, split='heldout', predictions=predictions)
    if checked['cases'] != rows or checked['metrics'] != report['metrics']:
        raise ValueError('Retrieval report scores do not reproduce.')
    passages = {p['id']:p for p in corpus}
    prompts = {}
    for c in selected:
        ids = predictions[c['id']]
        context = '\n\n'.join(
            f'[Passage {i}; page {passages[pid]["page"]}]\n{passages[pid]["text"]}'
            for i,pid in enumerate(ids,1))
        prompts[c['id']] = {
            'passage_ids':ids,
            'prompt':f'{RULES}\n\nReference passages:\n{context}\n\n\nQuestion:\n{c["question"]}',
        }
    return prompts

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--model',choices=['gemini-3.6-flash'],default='gemini-3.6-flash')
    parser.add_argument('--outdir',default='reports/agents_holdout_answers_v1')
    args = parser.parse_args()
    verify_data()
    data = ROOT/'private_data/agents_taxonomy'
    corpus,cases = read(data/'corpus.json'),read(data/'cases.json')
    retrieval_path = ROOT/'reports/agents_holdout_v1/retrieval.json'
    report = read(retrieval_path)
    config = read(data/'frozen_config.json')
    if report['frozen_pipeline']['identity'] != identity(config):
        raise ValueError('Retrieval provenance differs from the frozen runner/environment.')
    prompts = prepare(corpus,cases,report)
    output = ROOT/args.outdir
    output.mkdir(parents=True,exist_ok=True)
    write_once(output/'prompts.json',prompts)
    plan = {'dataset_sha256':report['dataset_sha256'],
            'retrieval_report_sha256':fingerprint(report),
            'prompts_sha256':fingerprint(prompts),
            'script_sha256':fingerprint(Path(__file__).read_text(encoding='utf-8')),
            'scope':'first held-out frozen-context answer evaluation; not Streamlit app evaluation',
            'generation_config':{'max_output_tokens':1500},
            'timeout_ms':60000,'retry_attempts':1,'minimum_request_start_interval_seconds':14,
            'requested_model':'gemini-3.6-flash'}
    write_once(output/'plan.json',plan)
    print(f'VERIFIED: {len(prompts)} prompts from saved top-three retrieval results.',flush=True)
    if not args.generate:
        print('DRY RUN: no API requests. Reference answers and review labels are absent from prompts.')
        print(f'Prompts saved to: {output / "prompts.json"}')
        return 0
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    load_dotenv(ROOT/'.env')
    expected_packages = read(ROOT/'reports/answer_evaluation_v2/generation_environment.json')['packages']
    if any(importlib.metadata.version(p)!=v for p,v in expected_packages.items()):
        raise ValueError('Gemini dependency versions differ from the development answer run.')
    key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    model = args.model
    if model != config['generation_model']:
        raise ValueError('Generation model differs from frozen configuration.')
    if not key or not model:
        raise ValueError('Set GOOGLE_API_KEY and GEMINI_MODEL in the project .env, or supply --model.')
    environment = {'requested_model':model,'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in ['google-genai','python-dotenv']}}
    write_once(output/'generation_environment.json',environment)
    records = output/'responses'
    records.mkdir(exist_ok=True)
    answers = {}
    previous_request_start = None
    with genai.Client(api_key=key,http_options=types.HttpOptions(
            timeout=60000,retry_options=types.HttpRetryOptions(attempts=1))) as client:
        for cid,item in prompts.items():
            target = records/f'{cid}.json'
            if target.exists():
                record = read(target)
                if record['prompt_sha256'] != fingerprint(item['prompt']) or not record['answer'].strip():
                    raise ValueError('Cached response mismatch.')
                answers[cid] = record['answer']
                print(f'{cid}: reused saved response',flush=True)
                continue
            if previous_request_start is not None:
                wait = max(0,14-(time.monotonic()-previous_request_start))
                if wait:
                    print(f'{cid}: pacing requests for {wait:.1f}s...',flush=True)
                    time.sleep(wait)
            previous_request_start = time.monotonic()
            print(f'{cid}: requesting answer...',flush=True)
            started = time.perf_counter()
            result = client.models.generate_content(model=model,contents=item['prompt'],
                config=types.GenerateContentConfig(max_output_tokens=1500))
            answer = result.text
            if not answer or not answer.strip():
                raise ValueError(f'{cid}: empty response; no review score assigned.')
            record = {'answer':answer,'prompt_sha256':fingerprint(item['prompt']),
                'passage_ids':item['passage_ids'],'elapsed_seconds':time.perf_counter()-started,
                'response':json.loads(result.model_dump_json(exclude_none=True))}
            write_once(target,record)
            answers[cid] = answer
            print(f'{cid}: saved',flush=True)
    write_once(output/'answers.json',answers)
    review_path = output/'answer_reviews.csv'
    if not review_path.exists():
        rows = template(corpus,cases,answers,'heldout')
        with review_path.open('x',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print('COMPLETE: answers and pending human-review template saved. No answer-quality score assigned.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
