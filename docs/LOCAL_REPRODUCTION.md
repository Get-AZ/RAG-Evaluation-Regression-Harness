# Local reproduction

## Public clone

The public synthetic demo and 44 standard-library tests are self-contained. Run the three commands in the README. Tests include simulated model objects where appropriate; they do not constitute real embedding inference. No Gemini requests run in CI.

## Existing author project

Copy the public release files into the existing project, preserving all `private_data`, `reports`, `.env` and virtual environments. Existing experiment scripts in this package were copied byte-for-byte from the supplied release archive. `.gitattributes` preserves script/JSON byte fingerprints across Git checkouts. Do not let an editor normalize original script line endings before replaying.

The `.gitignore` excludes local private evidence, raw responses, reports, PDFs, keys, virtual environments, cached bytecode and ZIP backups. It does not remove already tracked files; inspect the staged file list before the first commit. Publish only the public files, not the entire release-review archive.

## Full measured pipeline

Required private paths include:

- `private_data/transformer/` and `private_data/pdfs/transformer.pdf`.
- `private_data/agents_taxonomy/` and `private_data/pdfs/agents_taxonomy.pdf`.
- Original reports under `reports/semantic_experiment_v1`, `reports/filtered_reranking_experiment_v1`, `reports/answer_evaluation_v2`, and the successful replay certificate/report.

These are retained locally by the author. A public clone does not contain the human-review ground truth or original result records and therefore cannot independently reproduce the recorded real-PDF metrics without those inputs. PDF sources alone are insufficient: extraction-version differences can change native block IDs. Do not fabricate approval metadata to bypass validation.

The original Transformer PDF SHA256 is `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`; source: Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762. The agents paper is attributed in EVALUATION.md. Recheck source licensing and review metadata before choosing to distribute a separate data artifact.

The Windows environment snapshot is `requirements-windows-lock.txt`. It includes CPU PyTorch. For an equivalent Windows Python 3.14 environment, the original installation used the PyTorch CPU package index, then sentence-transformers and the Google Gen AI SDK. Fresh installation from this snapshot has not been tested as part of release packaging. Do not change a working environment to obtain prettier version strings.

With the measured environment and private files present:

```powershell
.\.venv-semantic\Scripts\python.exe verify_agents_holdout.py
.\.venv-semantic\Scripts\python.exe frozen_pipeline.py check
```

The archived successful replay is retained. A new replay needs a fresh output folder:

```powershell
.\.venv-semantic\Scripts\python.exe -u frozen_pipeline.py replay-dev --outdir reports/replay_release_check
```

Success requires exact rankings and fusion pools. If the candidate script changes, its previous replay certificate no longer applies. Preserve mismatches rather than weakening the check.

The first held-out run has already been observed and archived. Do not repeatedly tune on it or present later tuned runs as untouched held-out results. The runner can be used for reproduction, with explicit new output paths and the matching replay certificate, but this does not create another independent test set.

## Answer generation

The recorded prompt uses retrieved context and the question, not the reference answer or reviewer notes. `run_answer_evaluation.py` and `run_heldout_answers.py` default to dry runs. Only `--generate` enables requests. Store the real API key in the ignored local `.env`, using `.env.example` solely as a template.

The held-out script checks the original retrieval provenance and Gemini dependency versions, uses the recorded model name, paces new request starts, and reuses saved responses. Pacing cannot guarantee avoidance of daily quotas or server demand errors. API usage may incur charges or consume quota. Never change the model mid-run while claiming a single-model result.

Completed answer-review CSVs are bound to dataset and answer fingerprints. Editing generated answers invalidates the old review. Scoring reviews locally makes no API requests:

```powershell
.\.venv-semantic\Scripts\python.exe -m harness.answer_review score --corpus private_data/agents_taxonomy/corpus.json --cases private_data/agents_taxonomy/cases.json --answers reports/agents_holdout_answers_v1/answers.json --reviews reports/agents_holdout_answers_v1/answer_reviews.csv --split heldout --out reports/agents_holdout_answers_v1/human_review_scores.json
```
