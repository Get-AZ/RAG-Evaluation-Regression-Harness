# RAG Evaluation and Regression Harness

An evaluation project by **Getamesay Zegeye** that separates evidence retrieval from generated-answer quality, preserves human-reviewed ground truth, and detects regressions before a candidate is accepted.

The development workflow moved from lexical retrieval to MiniLM semantic search, reciprocal-rank fusion, cross-encoder reranking, and a fixed short-fragment filter. A pinned runner reproduced every saved development top-three ranking and fusion pool before its first evaluation on a different research PDF.

## Recorded results

| Measure | Development: Transformer paper | Held-out: AI Agents vs. Agentic AI paper |
|---|---:|---:|
| PDF pages / extracted passages | 15 / 484 | 30 / 754 |
| Answerable retrieval cases | 9 | 9 |
| Missing-information cases | 1 | 3 |
| Recall@3 | 0.9444 | 0.8333 |
| MRR@3 | 0.8889 | 0.8148 |
| nDCG@3 | 0.8909 | 0.7798 |
| Generated answers passing all four human-review criteria | 9/10 | 10/12 |
| Missing-information answers passing review | 1/1 | 3/3 |

These are results on two small, correlated, manually reviewed research-paper datasets. They are **not general accuracy estimates**, a production service benchmark, or measurements of the existing FAISS Streamlit app. Held-out means excluded from this project's tuning; model pretraining exposure is unknown. The reference answers and evidence labels were not sent to Gemini.

The machine-readable public summary is [docs/evaluation_summary.json](docs/evaluation_summary.json). Raw PDFs, source passages, reviewer records, prompts, API responses and local reports remain outside public Git history by default.

## Run the offline public checks

Python 3.14 is used by the proposed GitHub workflow. The evaluator, synthetic demo and helper tests use the Python standard library; no pip installation, model download, GPU or API key is needed for these checks.

From the repository root:

```powershell
python -m unittest discover -s tests -v
python -m unittest test_retrieval_experiment test_semantic_experiment test_frozen_pipeline test_ci_gate -v
python run_ci_gate.py
```

The gate prints `DEMO ONLY` and `PASS: no gated regression`, returning exit code 0. It evaluates the configured demo retriever against the frozen synthetic baseline. It does not refresh real-model predictions or score fresh Gemini answers.

## Demonstrate rejection

On a separate branch, change `config/ci_retriever.json` from `"lexical"` to `"empty"`. The same gate then returns 1 and reports declines from 1.0 to 0.0. In GitHub Actions, the enforcement step should fail. Follow [docs/CI_DEMONSTRATION.md](docs/CI_DEMONSTRATION.md) and retain links to an actual passing workflow and a failing pull request. Hosted execution is pending until those runs are observed; this repository does not claim branch protection is already enabled.

The unit test that expects a bad candidate to fail should itself pass. The separate workflow enforcement step rejects the configured bad candidate. This distinction prevents a green negative-control test from being mistaken for a demonstrated red pull request.

## Evaluation design

- **Identity and provenance:** SHA256 PDF identities, fixed passage IDs, dataset fingerprints and recorded model revisions.
- **Human evidence review:** approved questions, source excerpts, reference answers and missing-information judgments.
- **Separation:** different documents for development and held-out evaluation; overlapping documents are rejected by validation.
- **Retrieval:** fixed k=3, macro-average labeled-evidence recall, reciprocal rank and binary nDCG over answerable cases only.
- **Regression comparison:** same dataset fingerprint, split, k and case coverage required. Aggregate drops greater than 0.01 fail; improvements cannot compensate for a drop in another gated metric. Individual-case declines can still be hidden by aggregate scores, so case-level review is retained.
- **Answer review:** supported claims (`grounded`), completeness, appropriate answering/abstention (`behavior_correct`), and correct citations. Reviews are bound to the exact generated answer and dataset fingerprint.
- **Replay:** pinned model revisions, package/version checks and exact development ranking reproduction before held-out execution.

See [docs/EVALUATION.md](docs/EVALUATION.md) for settings, limitations, experiment history and retained failures.

## Local real-model workflow

The source scripts for real-model experiments are preserved, but their original private data and reports are not included in a public clone. Instructions and prerequisites are in [docs/LOCAL_REPRODUCTION.md](docs/LOCAL_REPRODUCTION.md). They require the original approved data artifacts and local reports; a fresh extraction is not automatically the same benchmark.

`requirements-windows-lock.txt` records the measured Windows environment. It is an environment snapshot, not a verified cross-platform installation recipe. Do not install it for the offline CI/demo checks. Gemini generation requires a local `.env`; only placeholders are included in `.env.example`.

## Main files

| File | Purpose |
|---|---|
| `harness/evaluate.py` | Validation, retrieval metrics and comparable-report regression gate |
| `harness/answer_review.py` | Human-review templates, answer fingerprints and scoring |
| `run_retrieval_experiment.py` | Controlled lexical/stopword/BM25 variants |
| `run_semantic_experiment.py` | MiniLM and hybrid retrieval experiments |
| `run_filtered_reranking_experiment.py` | Historical development candidate; preserve unchanged |
| `frozen_pipeline.py` | Pinned development replay and held-out retrieval |
| `run_answer_evaluation.py` / `run_heldout_answers.py` | Frozen-context Gemini generation and resumable outputs |
| `run_ci_gate.py` | Public synthetic regression enforcement |
| `.github/workflows/regression.yml` | Offline tests and gate on pushes/pull requests |

The real-model scripts are intentionally left unchanged to preserve recorded fingerprints. They can emit upstream deprecation warnings. Any future refactor needs its own replay rather than silently changing the measured candidate.

## Project status

Completed: human-reviewed development and held-out datasets; controlled retrieval experiments; exact development replay on the author's machine; generated-answer reviews; local negative-control regression rejection.

Remaining release evidence: an observed GitHub Actions passing run, an observed deliberately bad pull request rejected by the gate, and optional branch-protection configuration. CI supplied here protects synthetic evaluator behavior; it does not establish that the full real-model pipeline is evaluated on every commit.

Future work includes evaluating more documents/domains, more diverse abstention cases, explicit alternative-evidence judgments, context-budget analysis and service-level latency/cost measurements. Tuning after inspecting this held-out set requires a new untouched test set.
