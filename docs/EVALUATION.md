# Evaluation record and limits

## Measured development sequence

All rows use the same 484 Transformer PDF blocks, nine answerable development questions and final k=3. The tenth question requests an unreported electricity cost and is excluded from retrieval metrics.

| Candidate | Recall@3 | MRR@3 | nDCG@3 |
|---|---:|---:|---:|
| Original lexical | 0.3333 | 0.3333 | 0.3333 |
| Stopword-overlap | 0.4444 | 0.3333 | 0.3624 |
| MiniLM dense | 0.4444 | 0.2778 | 0.3214 |
| Dense + stopword-overlap RRF | 0.6667 | 0.4259 | 0.4881 |
| Cross-encoder reranking | 0.8333 | 0.7778 | 0.7798 |
| Same pool, five-word filter before reranking | 0.9444 | 0.8889 | 0.8909 |

Source: original locally saved experiment reports, inspected in the November 10, 2026 release archive. These are repeated development experiments, not independent held-out comparisons. Stopword-overlap helped one case but lowered another case's rank; aggregate gates alone did not expose that tradeoff. The unbudgeted neighbor-expansion diagnostic was not promoted or presented as Recall@3.

The filter removed 19 tiny fragments from TR-DEV-009's pool. It excluded no labeled reference evidence in this development run. This does not prove short passages are generally useless: short equations, definitions or captions can be valuable.

## Frozen settings

| Setting | Value |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding commit | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Dimensions / maximum sequence length | 384 / 256 tokens |
| Dense score | Exact cosine ranking of normalized vectors |
| Lexical arm | Unique-token overlap with the fixed stopword list in source |
| Fusion | Equal-weight RRF; top 20 per arm; constant 60 |
| Filter | At least five whitespace-separated words, after fusion and before reranking; no replacement candidates |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Reranker commit | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| Reranker maximum sequence length | 512 tokens |
| Tie-break | Descending score, then passage ID |
| Final context | At most three passages |
| Generation | `gemini-3.6-flash`, maximum 1500 output tokens; no explicit temperature override |
| Runtime | Windows, Python 3.14.4, CPU; package snapshot supplied |

Gemini's served model/version and defaults may change. Saved raw responses retain available provider metadata. A model name is not a guarantee of deterministic regeneration. Requests resumed after temporary 503 demand and 429 quota errors; completed responses were retained, and failed requests were not counted as incorrect answers. A single generated response per case was reviewed; no repeated-sampling study was performed.

## Held-out evaluation

Source: Sapkota, Roumeliotis and Karkee, *AI Agents vs. Agentic AI: A Conceptual taxonomy, applications and challenges*, Information Fusion 126 (2026), 103599, DOI `10.1016/j.inffus.2025.103599`. Supplied PDF SHA256: `70a0c89f937f35af4e6af06dff86be7121f67882130d787bdf6cc0ff9730bba7`.

The 30-page document produced 754 nonempty native text blocks. The user reviewed six direct, three multiple-passage and three missing-information cases before retrieval. Questions and references were assisted by an AI and explicitly approved by one human reviewer; this is not an independent blinded multi-reviewer study. Multiple questions share source passages.

All six direct questions recovered their labeled reference evidence. Multiple-passage reference coverage was 1/3 for AG-HOLD-007, 2/3 for AG-HOLD-008 and 1/2 for AG-HOLD-009. The macro-average Recall@3 was 0.8333. MRR@3 was 0.8148 and nDCG@3 was 0.7798.

The audit recorded 25 passage inputs beyond the embedding limit and two query-passage pairs beyond the reranker limit. No labeled reference passage was among those over-limit inputs. Other candidates can still be affected, so no causal conclusion that truncation caused or did not cause ranking failures is justified. Full selected passage text, rather than the encoder's truncated representation, was supplied to Gemini.

## Human-reviewed generated answers

Each criterion is binary. An answer passes overall only if all four criteria pass. Citation correctness evaluates support and page accuracy; page-only citations were accepted. Brief context-limited missing-information answers did not require fabricated citations. `behavior_correct` is not a separate comprehensive factual-correctness score.

| Split | Grounded | Complete | Behavior correct | Citations correct | All criteria |
|---|---:|---:|---:|---:|---:|
| Development, n=10 | 10/10 | 9/10 | 10/10 | 10/10 | 9/10 |
| Held-out, n=12 | 12/12 | 10/12 | 12/12 | 12/12 | 10/12 |

Retained failures:

- **TR-DEV-008:** omitted an explicit encoder self-attention description; relevant context was incomplete.
- **AG-HOLD-007:** described tool invocation and result integration but omitted structured request generation and orchestration-layer execution specified by the approved reference answer.
- **AG-HOLD-009:** omitted explainability deficits and adversarial vulnerabilities specified by the reference answer.

AG-HOLD-008 was judged adequately answered from overlapping retrieved evidence despite partial reference-ID recall. This illustrates that labels are not exhaustive and retrieval reference recall is not answer accuracy. The original labels and scores were retained rather than revised after seeing results.

All three held-out missing-information questions were answered without inventing the requested search expenditure, Cohen's kappa, or measured smart-home latency. Three related tests on one paper are insufficient to estimate general hallucination or abstention reliability.

## Interpretation

The evaluator fingerprints the full dataset and rejects incompatible report comparisons. Development and held-out scores must not be passed to the same regression gate because they describe different datasets. No minimum held-out threshold was declared; the first results were reported rather than optimized to pass a retroactive target.

The frozen pipeline is a block-based evaluation experiment. It does not use FAISS and differs from the earlier Streamlit chatbot's 120-word chunks and neighbor expansion. No production SLA, p95 latency, dollar cost, independent replication or large-scale benchmark is claimed.
