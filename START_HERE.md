# Transformer development evaluation data - Stage 2

This is an **add-on to the existing rag-evaluation-harness project**. The uploaded
CSV was validated: 10 approved development cases, 9 answerable and 1 missing-
information case. Its document hash matches the uploaded PDF exactly.

## Install this data pack

1. Extract this ZIP to a temporary folder.
2. Copy its **private_data** folder and **verify_transformer_data.py** into your
   existing `rag-evaluation-harness` folder. Merge the private_data folders when
   Windows asks. No existing harness code or original review CSV is replaced.
3. Keep `START_HERE.md` handy for these commands. Open the existing project in VS Code.
4. The PDF should be at `private_data/pdfs/transformer.pdf` inside the project.
   If not, copy your desktop PDF there using the PowerShell commands below.

```powershell
New-Item -ItemType Directory -Force "private_data\pdfs"
Copy-Item -LiteralPath "C:\Users\hiw\Desktop\Attention Is All You Need..pdf" -Destination "private_data\pdfs\transformer.pdf"
```

The PDF is not duplicated inside this ZIP. Do not copy a different edition under
that filename: the verifier checks the file contents, not just the name.

## Step 1 - verify the inputs

In the VS Code terminal, from the existing project root:

```powershell
.\.venv\Scripts\python.exe verify_transformer_data.py
```

Expect four VERIFIED lines. It checks the PDF hash, packaged file hashes, all
source-page positions, approved-case schema and baseline dataset fingerprint.
It also checks that your working review CSV has not changed from the uploaded
version. If that check fails, preserve your changes and regenerate the data pack.
No installs, embeddings, model downloads or API key are required.

## Step 2 - run retrieval on actual document text

```powershell
.\.venv\Scripts\python.exe -m harness.evaluate run --corpus private_data/transformer/corpus.json --cases private_data/transformer/cases.json --split dev --k 3 --out reports/transformer-lexical-candidate.json
```

Expected with the original Stage 1 runner and this fixed pack:

```text
{"recall": 0.3333333333333333, "mrr": 0.3333333333333333, "ndcg": 0.3333333333333333}
```

These are **real-document, development-only retrieval scores for the starter's
simple lexical retriever**, not synthetic demo scores. Do not add `--demo`.
They do not measure your MiniLM/FAISS/Gemini app, generation quality, held-out
performance or general accuracy. All three metrics happen to coincide here.
Three of nine answerable questions retrieve their labeled supporting evidence
at rank 1; six do not recover their required labeled evidence within top 3.
The missing-information case has no retrieval ranking score. No answer has been
generated or automatically judged.

## Step 3 - compare with the frozen lexical baseline

```powershell
.\.venv\Scripts\python.exe -m harness.evaluate compare --baseline private_data/transformer/lexical_baseline_k3.json --candidate reports/transformer-lexical-candidate.json
```

Expected: `PASS: no gated regression`. This means the candidate matches the
approved comparison rule, not that retrieval quality is adequate. A weak baseline
can be reproducible. Baseline promotion still requires explicit review; do not
regenerate it automatically during changes.

## Step 4 - deliberately break retrieval on the real development corpus

```powershell
.\.venv\Scripts\python.exe -m harness.evaluate run --corpus private_data/transformer/corpus.json --cases private_data/transformer/cases.json --split dev --k 3 --mode empty --out reports/transformer-broken.json
.\.venv\Scripts\python.exe -m harness.evaluate compare --baseline private_data/transformer/lexical_baseline_k3.json --candidate reports/transformer-broken.json
$LASTEXITCODE
```

Expected: `REGRESSION` and exit code 1. Inspect `$LASTEXITCODE` immediately after
the compare command. Exit code 2 means invalid inputs, not a successful test of
the quality gate. Hosted CI and required branch protection remain pending.

## What the files contain

- `corpus.json`: **484 native PDF text blocks from all 15 pages**, including
  headings, references, captions, equation fragments and distractor material.
  Each entry has a source SHA-256, PDF page, frozen block ID, original raw text,
  normalized searchable text, bounding box and raw-text hash.
- `cases.json`: the 10 CSV reviews converted to the Stage 1 schema. For case 008,
  the reviewed comparison maps to two source blocks on page 3, not an invented
  joined passage. Case 010 has an empty relevant_ids list and answerable=false.
- `review_source.csv`: an unchanged copy of your approved uploaded worksheet.
- `evidence_mapping.json`: case-to-source mapping with notes on omitted text and
  mathematical notation normalization. The review quotes were preserved.
- `manifest.json`: extraction version/settings, file hashes, counts and limitations.
- `lexical_baseline_k3.json`: complete per-case rankings, metrics and category slices.

The exact reviewed PDF SHA-256 is:
`bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`.

## Important interpretation limits

- These are layout-derived text blocks, **not the 120-word overlapping chunks
  used by your Single-PDF app**. Do not compare this report directly with an app
  report that has a different passage inventory or k. Next we will implement the
  adapter and an evidence mapping suitable for chunking comparisons.
- All native nonempty text blocks are retained to avoid using the gold labels to
  build an artificially easy corpus. This leaves tiny fragments and extraction
  noise in the corpus. A later cleaning/chunking experiment must be measured and
  versioned rather than silently replacing this inventory.
- Figure pixels are not represented and no OCR was performed. Equations and
  tables can lose spatial structure. The extraction includes text found on all
  pages; it does not claim to capture every fact visible in figures.
- The gold set lists the specific blocks reviewed with you. Other blocks can
  contain equivalent correct evidence but have not been exhaustively relevance-
  judged. A scored miss may therefore require manual review of alternative
  evidence. Treat these as reference-evidence recovery scores, not complete
  judgments of every possible supporting passage in the paper.
- The absence label in case 010 records the manual review and searches you
  reported. No quotation or page number is fabricated for an absent dollar cost.
- No held-out document or test question has been added. Transformer and YOLO
  remain familiar development material.
- Keep private_data out of public Git history. The original project's .gitignore
  already excludes it; the pack preserves review metadata for your local work.

## What to send back

First run **Step 1** and paste its output. Then run Steps 2 and 3. We will inspect
which cases fail before connecting or changing the real chatbot. None of these
commands opens a Streamlit UI or calls Gemini.
