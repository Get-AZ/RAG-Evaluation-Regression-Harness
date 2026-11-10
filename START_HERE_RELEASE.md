# Install the release files in your existing project

1. Extract this public release ZIP into a temporary folder.
2. Copy its contents, including `.github`, `.gitignore` and `.gitattributes`, into `C:\Users\hiw\Desktop\rag-evaluation-harness`. Merge folders. The supplied historical scripts are byte-for-byte copies from your release-review archive. Preserve your existing private_data, reports, .env and .venv-semantic folders; this ZIP does not contain replacements for them.
3. In VS Code PowerShell from the project root, run these one at a time:

```powershell
.\.venv-semantic\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-semantic\Scripts\python.exe -m unittest test_retrieval_experiment test_semantic_experiment test_frozen_pipeline test_ci_gate -v
.\.venv-semantic\Scripts\python.exe run_ci_gate.py
```

Expected: 16 tests pass, then 28 tests pass, then the synthetic gate prints PASS. No model downloads or API requests occur.

4. Share those outputs before initializing Git. GitHub publishing and the hosted bad-change demonstration are the following steps; do not upload the original release-review ZIP.

The public README includes verified real-paper summary results and known failures. Private passages, PDFs, reviewer records, raw API responses and local reports are excluded. The public CI currently enforces a synthetic fixture, not a fresh Gemini or full real-model evaluation.
