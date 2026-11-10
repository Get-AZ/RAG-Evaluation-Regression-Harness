# Demonstrate a real hosted rejection

Status: the public gate has been tested locally. Actual GitHub workflow URLs and branch protection are pending; do not claim hosted CI rejection until observed.

## Healthy run

1. Publish the prepared public repository with `config/ci_retriever.json` containing `{"mode":"lexical"}`.
2. Open its Actions tab and wait for **RAG regression checks** to finish.
3. Record the run URL, commit SHA and successful status below.

## Deliberately bad change

In PowerShell from the Git repository:

```powershell
git switch -c demo/reject-empty-retriever
```

In the VS Code editor, change only `config/ci_retriever.json` to:

```json
{"mode":"empty"}
```

Then run:

```powershell
python run_ci_gate.py
$LASTEXITCODE
```

The expected exit code is **1**, with `REGRESSION` for recall, MRR and nDCG. Synthetic metrics decline from 1.0 to 0.0. This does not change the recorded real-paper scores.

Commit only this intended negative-control change and push the branch:

```powershell
git add config/ci_retriever.json
git commit -m "demo: intentionally return no retrieval results"
git push -u origin demo/reject-empty-retriever
```

Open a pull request. The unit tests should pass, while **Enforce synthetic retrieval regression gate** must fail. Record that run URL and pull-request URL. Close the pull request **without merging** and return to `main`. Do not edit the baseline or remove the gate to make this branch pass.

## Evidence to fill after observation

| Evidence | Value |
|---|---|
| Healthy Actions run URL | Pending |
| Healthy commit SHA | Pending |
| Deliberately bad PR URL | Pending |
| Failed enforcement run URL | Pending |
| Observed failure | Pending; expected metric regression and exit 1 |
| Main branch stayed healthy | Pending |

Optional: configure the repository's branch protection/ruleset to require the `evaluation-gate` job before merging. A red check does not by itself prevent administrators or unprotected branches from merging. Branch protection is a separate configuration step.

## Scope

This workflow is intentionally offline and uses a synthetic fixture. It tests evaluator behavior and enforcement wiring without API credentials, public PDF downloads or model weights. It does not rerun real-model retrieval or Gemini inference on every pull request. That broader regression protection remains future work or a separate private/manual workflow.
