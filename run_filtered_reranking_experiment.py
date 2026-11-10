import hashlib
import json
import math
import time
from pathlib import Path

from huggingface_hub import HfApi
from sentence_transformers import CrossEncoder
from harness.evaluate import run, compare, fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    source = Path("reports/semantic_experiment_v1")
    output = Path("reports/filtered_reranking_experiment_v1")
    data = Path("private_data/transformer")

    corpus = read(data / "corpus.json")
    cases = read(data / "cases.json")
    baseline = read("reports/reranking_experiment_v1/reranked.json")
    pools_path = source / "candidate_rankings.json"
    pools = read(pools_path)

    expected = fingerprint({"corpus": corpus, "cases": cases})
    if baseline["dataset_sha256"] != expected:
        raise ValueError("Dataset differs from the hybrid report.")
    if baseline["k"] != 3 or baseline["demo"]:
        raise ValueError("Expected a real-data top-three baseline.")

    selected = [c for c in cases if c["split"] == baseline["split"]]
    if set(pools) != {c["id"] for c in selected}:
        raise ValueError("Candidate pools do not match the selected cases.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output folder is not empty; choose a new run folder.")

    passages = {p["id"]: p for p in corpus}

    # Validate candidates and reproduce the existing hybrid top three.
    baseline_rows = {c["id"]: c for c in baseline["cases"]}
    for case in selected:
        ids = [p["id"] for p in pools[case["id"]]["fusion_full_pool"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate IDs.")
        for pid in ids:
            if pid not in passages:
                raise ValueError("Unknown passage ID.")
            if passages[pid]["document_id"] not in case["document_ids"]:
                raise ValueError("Candidate outside the document scope.")
       

    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision = HfApi().model_info(model_name).sha
    model = CrossEncoder(
        model_name,
        revision=revision,
        device="cpu",
        trust_remote_code=False,
    )

    predictions = {}
    audit = {}
    started = time.perf_counter()

    for case in selected:
        cid = case["id"]
        original_ids = [
            p["id"] for p in pools[cid]["fusion_full_pool"]
        ]

        # Fixed rule applied identically to every question.
        excluded_ids = [
            pid for pid in original_ids
            if len(passages[pid]["text"].split()) < 5
        ]
        ids = [
            pid for pid in original_ids
            if len(passages[pid]["text"].split()) >= 5
        ]

        pairs = [
            [case["question"], passages[pid]["text"]]
            for pid in ids
        ]
        
        # Record inputs exceeding the model's limit.
        lengths = [
            len(model.tokenizer(q, text, truncation=False)["input_ids"])
            for q, text in pairs
        ]

        scores = model.predict(
            pairs, batch_size=16, show_progress_bar=False
        ) if pairs else []

        values = [float(score) for score in scores]
        if len(values) != len(ids) or not all(map(math.isfinite, values)):
            raise ValueError("Invalid reranker scores.")

        ranked = sorted(zip(ids, values), key=lambda item: (-item[1], item[0]))
        predictions[cid] = [pid for pid, score in ranked[:3]]
        audit[cid] = {
            "minimum_words": 5,
            "excluded_ids": excluded_ids,
            "excluded_labeled_evidence": [
                pid for pid in excluded_ids
                if pid in case["relevant_ids"]
            ],
            "ranking": [{"id": pid, "score": score} for pid, score in ranked],
            "over_limit_ids": [
                pid for pid, length in zip(ids, lengths)
                if length > model.max_seq_length
            ],
        }

    elapsed = time.perf_counter() - started
    report = run(
        corpus, cases, k=3, split=baseline["split"],
        predictions=predictions,
    )
    regressions = compare(baseline, report)
    report["experiment"] = {
        "model": model_name,
        "model_revision": revision,
        "max_length": model.max_seq_length,
        "candidate_source": str(pools_path),
        "candidate_source_sha256": hashlib.sha256(
            pools_path.read_bytes()
        ).hexdigest(),
        "scoring_and_ranking_seconds": elapsed,
        "note": "Development retrieval experiment; not answer accuracy.",
    }

    output.mkdir(parents=True, exist_ok=True)
    save(output / "reranked.json", report)
    save(output / "rankings.json", audit)

    print("\nMethod          Recall     MRR      nDCG")
    for name, result in [("Hybrid", baseline), ("Reranked", report)]:
        m = result["metrics"]
        print(f"{name:14} {m['recall']:.4f}   {m['mrr']:.4f}   {m['ndcg']:.4f}")

    print("\nGate against hybrid:", "FAIL" if regressions else "PASS")
    for reason in regressions:
        print(reason)

    print("\nPer-question results:")
    for row in report["cases"]:
        print(row["id"], row["metrics"])

    print("\nReports saved to:", output)


if __name__ == "__main__":
    main()