from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_golden_dataset(path: str | None = None) -> List[Dict[str, Any]]:
    path = path or str(Path(__file__).resolve().parent / "golden_dataset.json")
    if not Path(path).exists():
        return [
            {
                "query": "What is this document about?",
                "expected_answer": "A summary of the document content.",
                "contexts": ["The document discusses the main themes and supporting details."],
            }
        ]
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _simple_faithfulness(answer: str, contexts: List[str]) -> float:
    if not answer:
        return 0.0
    overlap = sum(1 for context in contexts if context.lower() in answer.lower())
    return round(overlap / max(1, len(contexts)), 2)


def run_evaluation(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for item in dataset:
        answer = item.get("expected_answer", "")
        contexts = item.get("contexts", [])
        results.append(
            {
                "query": item["query"],
                "expected_answer": answer,
                "faithfulness": _simple_faithfulness(answer, contexts),
                "context_recall": 1.0 if contexts else 0.0,
            }
        )

    return {
        "count": len(results),
        "results": results,
        "summary": {
            "avg_faithfulness": round(sum(item["faithfulness"] for item in results) / len(results), 2) if results else 0.0,
            "avg_context_recall": round(sum(item["context_recall"] for item in results) / len(results), 2) if results else 0.0,
        },
    }


if __name__ == "__main__":
    dataset = load_golden_dataset()
    print(json.dumps(run_evaluation(dataset), indent=2))
