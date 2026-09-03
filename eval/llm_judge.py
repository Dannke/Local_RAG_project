"""LLM-as-judge evaluation for faithfulness and relevance."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_project.config import load_settings  # noqa: E402
from rag_project.llm.llm_client import OpenRouterClient  # noqa: E402
from rag_project.pipelines.chat_pipeline import ChatSession  # noqa: E402

FAITHFULNESS_PROMPT = (
    "You are an expert evaluator. Assess whether the ANSWER is faithful "
    "to the provided CONTEXT.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Rate on a scale of 1-5:\n"
    "1 = Hallucinated/contradicts context\n"
    "2 = Mostly unfaithful\n"
    "3 = Partially faithful\n"
    "4 = Mostly faithful\n"
    "5 = Fully faithful to context\n\n"
    'Respond with ONLY a JSON object: {"score": <int>, "reasoning": "<brief>"}'
)

RELEVANCE_PROMPT = (
    "You are an expert evaluator. Assess whether the ANSWER is relevant "
    "and helpful for the QUESTION.\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Rate on a scale of 1-5:\n"
    "1 = Irrelevant/does not address question\n"
    "2 = Barely relevant\n"
    "3 = Partially relevant\n"
    "4 = Mostly relevant\n"
    "5 = Highly relevant and complete\n\n"
    'Respond with ONLY a JSON object: {"score": <int>, "reasoning": "<brief>"}'
)


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_judge_response(response: str) -> tuple[int, str] | None:
    try:
        data = json.loads(response.strip())
        return int(data.get("score", 0)), data.get("reasoning", "")
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def judge_answer(client: OpenRouterClient, prompt: str) -> tuple[int, str]:
    try:
        response = client.generate_answer("Evaluate:", [prompt])
        parsed = parse_judge_response(response)
        if parsed:
            return parsed
    except Exception as e:
        print(f"[WARN] Judge failed: {e}")
    return 0, "judge error"


def evaluate(golden_path: Path, index_dir: Path, output_path: Path, top_k: int = 5) -> dict:
    settings = load_settings()
    golden = load_golden(golden_path)

    client = OpenRouterClient.from_settings(settings, rate_limit_key="eval")
    session = ChatSession.from_faiss_index(index_dir=index_dir, settings=settings)

    results = []
    faithful_sum = 0
    relevance_sum = 0
    total = 0

    for item in golden:
        question = item["question"]
        expected_answer = item.get("expected_answer", "")

        try:
            response = session.ask(question, top_k=top_k)
            answer = response.answer
            context = "\n\n".join(
                f"[{i+1}] {r.document.text[:500]}"
                for i, r in enumerate(response.results)
            )

            # Faithfulness
            faithful_prompt = FAITHFULNESS_PROMPT.format(
                context=context, question=question, answer=answer
            )
            faithful_score, faithful_reason = judge_answer(client, faithful_prompt)

            # Relevance
            relevance_prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
            relevance_score, relevance_reason = judge_answer(client, relevance_prompt)

            results.append({
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "faithfulness": {"score": faithful_score, "reasoning": faithful_reason},
                "relevance": {"score": relevance_score, "reasoning": relevance_reason},
            })

            faithful_sum += faithful_score
            relevance_sum += relevance_score
            total += 1

            time.sleep(0.5)  # Be nice to API

        except Exception as e:
            print(f"[ERROR] Failed on question: {question[:50]}... - {e}")
            results.append({
                "question": question,
                "error": str(e),
            })

    return {
        "faithfulness_avg": faithful_sum / total if total else 0,
        "relevance_avg": relevance_sum / total if total else 0,
        "questions_evaluated": total,
        "details": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden_set.jsonl")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "llm_judge_results.json")
    args = parser.parse_args()

    settings = load_settings()
    index_dir = args.index_dir or settings.vector_store_dir

    if not settings.openrouter_api_key or settings.openrouter_api_key == "sk-or-v1-your-key":
        print("ERROR: OPENROUTER_API_KEY not set in environment")
        return 1

    metrics = evaluate(args.golden, index_dir, args.output, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("LLM Judge metrics:")
    print(f"  faithfulness_avg: {metrics['faithfulness_avg']:.2f}")
    print(f"  relevance_avg: {metrics['relevance_avg']:.2f}")
    print(f"  questions_evaluated: {metrics['questions_evaluated']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())