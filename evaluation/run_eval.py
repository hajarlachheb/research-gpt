"""CLI script to run the RAG evaluation pipeline and print a report."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluator import RAGEvaluator
from evaluation.dataset import EvalDatasetManager


async def main(dataset_path: str | None = None, output_path: str | None = None) -> None:
    from api.dependencies import get_app_state

    state = get_app_state()

    async def ask_fn(question: str) -> tuple[str, list[str]]:
        rewritten = await state.query_rewriter.rewrite(question)
        retrieved = state.hybrid_retriever.retrieve(rewritten)
        reranked = state.reranker.rerank(rewritten, retrieved)
        context_text, _ = state.context_builder.build(reranked)
        answer, _ = await state.llm_generator.generate(question, context_text)
        contexts = [rc.chunk.text for rc in reranked]
        return answer, contexts

    dataset_manager = EvalDatasetManager(dataset_path)
    samples = dataset_manager.samples
    print(f"Running evaluation on {len(samples)} samples...\n")

    evaluator = RAGEvaluator(ask_fn=ask_fn)
    result = await evaluator.evaluate(samples)

    report = result.model_dump()
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples evaluated : {report['num_samples']}")
    print(f"  Faithfulness      : {report.get('faithfulness', 'N/A')}")
    print(f"  Context Precision : {report.get('context_precision', 'N/A')}")
    print(f"  Context Recall    : {report.get('context_recall', 'N/A')}")
    print(f"  Answer Relevancy  : {report.get('answer_relevancy', 'N/A')}")
    print("=" * 60)

    if report.get("details"):
        print("\nPer-sample details:")
        for i, detail in enumerate(report["details"], 1):
            print(f"  [{i}] {detail}")

    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResearchGPT RAG evaluation")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON report")
    args = parser.parse_args()

    asyncio.run(main(dataset_path=args.dataset, output_path=args.output))
