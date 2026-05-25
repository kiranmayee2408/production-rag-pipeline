"""
RAG Evaluation Framework — RAGAS-style metrics

Metrics implemented:
  - Faithfulness     : Is the answer supported by the retrieved context?
  - Answer Relevancy : Does the answer address the question?
  - Context Precision: Are the retrieved chunks actually relevant to the question?
  - Context Recall   : Does the context contain the info needed to answer?

Each metric returns a score in [0, 1]. Higher is better.

Usage:
    evaluator = RAGEvaluator(llm_fn=my_llm_call)
    results = evaluator.evaluate(dataset)
    report = evaluator.report(results)
"""
import re
import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass
class EvalResult:
    sample: EvalSample
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def mean_score(self) -> float:
        return round((self.faithfulness + self.answer_relevancy +
                      self.context_precision + self.context_recall) / 4, 3)


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using LLM-as-a-judge + heuristic fallbacks.

    Args:
        llm_fn: callable(prompt: str) -> str  — your LLM call
        use_llm_judge: if False, use heuristic-only scoring (no API calls)
    """
    def __init__(self, llm_fn: Callable[[str], str] | None = None, use_llm_judge: bool = True):
        self.llm_fn = llm_fn
        self.use_llm_judge = use_llm_judge and llm_fn is not None

    # ── Faithfulness ──────────────────────────────────────────────────────────

    def _faithfulness_prompt(self, question: str, answer: str, contexts: list[str]) -> str:
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        return f"""Given the following contexts and an answer, determine what fraction of claims
in the answer are supported by the contexts.

Contexts:
{ctx}

Answer: {answer}

Return a JSON object with:
  "supported_claims": list of claims from the answer that are grounded in the contexts
  "unsupported_claims": list of claims not found in the contexts
  "faithfulness_score": float between 0 and 1

Return only valid JSON."""

    def _heuristic_faithfulness(self, answer: str, contexts: list[str]) -> float:
        answer_words = set(re.findall(r"\w+", answer.lower()))
        ctx_words = set(re.findall(r"\w+", " ".join(contexts).lower()))
        if not answer_words:
            return 0.0
        overlap = len(answer_words & ctx_words) / len(answer_words)
        return round(min(overlap * 1.5, 1.0), 3)  # scale up slightly

    def faithfulness(self, sample: EvalSample) -> float:
        if self.use_llm_judge:
            try:
                prompt = self._faithfulness_prompt(sample.question, sample.answer, sample.contexts)
                raw = self.llm_fn(prompt)
                data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
                return float(data.get("faithfulness_score", 0))
            except Exception:
                pass
        return self._heuristic_faithfulness(sample.answer, sample.contexts)

    # ── Answer Relevancy ──────────────────────────────────────────────────────

    def _heuristic_answer_relevancy(self, question: str, answer: str) -> float:
        q_words = set(re.findall(r"\w+", question.lower())) - {"what", "how", "why", "is", "the", "a", "an"}
        a_words = set(re.findall(r"\w+", answer.lower()))
        if not q_words:
            return 0.5
        return round(min(len(q_words & a_words) / len(q_words) * 1.2, 1.0), 3)

    def answer_relevancy(self, sample: EvalSample) -> float:
        if self.use_llm_judge:
            try:
                prompt = f"""Given the question and answer, generate 3 questions that the answer would be a good response to.
Question: {sample.question}
Answer: {sample.answer}
Return a JSON list of 3 question strings."""
                raw = self.llm_fn(prompt)
                generated_qs = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group())
                # Measure similarity of generated questions to original
                scores = [self._heuristic_answer_relevancy(sample.question, gq) for gq in generated_qs]
                return round(sum(scores) / len(scores), 3)
            except Exception:
                pass
        return self._heuristic_answer_relevancy(sample.question, sample.answer)

    # ── Context Precision ─────────────────────────────────────────────────────

    def context_precision(self, sample: EvalSample) -> float:
        """What fraction of retrieved chunks are relevant to the question?"""
        if not sample.contexts:
            return 0.0
        q_words = set(re.findall(r"\w+", sample.question.lower()))
        relevant = 0
        for ctx in sample.contexts:
            ctx_words = set(re.findall(r"\w+", ctx.lower()))
            overlap = len(q_words & ctx_words) / max(len(q_words), 1)
            if overlap > 0.2:
                relevant += 1
        return round(relevant / len(sample.contexts), 3)

    # ── Context Recall ────────────────────────────────────────────────────────

    def context_recall(self, sample: EvalSample) -> float:
        """Does context contain information to answer the question?"""
        if not sample.ground_truth:
            return self.context_precision(sample)  # fallback
        gt_words = set(re.findall(r"\w+", sample.ground_truth.lower()))
        ctx_words = set(re.findall(r"\w+", " ".join(sample.contexts).lower()))
        if not gt_words:
            return 0.0
        return round(len(gt_words & ctx_words) / len(gt_words), 3)

    # ── Full Evaluation ───────────────────────────────────────────────────────

    def evaluate(self, samples: list[EvalSample]) -> list[EvalResult]:
        results = []
        for i, sample in enumerate(samples):
            print(f"[eval] {i+1}/{len(samples)} — evaluating sample...")
            results.append(EvalResult(
                sample=sample,
                faithfulness=self.faithfulness(sample),
                answer_relevancy=self.answer_relevancy(sample),
                context_precision=self.context_precision(sample),
                context_recall=self.context_recall(sample),
            ))
        return results

    def report(self, results: list[EvalResult]) -> dict:
        if not results:
            return {}
        return {
            "num_samples": len(results),
            "avg_faithfulness": round(sum(r.faithfulness for r in results) / len(results), 3),
            "avg_answer_relevancy": round(sum(r.answer_relevancy for r in results) / len(results), 3),
            "avg_context_precision": round(sum(r.context_precision for r in results) / len(results), 3),
            "avg_context_recall": round(sum(r.context_recall for r in results) / len(results), 3),
            "avg_mean_score": round(sum(r.mean_score for r in results) / len(results), 3),
            "by_sample": [
                {
                    "question": r.sample.question[:80] + "...",
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "mean": r.mean_score,
                }
                for r in results
            ],
        }
