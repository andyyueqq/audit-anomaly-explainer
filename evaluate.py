"""
Evaluation Script for Audit Anomaly Explainer
===============================================
Runs both the RAG system and prompt-only baseline on the test set,
then uses a model-as-judge approach to score each observation on a 4-dimension rubric.

Usage:
    python evaluate.py --google-key <key>
"""

import os
import csv
import json
import re
import argparse
import time
from pathlib import Path

from rag_pipeline import build_or_load_index
from llm_client import generate_observation, generate_observation_baseline

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert audit quality reviewer. You will evaluate an AI-generated audit observation against a rubric.

Score each dimension from 1 to 4:
- 1 = Poor: Missing, incorrect, or irrelevant.
- 2 = Below average: Partially correct but with significant gaps or errors.
- 3 = Good: Mostly correct and useful, minor issues only.
- 4 = Excellent: Accurate, specific, well-cited, and professionally written.

You MUST respond with valid JSON only, no other text. Use this exact format:
{
  "policy_citation": <1-4>,
  "policy_citation_rationale": "<brief explanation>",
  "condition_accuracy": <1-4>,
  "condition_accuracy_rationale": "<brief explanation>",
  "reasoning_quality": <1-4>,
  "reasoning_quality_rationale": "<brief explanation>",
  "recommendation_quality": <1-4>,
  "recommendation_quality_rationale": "<brief explanation>"
}
"""

JUDGE_USER_TEMPLATE = """## Anomaly Details
- Entry ID: {entry_id}
- Date: {date}
- Account Debit: {account_debit}
- Account Credit: {account_credit}
- Amount: ${amount:,.2f}
- Posting User: {posting_user}
- Flag Reason: {flag_reason}
- Description: {description}

## Observation to Evaluate
{observation}

## Evaluation Rubric
1. **Policy Citation (1-4):** Does the observation correctly cite a specific, relevant policy? Does it reference real policy numbers and sections rather than generic standards? (If no policy context was provided to the model, evaluate whether the observation acknowledges this gap rather than fabricating citations.)
2. **Condition Accuracy (1-4):** Does the Condition section accurately describe the anomaly details (amounts, dates, accounts, users)?
3. **Reasoning Quality (1-4):** Is the Cause/Effect analysis plausible, specific, and grounded in the data? Does it avoid speculation?
4. **Recommendation Quality (1-4):** Is the recommendation actionable, specific, and proportionate to the finding?

Score the observation now. Respond with JSON only.
"""


# ---------------------------------------------------------------------------
# Judge function
# ---------------------------------------------------------------------------

def judge_observation(
    anomaly: dict,
    observation_text: str,
    api_key: str,
    model: str = "gemini-2.5-pro",
) -> dict:
    """Use a model-as-judge to score an observation on the 4-dimension rubric."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=JUDGE_SYSTEM,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=512,
        ),
    )

    user_msg = JUDGE_USER_TEMPLATE.format(
        entry_id=anomaly.get("entry_id", "N/A"),
        date=anomaly.get("date", "N/A"),
        account_debit=anomaly.get("account_debit", "N/A"),
        account_credit=anomaly.get("account_credit", "N/A"),
        amount=float(anomaly.get("amount", 0)),
        posting_user=anomaly.get("posting_user", "N/A"),
        flag_reason=anomaly.get("flag_reason", "N/A"),
        description=anomaly.get("description", "N/A"),
        observation=observation_text,
    )

    response = gemini_model.generate_content(user_msg)

    text = response.text.strip()
    # Parse JSON from response
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            scores = json.loads(match.group())
        else:
            scores = {
                "policy_citation": 0,
                "condition_accuracy": 0,
                "reasoning_quality": 0,
                "recommendation_quality": 0,
                "parse_error": text,
            }

    return scores


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    csv_path: str,
    google_key: str,
    model: str = "gemini-2.5-flash",
    judge_model: str = "gemini-2.5-pro",
    top_k: int = 3,
    output_path: str = "evaluation_results.json",
):
    """Run full evaluation: RAG vs baseline on all test cases."""

    app_dir = Path(__file__).parent
    policy_dir = str(app_dir / "policies")
    index_dir = str(app_dir / "index_cache")

    # Build index
    print("Building policy index...")
    index = build_or_load_index(policy_dir, index_dir, google_key)
    print(f"  -> {len(index.chunks)} chunks indexed.\n")

    # Load test data
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        anomalies = list(reader)
    print(f"Loaded {len(anomalies)} test anomalies from {csv_path}\n")

    results = []

    for i, anomaly in enumerate(anomalies):
        entry_id = anomaly.get("entry_id", f"entry_{i}")
        print(f"[{i+1}/{len(anomalies)}] Processing {entry_id}...")

        # --- RAG observation ---
        query = f"{anomaly.get('flag_reason', '')} {anomaly.get('account_debit', '')} {anomaly.get('account_credit', '')}"
        retrieved = index.search(query, google_key, top_k=top_k)
        rag_result = generate_observation(anomaly, retrieved, google_key, model=model)
        time.sleep(1)  # Rate limiting for Gemini free tier

        # --- Baseline observation ---
        baseline_result = generate_observation_baseline(anomaly, google_key, model=model)
        time.sleep(1)

        # --- Judge RAG ---
        print(f"  Judging RAG observation...")
        rag_scores = judge_observation(anomaly, rag_result["observation"], google_key, model=judge_model)
        time.sleep(1)

        # --- Judge Baseline ---
        print(f"  Judging baseline observation...")
        baseline_scores = judge_observation(anomaly, baseline_result["observation"], google_key, model=judge_model)
        time.sleep(1)

        result = {
            "entry_id": entry_id,
            "flag_reason": anomaly.get("flag_reason", ""),
            "rag": {
                "observation": rag_result["observation"],
                "scores": rag_scores,
                "input_tokens": rag_result.get("input_tokens", 0),
                "output_tokens": rag_result.get("output_tokens", 0),
                "retrieved_sources": [
                    {"source": c["doc_title"], "section": c["section"], "score": c["score"]}
                    for c in retrieved
                ],
            },
            "baseline": {
                "observation": baseline_result["observation"],
                "scores": baseline_scores,
                "input_tokens": baseline_result.get("input_tokens", 0),
                "output_tokens": baseline_result.get("output_tokens", 0),
            },
        }
        results.append(result)

        # Print comparison
        rag_avg = _avg_score(rag_scores)
        base_avg = _avg_score(baseline_scores)
        print(f"  RAG avg: {rag_avg:.2f} | Baseline avg: {base_avg:.2f} | Delta: {rag_avg - base_avg:+.2f}\n")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    _print_summary(results)

    return results


def _avg_score(scores: dict) -> float:
    dims = ["policy_citation", "condition_accuracy", "reasoning_quality", "recommendation_quality"]
    vals = [scores.get(d, 0) for d in dims]
    return sum(vals) / len(vals) if vals else 0


def _print_summary(results: list):
    dims = ["policy_citation", "condition_accuracy", "reasoning_quality", "recommendation_quality"]

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Dimension':<25} {'RAG Avg':>10} {'Baseline Avg':>14} {'Delta':>10}")
    print("-" * 60)

    for dim in dims:
        rag_vals = [r["rag"]["scores"].get(dim, 0) for r in results]
        base_vals = [r["baseline"]["scores"].get(dim, 0) for r in results]
        rag_avg = sum(rag_vals) / len(rag_vals)
        base_avg = sum(base_vals) / len(base_vals)
        delta = rag_avg - base_avg
        print(f"{dim:<25} {rag_avg:>10.2f} {base_avg:>14.2f} {delta:>+10.2f}")

    # Overall
    all_rag = [_avg_score(r["rag"]["scores"]) for r in results]
    all_base = [_avg_score(r["baseline"]["scores"]) for r in results]
    rag_overall = sum(all_rag) / len(all_rag)
    base_overall = sum(all_base) / len(all_base)
    print("-" * 60)
    print(f"{'OVERALL':<25} {rag_overall:>10.2f} {base_overall:>14.2f} {rag_overall - base_overall:>+10.2f}")
    print("=" * 70)

    # Token usage / cost estimate
    total_rag_in = sum(r["rag"].get("input_tokens", 0) for r in results)
    total_rag_out = sum(r["rag"].get("output_tokens", 0) for r in results)
    total_base_in = sum(r["baseline"].get("input_tokens", 0) for r in results)
    total_base_out = sum(r["baseline"].get("output_tokens", 0) for r in results)
    print(f"\nToken usage (generation only):")
    print(f"  RAG:      {total_rag_in:,} input + {total_rag_out:,} output")
    print(f"  Baseline: {total_base_in:,} input + {total_base_out:,} output")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Audit Anomaly Explainer: RAG vs baseline")
    parser.add_argument("--google-key", required=True, help="Google API key")
    parser.add_argument("--csv", default=str(Path(__file__).parent / "data" / "flagged_anomalies.csv"),
                        help="Path to flagged anomalies CSV")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model for generation")
    parser.add_argument("--judge-model", default="gemini-2.5-pro", help="Model for judging")
    parser.add_argument("--top-k", type=int, default=3, help="Number of policy chunks to retrieve")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON path")

    args = parser.parse_args()

    run_evaluation(
        csv_path=args.csv,
        google_key=args.google_key,
        model=args.model,
        judge_model=args.judge_model,
        top_k=args.top_k,
        output_path=args.output,
    )
