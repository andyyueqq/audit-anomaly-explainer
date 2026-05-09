"""
LLM Client for Audit Anomaly Explainer
========================================
Generates structured audit observations using Google Gemini,
with and without RAG context (for baseline comparison).
"""

import json
from typing import Optional

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an experienced internal auditor drafting concise audit observations for journal entry testing.

Given a flagged journal entry anomaly and (optionally) relevant policy excerpts, generate a SHORT structured observation.

Format — use exactly these five headings:

**Condition:** 1-2 sentences. State the key facts: entry ID, date, amount, accounts, and the specific issue.

**Criteria:** 1-2 sentences. Cite the specific policy number/section violated. If no policy provided, state the applicable standard briefly.

**Cause:** 1 sentence. Most likely reason for the anomaly.

**Effect:** 1 sentence. Potential financial or control impact. Quantify if possible.

**Recommendation:** 1-2 sentences. One specific, actionable step.

CRITICAL RULES:
- Keep the TOTAL observation under 150 words. Be direct and concise — no filler.
- Do NOT start with "Here is the audit observation" or similar preamble.
- Start directly with **Condition:**
- If policy excerpts are provided, cite the policy number and section.
- Do not fabricate policy numbers or thresholds not in the provided context.
- Use professional audit language.
"""

# ---------------------------------------------------------------------------
# Observation generation
# ---------------------------------------------------------------------------

def generate_observation(
    anomaly: dict,
    policy_context: list[dict],
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> dict:
    """
    Generate an audit observation for a single anomaly.

    Args:
        anomaly: Dict with keys like entry_id, date, account_debit, account_credit,
                 amount, posting_user, description, flag_reason.
        policy_context: List of retrieved policy chunks (each with source, section, text, score).
                       Pass empty list for baseline (no-RAG) mode.
        api_key: Google API key.
        model: Gemini model name to use.

    Returns:
        Dict with keys: observation (str), model (str), had_context (bool),
                        input_tokens (int), output_tokens (int).
    """
    import time
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=2048,
        ),
    )

    # Build the user message
    user_message = _build_user_message(anomaly, policy_context)

    # Retry logic for rate-limited / truncated responses
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(user_message)
            observation_text = response.text

            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0)
            output_tokens = getattr(usage, "candidates_token_count", 0)

            # If output is too short (< 100 tokens), likely rate-limited — retry
            if output_tokens < 100 and attempt < max_retries - 1:
                wait = (attempt + 1) * 5  # 5s, 10s, 15s
                time.sleep(wait)
                continue

            return {
                "entry_id": anomaly.get("entry_id", ""),
                "observation": observation_text,
                "model": model,
                "had_context": len(policy_context) > 0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 5)
                continue
            raise e

    # Fallback (should not reach here)
    return {
        "entry_id": anomaly.get("entry_id", ""),
        "observation": "Generation failed after retries.",
        "model": model,
        "had_context": len(policy_context) > 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def generate_observation_baseline(
    anomaly: dict,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> dict:
    """Generate an observation WITHOUT policy context (baseline)."""
    return generate_observation(anomaly, [], api_key, model)


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def _build_user_message(anomaly: dict, policy_context: list[dict]) -> str:
    """Build the user-facing prompt with anomaly details and optional policy context."""

    parts = []

    # Anomaly details
    parts.append("## Flagged Journal Entry Anomaly\n")
    parts.append(f"- **Entry ID:** {anomaly.get('entry_id', 'N/A')}")
    parts.append(f"- **Date:** {anomaly.get('date', 'N/A')}")
    parts.append(f"- **Account (Debit):** {anomaly.get('account_debit', 'N/A')}")
    parts.append(f"- **Account (Credit):** {anomaly.get('account_credit', 'N/A')}")
    parts.append(f"- **Amount:** ${float(anomaly.get('amount', 0)):,.2f}")
    parts.append(f"- **Posting User:** {anomaly.get('posting_user', 'N/A')}")
    parts.append(f"- **Description:** {anomaly.get('description', 'N/A')}")
    parts.append(f"- **Flag Reason:** {anomaly.get('flag_reason', 'N/A')}")

    # Policy context (only if provided — skip for baseline)
    if policy_context:
        parts.append("\n## Relevant Policy Excerpts\n")
        for i, chunk in enumerate(policy_context, 1):
            source = chunk.get("doc_title", chunk.get("source", "Unknown"))
            section = chunk.get("section", "")
            score = chunk.get("score", 0)
            parts.append(f"### Policy Excerpt {i} (from: {source}, Section: {section}, Relevance: {score:.3f})")
            parts.append(chunk.get("text", ""))
            parts.append("")

    parts.append("\nPlease generate the audit observation in the five-part format (Condition, Criteria, Cause, Effect, Recommendation).")

    return "\n".join(parts)
