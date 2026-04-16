"""
agents/explanation.py

The Explanation Agent.

Responsibility: Use Claude to explain *why* each flagged test is likely
to fail, given the diff and the test name.

This is the only agent that calls the LLM. All the other agents use
deterministic code (API calls, vector math, arithmetic). We only bring
Claude in at the end to generate human-readable justifications.

Why keep Claude usage minimal?
  - LLM calls are slower and more expensive than deterministic code
  - The retrieval + ranking pipeline is already doing the heavy lifting
  - Claude's job here is just to translate the prediction into plain English

The agent makes one Claude API call per test prediction (or one batched
call for all predictions at once to save latency).
"""

import anthropic
from models.schemas import TestPrediction
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


# Initialize the Anthropic client once at module load time
# It reads ANTHROPIC_API_KEY from the environment automatically
_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def run(diff_text: str, predictions: list[TestPrediction]) -> list[TestPrediction]:
    """
    Main entry point for the Explanation Agent.

    For each test prediction, asks Claude to explain why that test
    might be affected by the diff.

    We make a SINGLE Claude API call with all predictions batched together
    (one request, multiple explanations) to minimize latency and cost.

    Args:
        diff_text:   The cleaned git diff being analyzed
        predictions: Output from ranking.run() — tests with failure scores

    Returns:
        The same list of TestPrediction objects, with the `explanation`
        field filled in for each one.

    Example output:
        prediction.explanation = "This test covers the `authenticate()` function
        in auth/middleware.py, which was modified in this diff to add MFA support.
        The test likely validates the existing password-only flow which may break."
    """
    if not predictions:
        return []

    print(f"[Explanation] Generating explanations for {len(predictions)} test predictions...")

    # Build a single prompt that asks Claude to explain all predictions at once
    # This is more efficient than one API call per test
    prompt = _build_prompt(diff_text, predictions)

    try:
        # Call Claude with streaming — explanations can be long
        explanations = _call_claude(prompt, len(predictions))

        # Attach each explanation to its corresponding TestPrediction
        for i, prediction in enumerate(predictions):
            if i < len(explanations):
                prediction.explanation = explanations[i]
            else:
                prediction.explanation = "No explanation available."

        print(f"[Explanation] Done generating explanations.")

    except Exception as e:
        print(f"[Explanation] Warning: Could not generate explanations ({e})")
        print(f"[Explanation] Returning predictions without explanations.")
        for prediction in predictions:
            prediction.explanation = "(Explanation unavailable — check ANTHROPIC_API_KEY credits)"

    return predictions


def _build_prompt(diff_text: str, predictions: list[TestPrediction]) -> str:
    """
    Builds the prompt we send to Claude.

    We show Claude:
      1. The diff (what changed)
      2. The ranked test list with failure scores
      3. Ask for a brief explanation for each test

    We keep explanations short (1-2 sentences) because they're displayed
    inline in the final report — we don't need an essay per test.
    """
    # Format the test list for the prompt
    test_list = "\n".join([
        f"{i+1}. {p.test_name} (failure score: {p.failure_score:.2f})"
        for i, p in enumerate(predictions)
    ])

    # Truncate the diff if it's very long — we don't need all of it for explanation
    display_diff = diff_text[:3000] + "\n... (truncated)" if len(diff_text) > 3000 else diff_text

    prompt = f"""You are analyzing a code diff to explain why certain tests might fail.

## The Diff
```
{display_diff}
```

## Tests Predicted to Fail
{test_list}

## Your Task
For each test above, write ONE concise sentence (max 20 words) explaining WHY
that test is likely to be affected by this diff. Focus on which specific function,
class, or behavior in the diff connects to what the test is checking.

Format your response as a numbered list matching the test numbers above.
Only output the numbered list — no preamble or summary.

Example format:
1. This test validates token expiry logic in auth.py which was modified in the diff.
2. This test checks the login flow that now requires the new MFA parameter.
"""
    return prompt


def _call_claude(prompt: str, expected_count: int) -> list[str]:
    """
    Makes the actual Claude API call and parses the response.

    Uses streaming so we see output progressively (useful for debugging).
    Then parses the numbered list into individual explanation strings.

    Args:
        prompt:         The formatted prompt string
        expected_count: How many explanations we expect back

    Returns:
        List of explanation strings, one per test.
    """
    # Use streaming to avoid timeout on long responses
    # .get_final_message() waits for the full response and returns a Message object
    with _client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2048,   # explanations are short, 2K tokens is plenty
        messages=[{"role": "user", "content": prompt}],
        # Use adaptive thinking for better reasoning about code
        thinking={"type": "adaptive"},
    ) as stream:
        response = stream.get_final_message()

    # Extract the text content from the response
    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text = block.text
            break

    # Parse the numbered list into individual strings
    # Input:  "1. Explanation A\n2. Explanation B\n3. Explanation C"
    # Output: ["Explanation A", "Explanation B", "Explanation C"]
    explanations = _parse_numbered_list(raw_text, expected_count)
    return explanations


def _parse_numbered_list(text: str, expected_count: int) -> list[str]:
    """
    Parses Claude's numbered list response into a Python list.

    Handles variations in formatting:
      "1. Explanation"
      "1) Explanation"
      "1 - Explanation"
    """
    import re
    lines = text.strip().split("\n")
    explanations = []

    for line in lines:
        line = line.strip()
        # Match lines that start with a number followed by . ) or -
        match = re.match(r"^\d+[.):\-]\s*(.+)$", line)
        if match:
            explanations.append(match.group(1).strip())

    # Pad with fallback messages if Claude returned fewer than expected
    while len(explanations) < expected_count:
        explanations.append("No explanation available.")

    return explanations[:expected_count]
