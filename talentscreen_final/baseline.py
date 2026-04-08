"""
TalentScreen OpenEnv — Baseline Inference Script

Runs an OpenAI model against all 3 tasks and prints reproducible scores.
Usage:
    OPENAI_API_KEY=sk-xxx python baseline.py

Reads OPENAI_API_KEY from environment variable.
Uses seed=42 for reproducibility.
"""

import os
import json
import random
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL   = os.environ.get("TALENTSCREEN_URL", "http://localhost:7860")
API_KEY    = os.environ.get("OPENAI_API_KEY")
MODEL      = "gpt-4o-mini"   # cost-efficient baseline model
SEED       = 42
TASKS      = ["easy", "medium", "hard"]

if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=API_KEY)

# ---------------------------------------------------------------------------
# System prompt for the agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a fraud detection specialist for NovaTalent, a hiring agency.
You will receive a structured candidate profile as JSON.
Your job is to:
1. Read the candidate's education, employment, certifications, interview Q&A, and references
2. Apply the protocol_rules provided in the profile — use ONLY these rules, not outside knowledge
3. Identify any fraud flags — fields that violate the protocol rules
4. Make a hiring decision: PASS, FLAG, or REJECT

You MUST respond with valid JSON in exactly this format:
{
  "decision": "PASS" | "FLAG" | "REJECT",
  "fraud_flags": [
    {
      "field": "<exact field path e.g. education[0].gpa>",
      "reason_code": "<exact code from protocol_rules.reason_codes>",
      "severity": <float between 0.0 and 1.0>
    }
  ],
  "confidence": <float between 0.0 and 1.0>
}

Respond with JSON only. No explanation. No markdown.
"""


def run_task(task_id: str) -> float:
    """Run baseline agent on one task. Returns (score, feedback)."""

    # 1. Reset environment
    resp = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    resp.raise_for_status()
    observation = resp.json()

    # 2. Call OpenAI with the observation
    random.seed(SEED)
    completion = client.chat.completions.create(
        model    = MODEL,
        seed     = SEED,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(observation, indent=2)}
        ],
        temperature = 0.0,   # deterministic
        max_tokens  = 1000,
    )

    raw = completion.choices[0].message.content.strip()

    # 3. Parse agent response
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action_dict = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse agent response for task '{task_id}'. Score: 0.0")
        print(f"  Raw response: {raw[:200]}")
        return 0.0, "Parse error"

    # 4. Submit action to environment
    step_resp = requests.post(
        f"{BASE_URL}/step",
        params = {"task_id": task_id},
        json   = action_dict
    )
    step_resp.raise_for_status()
    result = step_resp.json()

    score    = result["reward"]["total"]
    feedback = result["reward"]["feedback"]
    return score, feedback


def main():
    print("=" * 55)
    print("  TalentScreen OpenEnv — Baseline Scores")
    print(f"  Model : {MODEL}  |  Seed : {SEED}")
    print("=" * 55)

    scores = {}
    total = 0.0
    for task_id in TASKS:
        print(f"\nRunning task: {task_id.upper()}")
        try:
            score, feedback = run_task(task_id)
            scores[task_id] = score
            print(f"  Score   : {score:.4f}")
            print(f"  Feedback: {feedback}")
            total += score
        except Exception as e:
            print(f"  ERROR on task '{task_id}': {e}")
            scores[task_id] = 0.0
            total += 0.0

    avg = total / len(TASKS)
    print("\n" + "=" * 55)
    print(f"  Average score across all tasks: {avg:.4f}")
    print("=" * 55)

    # Machine-readable output for automated validators (no re-run!)
    parts = " ".join(f"{t}={scores[t]:.4f}" for t in TASKS)
    print(f"\nBASELINE_SCORES: {parts}")


if __name__ == "__main__":
    main()
