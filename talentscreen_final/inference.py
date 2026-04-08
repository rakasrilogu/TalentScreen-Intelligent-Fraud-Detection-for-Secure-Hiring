"""
TalentScreen OpenEnv — inference.py
Mandatory script name: inference.py (root directory)

Environment variables required:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — HuggingFace / API key

Uses OpenAI client pointed at API_BASE_URL.
Emits [START] [STEP] [END] structured stdout logs.
"""

import os
import json
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — reads from environment variables (mandatory per spec)
# ---------------------------------------------------------------------------
API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api-inference.huggingface.co/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN      = os.environ.get("HF_TOKEN",      "")
ENV_URL       = os.environ.get("TALENTSCREEN_URL", "http://localhost:7860")
SEED          = 42
TASKS         = ["easy", "medium", "hard"]

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
        "Get your free token at huggingface.co/settings/tokens"
    )

# ---------------------------------------------------------------------------
# OpenAI client pointed at API_BASE_URL (mandatory per spec)
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url = API_BASE_URL,
    api_key  = HF_TOKEN,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a fraud detection specialist for NovaTalent, a hiring agency.
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

Respond with JSON only. No explanation. No markdown. No code fences."""


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------
def run_task(task_id: str):

    # reset environment
    reset_resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    reset_resp.raise_for_status()
    observation = reset_resp.json()

    # call LLM via OpenAI client at API_BASE_URL
    completion = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(observation, indent=2)},
        ],
        temperature = 0.0,
        max_tokens  = 1000,
        seed        = SEED,
    )

    raw = completion.choices[0].message.content.strip()

    # strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        action_dict = json.loads(raw)
    except json.JSONDecodeError:
        action_dict = {"decision": "PASS", "fraud_flags": [], "confidence": 0.0}

    # submit action to environment
    step_resp = requests.post(
        f"{ENV_URL}/step",
        params = {"task_id": task_id},
        json   = action_dict,
    )
    step_resp.raise_for_status()
    result    = step_resp.json()
    reward    = result["reward"]
    done      = result["done"]

    return observation, action_dict, reward, done


# ---------------------------------------------------------------------------
# Main — mandatory [START] [STEP] [END] log format
# ---------------------------------------------------------------------------
def main():

    all_scores = {}

    for task_id in TASKS:

        # [START]
        print(json.dumps({
            "event":    "START",
            "task_id":  task_id,
            "model":    MODEL_NAME,
            "api_base": API_BASE_URL,
            "seed":     SEED,
        }))

        try:
            observation, action, reward, done = run_task(task_id)

            # [STEP]
            print(json.dumps({
                "event":   "STEP",
                "task_id": task_id,
                "step":    1,
                "observation": {
                    "candidate_id": observation.get("candidate_id"),
                    "task_id":      observation.get("task_id"),
                    "step":         observation.get("step"),
                },
                "action":  action,
                "reward":  reward["total"],
                "done":    done,
                "info": {
                    "flags_score":     reward["flags_score"],
                    "decision_score":  reward["decision_score"],
                    "precision_score": reward["precision_score"],
                    "feedback":        reward["feedback"],
                },
            }))

            score = reward["total"]

        except Exception as e:
            # [STEP] with error
            print(json.dumps({
                "event":   "STEP",
                "task_id": task_id,
                "step":    1,
                "error":   str(e),
                "reward":  0.0,
                "done":    True,
            }))
            score = 0.0

        all_scores[task_id] = score

        # [END]
        print(json.dumps({
            "event":   "END",
            "task_id": task_id,
            "score":   score,
            "model":   MODEL_NAME,
        }))

    # SUMMARY
    avg = sum(all_scores.values()) / len(all_scores)
    print(json.dumps({
        "event":     "SUMMARY",
        "scores":    all_scores,
        "average":   round(avg, 4),
        "model":     MODEL_NAME,
        "tasks_run": len(TASKS),
    }))


if __name__ == "__main__":
    main()
