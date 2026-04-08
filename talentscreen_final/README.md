---
title: TalentScreen
sdk: docker
tags:
  - openenv
---

# TalentScreen — Hiring Pipeline Fraud Detection

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.dev)

An OpenEnv-compliant environment where an AI agent reads structured candidate profiles and detects resume fraud, credential misrepresentation, and interview inconsistencies.

---

## What is TalentScreen?

HR teams and background check companies manually cross-reference resumes vs employment history vs interview answers vs references every day. A wrong hire costs $15,000–$240,000. TalentScreen simulates this task as a reinforcement learning environment.

The agent receives a structured synthetic candidate profile and must:
1. Identify fraudulent fields (timeline conflicts, fake credentials, skill gaps)
2. Assign reason codes from the NovaTalent protocol
3. Make a hiring decision: `PASS`, `FLAG`, or `REJECT`

All candidate data is **synthetic**. Fraud flags are **pre-planted** with a fixed ground truth for fully deterministic grading.

---

## Action Space

```python
class Action(BaseModel):
    decision:    Decision          # PASS | FLAG | REJECT
    fraud_flags: List[FraudFlag]   # field + reason_code + severity
    confidence:  float             # 0.0 – 1.0
```

## Observation Space

```python
class Observation(BaseModel):
    candidate_id:   str
    task_id:        str
    step:           int
    education:      List[EducationRecord]
    employment:     List[EmploymentRecord]
    skills_claimed: List[str]
    certifications: List[CertificationRecord]
    interview_qa:   List[QAPair]
    references:     List[ReferenceRecord]
    protocol_rules: Dict   # NovaTalent rulebook — agent must use these
    done:           bool
```

## Reward Function

Partial reward fires on every step (not just episode end):

| Component       | Weight | Description                          |
|-----------------|--------|--------------------------------------|
| `flags_score`   | 0.40   | Correct fraud flags detected         |
| `decision_score`| 0.30   | Correct final hiring decision        |
| `precision_score`| 0.30  | No false positives penalised         |

---

## Tasks

| Task   | Flags | Expected Score | Description                              |
|--------|-------|----------------|------------------------------------------|
| easy   | 1     | ~0.88          | Single timeline conflict                 |
| medium | 3     | ~0.55          | GPA violation + title inflation + skill gap |
| hard   | 5     | ~0.27          | 5 layered flags across 12 profile fields |

---

## Setup & Usage

### Local

```bash
# Install
pip install -r requirements.txt

# Run server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run baseline
OPENAI_API_KEY=sk-xxx python baseline.py
```

### Docker

```bash
docker build -t talentscreen .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-xxx talentscreen
```

---

## API Endpoints

| Method | Route     | Description                        |
|--------|-----------|------------------------------------|
| POST   | `/reset`  | Start new episode, get observation |
| POST   | `/step`   | Submit action, get reward          |
| GET    | `/state`  | Get current environment state      |
| GET    | `/tasks`  | List all tasks                     |
| GET    | `/health` | Health check                       |

### Example

```python
import requests

# Reset
obs = requests.post("http://localhost:7860/reset", params={"task_id": "easy"}).json()

# Step
action = {
    "decision": "FLAG",
    "fraud_flags": [
        {
            "field": "employment[0].start_year",
            "reason_code": "TIMELINE_CONFLICT",
            "severity": 0.8
        }
    ],
    "confidence": 0.9
}
result = requests.post("http://localhost:7860/step", params={"task_id": "easy"}, json=action).json()
print(result["reward"]["total"])   # e.g. 0.85
```

---

## Baseline Scores

| Model       | Easy   | Medium | Hard   | Average |
|-------------|--------|--------|--------|---------|
| gpt-4o-mini | ~0.85  | ~0.52  | ~0.27  | ~0.55   |

---

## Project Structure

```
talentscreen/
├── talentscreen/
│   ├── __init__.py   # package exports
│   ├── models.py     # Pydantic: Observation, Action, Reward
│   ├── env.py        # TalentScreenEnv: reset() step() state()
│   ├── tasks.py      # deterministic grader
│   └── data.py       # synthetic candidate generator
├── app.py            # FastAPI server
├── baseline.py       # OpenAI baseline agent
├── openenv.yaml      # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

MIT
