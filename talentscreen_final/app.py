"""
TalentScreen OpenEnv — FastAPI Server
3 routes: POST /reset  POST /step  GET /state
This is the HTTP wrapper that HuggingFace Space serves.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from talentscreen.env    import TalentScreenEnv
from talentscreen.models import Action, Observation, Reward
from typing import Dict, Any
import uvicorn

app = FastAPI(
    title       = "TalentScreen OpenEnv",
    description = "Hiring Pipeline Fraud Detection — OpenEnv compliant environment",
    version     = "1.0.0"
)

# One environment instance per task — keyed by task_id
_envs: Dict[str, TalentScreenEnv] = {
    "easy":   TalentScreenEnv("easy"),
    "medium": TalentScreenEnv("medium"),
    "hard":   TalentScreenEnv("hard"),
}


def _get_env(task_id: str) -> TalentScreenEnv:
    if task_id not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{task_id}'. Choose from: easy, medium, hard"
        )
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "environment": "TalentScreen", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------
@app.post("/reset", response_model=Observation)
def reset(task_id: str = "easy"):
    """
    Start a new episode. Returns a fresh candidate profile observation.
    task_id: easy | medium | hard
    """
    env = _get_env(task_id)
    obs = env.reset()
    return obs


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------
@app.post("/step")
def step(action: Action, task_id: str = "easy") -> Dict[str, Any]:
    """
    Submit a fraud assessment action.
    Returns observation, reward, done, info.
    """
    env = _get_env(task_id)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------
@app.get("/state")
def state(task_id: str = "easy") -> Dict[str, Any]:
    """Return current environment state."""
    env = _get_env(task_id)
    return env.state()


# ---------------------------------------------------------------------------
# GET /tasks — bonus: list all tasks
# ---------------------------------------------------------------------------
@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id":          "easy",
                "description": "Single fraud flag — timeline conflict",
                "difficulty":  "easy",
                "expected_score": 0.88
            },
            {
                "id":          "medium",
                "description": "3 subtle inconsistencies across profile fields",
                "difficulty":  "medium",
                "expected_score": 0.55
            },
            {
                "id":          "hard",
                "description": "5 layered flags — multi-field cross-referencing required",
                "difficulty":  "hard",
                "expected_score": 0.27
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
