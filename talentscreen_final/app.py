"""
TalentScreen OpenEnv — FastAPI Server
Routes:
GET  /
GET  /health
POST /reset
POST /step
GET  /state
GET  /tasks
"""

from fastapi import FastAPI, HTTPException
from talentscreen.env import TalentScreenEnv
from talentscreen.models import Action
from typing import Dict, Any

app = FastAPI(
    title="TalentScreen OpenEnv",
    description="Hiring Pipeline Fraud Detection — OpenEnv compliant environment",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Root route (IMPORTANT for HuggingFace)
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "TalentScreen OpenEnv API is running 🚀",
        "routes": {
            "health": "/health",
            "reset": "POST /reset?task_id=easy|medium|hard",
            "step": "POST /step?task_id=easy|medium|hard",
            "state": "/state?task_id=easy|medium|hard",
            "tasks": "/tasks"
        }
    }


# ---------------------------------------------------------------------------
# Environment instances
# ---------------------------------------------------------------------------
_envs: Dict[str, TalentScreenEnv] = {
    "easy": TalentScreenEnv("easy"),
    "medium": TalentScreenEnv("medium"),
    "hard": TalentScreenEnv("hard"),
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
    return {
        "status": "ok",
        "environment": "TalentScreen",
        "version": "1.0.0"
    }


# ---------------------------------------------------------------------------
# POST /reset  ✅ FIXED
# ---------------------------------------------------------------------------
@app.post("/reset")
def reset(task_id: str = "easy") -> Dict[str, Any]:
    env = _get_env(task_id)
    obs = env.reset()

    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
        "reward": 0.0,
        "done": False,
        "info": {}
    }


# ---------------------------------------------------------------------------
# POST /step  ✅ FIXED
# ---------------------------------------------------------------------------
@app.post("/step")
def step(action: Action, task_id: str = "easy") -> Dict[str, Any]:
    env = _get_env(task_id)

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
        "reward": float(reward),  # MUST be float
        "done": done,
        "info": info,
    }


# ---------------------------------------------------------------------------
# GET /state  ✅ FIXED
# ---------------------------------------------------------------------------
@app.get("/state")
def state(task_id: str = "easy") -> Dict[str, Any]:
    env = _get_env(task_id)
    obs = env.state()

    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
        "done": False
    }


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------
@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Single fraud flag — timeline conflict",
                "difficulty": "easy",
                "expected_score": 0.88
            },
            {
                "id": "medium",
                "description": "3 subtle inconsistencies across profile fields",
                "difficulty": "medium",
                "expected_score": 0.55
            },
            {
                "id": "hard",
                "description": "5 layered flags — multi-field cross-referencing required",
                "difficulty": "hard",
                "expected_score": 0.27
            }
        ]
    }
       




   



       
