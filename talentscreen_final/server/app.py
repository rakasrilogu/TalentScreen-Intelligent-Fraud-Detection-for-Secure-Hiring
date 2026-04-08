from fastapi import FastAPI
from talentscreen.env import TalentScreenEnv
from talentscreen.models import Action

app = FastAPI()

_envs = {
    "easy": TalentScreenEnv("easy"),
    "medium": TalentScreenEnv("medium"),
    "hard": TalentScreenEnv("hard"),
}

@app.post("/reset")
def reset(task_id: str):
    env = _envs[task_id]
    obs = env.reset()
    return {"observation": obs.dict()}

@app.post("/step")
def step(task_id: str, action: Action):
    env = _envs[task_id]
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward.value if hasattr(reward, "value") else reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state(task_id: str):
    return {"state": _envs[task_id].get_state()}
