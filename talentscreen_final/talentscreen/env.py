"""
TalentScreen OpenEnv — Main Environment
Implements full OpenEnv spec: reset() / step() / state()
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
from talentscreen.models   import Observation, Action, Reward
from talentscreen.data     import TASK_REGISTRY
from talentscreen.tasks    import grade_action


class TalentScreenEnv:
    """
    TalentScreen: Hiring Pipeline Fraud Detection Environment.

    An AI agent receives a structured candidate profile and must
    detect fraudulent fields, assign reason codes, and make a
    hiring decision. Graded deterministically against a fixed
    ground truth embedded in the environment.

    Tasks:
        easy   — 1 obvious fraud flag, expected score ~0.88
        medium — 3 subtle flags,       expected score ~0.55
        hard   — 5 layered flags,      expected score ~0.27

    Action space:
        decision:    PASS | FLAG | REJECT
        fraud_flags: List[FraudFlag] — field + reason_code + severity
        confidence:  float 0.0–1.0

    Observation space:
        Structured candidate profile with education, employment,
        certifications, interview Q&A, references, and the
        NovaTalent protocol rules the agent must apply.

    Reward:
        Partial reward at every step (not just episode end).
        flags_score (0–0.40) + decision_score (0–0.30) + precision_score (0–0.30)
    """

    def __init__(self, task_id: str = "easy"):
        assert task_id in TASK_REGISTRY, f"task_id must be one of {list(TASK_REGISTRY)}"
        self.task_id        = task_id
        self._obs: Optional[Observation]       = None
        self._ground_truth: Optional[Dict]     = None
        self._last_reward: Optional[Reward]    = None
        self._done: bool                       = False
        self._step_count: int                  = 0
        self.MAX_STEPS: int                    = 5   # agent gets up to 5 attempts

    # -----------------------------------------------------------------------
    # reset() — required by OpenEnv spec
    # -----------------------------------------------------------------------
    def reset(self) -> Observation:
        """Return a fresh candidate profile. Called at episode start."""
        loader                 = TASK_REGISTRY[self.task_id]
        self._obs, self._ground_truth = loader()
        self._last_reward      = None
        self._done             = False
        self._step_count       = 0
        self._obs.step         = self._step_count
        self._obs.done         = False
        return self._obs

    # -----------------------------------------------------------------------
    # step() — required by OpenEnv spec
    # -----------------------------------------------------------------------
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Submit a fraud assessment action.
        Returns: (observation, reward, done, info)
        """
        if self._obs is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Grade the action deterministically
        reward             = grade_action(action, self._ground_truth)
        self._last_reward  = reward
        self._step_count  += 1

        # Episode ends when agent makes a decision or max steps reached
        done = True   # TalentScreen is a single-decision environment
        self._done     = done
        self._obs.done = done
        self._obs.step = self._step_count

        info = {
            "step":       self._step_count,
            "task_id":    self.task_id,
            "score":      reward.total,
            "feedback":   reward.feedback,
        }

        return self._obs, reward, done, info

    # -----------------------------------------------------------------------
    # state() — required by OpenEnv spec
    # -----------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "task_id":      self.task_id,
            "step":         self._step_count,
            "done":         self._done,
            "last_score":   self._last_reward.total if self._last_reward else None,
            "last_feedback":self._last_reward.feedback if self._last_reward else None,
            "observation":  self._obs.model_dump() if self._obs else None,
        }
