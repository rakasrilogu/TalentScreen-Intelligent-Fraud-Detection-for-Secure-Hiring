"""
TalentScreen OpenEnv — Pydantic Models
Observation, Action, Reward — fully OpenEnv spec compliant
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models inside Observation
# ---------------------------------------------------------------------------

class EducationRecord(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    gpa: Optional[float] = None
    verified: bool = False


class EmploymentRecord(BaseModel):
    company: str
    title: str
    start_year: int
    end_year: Optional[int] = None  # None = current
    responsibilities: List[str]
    reference_contact: Optional[str] = None


class CertificationRecord(BaseModel):
    name: str
    issuing_body: str
    year_obtained: int
    credential_id: str


class QAPair(BaseModel):
    question: str
    answer: str
    expected_skill: str


class ReferenceRecord(BaseModel):
    name: str
    relationship: str
    company: str
    phone: str
    email: str


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    candidate_id: str
    task_id: str                          # "easy" | "medium" | "hard"
    step: int
    education: List[EducationRecord]
    employment: List[EmploymentRecord]
    skills_claimed: List[str]
    certifications: List[CertificationRecord]
    interview_qa: List[QAPair]
    references: List[ReferenceRecord]
    protocol_rules: Dict[str, Any]        # NovaTalent fraud detection rulebook
    done: bool = False


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    PASS   = "PASS"    # candidate is clean — move to next round
    FLAG   = "FLAG"    # inconsistencies found — needs further review
    REJECT = "REJECT"  # clear fraud — do not proceed


class FraudFlag(BaseModel):
    field: str          # e.g. "education[0].end_year"
    reason_code: str    # e.g. "TIMELINE_CONFLICT"
    severity: float     # 0.0 to 1.0


class Action(BaseModel):
    decision: Decision
    fraud_flags: List[FraudFlag] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    total: float = Field(ge=0.0, le=1.0)
    flags_score: float    # 0.0–0.40 — correct fraud flags detected
    decision_score: float # 0.0–0.30 — correct final decision
    precision_score: float # 0.0–0.30 — no false positives
    step_bonus: float = Field(default=0.0, ge=0.0, le=0.10)  # bonus for multi-step reasoning
    feedback: str         # human-readable explanation
    hints: List[str] = Field(default_factory=list)  # progressive hints for the agent
