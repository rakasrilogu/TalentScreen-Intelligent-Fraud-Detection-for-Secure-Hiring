"""
TalentScreen OpenEnv — Synthetic Data Generator
All candidate profiles are fictional. Fraud flags are pre-planted
and stored as ground truth for deterministic grading.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
from talentscreen.models import (
    Observation, EducationRecord, EmploymentRecord,
    CertificationRecord, QAPair, ReferenceRecord
)

# ---------------------------------------------------------------------------
# NovaTalent Protocol Rules — injected into every observation
# Agent must use THESE rules, not real-world knowledge
# ---------------------------------------------------------------------------

PROTOCOL_RULES: Dict[str, Any] = {
    "version": "NovaTalent-v2.1",
    "reason_codes": {
        "TIMELINE_CONFLICT":    "Employment or education dates overlap impossibly",
        "CREDENTIAL_MISMATCH":  "Certification ID format does not match issuing body pattern",
        "SKILL_ANSWER_GAP":     "Interview answer contradicts claimed skill proficiency",
        "REFERENCE_DUPLICATE":  "Reference contact details match candidate's own details",
        "TITLE_INFLATION":      "Job title inconsistent with stated responsibilities",
        "GPA_RANGE_VIOLATION":  "GPA outside valid range for stated institution",
        "GAP_UNEXPLAINED":      "Employment gap > 12 months with no explanation",
    },
    "severity_thresholds": {
        "low":      [0.0, 0.39],
        "medium":   [0.4, 0.69],
        "high":     [0.7, 1.0],
    },
    "decision_rules": {
        "PASS":   "0 high-severity flags, <= 1 medium flag",
        "FLAG":   "1+ medium flags OR 1 high flag — send to senior review",
        "REJECT": "2+ high-severity flags OR confirmed credential fraud",
    }
}


# ---------------------------------------------------------------------------
# TASK 1 — EASY
# Single obvious timeline conflict
# Ground truth: 1 flag, decision = FLAG
# ---------------------------------------------------------------------------

def get_easy_candidate() -> Tuple[Observation, Dict]:
    obs = Observation(
        candidate_id="CAND-001",
        task_id="easy",
        step=0,
        education=[
            EducationRecord(
                institution="Novaford University",
                degree="Bachelor of Science",
                field_of_study="Computer Science",
                start_year=2015,
                end_year=2019,   # graduated 2019
                gpa=3.7,
                verified=False
            )
        ],
        employment=[
            EmploymentRecord(
                company="TechNova Solutions",
                title="Software Engineer",
                start_year=2017,   # started work 2017 — BEFORE graduation 2019
                end_year=2022,
                responsibilities=["Built REST APIs", "Code reviews"],
                reference_contact="manager.nova@technovasolutions.com"
            )
        ],
        skills_claimed=["Python", "REST APIs", "Git"],
        certifications=[
            CertificationRecord(
                name="NovaCert Python Associate",
                issuing_body="NovaCert Institute",
                year_obtained=2020,
                credential_id="NCP-2020-44821"  # valid format: NCP-YYYY-#####
            )
        ],
        interview_qa=[
            QAPair(
                question="Describe your experience with REST APIs.",
                answer="I have built several REST APIs using Flask and FastAPI during my time at TechNova.",
                expected_skill="REST APIs"
            )
        ],
        references=[
            ReferenceRecord(
                name="James Holt",
                relationship="Direct Manager",
                company="TechNova Solutions",
                phone="+1-555-0192",
                email="manager.nova@technovasolutions.com"
            )
        ],
        protocol_rules=PROTOCOL_RULES,
        done=False
    )

    ground_truth = {
        "flags": [
            {
                "field": "employment[0].start_year",
                "reason_code": "TIMELINE_CONFLICT",
                "severity_min": 0.7
            }
        ],
        "correct_decision": "FLAG",
        "total_flags": 1
    }

    return obs, ground_truth


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM
# 3 subtle inconsistencies across different fields
# Ground truth: 3 flags, decision = FLAG
# ---------------------------------------------------------------------------

def get_medium_candidate() -> Tuple[Observation, Dict]:
    obs = Observation(
        candidate_id="CAND-002",
        task_id="medium",
        step=0,
        education=[
            EducationRecord(
                institution="Eastbrook College",
                degree="Bachelor of Engineering",
                field_of_study="Software Engineering",
                start_year=2013,
                end_year=2017,
                gpa=4.3,   # GPA violation: max is 4.0 at Eastbrook per protocol
                verified=False
            )
        ],
        employment=[
            EmploymentRecord(
                company="DataBridge Corp",
                title="Senior Data Architect",   # title inflation
                start_year=2017,
                end_year=2023,
                responsibilities=["Wrote SQL queries", "Attended meetings"],  # junior responsibilities
                reference_contact="hr@databridgecorp.com"
            )
        ],
        skills_claimed=["Kubernetes", "Docker", "Python", "Data Architecture"],
        certifications=[
            CertificationRecord(
                name="NovaCert Cloud Practitioner",
                issuing_body="NovaCert Institute",
                year_obtained=2021,
                credential_id="NCP-2021-88234"
            )
        ],
        interview_qa=[
            QAPair(
                question="Walk me through how you would deploy a Kubernetes cluster.",
                answer="I would use a cloud provider dashboard to click through the setup wizard.",
                expected_skill="Kubernetes"   # answer reveals no real K8s knowledge
            )
        ],
        references=[
            ReferenceRecord(
                name="Sandra Voss",
                relationship="HR Manager",
                company="DataBridge Corp",
                phone="+1-555-0341",
                email="hr@databridgecorp.com"
            )
        ],
        protocol_rules=PROTOCOL_RULES,
        done=False
    )

    ground_truth = {
        "flags": [
            {
                "field": "education[0].gpa",
                "reason_code": "GPA_RANGE_VIOLATION",
                "severity_min": 0.5
            },
            {
                "field": "employment[0].title",
                "reason_code": "TITLE_INFLATION",
                "severity_min": 0.5
            },
            {
                "field": "interview_qa[0].answer",
                "reason_code": "SKILL_ANSWER_GAP",
                "severity_min": 0.6
            }
        ],
        "correct_decision": "FLAG",
        "total_flags": 3
    }

    return obs, ground_truth


# ---------------------------------------------------------------------------
# TASK 3 — HARD
# 5 layered flags across 12 fields — requires multi-step cross-referencing
# Ground truth: 5 flags, decision = REJECT
# ---------------------------------------------------------------------------

def get_hard_candidate() -> Tuple[Observation, Dict]:
    obs = Observation(
        candidate_id="CAND-003",
        task_id="hard",
        step=0,
        education=[
            EducationRecord(
                institution="Novaford University",
                degree="Master of Science",
                field_of_study="Machine Learning",
                start_year=2016,
                end_year=2018,
                gpa=3.9,
                verified=False
            ),
            EducationRecord(
                institution="Westgate Community College",
                degree="Bachelor of Science",
                field_of_study="Mathematics",
                start_year=2012,
                end_year=2016,
                gpa=3.5,
                verified=False
            )
        ],
        employment=[
            EmploymentRecord(
                company="QuantumAI Labs",
                title="Principal ML Engineer",
                start_year=2018,
                end_year=2023,
                responsibilities=[
                    "Led team of 12 engineers",
                    "Designed transformer architectures",
                    "Published 3 internal research papers"
                ],
                reference_contact="cto@quantumailabs.com"
            ),
            EmploymentRecord(
                company="NovaData Inc",
                title="Data Analyst",
                start_year=2016,    # overlaps with Master's degree 2016-2018
                end_year=2018,
                responsibilities=["Built dashboards", "SQL reporting"],
                reference_contact="hr@novadatainc.com"
            )
        ],
        skills_claimed=[
            "PyTorch", "TensorFlow", "Transformer Architecture",
            "Team Leadership", "Research", "Kubernetes", "Rust"
        ],
        certifications=[
            CertificationRecord(
                name="NovaCert ML Professional",
                issuing_body="NovaCert Institute",
                year_obtained=2019,
                credential_id="NCML-2019-00291"  # valid format: NCML-YYYY-#####
            ),
            CertificationRecord(
                name="NovaCert Kubernetes Expert",
                issuing_body="NovaCert Institute",
                year_obtained=2020,
                credential_id="NCK-20-5511"  # INVALID format — should be NCK-YYYY-#####
            )
        ],
        interview_qa=[
            QAPair(
                question="Describe the transformer architecture you designed at QuantumAI.",
                answer="I worked closely with the team that designed it. I mostly handled the training pipelines.",
                expected_skill="Transformer Architecture"
            ),
            QAPair(
                question="How do you manage a team of 12 engineers?",
                answer="I have not directly managed engineers before. I prefer individual contributor work.",
                expected_skill="Team Leadership"
            ),
            QAPair(
                question="Walk me through your experience with Rust.",
                answer="I have read about Rust and completed a few online tutorials.",
                expected_skill="Rust"
            )
        ],
        references=[
            ReferenceRecord(
                name="Dr. Alan Mercer",
                relationship="CTO",
                company="QuantumAI Labs",
                phone="+1-555-0874",
                email="cto@quantumailabs.com"
            ),
            ReferenceRecord(
                name="Candidate Self",        # reference contact is the candidate themselves
                relationship="HR Contact",
                company="NovaData Inc",
                phone="+1-555-0291",
                email="candidate.self@gmail.com"   # matches candidate email pattern
            )
        ],
        protocol_rules=PROTOCOL_RULES,
        done=False
    )

    ground_truth = {
        "flags": [
            {
                "field": "employment[1].start_year",
                "reason_code": "TIMELINE_CONFLICT",
                "severity_min": 0.7
            },
            {
                "field": "certifications[1].credential_id",
                "reason_code": "CREDENTIAL_MISMATCH",
                "severity_min": 0.8
            },
            {
                "field": "interview_qa[1].answer",
                "reason_code": "SKILL_ANSWER_GAP",
                "severity_min": 0.7
            },
            {
                "field": "references[1].email",
                "reason_code": "REFERENCE_DUPLICATE",
                "severity_min": 0.9
            },
            {
                "field": "employment[0].title",
                "reason_code": "TITLE_INFLATION",
                "severity_min": 0.6
            }
        ],
        "correct_decision": "REJECT",
        "total_flags": 5
    }

    return obs, ground_truth


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "easy":   get_easy_candidate,
    "medium": get_medium_candidate,
    "hard":   get_hard_candidate,
}
