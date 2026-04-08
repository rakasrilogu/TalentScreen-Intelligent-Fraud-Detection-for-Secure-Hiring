"""
TalentScreen OpenEnv — Task Graders
Fully deterministic. Compares agent action against embedded ground truth.
No LLM-as-judge. No randomness. Same input = same score always.
"""

from __future__ import annotations
from typing import Dict, Any
from talentscreen.models import Action, Reward, Decision


def grade_action(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Score an agent's action against ground truth.
    
    Reward breakdown:
      flags_score    (0.0 - 0.40): correct fraud flags detected
      decision_score (0.0 - 0.30): correct final decision
      precision_score(0.0 - 0.30): no false positives penalised
    """

    gt_flags      = ground_truth["flags"]           # list of expected flags
    gt_decision   = ground_truth["correct_decision"] # string
    total_gt      = ground_truth["total_flags"]      # int

    agent_flags   = action.fraud_flags
    agent_decision = action.decision.value

    # -----------------------------------------------------------------------
    # 1. Flags score — how many correct flags did agent detect?
    # -----------------------------------------------------------------------
    detected_correct = 0
    matched_fields   = set()

    for af in agent_flags:
        for gt_flag in gt_flags:
            if (af.field == gt_flag["field"]
                    and af.reason_code == gt_flag["reason_code"]
                    and af.severity   >= gt_flag["severity_min"]
                    and gt_flag["field"] not in matched_fields):
                detected_correct += 1
                matched_fields.add(gt_flag["field"])
                break

    flags_score = round((detected_correct / max(total_gt, 1)) * 0.40, 4)

    # -----------------------------------------------------------------------
    # 2. Decision score — is the final decision correct?
    # -----------------------------------------------------------------------
    decision_score = 0.30 if agent_decision == gt_decision else 0.0

    # partial credit: FLAG when correct is REJECT = 0.10
    if agent_decision == "FLAG" and gt_decision == "REJECT":
        decision_score = 0.10

    # -----------------------------------------------------------------------
    # 3. Precision score — penalise false positives
    # -----------------------------------------------------------------------
    false_positives = max(0, len(agent_flags) - detected_correct)
    precision_score = max(0.0, round(0.30 - (false_positives * 0.10), 4))

    # -----------------------------------------------------------------------
    # Total
    # -----------------------------------------------------------------------
    total = round(min(1.0, flags_score + decision_score + precision_score), 4)

    # -----------------------------------------------------------------------
    # Feedback message
    # -----------------------------------------------------------------------
    feedback_parts = []
    feedback_parts.append(f"Detected {detected_correct}/{total_gt} correct flags.")
    if agent_decision == gt_decision:
        feedback_parts.append(f"Decision '{agent_decision}' is correct.")
    else:
        feedback_parts.append(
            f"Decision '{agent_decision}' is wrong. Expected '{gt_decision}'."
        )
    if false_positives > 0:
        feedback_parts.append(f"{false_positives} false positive flag(s) penalised.")

    feedback = " ".join(feedback_parts)

    # Build progressive hints
    hints = []
    missed = [f for f in gt_flags if f["field"] not in matched_fields]
    if missed:
        hints.append(f"You missed {len(missed)} flag(s). Look more carefully at the profile fields.")
    if agent_decision != gt_decision:
        hints.append(f"Re-read the decision_rules in protocol_rules to reconsider your decision.")

    return Reward(
        total           = total,
        flags_score     = flags_score,
        decision_score  = decision_score,
        precision_score = precision_score,
        step_bonus      = 0.0,
        feedback        = feedback,
        hints           = hints,
    )
