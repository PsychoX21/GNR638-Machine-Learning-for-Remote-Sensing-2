import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ANSWER_THRESHOLD = 0.55


@dataclass
class PipelineResult:
    pot_answer: Optional[int] = None
    elimination_answer: Optional[int] = None
    elimination_max_prob: float = 0.0
    vote_answers: list = field(default_factory=list)
    vote_majority: Optional[int] = None
    vote_unanimity: bool = False
    vote_agreement_ratio: float = 0.0
    secondary_answer: Optional[int] = None
    secondary_agrees: Optional[bool] = None
    reflexion_answer: Optional[int] = None
    reflexion_changed: bool = False
    reflexion_found_counter: bool = False


def compute_confidence(result: PipelineResult) -> float:
    if result.pot_answer is not None:
        return 1.00

    if result.reflexion_found_counter and result.reflexion_answer is None:
        return 0.00

    if result.elimination_max_prob < 0.40:
        return 0.00

    base = result.elimination_max_prob

    # Single-vote mode (FAST or STANDARD with self-consistency)
    if len(result.vote_answers) < 2:
        if result.secondary_agrees is True:
            confidence = min(0.80, base + 0.10)
        elif result.secondary_agrees is False:
            confidence = max(0.40, base - 0.10)
        else:
            confidence = base
        return confidence

    # All three votes different
    if len(result.vote_answers) >= 3 and result.vote_agreement_ratio < 0.5:
        return 0.00

    # 3/3 unanimous
    if result.vote_unanimity:
        if base > 0.80:
            return 0.95
        elif base > 0.60:
            return 0.85
        else:
            return 0.75

    # 2/3 majority
    if result.vote_majority is not None:
        if result.secondary_agrees is True:
            confidence = 0.75 if base > 0.60 else 0.65
        elif result.secondary_agrees is False:
            confidence = 0.65 if base > 0.70 else 0.50
        else:
            confidence = min(0.70, base)

        if result.reflexion_changed:
            confidence -= 0.10

        return confidence

    return base * 0.8


def decide_answer(result: PipelineResult) -> int:
    confidence = compute_confidence(result)

    if result.pot_answer is not None:
        answer = result.pot_answer
    elif result.reflexion_answer is not None and result.reflexion_changed:
        answer = result.reflexion_answer
    elif result.vote_majority is not None:
        answer = result.vote_majority
    elif result.elimination_answer is not None:
        answer = result.elimination_answer
    else:
        return 5

    if confidence >= ANSWER_THRESHOLD:
        logger.info(f"Decision: ANSWER {answer} (confidence={confidence:.2f})")
        return answer
    else:
        logger.info(f"Decision: SKIP (confidence={confidence:.2f})")
        return 5
