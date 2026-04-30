import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter
from PIL import Image

from src.model_manager import ModelManager
from src.elimination import run_elimination
from src.extraction import format_question_for_reasoning

logger = logging.getLogger(__name__)

VERIFICATION_PROMPT = """You are a deep learning expert. Given this question and 4 options, select the most likely correct answer.

{question_text}

Think carefully, then output ONLY one integer: 1, 2, 3, or 4."""


def run_voting(
    model_manager: ModelManager,
    extraction: Dict[str, Any],
    first_vote_answer: Optional[int],
    first_vote_prob: float,
    images: Optional[List[Image.Image]] = None,
) -> Tuple[Optional[int], bool, float, List[int]]:
    all_votes = []

    if first_vote_answer is not None:
        all_votes.append(first_vote_answer)

    logger.info("Stage 4: Running Vote 2 (mathematical)...")
    answer2, _, _ = run_elimination(
        model_manager, extraction, images,
        temperature=0.4,
        prompt_variant="mathematical",
    )
    if answer2 is not None:
        all_votes.append(answer2)

    logger.info("Stage 4: Running Vote 3 (anti-trap)...")
    answer3, _, _ = run_elimination(
        model_manager, extraction, images,
        temperature=0.4,
        prompt_variant="anti_trap",
    )
    if answer3 is not None:
        all_votes.append(answer3)

    return _interpret_votes(all_votes)


def _interpret_votes(votes: List[int]) -> Tuple[Optional[int], bool, float, List[int]]:
    if not votes:
        return None, False, 0.0, votes

    counter = Counter(votes)
    majority_answer, majority_count = counter.most_common(1)[0]
    total_votes = len(votes)
    is_unanimous = (majority_count == total_votes and total_votes >= 2)
    agreement_ratio = majority_count / total_votes

    if is_unanimous:
        logger.info(f"Stage 4: UNANIMOUS ({total_votes}/{total_votes}) -> {majority_answer}")
    elif agreement_ratio >= 0.5:
        logger.info(f"Stage 4: MAJORITY ({majority_count}/{total_votes}) -> {majority_answer}")
    else:
        logger.info(f"Stage 4: SPLIT. Votes: {votes}")
        majority_answer = None

    return majority_answer, is_unanimous, agreement_ratio, votes


def run_cross_model_verification(
    model_manager: ModelManager,
    extraction: Dict[str, Any],
    majority_answer: int,
) -> Tuple[Optional[int], bool]:
    """Self-consistency verification using the same model with a different prompt."""
    logger.info("Stage 5: Running self-consistency verification...")

    question_text = format_question_for_reasoning(extraction)
    prompt = VERIFICATION_PROMPT.format(question_text=question_text)

    try:
        response = model_manager.call_secondary(
            text_prompt=prompt,
            images=None,  # Text-only — avoids re-reading the image
            max_tokens=256,
        )
    except Exception as e:
        logger.error(f"Stage 5: Verification failed: {e}")
        return None, False

    secondary_answer = _parse_verification_response(response)
    if secondary_answer is None:
        logger.warning(f"Stage 5: Could not parse answer from: {response[:200]}")
        return None, False

    agrees = (secondary_answer == majority_answer)
    logger.info(f"Stage 5: verification={secondary_answer}, majority={majority_answer}, agrees={agrees}")
    return secondary_answer, agrees


def _parse_verification_response(response: str) -> Optional[int]:
    if not response:
        return None

    matches = re.findall(r'\b([1-4])\b', response)
    if matches:
        return int(matches[-1])

    for char in reversed(response):
        if char in '1234':
            return int(char)
    return None
