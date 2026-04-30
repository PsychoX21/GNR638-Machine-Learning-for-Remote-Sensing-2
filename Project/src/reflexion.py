import re
import logging
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

from src.model_manager import ModelManager
from src.extraction import format_question_for_reasoning

logger = logging.getLogger(__name__)

REFLEXION_PROMPT = """You previously analyzed this deep learning MCQ and concluded the answer is Option {current_answer}.

{question_text}

TASK: Act as a HARSH ACADEMIC CRITIC. Before committing to Option {current_answer}:

1. What is the SINGLE STRONGEST argument that Option {current_answer} is WRONG?
2. Is there an option you may have UNDER-EVALUATED or dismissed too quickly?
3. Check for common traps:
   - Is there a negation in the question ("which does NOT...")?
   - Are there subtle differences between options you might have missed?
   - Could a default parameter value change the answer?

After your critique, state your final decision on the LAST LINE as exactly:
FINAL: <1|2|3|4|5>
where 5 means you are now genuinely uncertain after reflection."""


def should_trigger_reflexion(
    pot_answer: Optional[int],
    max_prob: float,
    is_unanimous: bool,
    time_remaining_per_q: float,
) -> bool:
    if pot_answer is not None:
        return False
    if not (0.50 <= max_prob <= 0.75):
        return False
    if is_unanimous:
        return False
    if time_remaining_per_q <= 20:
        return False
    return True


def run_reflexion(
    model_manager: ModelManager,
    extraction: Dict[str, Any],
    current_answer: int,
    images: Optional[List[Image.Image]] = None,
) -> Tuple[Optional[int], bool, bool]:
    logger.info(f"Stage 6: Running reflexion on answer {current_answer}...")

    question_text = format_question_for_reasoning(extraction)
    prompt = REFLEXION_PROMPT.format(
        current_answer=current_answer,
        question_text=question_text,
    )

    try:
        response = model_manager.call_primary(
            text_prompt=prompt,
            images=images,
            temperature=0.0,
            max_tokens=2048,
            thinking_mode=True,
        )
    except Exception as e:
        logger.error(f"Stage 6: Reflexion failed: {e}")
        return current_answer, False, False

    return _parse_reflexion_response(response, current_answer)


def _parse_reflexion_response(response: str, current_answer: int) -> Tuple[Optional[int], bool, bool]:
    if not response:
        return current_answer, False, False

    matches = re.findall(r'FINAL\s*:\s*([1-5])', response)
    if not matches:
        matches = re.findall(r'final\s*(?:answer|decision)?\s*[:=]\s*([1-5])', response, re.IGNORECASE)

    if not matches:
        return current_answer, False, False

    final = int(matches[-1])

    if final == 5:
        return None, True, True
    elif final != current_answer:
        return final, True, True
    else:
        return final, False, False
