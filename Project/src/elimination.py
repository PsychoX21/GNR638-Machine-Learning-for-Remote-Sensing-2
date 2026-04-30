import re
import logging
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

from src.model_manager import ModelManager
from src.extraction import format_question_for_reasoning
from src.formulas import get_formula_injection

logger = logging.getLogger(__name__)

ELIMINATION_PROMPT_TEMPLATE = """{formula_injection}

{question_text}

{diagram_instruction}

TASK: Analyze this deep learning MCQ using elimination reasoning.

For EACH of the 4 options:
1. Construct the STRONGEST possible argument that it is WRONG.
   - Cite the specific flaw: wrong formula, incorrect API behavior, wrong tensor shape, false claim about a mathematical property, etc.
2. Rate your falsification argument:
   - DEFINITE: Certain this option is wrong
   - PROBABLE: Likely wrong
   - WEAK: Could be wrong but uncertain
   - NONE: Cannot falsify — this is probably correct

3. After analyzing all options, assign a probability that each option is correct.
   The probabilities must sum to approximately 100%.

Format your final answer on the LAST LINE as exactly:
ANSWER: <1|2|3|4|5>
where 5 means your maximum probability is below 40% (genuinely uncertain).

Also state probabilities like:
Option 1: XX%
Option 2: XX%
Option 3: XX%
Option 4: XX%"""

DIAGRAM_INSTRUCTION = (
    "IMPORTANT: This question contains a visual diagram. Before answering, "
    "first describe the diagram in precise technical terms: its type, components, "
    "connections, and what it represents. Use this description as the basis for your reasoning."
)


def run_elimination(
    model_manager: ModelManager,
    extraction: Dict[str, Any],
    images: Optional[List[Image.Image]] = None,
    temperature: float = 0.0,
    prompt_variant: str = "standard",
) -> Tuple[Optional[int], float, str]:
    question_text = format_question_for_reasoning(extraction)
    formula_text = get_formula_injection(extraction.get("key_concept", ""))
    formula_injection = formula_text if formula_text else ""
    has_diagram = extraction.get("has_diagram", False)
    diagram_instruction = DIAGRAM_INSTRUCTION if has_diagram else ""

    if prompt_variant == "mathematical":
        prompt = _get_mathematical_prompt(question_text, formula_injection)
    elif prompt_variant == "anti_trap":
        prompt = _get_anti_trap_prompt(question_text, formula_injection)
    else:
        prompt = ELIMINATION_PROMPT_TEMPLATE.format(
            formula_injection=formula_injection,
            question_text=question_text,
            diagram_instruction=diagram_instruction,
        )

    call_images = images if has_diagram else None

    try:
        response = model_manager.call_primary(
            text_prompt=prompt,
            images=call_images,
            temperature=temperature,
            max_tokens=4096,
            thinking_mode=True,  # Deep reasoning with thinking trace
        )
    except Exception as e:
        logger.error(f"Stage 3: Elimination failed: {e}")
        return None, 0.0, ""

    answer, max_prob = _parse_elimination_response(response)
    logger.info(f"Stage 3 ({prompt_variant}): answer={answer}, max_prob={max_prob:.2f}")
    return answer, max_prob, response


def _parse_elimination_response(response: str) -> Tuple[Optional[int], float]:
    if not response:
        return None, 0.0

    answer_matches = re.findall(r'ANSWER\s*:\s*([1-5])', response)
    answer = int(answer_matches[-1]) if answer_matches else None

    if answer == 5:
        return None, 0.0

    prob_matches = re.findall(
        r'[Oo]ption\s*(\d)\s*[:\-—=\s]+(\d+(?:\.\d+)?)\s*%',
        response
    )

    if prob_matches:
        probs = {}
        for opt, prob_str in prob_matches:
            probs[int(opt)] = float(prob_str) / 100.0
        max_prob = max(probs.values()) if probs else 0.65
    else:
        max_prob = 0.65

    return answer, max_prob


def _get_mathematical_prompt(question_text: str, formula_injection: str) -> str:
    return f"""{formula_injection}

{question_text}

TASK: Approach this PURELY MATHEMATICALLY. Ignore intuition entirely.

1. Set up the formal equations and definitions relevant to this question.
2. Trace through the calculation step by step, showing every intermediate value.
3. Consider each option based ONLY on mathematical correctness.
4. For each option, determine if it is mathematically valid or invalid.

Assign probability that each option is correct (must sum to ~100%):
Option 1: XX%
Option 2: XX%
Option 3: XX%
Option 4: XX%

State final answer on the LAST LINE as exactly: ANSWER: <1|2|3|4|5>"""


def _get_anti_trap_prompt(question_text: str, formula_injection: str) -> str:
    return f"""{formula_injection}

{question_text}

TASK: This is an exam question DESIGNED TO TRICK students.

1. Think about what COMMON MISCONCEPTIONS would lead a student to each wrong answer.
2. Identify which options are TRAPS (designed to catch common errors).
3. For each option, determine: Is this the trap answer, or is it genuinely correct?
4. Pay special attention to:
   - Negation traps ("which does NOT...")
   - Off-by-one errors in shape calculations
   - Confusion between training and inference behavior
   - Default parameter values that students often get wrong

Assign probability that each option is correct (must sum to ~100%):
Option 1: XX%
Option 2: XX%
Option 3: XX%
Option 4: XX%

State final answer on the LAST LINE as exactly: ANSWER: <1|2|3|4|5>"""
