import json
import re
import copy
import logging
from typing import Optional, Dict, Any, List
from PIL import Image

from src.model_manager import ModelManager

logger = logging.getLogger(__name__)

EMPTY_EXTRACTION = {
    "question": "",
    "options": {"1": "", "2": "", "3": "", "4": ""},
    "has_code": False,
    "has_math": False,
    "has_diagram": False,
    "is_computational": False,
    "key_concept": "",
}

EXTRACTION_PROMPT = """You are reading a scanned exam image containing a deep learning multiple-choice question.

Your ONLY task is to faithfully transcribe the content. Do NOT answer the question.

Extract the following and return as a JSON object with EXACTLY this schema:
{
  "question": "complete verbatim question text including all math/code",
  "options": {
    "1": "complete text of option A/1",
    "2": "complete text of option B/2",
    "3": "complete text of option C/3",
    "4": "complete text of option D/4"
  },
  "has_code": true/false,
  "has_math": true/false,
  "has_diagram": true/false,
  "is_computational": true/false,
  "key_concept": "brief topic, e.g. conv output shape, BatchNorm inference, LSTM parameter count"
}

Rules:
- Map options labeled A/B/C/D or (a)/(b)/(c)/(d) to keys "1"/"2"/"3"/"4"
- Preserve ALL mathematical notation, subscripts, Greek letters exactly
- Preserve ALL code exactly including indentation and variable names
- is_computational = true if the answer could be derived by running Python code
- has_code = true if question or options contain code snippets
- has_math = true if question involves mathematical formulas or calculations
- has_diagram = true if there is a neural network diagram or architecture figure
- Return ONLY the JSON object — no preamble, no markdown fences, no explanation"""

# Retry prompt for when first extraction fails
RETRY_PROMPT = """The previous extraction was incomplete or failed to parse.

This is a scanned exam image with a deep learning MCQ. Carefully read every character including subscripts, code, and mathematical notation.

Return a JSON object with this EXACT schema:
{
  "question": "complete verbatim question text",
  "options": {"1": "option A text", "2": "option B text", "3": "option C text", "4": "option D text"},
  "has_code": true/false,
  "has_math": true/false,
  "has_diagram": true/false,
  "is_computational": true/false,
  "key_concept": "brief topic keyword"
}

CRITICAL: Each option must be non-empty. Return ONLY valid JSON."""


def extract_question(model_manager: ModelManager, images: List[Image.Image]) -> Dict[str, Any]:
    result = _attempt_extraction(model_manager, images, EXTRACTION_PROMPT)
    if result is not None:
        logger.info(f"Stage 1: Extraction OK. Concept: {result.get('key_concept', 'N/A')}")
        return result

    logger.warning("Stage 1: First extraction failed, retrying...")
    result = _attempt_extraction(model_manager, images, RETRY_PROMPT)
    if result is not None:
        logger.info(f"Stage 1: Retry OK. Concept: {result.get('key_concept', 'N/A')}")
        return result

    logger.error("Stage 1: Both extraction attempts failed.")
    return copy.deepcopy(EMPTY_EXTRACTION)


def _attempt_extraction(model_manager: ModelManager, images: List[Image.Image], prompt: str) -> Optional[Dict[str, Any]]:
    try:
        response = model_manager.call_primary(
            text_prompt=prompt,
            images=images,
            temperature=0.0,
            max_tokens=2048,
            thinking_mode=False,  # OCR = no thinking needed
        )
        return _parse_extraction_response(response)
    except Exception as e:
        logger.error(f"Extraction attempt failed: {e}")
        return None


def _parse_extraction_response(response: str) -> Optional[Dict[str, Any]]:
    if not response:
        return None

    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        logger.warning(f"No JSON found in response: {response[:200]}")
        return None

    json_str = match.group()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}. Raw: {json_str[:300]}")
        data = _try_fix_json(json_str)
        if data is None:
            return None

    if not _validate_extraction(data):
        return None

    result = copy.deepcopy(EMPTY_EXTRACTION)
    result["question"] = str(data.get("question", ""))

    options = data.get("options", {})
    if isinstance(options, dict):
        for key in ["1", "2", "3", "4"]:
            result["options"][key] = str(options.get(key, ""))

    result["has_code"] = bool(data.get("has_code", False))
    result["has_math"] = bool(data.get("has_math", False))
    result["has_diagram"] = bool(data.get("has_diagram", False))
    result["is_computational"] = bool(data.get("is_computational", False))
    result["key_concept"] = str(data.get("key_concept", ""))

    return result


def _validate_extraction(data: dict) -> bool:
    if not isinstance(data, dict):
        return False

    question = data.get("question", "")
    if not question or not str(question).strip():
        return False

    options = data.get("options", {})
    if not isinstance(options, dict):
        return False

    for key in ["1", "2", "3", "4"]:
        val = options.get(key, "")
        if not val or not str(val).strip():
            return False

    return True


def _try_fix_json(json_str: str) -> Optional[dict]:
    fixed = re.sub(r',\s*}', '}', json_str)
    fixed = re.sub(r',\s*]', ']', fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return None


def format_question_for_reasoning(extraction: Dict[str, Any]) -> str:
    q = extraction["question"]
    opts = extraction["options"]
    text = f"QUESTION:\n{q}\n\nOPTIONS:\n"
    text += f"Option 1: {opts['1']}\n"
    text += f"Option 2: {opts['2']}\n"
    text += f"Option 3: {opts['3']}\n"
    text += f"Option 4: {opts['4']}\n"

    return text
