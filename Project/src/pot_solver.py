import os
import re
import sys
import subprocess
import tempfile
import logging
from typing import Optional, Dict, Any, List
from PIL import Image

from src.model_manager import ModelManager
from src.extraction import format_question_for_reasoning

logger = logging.getLogger(__name__)

CODE_TIMEOUT = 15

POT_PROMPT = """You are a Python expert. Given the following deep learning MCQ, write Python code that VERIFIES each of the four options.

{question_text}

INSTRUCTIONS:
1. Compute the correct answer from first principles using numpy, torch, math, or sympy as needed.
2. Check which of the 4 options matches your computation.
3. The ABSOLUTE LAST LINE of your code's output must be exactly one digit: 1, 2, 3, or 4 — representing the correct option number.
4. If you cannot determine the answer computationally, print "CANNOT_VERIFY" as the last line.
5. Do NOT use any imports besides: numpy, torch, math, sympy, itertools, collections.
6. Do NOT use any network calls, file I/O, or GUI operations.
7. Keep the code concise — no more than 100 lines.
8. Use print() for the final answer on the last line.

Write ONLY the Python code — no markdown fences, no explanation, no comments outside the code."""


SAFE_PREAMBLE = """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sympy
from itertools import *
from collections import *
"""


def run_pot(model_manager: ModelManager, extraction: Dict[str, Any]) -> Optional[int]:
    should_trigger = (
        extraction.get("has_code", False) or
        extraction.get("is_computational", False) or
        extraction.get("has_math", False)
    )

    if not should_trigger:
        logger.info("Stage 2: PoT not triggered (no code/computational/math flags)")
        return None

    logger.info("Stage 2: Running Program of Thought...")

    question_text = format_question_for_reasoning(extraction)
    prompt = POT_PROMPT.format(question_text=question_text)

    try:
        code_response = model_manager.call_primary(
            text_prompt=prompt,
            images=None,  # Text-only for code generation
            temperature=0.0,
            max_tokens=2048,
            thinking_mode=True,  # Use thinking for careful code generation
        )
    except Exception as e:
        logger.error(f"Stage 2: Code generation failed: {e}")
        return None

    code = _clean_code(code_response)
    if not code:
        logger.warning("Stage 2: No valid code extracted")
        return None

    return _execute_code(code)


def _clean_code(response: str) -> str:
    if not response:
        return ""

    fence_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    lines = response.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ', 'def ', 'class ', 'for ', 'if ',
                                'while ', 'try:', 'with ', 'print(', '#', 'result',
                                'answer', 'output', 'x ', 'y ', 'model', 'layer',
                                'input', 'conv', 'linear', 'torch', 'np.', 'numpy')):
            in_code = True
        if in_code:
            code_lines.append(line)

    return '\n'.join(code_lines).strip()


def _execute_code(code: str) -> Optional[int]:
    full_code = SAFE_PREAMBLE + "\n" + code

    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir='.', encoding='utf-8'
        ) as f:
            f.write(full_code)
            temp_path = f.name
    except Exception as e:
        logger.error(f"Failed to write temp file: {e}")
        return None

    try:
        # Force CPU-only execution so PoT doesn't compete with vLLM for GPU
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
            env=env,
        )

        stdout = result.stdout.strip()

        if result.returncode != 0:
            logger.warning(f"Stage 2: Code error (rc={result.returncode})")
            return None

        if not stdout:
            return None

        if "CANNOT_VERIFY" in stdout:
            logger.info("Stage 2: Code reported CANNOT_VERIFY")
            return None

        last_line = stdout.split('\n')[-1].strip()

        if last_line in {"1", "2", "3", "4"}:
            answer = int(last_line)
            logger.info(f"Stage 2: PoT answer: {answer}")
            return answer

        logger.warning(f"Stage 2: Invalid last line: '{last_line}'")
        return None

    except subprocess.TimeoutExpired:
        logger.warning(f"Stage 2: Timed out ({CODE_TIMEOUT}s)")
        return None
    except Exception as e:
        logger.error(f"Stage 2: Unexpected error: {e}")
        return None
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
