#!/usr/bin/env python3
"""
DL MCQ Solver — Main Inference Pipeline
Usage: python inference.py --test_dir <absolute_path_to_test_dir>
"""

import argparse
import os
import sys
import logging
import traceback
import pandas as pd
from typing import List, Tuple, Optional

from src.model_manager import ModelManager
from src.preprocessing import preprocess_image
from src.extraction import extract_question
from src.pot_solver import run_pot
from src.elimination import run_elimination
from src.voting import run_voting, run_cross_model_verification
from src.reflexion import should_trigger_reflexion, run_reflexion
from src.confidence import PipelineResult, decide_answer
from src.output_guard import safe_output, write_submission_csv
from src.timer import Timer, Strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference.log", mode="w"),
    ],
)
logger = logging.getLogger("inference")


def process_single_question(
    model_manager: ModelManager,
    image_path: str,
    image_name: str,
    strategy: Strategy,
    timer: Timer,
) -> int:
    result = PipelineResult()

    if strategy == Strategy.ABORT:
        return 5

    # Stage 0: Preprocessing
    try:
        images = preprocess_image(image_path)
    except Exception as e:
        logger.error(f"[{image_name}] Preprocessing error: {e}")
        return 5
    if not images:
        return 5

    # Stage 1: Extraction
    try:
        extraction = extract_question(model_manager, images)
    except Exception as e:
        logger.error(f"[{image_name}] Extraction error: {e}")
        return 5
    if not extraction["question"] or not all(extraction["options"].values()):
        logger.warning(f"[{image_name}] Extraction incomplete -> skip")
        return 5

    logger.info(
        f"[{image_name}] concept={extraction['key_concept']}, "
        f"code={extraction['has_code']}, math={extraction['has_math']}, "
        f"diagram={extraction['has_diagram']}, comp={extraction['is_computational']}"
    )

    # Stage 2: Program of Thought
    if strategy in (Strategy.FULL, Strategy.STANDARD):
        try:
            pot_answer = run_pot(model_manager, extraction)
        except Exception as e:
            logger.error(f"[{image_name}] PoT error: {e}")
            pot_answer = None
        if pot_answer is not None:
            result.pot_answer = pot_answer
            logger.info(f"[{image_name}] PoT -> {pot_answer}")
            return safe_output(pot_answer)

    # Stage 3: Elimination Reasoning
    try:
        elim_answer, elim_prob, _ = run_elimination(
            model_manager, extraction, images,
            temperature=0.0, prompt_variant="standard",
        )
    except Exception as e:
        logger.error(f"[{image_name}] Elimination error: {e}")
        return 5

    result.elimination_answer = elim_answer
    result.elimination_max_prob = elim_prob

    if elim_prob >= 0.85 and elim_answer is not None:
        return safe_output(decide_answer(result))

    if elim_prob < 0.40 or elim_answer is None:
        return 5

    # Stage 4: Multi-Angle Voting (FULL only)
    if strategy == Strategy.FULL:
        try:
            majority, is_unanimous, agreement_ratio, all_votes = run_voting(
                model_manager, extraction,
                first_vote_answer=elim_answer,
                first_vote_prob=elim_prob,
                images=images,
            )
        except Exception as e:
            logger.error(f"[{image_name}] Voting error: {e}")
            majority, is_unanimous, agreement_ratio, all_votes = elim_answer, False, 1.0, [elim_answer]

        result.vote_answers = all_votes
        result.vote_majority = majority
        result.vote_unanimity = is_unanimous
        result.vote_agreement_ratio = agreement_ratio

        if majority is None:
            return 5

        # Stage 5: Self-consistency verification (contested 2/3 only)
        if not is_unanimous and majority is not None:
            try:
                sec_answer, sec_agrees = run_cross_model_verification(
                    model_manager, extraction, majority,
                )
                result.secondary_answer = sec_answer
                result.secondary_agrees = sec_agrees
            except Exception as e:
                logger.error(f"[{image_name}] Verification error: {e}")

    elif strategy == Strategy.STANDARD:
        result.vote_answers = [elim_answer]
        result.vote_majority = elim_answer

        try:
            sec_answer, sec_agrees = run_cross_model_verification(
                model_manager, extraction, elim_answer,
            )
            result.secondary_answer = sec_answer
            result.secondary_agrees = sec_agrees
        except Exception as e:
            logger.error(f"[{image_name}] Verification error: {e}")
    else:
        # FAST
        result.vote_answers = [elim_answer]
        result.vote_majority = elim_answer

    # Stage 6: Reflexion (FULL, borderline only)
    if strategy == Strategy.FULL:
        current_answer = result.vote_majority or result.elimination_answer
        if current_answer is not None and should_trigger_reflexion(
            pot_answer=result.pot_answer,
            max_prob=result.elimination_max_prob,
            is_unanimous=result.vote_unanimity,
            time_remaining_per_q=timer.time_per_question,
        ):
            try:
                ref_answer, ref_changed, ref_counter = run_reflexion(
                    model_manager, extraction, current_answer, images,
                )
                result.reflexion_answer = ref_answer
                result.reflexion_changed = ref_changed
                result.reflexion_found_counter = ref_counter
            except Exception as e:
                logger.error(f"[{image_name}] Reflexion error: {e}")

    return safe_output(decide_answer(result))


def main():
    parser = argparse.ArgumentParser(description="DL MCQ Solver")
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()

    test_dir = args.test_dir
    logger.info(f"Test directory: {test_dir}")

    # Read test.csv
    test_csv_path = os.path.join(test_dir, "test.csv")
    try:
        df = pd.read_csv(test_csv_path)
        image_names = df["image_name"].tolist()
    except Exception as e:
        logger.critical(f"Failed to read test.csv: {e}")
        # Try to produce empty submission
        try:
            write_submission_csv([], "submission.csv")
        except Exception:
            pass
        sys.exit(1)

    total_questions = len(image_names)
    logger.info(f"Found {total_questions} questions")

    if total_questions == 0:
        write_submission_csv([], "submission.csv")
        logger.info("No questions to process. Empty submission written.")
        return

    # SAFETY: Write initial submission.csv with all skips IMMEDIATELY.
    # If the process gets killed at any point, a valid submission.csv exists.
    write_submission_csv([(name, 5) for name in image_names], "submission.csv")
    logger.info("Safety submission.csv written (all skips)")

    # Timer starts NOW — the 1-hour clock began when this script started
    timer = Timer(total_questions)
    model_manager = ModelManager()

    try:
        logger.info("Starting primary model...")
        model_manager.start_primary_model()
        logger.info(f"Model loaded. {timer.remaining_time:.0f}s remaining for questions.")

        results: List[Tuple[str, int]] = []

        for idx, image_name in enumerate(image_names):
            timer.start_question()
            strategy = timer.get_strategy()

            image_path = os.path.join(test_dir, "images", f"{image_name}.png")
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                results.append((image_name, 5))
                timer.end_question()
                continue

            logger.info(f"[{idx+1}/{total_questions}] {image_name} ({strategy.value})")

            try:
                answer = process_single_question(
                    model_manager, image_path, image_name, strategy, timer,
                )
            except Exception as e:
                logger.error(f"Error processing {image_name}: {e}")
                answer = 5

            answer = safe_output(answer)
            results.append((image_name, answer))
            elapsed = timer.end_question()
            logger.info(f"[{image_name}] -> {answer} ({elapsed:.1f}s)")

            # Periodically update submission.csv so partial results survive kills
            if (idx + 1) % 5 == 0 or (idx + 1) == total_questions:
                partial = list(results)
                processed = {r[0] for r in partial}
                for name in image_names:
                    if name not in processed:
                        partial.append((name, 5))
                write_submission_csv(partial, "submission.csv")

        # Final write with all results
        write_submission_csv(results, "submission.csv")

        answered = sum(1 for _, a in results if a != 5)
        skipped = sum(1 for _, a in results if a == 5)
        logger.info(f"Done: {answered} answered, {skipped} skipped, {timer.elapsed:.0f}s total")

    except Exception as e:
        logger.critical(f"Fatal error: {e}\n{traceback.format_exc()}")
        # Write whatever results we have
        if 'results' in locals() and results:
            processed = {r[0] for r in results}
            for name in image_names:
                if name not in processed:
                    results.append((name, 5))
            write_submission_csv(results, "submission.csv")
        # else: safety CSV from the beginning is still valid
        logger.info("Emergency submission.csv updated")
    finally:
        model_manager.cleanup()
        logger.info("Cleanup done.")


if __name__ == "__main__":
    main()
