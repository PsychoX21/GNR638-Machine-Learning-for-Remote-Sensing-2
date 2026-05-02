import os
import csv
import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)


def safe_output(raw_answer: Any) -> int:
    try:
        val = int(raw_answer)
        if val in {1, 2, 3, 4, 5}:
            return val
    except (TypeError, ValueError):
        pass
    logger.warning(f"Output guard: {raw_answer!r} -> 5 (skip)")
    return 5


def write_submission_csv(results: List[Tuple[str, int]], output_path: str = "submission.csv") -> None:
    safe_results = [(name, safe_output(ans)) for name, ans in results]

    # Write to temp file first, then atomic rename.
    # If process is killed mid-write, the previous submission.csv stays intact.
    temp_path = output_path + ".tmp"
    try:
        with open(temp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "image_name", "option"])
            for image_name, answer in safe_results:
                writer.writerow([image_name, image_name, answer])
        os.replace(temp_path, output_path)
        logger.info(f"Submission CSV: {output_path} ({len(safe_results)} entries)")
    except Exception as e:
        logger.error(f"Failed to write CSV: {e}")
        # Fallback: try direct write
        try:
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "image_name", "option"])
                for image_name, answer in safe_results:
                    writer.writerow([image_name, image_name, answer])
        except Exception as e2:
            logger.critical(f"CSV write completely failed: {e2}")
