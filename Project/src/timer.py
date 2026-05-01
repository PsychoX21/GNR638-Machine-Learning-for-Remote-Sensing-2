import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

TOTAL_BUDGET_SECONDS = 55 * 60


class Strategy(Enum):
    FULL = "FULL"
    STANDARD = "STANDARD"
    FAST = "FAST"
    ABORT = "ABORT"


class Timer:
    def __init__(self, total_questions: int):
        self.total_questions = max(1, total_questions)
        self.start_time = time.time()
        self.questions_completed = 0
        self._question_start = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining_time(self) -> float:
        return max(0, TOTAL_BUDGET_SECONDS - self.elapsed)

    @property
    def remaining_questions(self) -> int:
        return max(1, self.total_questions - self.questions_completed)

    @property
    def time_per_question(self) -> float:
        return self.remaining_time / self.remaining_questions

    def get_strategy(self) -> Strategy:
        tpq = self.time_per_question

        if tpq > 65:
            strategy = Strategy.FULL
        elif tpq > 40:
            strategy = Strategy.STANDARD
        elif tpq > 15:
            strategy = Strategy.FAST
        else:
            strategy = Strategy.ABORT

        logger.info(
            f"Strategy: {strategy.value} | "
            f"{self.remaining_time:.0f}s for {self.remaining_questions}q | "
            f"{tpq:.1f}s/q"
        )
        return strategy

    def start_question(self) -> None:
        self._question_start = time.time()

    def end_question(self) -> float:
        self.questions_completed += 1
        if self._question_start is not None:
            elapsed = time.time() - self._question_start
            self._question_start = None
            logger.info(f"Question {self.questions_completed}/{self.total_questions}: {elapsed:.1f}s")
            return elapsed
        return 0.0
