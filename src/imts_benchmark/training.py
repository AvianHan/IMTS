"""Treat library-swallowed interruptions as failed experiments."""
from contextlib import contextmanager
import logging


class _TrainingLog(logging.Handler):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.failures = []

    def emit(self, record):
        message = record.getMessage()
        self.stream.write(f"[{record.levelname}] {message}\n")
        if record.levelno >= logging.ERROR or "training got interrupted" in message.lower():
            self.failures.append(message)


@contextmanager
def checked_training(neural, log):
    """PyPOTS 1.5 can return a checkpoint after errors; reject that outcome.

    The handler is scoped to this fit and removed even when fitting raises.
    Normal patience-based early stopping is allowed.
    """
    if not neural:
        yield
        return
    from pypots.utils.logging import logger
    handler = _TrainingLog(log)
    logger.addHandler(handler)
    try:
        yield
        if handler.failures:
            raise RuntimeError("Neural training aborted despite returning a checkpoint: " + " | ".join(handler.failures))
    finally:
        logger.removeHandler(handler)
