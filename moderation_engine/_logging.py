import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, stream=sys.stdout, format="%(message)s")

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    # JSON when stdout isn't a terminal (containers, CI); pretty in interactive dev.
    renderer = (
        structlog.dev.ConsoleRenderer()
        if sys.stdout.isatty()
        else structlog.processors.JSONRenderer()
    )
    processors.append(renderer)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
