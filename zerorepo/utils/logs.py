import logging
from pathlib import Path
from typing import Optional, Union

def setup_logger(
    logger: Optional[logging.Logger] = None,
    file_path: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup a logger with console output by default.
    If `file_path` is provided, also log to that file.

    - Avoids duplicate handlers when called multiple times.
    - Creates parent directory for file_path if needed.
    """
    if logger is None:
        logger = logging.getLogger("Agent")

    logger.setLevel(level)
    logger.propagate = False  # avoid double logging via root logger

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # Ensure a stream handler exists
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # Optionally add a file handler
    if file_path is not None:
        fp = Path(file_path)
        fp.parent.mkdir(parents=True, exist_ok=True)

        # Avoid adding the same file handler multiple times
        has_same_file = any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == fp
            for h in logger.handlers
        )
        if not has_same_file:
            fh = logging.FileHandler(fp, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger