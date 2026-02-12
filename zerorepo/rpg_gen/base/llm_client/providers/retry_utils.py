"""Retry utility with randomized backoff for LLM API calls."""

import random
import time
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_with(
    func: Callable[..., T],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
) -> Callable[..., T]:
    """
    Decorator that adds retry logic with randomized backoff.

    Args:
        func: The function to decorate.
        provider_name: The name of the model provider (for logging).
        max_retries: Maximum number of retry attempts.

    Returns:
        Decorated function with retry logic.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == max_retries:
                    raise

                sleep_time = random.randint(3, 30)
                this_error_message = str(e)
                print(
                    f"{provider_name} API call failed: {this_error_message}. "
                    f"Will sleep for {sleep_time} seconds and retry.\n"
                    f"{traceback.format_exc()}"
                )
                time.sleep(sleep_time)

        raise last_exception or Exception("Retry failed for unknown reason")

    return wrapper
