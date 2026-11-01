import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger("findex.timing")


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор: логує, скільки виконувалась функція.

    @functools.wraps копіює __name__, __qualname__, __doc__ оригіналу на обгортку —
    без нього всі задекоровані функції називались би 'wrapper'.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:  # час логуємо навіть якщо функція впала
            ms = (time.perf_counter() - t0) * 1000
            log.info("%s took %.2f ms", func.__qualname__, ms)

    return wrapper
