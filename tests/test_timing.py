import logging

import pytest

from findex.timing import timed


@timed
def add(a, b):
    """Складає два числа."""
    return a + b


@timed
def boom():
    raise RuntimeError


def test_wraps_keeps_metadata():
    assert add.__name__ == "add"
    assert add.__doc__ == "Складає два числа."
    assert add.__wrapped__(1, 2) == 3  # оригінал доступний через __wrapped__


def test_logs_time(caplog):
    with caplog.at_level(logging.INFO, logger="findex.timing"):
        assert add(1, 2) == 3
    assert "add took" in caplog.text and "ms" in caplog.text


def test_logs_even_on_exception(caplog):
    with caplog.at_level(logging.INFO, logger="findex.timing"), pytest.raises(RuntimeError):
        boom()
    assert "boom took" in caplog.text
