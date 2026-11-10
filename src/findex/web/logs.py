import logging
from contextvars import ContextVar
from typing import Any

# id поточного запиту; middleware ставить його на вході, фільтр дописує в кожен запис логу.
# ContextVar, а не глобальна змінна: одночасні запити (корутини і потоки пулу) не плутаються.
request_id: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


class QuietHealthFilter(logging.Filter):
    # Платформа смикає /health кожні кілька секунд — успішні перевірки в лог не пишемо.
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 5:
            path, status = args[2], args[4]
            return not (str(path).startswith("/health") and status == 200)
        return True


FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"


def log_config(level: str = "INFO") -> dict[str, Any]:
    # Словник для logging.config.dictConfig. Його ж отримує uvicorn (log_config=...),
    # тож кожен воркер-процес налаштовує логи однаково.
    handler = {
        "class": "logging.StreamHandler",
        "stream": "ext://sys.stderr",
        "formatter": "default",
        "filters": ["request_id"],
    }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {"()": "findex.web.logs.RequestIdFilter"},
            "quiet_health": {"()": "findex.web.logs.QuietHealthFilter"},
        },
        "formatters": {"default": {"format": FORMAT}},
        "handlers": {
            "default": handler,
            "access": {**handler, "filters": ["request_id", "quiet_health"]},
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": level, "propagate": False},
            "findex": {"handlers": ["default"], "level": level, "propagate": False},
        },
    }
