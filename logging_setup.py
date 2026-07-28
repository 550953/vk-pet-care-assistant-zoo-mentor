"""
Structured JSON logging for Better Stack.

Каждая запись лога — одна строка JSON. Это даёт возможность строить
дашборды/алерты в Better Stack по полям (event, peer_id, triage_level и т.д.),
а не парсить текст регулярками.

Использование:

    from logging_setup import setup_logging, new_correlation_id, set_peer_id, log_event

    setup_logging()                    # один раз при старте приложения

    cid = new_correlation_id()         # в начале обработки каждого сообщения/задачи
    set_peer_id(vk_id)

    log_event(logger, logging.INFO, "message_received", peer_id=vk_id, text_len=len(text))
"""
import datetime
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

from pythonjsonlogger.json import JsonFormatter


class _UtcJsonFormatter(JsonFormatter):
    """JsonFormatter с корректным UTC ISO8601 timestamp (стандартный %f не
    разворачивается через time.strftime, поэтому переопределяем formatTime)."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

# ─── Контекст, сквозной для одной обработки апдейта ──────────────────────────
# ContextVar безопасен для asyncio: у каждой задачи (asyncio.Task) свой набор
# значений, поэтому конкурентные обработки разных сообщений не путают peer_id.

correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
peer_id_var: ContextVar[Optional[int]] = ContextVar("peer_id", default=None)
langfuse_trace_id_var: ContextVar[Optional[str]] = ContextVar("langfuse_trace_id", default=None)


class _ContextFilter(logging.Filter):
    """Автоматически добавляет correlation_id/peer_id/langfuse_trace_id в каждую запись."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get()
        record.peer_id = peer_id_var.get()
        record.langfuse_trace_id = langfuse_trace_id_var.get()
        return True


def setup_logging(level: int = logging.INFO) -> None:
    """Настраивает root-логгер на вывод JSON в stdout. Вызывать один раз при старте."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())

    formatter = _UtcJsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s %(peer_id)s %(langfuse_trace_id)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
        },
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Приглушаем шумные библиотеки — их текстовые access-логи не несут пользы в JSON-виде.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def new_correlation_id() -> str:
    """Генерирует новый correlation_id и устанавливает его в контекст текущей задачи."""
    cid = uuid.uuid4().hex[:12]
    correlation_id_var.set(cid)
    return cid


def set_peer_id(peer_id: Optional[int]) -> None:
    peer_id_var.set(peer_id)


def set_langfuse_trace_id(trace_id: Optional[str]) -> None:
    langfuse_trace_id_var.set(trace_id)


def log_event(logger: logging.Logger, level: int, event: str, **fields) -> None:
    """
    Структурированная запись лога.

    log_event(logger, logging.INFO, "message_received", peer_id=123, text_len=42)

    Поля со значением None не попадают в лог (чтобы не засорять JSON пустыми ключами).
    correlation_id/peer_id добавляются автоматически через _ContextFilter — их сюда
    передавать не нужно, если только не требуется переопределить контекстное значение.
    """
    extra = {"event": event}
    extra.update({k: v for k, v in fields.items() if v is not None})
    logger.log(level, event, extra=extra)
