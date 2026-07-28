"""
Langfuse — трейсинг LLM-вызовов (Gemini + GigaChat fallback).

В отличие от Gemini/Groq/Tavily это не пул ключей (round-robin не нужен) —
одна тройка LANGFUSE_PUBLIC_KEY_<suffix> / LANGFUSE_SECRET_KEY_<suffix> /
LANGFUSE_BASE_URL_<suffix> из Infisical (см. infisical_loader.extract_langfuse_config).

ВАЖНО про порядок инициализации: llm.py и gigachat_client.py делают
`from observability import langfuse` на уровне модуля — это происходит при
старте процесса, ДО того как main.py:lifespan успевает асинхронно сходить
в Infisical за секретами. Поэтому `langfuse` здесь — не сам клиент, а лёгкий
прокси: до вызова init_langfuse() он резолвится в дефолтный (no-op) клиент,
после — в настоящий, настроенный клиент. Код в llm.py/gigachat_client.py
менять не нужно, `langfuse.start_as_current_observation(...)` работает как
обычно в обоих случаях.
"""
import logging
import os
from typing import Any, Optional

from langfuse import Langfuse, get_client

logger = logging.getLogger(__name__)

_client: Optional[Langfuse] = None


class _LazyLangfuseProxy:
    """Прокси-обёртка: каждый доступ к атрибуту резолвится в актуальный клиент."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_resolve_client(), name)


def _resolve_client() -> Langfuse:
    global _client
    if _client is None:
        _client = get_client()  # без ключей — no-op режим, бота не ломает
    return _client


langfuse = _LazyLangfuseProxy()


def init_langfuse(secrets: Optional[dict[str, str]] = None) -> None:
    """
    Настраивает Langfuse-клиент. Вызывать один раз из main.py:lifespan,
    после того как секреты Infisical загружены — тем же паттерном, что
    init_key_pool()/init_groq_pool()/init_tavily_pool() в main.py.

    secrets — словарь секретов Infisical. Если в нём не нашлось
    LANGFUSE_PUBLIC_KEY_*/LANGFUSE_SECRET_KEY_* — падаем на обычные
    переменные окружения (LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST,
    без суффикса) для локальной разработки.
    """
    global _client

    config: Optional[dict[str, str]] = None
    if secrets:
        from infisical_loader import extract_langfuse_config
        config = extract_langfuse_config(secrets)

    if not config:
        pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        sec = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        host = os.environ.get("LANGFUSE_HOST", "").strip() or "https://cloud.langfuse.com"
        if pub and sec:
            config = {"public_key": pub, "secret_key": sec, "host": host}

    if config:
        _client = Langfuse(
            public_key=config["public_key"],
            secret_key=config["secret_key"],
            host=config["host"],
        )
        logger.info("Langfuse: трейсинг включён (host=%s)", config["host"])
    else:
        _client = get_client()  # no-op режим
        logger.warning(
            "Langfuse: LANGFUSE_PUBLIC_KEY_*/LANGFUSE_SECRET_KEY_* не найдены "
            "ни в Infisical, ни в окружении — трейсинг отключён (no-op режим)"
        )


def log_status() -> None:
    """Оставлено для обратной совместимости — статус теперь логирует init_langfuse()."""
    if _client is None:
        logger.warning("Langfuse: init_langfuse() ещё не вызывался")
