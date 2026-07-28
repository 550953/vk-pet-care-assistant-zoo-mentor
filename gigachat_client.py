"""
GigaChat — резервный (fallback) LLM, подключается ТОЛЬКО когда Gemini
полностью в глобальном blackout (llm.is_overloaded() == True).

Только текстовые запросы — GigaChat не участвует в анализе фото/аудио/PDF,
это ограничение проверяется на стороне вызывающего кода (llm.py).

Авторизация: используем готовый Authorization Key (Base64 от client_id:client_secret),
как в проверенном локальном скрипте — GigaChat(credentials=auth_key, ...) сам
обменивает его на access-токен и обновляет по истечении, без ручного OAuth-обмена
с нашей стороны.

Секреты в Infisical по префиксу GIGACHAT_AUTH_KEY_ (по аналогии с GEMINI_API_KEY_/
GROQ_API_KEY_) — один аккаунт сейчас (GIGACHAT_AUTH_KEY_550953), но пул готов
к нескольким, если появятся ещё.
"""
import asyncio
import logging
from typing import Optional, Tuple, Callable, Awaitable, Any

from gigachat import GigaChat
from gigachat.exceptions import GigaChatException
from sqlalchemy import select, func

from langfuse import propagate_attributes

from key_pool import KeyPool
from db import async_session
from models import GigaChatUsage
from observability import langfuse

logger = logging.getLogger(__name__)

# Пул авторизационных ключей GigaChat — использует общий KeyPool (как Groq/Tavily),
# failover/exhausted/db_log колбэки уже установлены глобально через kp.set_callbacks()
# в main.py lifespan — отдельная привязка колбэков здесь не нужна.
_pool = KeyPool("gigachat", blackout_min=120.0)

# 1 одновременный запрос — как и требует freemium-лимит физлиц GigaChat
_GIGACHAT_SEMAPHORE = asyncio.Semaphore(1)

DEFAULT_MODEL = "GigaChat-2"  # актуальность модели/лимитов сверить перед проды на масштаб

# Актуальные годовые лимиты freemium физлиц на момент внедрения (2026) —
# сверить перед изменением, GigaChat может поменять условия.
GIGACHAT_YEARLY_LIMITS = {
    "GigaChat-2": 250_000_000,
    "GigaChat-2-Pro": 40_000_000,
    "GigaChat-2-Max": 25_000_000,
    "GigaChat-2-Ultra": 50_000_000,
}
GIGACHAT_WARN_THRESHOLD = 0.9  # 90% — предупреждаем администратора заранее

# ── Колбэк учёта токенов, устанавливается из main.py ───────────────────────
# async (model: str, input_tokens: int, output_tokens: int) -> None
_on_usage_cb: Optional[Callable[..., Awaitable[None]]] = None

# ── Колбэк предупреждения о приближении к годовому лимиту ──────────────────
# async (service_label: str) -> None — переиспользует тот же паттерн,
# что и _on_exhausted_cb у KeyPool (уведомление администратора в VK)
_on_budget_warning_cb: Optional[Callable[..., Awaitable[None]]] = None


def set_usage_callback(cb: Optional[Callable] = None) -> None:
    """Устанавливает колбэк для записи usage в БД (gigachat_usage)."""
    global _on_usage_cb
    _on_usage_cb = cb


def set_budget_warning_callback(cb: Optional[Callable] = None) -> None:
    """Устанавливает колбэк предупреждения при приближении к годовому лимиту."""
    global _on_budget_warning_cb
    _on_budget_warning_cb = cb


def _fire(coro_or_none: Any) -> None:
    if coro_or_none is None:
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro_or_none)
    except Exception as e:
        logger.debug("_fire error: %s", e)


def init_gigachat_pool(secrets: dict) -> None:
    """Инициализирует пул из словаря секретов Infisical (префикс GIGACHAT_AUTH_KEY_)."""
    _pool.init(secrets, "GIGACHAT_AUTH_KEY_")


def is_gigachat_available() -> bool:
    return not _pool.is_empty()


async def _usage_ytd(model: str) -> int:
    """
    Суммарные токены (in+out) по модели за всё время учёта.

    ПРИБЛИЖЕНИЕ: считаем от начала таблицы, а не от точной даты активации
    годового freemium-лимита GigaChat (эта дата нам неизвестна) — если разница
    станет критичной, скорректировать вручную по факту первого запроса.
    """
    try:
        async with async_session() as session:
            result = await session.execute(
                select(
                    func.coalesce(func.sum(GigaChatUsage.input_tokens), 0)
                    + func.coalesce(func.sum(GigaChatUsage.output_tokens), 0)
                ).where(GigaChatUsage.model == model)
            )
            return int(result.scalar() or 0)
    except Exception as e:
        logger.error("GigaChat: не удалось прочитать usage: %s", e)
        return 0


async def gigachat_budget_ok(model: str = DEFAULT_MODEL) -> bool:
    """
    Проверяет остаток годового лимита ПЕРЕД вызовом GigaChat API.

    При >=90% использования — шлёт предупреждение через колбэк (как у KeyPool
    при исчерпании ключей), чтобы узнать заранее, а не по факту отказа API.
    При 100% — возвращает False, вызывающий код не должен дёргать API вообще.
    """
    limit = GIGACHAT_YEARLY_LIMITS.get(model)
    if not limit:
        return True  # неизвестная модель — не блокируем, но лимит проверить вручную

    used = await _usage_ytd(model)
    ratio = used / limit if limit else 0.0

    if ratio >= 1.0:
        logger.critical("GigaChat: годовой лимит %s исчерпан (%d/%d)", model, used, limit)
        return False

    if ratio >= GIGACHAT_WARN_THRESHOLD:
        logger.warning(
            "GigaChat: приближение к годовому лимиту %s — %.1f%% (%d/%d)",
            model, ratio * 100, used, limit,
        )
        if _on_budget_warning_cb:
            _fire(_on_budget_warning_cb(f"gigachat_budget_{model}"))

    return True


async def generate_gigachat(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    current_vk_id: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Текстовый запрос к GigaChat.

    Возвращает (text, error). При успехе error = None.
    При ошибке text = None, error содержит причину.

    Не принимает media — вызывающий код (llm.py) сам решает, что делать
    с фото/аудио, когда Gemini лежит (GigaChat их не обрабатывает).
    """
    if _pool.is_empty():
        logger.debug("GigaChat: пул пуст — fallback недоступен")
        return None, "GigaChat не настроен"

    overloaded, wait_secs = _pool.is_overloaded()
    if overloaded:
        logger.debug("GigaChat: собственный blackout активен (%.0f с)", wait_secs)
        return None, "GigaChat временно недоступен"

    idx, api_key = _pool.get_next_key()
    if idx is None or api_key is None:
        logger.debug("GigaChat: все ключи на cooldown")
        return None, "GigaChat временно недоступен"

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    propagate_kwargs: dict[str, Any] = {"tags": ["gigachat", "fallback", model]}
    if current_vk_id:
        propagate_kwargs["user_id"] = str(current_vk_id)

    async with _GIGACHAT_SEMAPHORE:
      with langfuse.start_as_current_observation(
          as_type="generation",
          name="gigachat-fallback-generate",
          model=model,
          input=messages,
          metadata={"fallback_reason": "gemini_blackout"},
      ) as generation:
        with propagate_attributes(**propagate_kwargs):
            try:
                async with GigaChat(
                    credentials=api_key,
                    scope="GIGACHAT_API_PERS",
                    model=model,
                    verify_ssl_certs=False,  # см. примечание в SKILL/README — заменить на
                                             # verify_ssl_certs=True + сертификаты НУЦ Минцифры
                                             # при переходе с теста на постоянную эксплуатацию
                ) as client:
                    response = await client.achat({"messages": messages})

                choice = response.choices[0] if response.choices else None
                text = choice.message.content.strip() if choice else ""

                usage = getattr(response, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", 0) or 0
                out_tok = getattr(usage, "completion_tokens", 0) or 0
                logger.info("GigaChat tokens: in=%s out=%s model=%s", in_tok, out_tok, model)

                generation.update(
                    output=text,
                    usage_details={"input": in_tok, "output": out_tok},
                )

                if _on_usage_cb:
                    _fire(_on_usage_cb(model, in_tok, out_tok))

                if not text:
                    generation.update(level="WARNING", status_message="empty response")
                    return None, "GigaChat вернул пустой ответ"
                return text, None

            except GigaChatException as exc:
                err_str = str(exc)
                logger.warning("GigaChat: ошибка — %s", err_str)
                generation.update(level="ERROR", status_message=err_str[:500])
                # Не знаем точных кодов rate-limit у GigaChat заранее — ставим
                # консервативный cooldown на конкретный ключ через mark_error,
                # это же логирует событие в БД через уже подключённый db_log колбэк.
                _pool.mark_error(
                    idx, "error_gigachat", err_str,
                    cooldown=_pool.blackout_min * 60.0,
                    current_vk_id=current_vk_id,
                )
                return None, "GigaChat временно недоступен"

            except Exception as exc:
                logger.error("GigaChat: неожиданная ошибка — %s", exc)
                generation.update(level="ERROR", status_message=str(exc)[:500])
                _pool.mark_error(
                    idx, "error_unknown", str(exc),
                    cooldown=60.0,
                    current_vk_id=current_vk_id,
                )
                return None, "GigaChat временно недоступен"
