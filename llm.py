"""
Gemini API — минимальная надёжная реализация.
Два рабочих ключа, Round-Robin, защита от 429.
Задача 4: failover-уведомления, логирование ApiKeyEvent в БД.

SDK: google-genai
Модель: gemini-flash-latest
"""
import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Optional, Any, Callable, Awaitable

from google import genai
from google.genai import types

from prompt import (
    build_system_prompt, build_extraction_prompt,
    build_triage_prompt, build_intent_prompt,
    EXTRACTION_SCHEMA, TRIAGE_SCHEMA, INTENT_SCHEMA,
)

logger = logging.getLogger(__name__)

# ─── Конфиг ───────────────────────────────────────────────────────────────────

MODEL = "gemini-flash-latest"

# Строго 1 одновременный LLM-запрос — ключи никогда не бьются параллельно
_LLM_SEMAPHORE = asyncio.Semaphore(1)

_BLACKOUT_MULT    = 1   # retry_delay × 1 = cooldown
_BLACKOUT_MIN     = 30  # минимум 30 с, если retry_delay не найден

# ─── Состояние пула (module-level, asyncio-safe) ──────────────────────────────

_keys:    list[str]                    = []   # значения ключей
_key_names: list[str]                  = []   # имена секретов (GEMINI_API_KEY_*)
_clients: list[Optional[genai.Client]] = []   # lazy-init клиенты
_blocked: list[float]                  = []   # monotonic time — до которого ключ на cooldown
_blackout_until: float                 = 0.0  # глобальный blackout
_rr_index: int                         = 0    # счётчик Round-Robin

# ─── Текущий vk_id (устанавливается в main.py перед каждым LLM-вызовом) ───────
# Семафор _LLM_SEMAPHORE(1) гарантирует что только один вызов активен → нет гонки.
_current_vk_id: Optional[int] = None


def set_current_vk_id(vk_id: Optional[int]) -> None:
    global _current_vk_id
    _current_vk_id = vk_id


# ─── Failover/exhausted колбэки (устанавливаются из main.py) ─────────────────
# async (vk_id: int, service: str) -> None
_on_failover_cb: Optional[Callable[..., Awaitable[None]]] = None
# async (service: str) -> None
_on_exhausted_cb: Optional[Callable[..., Awaitable[None]]] = None
# async (service: str, key_name: str, status: str, error_text: str) -> None
_db_log_cb: Optional[Callable[..., Awaitable[None]]] = None


def set_llm_callbacks(
    on_failover: Optional[Callable] = None,
    on_exhausted: Optional[Callable] = None,
    db_log: Optional[Callable] = None,
) -> None:
    """Устанавливает колбэки для failover-уведомлений и DB-логирования."""
    global _on_failover_cb, _on_exhausted_cb, _db_log_cb
    if on_failover is not None:
        _on_failover_cb = on_failover
    if on_exhausted is not None:
        _on_exhausted_cb = on_exhausted
    if db_log is not None:
        _db_log_cb = db_log


def _fire_callback(coro: Any) -> None:
    """Запускает корутину в фоне, не блокируя текущий поток."""
    if coro is None:
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro)
    except Exception as e:
        logger.debug("_fire_callback error: %s", e)


# ─── Парсинг retry_delay из ошибки Gemini ─────────────────────────────────────

_RETRY_RE = re.compile(r'retry[_\s\-]?delay[^0-9]*(\d+(?:\.\d+)?)', re.IGNORECASE)


def _parse_retry(err_text: str) -> float:
    m = _RETRY_RE.search(err_text)
    return float(m.group(1)) if m else 0.0


# ─── Строка-заглушка ───────────────────────────────────────────────────────────

def _stub(secs: float) -> str:
    if secs >= 60:
        s = f"{math.ceil(secs / 60)} мин."
    else:
        s = f"{math.ceil(secs)} сек."
    return (
        f"⌛ Ева временно перегружена запросами, попробуйте через {s}. "
        f"Обычные команды бота работают!"
    )


# ─── Инициализация пула ────────────────────────────────────────────────────────

def init_key_pool(source: Any) -> None:
    """
    Инициализирует пул ключей.

    source = dict  — берёт все значения где ключ начинается с GEMINI_API_KEY_
                     (Infisical-словарь или env-dict); дедупликация по значению.
    source = list  — список строк-ключей напрямую.
    source = str   — одиночный ключ.
    """
    global _keys, _key_names, _clients, _blocked, _blackout_until, _rr_index

    if isinstance(source, dict):
        seen: set[str] = set()
        extracted: list[str] = []
        names: list[str] = []
        for k, v in source.items():
            if k.startswith("GEMINI_API_KEY_") and isinstance(v, str):
                val = v.strip()
                if val and val not in seen:
                    seen.add(val)
                    extracted.append(val)
                    names.append(k)
                    logger.info("LLM init: %s — добавлен ✓", k)
        _keys = extracted
        _key_names = names

    elif isinstance(source, list):
        seen = set()
        flat: list[str] = []
        flat_names: list[str] = []
        for i, v in enumerate(source):
            if isinstance(v, str) and v.strip() and v.strip() not in seen:
                seen.add(v.strip())
                flat.append(v.strip())
                flat_names.append(f"GEMINI_API_KEY_{i}")
        _keys = flat
        _key_names = flat_names
        logger.info("LLM init: list-режим, %d ключ(ей)", len(_keys))

    elif isinstance(source, str) and source.strip():
        _keys = [source.strip()]
        _key_names = ["GEMINI_API_KEY_0"]
        logger.info("LLM init: одиночный ключ")

    else:
        logger.error("LLM init: неожиданный тип source: %s", type(source))
        _keys = []
        _key_names = []

    # Сбрасываем состояние
    _clients         = [None] * len(_keys)
    _blocked         = [0.0]  * len(_keys)
    _blackout_until  = 0.0
    _rr_index        = 0
    logger.info("LLM init: пул готов, %d ключ(ей)", len(_keys))


def _lazy_client(idx: int) -> genai.Client:
    """Создаёт genai.Client для ключа #idx при первом использовании."""
    if _clients[idx] is None:
        _clients[idx] = genai.Client(api_key=_keys[idx])
    return _clients[idx]  # type: ignore[return-value]


def _ensure_keys() -> bool:
    """Возвращает True если пул готов. Если нет — пробует взять GEMINI_API_KEY из env."""
    global _keys, _key_names, _clients, _blocked, _rr_index
    if _keys:
        return True
    # fallback: одиночный ключ из env
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        _keys    = [key]
        _key_names = ["GEMINI_API_KEY"]
        _clients = [None]
        _blocked = [0.0]
        _rr_index = 0
        logger.info("LLM: fallback на GEMINI_API_KEY из env")
        return True
    return False


# ─── Ядро: вызов с Round-Robin и защитой от 429 ───────────────────────────────

async def _call(
    contents: list[Any],
    config: types.GenerateContentConfig,
) -> str:
    """
    Отправляет запрос в Gemini.

    Алгоритм:
    1. Если глобальный blackout активен → сразу возвращаем заглушку, 0 запросов.
    2. Семафор (_LLM_SEMAPHORE) гарантирует не более 1 одновременного запроса.
    3. Round-Robin по доступным ключам (пропускаем ключи на cooldown).
    4. При 429 на ключе → cooldown = max(30, retry_delay).
       Если оба ключа на cooldown → устанавливаем глобальный blackout, возвращаем заглушку.
    5. При 5xx/timeout → одна повторная попытка через 5 с на том же ключе.
    6. Любая другая ошибка → пробрасывается выше.
    """
    global _rr_index, _blackout_until

    # 1. Глобальный blackout — ни одного сетевого запроса
    now = time.monotonic()
    if now < _blackout_until:
        remaining = _blackout_until - now
        logger.debug("LLM: blackout активен, осталось %.0f с", remaining)
        return _stub(remaining)

    # 2. Семафор — не более 1 одновременного запроса к API
    async with _LLM_SEMAPHORE:
        return await _call_inner(contents, config)


async def _call_inner(
    contents: list[Any],
    config: types.GenerateContentConfig,
) -> str:
    """Внутренняя реализация _call, вызывается уже под семафором."""
    global _rr_index, _blackout_until

    # Повторно проверяем blackout — пока ждали семафор, мог выставиться
    now = time.monotonic()
    if now < _blackout_until:
        remaining = _blackout_until - now
        logger.debug("LLM: blackout активен (post-semaphore), осталось %.0f с", remaining)
        return _stub(remaining)

    if not _ensure_keys():
        return "⚠️ API-ключи не настроены. Обратитесь к администратору."

    n = len(_keys)
    tried: set[int] = set()

    for _ in range(n):
        idx = _rr_index % n
        _rr_index += 1

        now2 = time.monotonic()
        if now2 < _blocked[idx]:
            continue  # этот ключ на per-key cooldown — пропускаем

        if idx in tried:
            continue  # уже пробовали в этом запросе
        tried.add(idx)

        client = _lazy_client(idx)

        for attempt in range(2):
            try:
                resp = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=config,
                )
                # Логируем токены и finish_reason для диагностики
                try:
                    reason = resp.candidates[0].finish_reason if resp.candidates else None
                    usage = getattr(resp, "usage_metadata", None)
                    in_tok  = getattr(usage, "prompt_token_count", "?")
                    out_tok = getattr(usage, "candidates_token_count", "?")
                    logger.info("LLM tokens: in=%s out=%s finish=%s", in_tok, out_tok, reason)
                    if reason and str(reason) not in ("FinishReason.STOP", "STOP", "1"):
                        logger.warning("LLM: finish_reason=%s — ответ обрезан!", reason)
                except Exception:
                    pass
                return resp.text or "Произошла ошибка, попробуй ещё раз."

            except Exception as exc:
                err_str = str(exc)
                err_up  = err_str.upper()
                key_name = _key_names[idx] if idx < len(_key_names) else f"gemini_key_{idx}"

                # ── 429 / лимит запросов ──────────────────────────────────────
                if "429" in err_up or "RESOURCE_EXHAUSTED" in err_up:
                    base     = _parse_retry(err_str)
                    cooldown = max(_BLACKOUT_MIN, base * _BLACKOUT_MULT)
                    _blocked[idx] = time.monotonic() + cooldown
                    logger.warning(
                        "LLM: ключ %s → 429, cooldown %.0f с", key_name, cooldown
                    )

                    # Логируем в БД
                    if _db_log_cb:
                        _fire_callback(_db_log_cb("gemini", key_name, "error_429", err_str[:500]))

                    # Проверяем: заблокированы ли ВСЕ ключи?
                    now3 = time.monotonic()
                    if all(now3 < t for t in _blocked):
                        # Blackout = до момента когда самый поздний ключ освободится
                        new_blackout = max(_blocked)
                        if new_blackout > _blackout_until:
                            _blackout_until = new_blackout
                        wait = _blackout_until - now3
                        logger.warning(
                            "LLM: все ключи на 429 → глобальный blackout %.0f с", wait
                        )
                        # Уведомляем об исчерпании
                        if _on_exhausted_cb:
                            _fire_callback(_on_exhausted_cb("gemini"))
                        if _db_log_cb:
                            _fire_callback(_db_log_cb(
                                "gemini", "ALL", "all_keys_exhausted",
                                f"All {len(_keys)} Gemini key(s) are rate-limited",
                            ))
                        return _stub(wait)

                    # Есть резервный ключ — уведомляем пользователя
                    if _on_failover_cb and _current_vk_id:
                        _fire_callback(_on_failover_cb(_current_vk_id, "gemini"))

                    break  # перейти к следующему ключу

                # ── Ошибка авторизации (невалидный ключ) ─────────────────────
                if "401" in err_up or "UNAUTHENTICATED" in err_up or "API_KEY_INVALID" in err_up:
                    _blocked[idx] = time.monotonic() + 3600.0  # на 1 час
                    logger.error("LLM: ключ %s → ошибка авторизации", key_name)
                    if _db_log_cb:
                        _fire_callback(_db_log_cb("gemini", key_name, "error_auth", err_str[:500]))
                    # Уведомляем если есть резерв
                    now3 = time.monotonic()
                    has_reserve = any(now3 >= t for i, t in enumerate(_blocked) if i != idx)
                    if has_reserve and _on_failover_cb and _current_vk_id:
                        _fire_callback(_on_failover_cb(_current_vk_id, "gemini"))
                    break

                # ── Временные ошибки сервера ──────────────────────────────────
                if (
                    "500" in err_up or "503" in err_up
                    or "UNAVAILABLE" in err_up or "TIMEOUT" in err_up
                ) and attempt == 0:
                    logger.warning(
                        "LLM: transient error ключ %s, retry через 5 с: %s",
                        key_name, exc,
                    )
                    await asyncio.sleep(5)
                    continue  # повторить на том же ключе

                # ── Неизвестная ошибка — пробрасываем ────────────────────────
                logger.error("LLM: ошибка ключ %s: %s", key_name, exc)
                raise

    # Все доступные ключи исчерпаны
    now4 = time.monotonic()
    if now4 < _blackout_until:
        return _stub(_blackout_until - now4)
    if _blocked:
        remaining = max(0.0, max(_blocked) - now4)
        if remaining > 0:
            return _stub(remaining)
    return "⚠️ Сервис временно недоступен — попробуй ещё раз."


# ─── Статус пула (для внешних проверок) ───────────────────────────────────────

def is_overloaded() -> tuple[bool, float]:
    """
    Возвращает (True, remaining_seconds) если глобальный blackout активен,
    иначе (False, 0.0).
    """
    now = time.monotonic()
    if now < _blackout_until:
        return True, _blackout_until - now
    return False, 0.0


# ─── Public API ────────────────────────────────────────────────────────────────

async def _analyze_one_photo(
    photo_data: bytes,
    mime: str,
    photo_num: int,
    total: int,
    user_request: str,
    pet_name: str,
    species: str,
    pet_facts: str,
    other_pets_hint: str,
    history: str,
    scope_instruction: Optional[str],
    emergency_active: bool,
) -> str:
    """Анализирует ОДНО фото — только разбор, без вступления и финального вывода."""
    system = build_system_prompt(
        pet_name=pet_name,
        species=species,
        pet_facts=pet_facts,
        other_pets_hint=other_pets_hint,
        history=history if photo_num == 1 else "",
        scope_instruction=scope_instruction,
        emergency_active=emergency_active,
        num_photos=1,
    )

    instruction_parts: list[str] = [
        f"[Фото {photo_num} из {total}]",
        "Без вступления и без финального совета — только разбор этого фото.",
    ]
    if user_request:
        instruction_parts.append(user_request)

    instruction = "\n".join(instruction_parts)

    return await _call(
        contents=[instruction, types.Part.from_bytes(data=photo_data, mime_type=mime)],
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )


async def _synthesize_photo_analyses(
    analyses: list[str],
    user_request: str,
    pet_name: str,
    species: str,
    pet_facts: str,
) -> str:
    """Финальный текстовый вызов (без картинок) — синтез вступления и совета."""
    from prompt import build_synthesis_prompt

    prompt_text = build_synthesis_prompt(
        analyses=analyses,
        user_request=user_request,
        pet_name=pet_name,
        species=species,
        pet_facts=pet_facts,
    )

    return await _call(
        contents=[prompt_text],
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )


async def generate_response(
    user_text: str,
    pet_name: str,
    species: str,
    pet_facts: str,
    other_pets_hint: str,
    history: str,
    media_items: Optional[list[tuple[bytes, str]]] = None,
    scope_instruction: Optional[str] = None,
    emergency_active: bool = False,
    web_search_context: Optional[str] = None,
    toxicology_active: bool = False,
) -> str:
    """Генерирует основной ответ бота.

    При нескольких фото используется двухэтапный пайплайн:
      Шаг 1 — каждое фото анализируется отдельным вызовом (только разбор).
      Шаг 2 — один финальный текстовый вызов синтезирует вступление,
               практический совет и вопрос по всему набору.

    web_search_context — результаты Tavily Search, инжектируются в системный промпт.
    """
    num_photos = sum(1 for _, mime in (media_items or []) if mime.startswith("image/"))
    logger.info(
        "generate_response: num_photos=%d, has_text=%s, media_items=%d",
        num_photos, bool(user_text), len(media_items or []),
    )

    if num_photos > 1:
        photo_items = [(d, m) for d, m in (media_items or []) if m.startswith("image/")]

        # ── Шаг 1: анализируем каждое фото по отдельности ──────────────────
        tasks = [
            _analyze_one_photo(
                photo_data=data,
                mime=mime,
                photo_num=i + 1,
                total=num_photos,
                user_request=user_text,
                pet_name=pet_name,
                species=species,
                pet_facts=pet_facts,
                other_pets_hint=other_pets_hint,
                history=history,
                scope_instruction=scope_instruction,
                emergency_active=emergency_active,
            )
            for i, (data, mime) in enumerate(photo_items)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        analyses: list[str] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.error("Photo %d analysis failed: %s", i + 1, r)
                analyses.append(f"[Фото {i + 1}: не удалось проанализировать]")
            else:
                analyses.append(str(r))

        # ── Шаг 2: финальный синтез (текст, без картинок) ──────────────────
        try:
            synthesis = await _synthesize_photo_analyses(
                analyses=analyses,
                user_request=user_text,
                pet_name=pet_name,
                species=species,
                pet_facts=pet_facts,
            )
        except Exception as exc:
            logger.error("Photo synthesis failed: %s", exc)
            synthesis = ""

        # ── Сборка финального ответа ─────────────────────────────────────────
        body = "\n\n".join(analyses)
        if synthesis:
            sep_idx = synthesis.find("\n\n")
            if sep_idx != -1:
                intro_part = synthesis[:sep_idx].strip()
                outro_part = synthesis[sep_idx:].strip()
                return f"{intro_part}\n\n{body}\n\n{outro_part}"
            else:
                return f"{body}\n\n{synthesis}"
        else:
            return body

    # ── Одно фото, PDF, аудио или только текст — стандартный единый вызов ────
    system = build_system_prompt(
        pet_name=pet_name,
        species=species,
        pet_facts=pet_facts,
        other_pets_hint=other_pets_hint,
        history=history,
        scope_instruction=scope_instruction,
        emergency_active=emergency_active,
        num_photos=num_photos,
        web_search_context=web_search_context,
        toxicology_active=toxicology_active,
    )

    contents: list[Any] = []
    if user_text:
        contents.append(user_text)
    for data, mime in (media_items or []):
        contents.append(types.Part.from_bytes(data=data, mime_type=mime))

    return await _call(
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=4096,
        ),
    )


async def extract_facts(
    message: str,
    pet_name: str,
    species: str,
    media_items: Optional[list[tuple[bytes, str]]] = None,
) -> list[dict]:
    """Извлекает структурированные факты из сообщения."""
    contents: list[Any] = [build_extraction_prompt(message, pet_name, species)]
    for data, mime in (media_items or []):
        contents.append(types.Part.from_bytes(data=data, mime_type=mime))

    try:
        raw = await _call(
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,   # было 1024 — обрывало JSON при включённом thinking
                response_mime_type="application/json",
                # Отключаем thinking: без этого gemini-flash-latest тратит токены на
                # внутренние размышления, оставляя ~28 токенов на JSON → Unterminated string
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.warning("extract_facts error: %s", e)
        return []


async def classify_triage(
    message: str,
    pet_name: str,
    species: str,
    emergency_active: bool,
) -> dict:
    """Быстрая триаж-классификация."""
    try:
        raw = await _call(
            contents=[build_triage_prompt(message, pet_name, species, emergency_active)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.warning("classify_triage error: %s", e)
        return {"level": "observation", "topic_changed": False, "brief_action": ""}


async def classify_intent(message: str) -> dict:
    """Быстрая классификация намерения."""
    try:
        raw = await _call(
            contents=[build_intent_prompt(message)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.warning("classify_intent error: %s", e)
        return {"intent": "general"}
