"""
ZOO Ментор — VK Bot
FastAPI app + VK Long Poll worker

Задача 1: Groq Whisper — расшифровка голосовых сообщений
Задача 2: Tavily — живой веб-поиск при явном намерении web_search_needed
Задача 3: PDF через Gemini нативно (уже работало через extract_media — явная проверка)
Задача 4: Универсальный failover — ApiKeyEvent в БД, уведомление пользователя/админа
Задача 5: TOXICOLOGY_RULES engine + Tavily с фильтром качества источников для отравлений
"""
import asyncio
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, Response
import uvicorn
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db import init_db, async_session
from models import User, Pet, ApiKeyEvent, GigaChatUsage
from billing import ensure_user, check_and_increment
from memory import (
    get_user_state_data, set_user_state,
    get_pet_facts_text, get_chat_history_text,
    save_message, process_extracted_facts,
)
from llm import (
    generate_response, extract_facts, init_key_pool,
    is_overloaded, set_current_vk_id, set_llm_callbacks,
)
from media import extract_media, extract_all_photos, has_photos, has_non_photo_media, get_audio_message
from intent import classify, is_toxicology_message, extract_toxin_from_message
from triage import analyze_triage, is_emergency_active, should_send_followup
from vk_client import VKClient, yes_no_keyboard, pets_keyboard, main_menu_keyboard
from scheduler import setup_scheduler
from infisical_loader import load_infisical_secrets, extract_gemini_keys
from groq_client import init_groq_pool, transcribe_audio, is_groq_available
from tavily_client import init_tavily_pool, search as tavily_search, is_tavily_available
from gigachat_client import (
    init_gigachat_pool, set_usage_callback as set_gigachat_usage_cb,
    set_budget_warning_callback as set_gigachat_budget_cb, is_gigachat_available,
)
import key_pool as kp
import handlers
import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

vk: VKClient | None = None
_admin_vk_id: int | None = None   # VK ID администратора для критических уведомлений

# ─── Per-user rate limiter (in-memory) ───────────────────────────────────────
_user_last_llm: dict[int, float] = {}
_USER_LLM_COOLDOWN = 4.0

# ─── Photo accumulation buffer ───────────────────────────────────────────────
import dataclasses

@dataclasses.dataclass
class _PhotoBatch:
    photos: list          # list of (bytes, mime_type)
    text: str             # текст первого сообщения в серии
    task: asyncio.Task    # задача-таймер

_photo_buffers: dict[int, _PhotoBatch] = {}
PHOTO_BATCH_WINDOW = 30.0


# ─── Задача 4: DB-логирование и VK-уведомления для KeyPool ───────────────────

async def _db_log_key_event(service: str, key_name: str, status: str, error_text: str) -> None:
    """Записывает событие API-ключа в БД."""
    try:
        async with async_session() as session:
            event = ApiKeyEvent(
                service=service,
                key_name=key_name,
                status=status,
                error_text=(error_text or "")[:500],
                created=datetime.utcnow(),
            )
            session.add(event)
            await session.commit()
        logger.debug("ApiKeyEvent: %s / %s / %s", service, key_name, status)
    except Exception as e:
        logger.error("Не удалось записать ApiKeyEvent: %s", e)


# ─── GigaChat: учёт токенов (годовой freemium-лимит, не помесячный) ──────────
# Сама проверка бюджета (gigachat_budget_ok) живёт в gigachat_client.py —
# здесь только запись в БД, т.к. только main.py открывает сессии для
# фонового логирования (тот же паттерн, что и у _db_log_key_event выше).

async def _record_gigachat_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """UPSERT дневной статистики использования GigaChat по (date, model)."""
    try:
        today = datetime.utcnow().date()
        async with async_session() as session:
            stmt = pg_insert(GigaChatUsage).values(
                date=today,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                requests=1,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["date", "model"],
                set_={
                    "input_tokens": GigaChatUsage.input_tokens + input_tokens,
                    "output_tokens": GigaChatUsage.output_tokens + output_tokens,
                    "requests": GigaChatUsage.requests + 1,
                },
            )
            await session.execute(stmt)
            await session.commit()
        logger.debug("GigaChatUsage: %s in=%d out=%d", model, input_tokens, output_tokens)
    except Exception as e:
        logger.error("Не удалось записать GigaChatUsage: %s", e)


async def _gigachat_usage_ytd(model: str) -> int:
    """
    Суммарные токены (in+out) по модели за всё время учёта.

    ПРИБЛИЖЕНИЕ: считаем от начала таблицы, а не от точной даты активации
    годового freemium-лимита GigaChat (эта дата нам неизвестна) — если разница
    станет критичной, скорректировать вручную по факту первого запроса к GigaChat.
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
        logger.error("Не удалось прочитать GigaChatUsage: %s", e)
        return 0


async def _on_key_failover(vk_id: int, service: str) -> None:
    """Тихое переключение на резервный ключ — пользователь НЕ уведомляется.

    Раньше здесь отправлялось «Секундочку, у меня что-то зависло» — но это
    приводило к спаму: при 12 ключах и массовом 429 пользователь получал
    12 сообщений «зависло», а потом нормальный ответ.
    Failover между ключами — это внутреннее дело бота, пользователю достаточно
    просто получить ответ с небольшой задержкой.
    Уведомление появится только в _on_keys_exhausted (все ключи исчерпаны).
    """
    logger.info("_on_key_failover: %s — переключение на следующий ключ (без уведомления пользователя)", service)


async def _on_keys_exhausted(service: str) -> None:
    """
    Критическое событие: все ключи сервиса исчерпаны.
    Уведомляет администратора через VK, если ADMIN_VK_ID задан.
    """
    logger.critical("ALL KEYS EXHAUSTED for service: %s", service)
    if vk and _admin_vk_id:
        try:
            await vk.send_message(
                _admin_vk_id,
                f"⚠️ КРИТИЧНО: все ключи сервиса {service.upper()} исчерпаны! "
                f"Бот не может выполнять запросы через этот сервис. "
                f"Проверь ключи в Infisical.",
            )
        except Exception as e:
            logger.error("_on_keys_exhausted: не удалось уведомить администратора: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vk, _admin_vk_id

    # ── Load secrets from Infisical FIRST (нужен DATABASE_URL до init_db) ──
    infisical_secrets = await load_infisical_secrets()

    if infisical_secrets:
        # Supabase connection string → DATABASE_URL (для db.py)
        db_url = infisical_secrets.get("SUPABASE_CP_babfrost2pdc9_zoo-mentor", "").strip()
        if db_url:
            os.environ["DATABASE_URL"] = db_url
            logger.info("DATABASE_URL: загружен из Infisical (Supabase)")
        else:
            logger.error("Infisical: SUPABASE_CP_babfrost2pdc9_zoo-mentor не найден!")

    await init_db()
    logger.info("Database initialised")

    if infisical_secrets:
        # Gemini key pool
        init_key_pool(infisical_secrets)

        # Groq Whisper key pool (Задача 1)
        init_groq_pool(infisical_secrets)
        if is_groq_available():
            logger.info("Groq: пул ключей инициализирован")
        else:
            logger.warning("Groq: GROQ_API_KEY_* не найдены в Infisical — голосовые будут через Gemini")

        # Tavily Search key pool (Задача 2)
        init_tavily_pool(infisical_secrets)
        if is_tavily_available():
            logger.info("Tavily: пул ключей инициализирован")
        else:
            logger.warning("Tavily: TAVILY_API_KEY_* не найдены в Infisical — поиск отключён")

        # GigaChat key pool — резервный LLM, срабатывает только когда Gemini
        # полностью в blackout (см. llm.is_overloaded() + правки в llm.py)
        init_gigachat_pool(infisical_secrets)
        set_gigachat_usage_cb(_record_gigachat_usage)
        set_gigachat_budget_cb(_on_keys_exhausted)  # переиспользуем существующее уведомление админу
        if is_gigachat_available():
            logger.info("GigaChat: пул ключей инициализирован (fallback LLM)")
        else:
            logger.warning("GigaChat: GIGACHAT_AUTH_KEY_* не найдены в Infisical — резервный LLM отключён")

        # VK credentials from Infisical
        vk_token_inf = infisical_secrets.get("VK_TOKEN", "")
        vk_gid_inf = infisical_secrets.get("VK_GROUP_ID", "")
        if vk_token_inf:
            os.environ["VK_TOKEN"] = vk_token_inf
        if vk_gid_inf:
            os.environ["VK_GROUP_ID"] = vk_gid_inf

        # Admin VK ID for critical notifications
        admin_id_str = infisical_secrets.get("ADMIN_VK_ID", "") or os.environ.get("ADMIN_VK_ID", "")
        if admin_id_str:
            try:
                _admin_vk_id = int(admin_id_str)
                logger.info("Admin VK ID: %d", _admin_vk_id)
            except ValueError:
                logger.warning("ADMIN_VK_ID не является числом: %s", admin_id_str)
    else:
        # Fallback: читаем ключи из env
        env_dict: dict[str, str] = {}
        for kid in ("GEMINI_API_KEY_550953", "GEMINI_API_KEY_s89781248490"):
            v = os.environ.get(kid, "").strip()
            if v:
                env_dict[kid] = v
        if env_dict:
            init_key_pool(env_dict)
            logger.info("Gemini key pool: %d ключ(ей) из env", len(env_dict))
        else:
            single_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if single_key:
                init_key_pool([single_key])
            else:
                logger.error("Gemini-ключи не найдены — LLM не будет работать")

        # Groq из env (fallback)
        groq_env: dict[str, str] = {}
        for kid in ("GROQ_API_KEY_550953", "GROQ_API_KEY_skukolka01"):
            v = os.environ.get(kid, "").strip()
            if v:
                groq_env[kid] = v
        if groq_env:
            init_groq_pool(groq_env)

        # Tavily из env (fallback)
        tavily_env: dict[str, str] = {}
        for kid in ("TAVILY_API_KEY_550953", "TAVILY_API_KEY_skukolka01"):
            v = os.environ.get(kid, "").strip()
            if v:
                tavily_env[kid] = v
        if tavily_env:
            init_tavily_pool(tavily_env)

    # ── Задача 4: устанавливаем failover-колбэки для ВСЕХ пулов ─────────────
    kp.set_callbacks(
        on_failover=_on_key_failover,
        on_exhausted=_on_keys_exhausted,
        db_log=_db_log_key_event,
    )
    set_llm_callbacks(
        on_failover=_on_key_failover,
        on_exhausted=_on_keys_exhausted,
        db_log=_db_log_key_event,
    )

    # ── VK Client ─────────────────────────────────────────────────────────
    token = os.environ.get("VK_TOKEN", "")
    group_id = os.environ.get("VK_GROUP_ID", "")
    if not token or not group_id:
        logger.error("VK_TOKEN or VK_GROUP_ID not set — bot will not connect to VK")
    else:
        vk = VKClient(token=token, group_id=group_id)
        await vk.start()
        logger.info("VK client started")

        def make_send(client):
            async def _send(vk_id: int, text: str):
                await client.send_message_plain(vk_id, text)
            return _send

        scheduler = setup_scheduler(async_session, make_send(vk))
        scheduler.start()
        logger.info("Scheduler started")

        asyncio.create_task(long_poll_loop())

    yield

    if vk:
        await vk.close()


app = FastAPI(lifespan=lifespan, title="ZOO Ментор")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ─── Long Poll Loop ──────────────────────────────────────────────────────────

async def long_poll_loop():
    logger.info("Starting Long Poll loop")
    await vk.run_long_poll(handle_update)


# ─── Update Dispatcher ───────────────────────────────────────────────────────

async def handle_update(update: dict) -> None:
    update_type = update.get("type")

    if update_type == "message_new":
        msg = update.get("object", {}).get("message", {})
        await handle_message(msg)

    elif update_type == "message_event":
        obj = update.get("object", {})
        await handle_callback_event(obj)


# ─── Message Handler ─────────────────────────────────────────────────────────

async def handle_message(msg: dict) -> None:
    vk_id = msg.get("from_id")
    text = (msg.get("text") or "").strip()

    if not vk_id or vk_id < 0:
        return  # Skip group/bot messages

    async with async_session() as session:
        user = await ensure_user(session, vk_id)

        # ── FSM: check active state first ──────────────────────────────────
        state, state_data = await get_user_state_data(session, user.id)

        if state == "awaiting_pet_species":
            await handlers.fsm_awaiting_pet_species(vk, session, user, vk_id, text)
            return

        if state == "awaiting_pet_name":
            await handlers.fsm_awaiting_pet_name(vk, session, user, vk_id, text, state_data)
            return

        if state == "awaiting_pet_age":
            await handlers.fsm_awaiting_pet_age(vk, session, user, vk_id, text, state_data)
            return

        if state == "awaiting_pet_switch":
            await handlers.fsm_awaiting_pet_switch(vk, session, user, vk_id, text)
            return

        if state == "awaiting_confirm_reset":
            await handlers.fsm_awaiting_confirm_reset(vk, session, user, vk_id, text, state_data)
            return

        if state == "awaiting_remind_text":
            await handlers.fsm_awaiting_remind_text(vk, session, user, vk_id, text, state_data)
            return

        if state == "awaiting_remind_time":
            await handlers.fsm_awaiting_remind_time(vk, session, user, vk_id, text, state_data)
            return

        # ── Explicit commands ──────────────────────────────────────────────
        lower = text.lower()

        if lower.startswith("/start"):
            await handlers.handle_start(vk, session, user, vk_id)
            return

        _BUTTON_MAP = {
            "🐾 мой питомец": "/me",
            "➕ добавить питомца": "/addpet",
            "⏰ напоминания": "/reminders",
            "🔄 сменить питомца": "/pets",
            "💎 подписка": "/subscription",
            "❓ помощь": "/help",
        }
        mapped = _BUTTON_MAP.get(lower.strip())
        if mapped:
            lower = mapped

        if lower.startswith("/addpet"):
            await handlers.handle_addpet(vk, session, user, vk_id)
            return

        if lower.startswith("/pets") or lower.startswith("/switch"):
            await handlers.handle_pets(vk, session, user, vk_id)
            return

        if lower.startswith("/me"):
            await handlers.handle_me(vk, session, user, vk_id)
            return

        if lower.startswith("/reset"):
            await handlers.handle_reset(vk, session, user, vk_id)
            return

        if lower.startswith("/delremind"):
            args = text[len("/delremind"):].strip()
            await handlers.handle_delremind(vk, session, user, vk_id, args)
            return

        if lower.startswith("/reminders"):
            await handlers.handle_reminders(vk, session, user, vk_id)
            return

        if lower.startswith("/remind"):
            await handlers.handle_remind(vk, session, user, vk_id)
            return

        if lower.startswith("/subscription"):
            await handlers.handle_subscription(vk, session, user, vk_id)
            return

        if lower.startswith("/pay"):
            await handlers.handle_pay(vk, session, user, vk_id)
            return

        if lower.startswith("/cancel"):
            await handlers.handle_cancel(vk, session, user, vk_id)
            return

        if lower.startswith("/referral"):
            await handlers.handle_referral(vk, session, user, vk_id)
            return

        if lower.startswith("/help") or lower.startswith("/skills"):
            await handlers.handle_help(vk, session, user, vk_id)
            return

        if lower.startswith("/quiet"):
            await handlers.handle_quiet(vk, session, user, vk_id)
            return

        if lower.startswith("/loud"):
            await handlers.handle_loud(vk, session, user, vk_id)
            return

        # ── No active pet → prompt to add ─────────────────────────────────
        from sqlalchemy import select
        from models import Pet as PetModel

        result = await session.execute(select(PetModel).where(PetModel.user_id == user.id))
        all_pets = result.scalars().all()

        if not all_pets:
            await vk.send_message(
                vk_id,
                "Привет! Чтобы я мог помочь, сначала познакомь меня со своим питомцем.\nНапиши /addpet"
            )
            return

        if user.active_pet_id is None and all_pets:
            user.active_pet_id = all_pets[0].id
            await session.commit()

        # ── Если Long Poll прислал обрезанное сообщение — дозапрашиваем полное ──
        if msg.get("is_cropped"):
            peer_id = msg.get("peer_id") or vk_id
            cmid = msg.get("conversation_message_id")
            if cmid and vk:
                full_msg = await vk.get_full_message(peer_id, cmid)
                if full_msg:
                    logger.info(
                        "is_cropped: заменяем обрезанное сообщение полным "
                        "(было %d вложений → стало %d)",
                        len(msg.get("attachments", [])),
                        len(full_msg.get("attachments", [])),
                    )
                    msg = full_msg

        # ── Detect media early ─────────────────────────────────────────────
        has_media = bool(msg.get("attachments"))
        _att_types = [a.get("type") for a in msg.get("attachments", [])]
        if _att_types:
            logger.info("VK msg from %s: text=%r attachments=%s", vk_id, text[:60] if text else "", _att_types)

        # ── Ранняя проверка: LLM перегружен? ──────────────────────────────
        overloaded, remaining = is_overloaded()
        if overloaded:
            wait_str = f"{math.ceil(remaining / 60)} мин." if remaining >= 60 else f"{math.ceil(remaining)} сек."
            await vk.send_message(
                vk_id,
                f"⌛ Ева временно перегружена запросами, попробуйте через {wait_str}. "
                f"Обычные команды бота работают!"
            )
            return

        # ── Per-user rate limit ────────────────────────────────────────────
        is_photo_only = has_media and has_photos(msg) and not text
        now_mono = time.monotonic()
        last_req = _user_last_llm.get(vk_id, 0.0)
        user_wait = _USER_LLM_COOLDOWN - (now_mono - last_req)
        if user_wait > 0 and not is_photo_only:
            await vk.send_message(
                vk_id,
                f"⏳ Подожди ещё {math.ceil(user_wait)} сек. — Ева обрабатывает предыдущий запрос."
            )
            return
        _user_last_llm[vk_id] = now_mono

        # ── Устанавливаем текущий vk_id для failover-уведомлений ──────────
        set_current_vk_id(vk_id)

        # ── Intent classification ──────────────────────────────────────────
        web_search_context: str | None = None
        toxicology_active: bool = False

        if text:
            intent, mapped_command, scope_restriction = await classify(text)

            if intent == "command" and mapped_command:
                await _route_mapped_command(vk, session, user, vk_id, mapped_command, text)
                return

            if intent == "profile_query":
                await _handle_profile_query(vk, session, user, vk_id)
                return

            if intent == "scope_instruction" and scope_restriction:
                _, data = await get_user_state_data(session, user.id)
                data["scope"] = scope_restriction
                await set_user_state(session, user.id, None, data)

            # ── Задача 2: Tavily веб-поиск (по явному намерению пользователя) ──
            if intent == "web_search_needed" and is_tavily_available():
                logger.info("Tavily: запрос веб-поиска для vk_id=%d: %r", vk_id, text[:80])
                web_search_context = await tavily_search(
                    query=text,
                    max_results=5,
                    current_vk_id=vk_id,
                    toxicology_mode=False,
                )
                if web_search_context:
                    logger.info("Tavily: получен контекст (%d символов)", len(web_search_context))
                # Не прерываем — продолжаем генерировать ответ с контекстом

            # ── Задача 5: Токсикология — автоматический триггер ──────────────
            # Независим от intent == "web_search_needed".
            # Активируем TOXICOLOGY_RULES engine + запускаем Tavily с фильтром качества.
            if is_toxicology_message(text):
                toxicology_active = True
                logger.info(
                    "Токсикология: обнаружены маркеры отравления для vk_id=%d: %r",
                    vk_id, text[:80],
                )

                if is_tavily_available() and web_search_context is None:
                    # Получаем активного питомца для формирования запроса
                    from sqlalchemy import select as _sel
                    from models import Pet as _Pet
                    _pet_result = await session.execute(
                        _sel(_Pet).where(_Pet.id == user.active_pet_id)
                    )
                    _pet_for_query = _pet_result.scalar_one_or_none()
                    _species_for_query = _pet_for_query.species if _pet_for_query else "питомец"

                    toxicology_query = extract_toxin_from_message(text, _species_for_query)
                    logger.info(
                        "Tavily токсикология: запрос «%s» для vk_id=%d",
                        toxicology_query, vk_id,
                    )
                    web_search_context = await tavily_search(
                        query=toxicology_query,
                        max_results=5,
                        current_vk_id=vk_id,
                        toxicology_mode=True,  # строгий фильтр источников
                    )
                    if web_search_context:
                        logger.info(
                            "Tavily токсикология: получен контекст с проверенными источниками (%d символов)",
                            len(web_search_context),
                        )
                    else:
                        logger.info(
                            "Tavily токсикология: нет доверенных источников — ответ без ссылок"
                        )

        # ── Usage limit check ──────────────────────────────────────────────
        allowed, limit_msg = await check_and_increment(session, user, is_media=has_media)
        if not allowed:
            await vk.send_message(vk_id, limit_msg)
            return

        # ── Get active pet ─────────────────────────────────────────────────
        result2 = await session.execute(
            select(PetModel).where(PetModel.id == user.active_pet_id)
        )
        pet = result2.scalar_one_or_none()
        if not pet:
            await vk.send_message(vk_id, "Добавь питомца: /addpet")
            return

        # ── Download media if any ──────────────────────────────────────────
        media_items: list[tuple[bytes, str]] = []
        media_error = None
        voice_transcribed = False  # флаг: голосовое уже преобразовано в текст

        if has_media:
            async with aiohttp.ClientSession() as http_session:
                if has_photos(msg):
                    new_photos, media_error = await extract_all_photos(msg, http_session)

                    if media_error and not new_photos:
                        await vk.send_message(vk_id, media_error)
                        return

                    # ── Photo batch buffer ─────────────────────────────────
                    if is_photo_only and new_photos:
                        existing = _photo_buffers.get(vk_id)
                        if existing is not None:
                            existing.photos.extend(new_photos)
                            logger.info(
                                "photo_buffer[%s]: добавлено %d фото, итого %d",
                                vk_id, len(new_photos), len(existing.photos),
                            )
                            return
                        else:
                            batch = _PhotoBatch(
                                photos=list(new_photos),
                                text=text,
                                task=None,  # type: ignore[arg-type]
                            )
                            _photo_buffers[vk_id] = batch
                            logger.info(
                                "photo_buffer[%s]: открыт, первые %d фото, ждём %.1f с",
                                vk_id, len(new_photos), PHOTO_BATCH_WINDOW,
                            )

                            _flush_uid = vk_id
                            _flush_user_id = user.id
                            _flush_pet_id = user.active_pet_id

                            async def _flush(uid=_flush_uid, user_id=_flush_user_id,
                                             active_pet_id=_flush_pet_id):
                                await asyncio.sleep(PHOTO_BATCH_WINDOW)
                                buf = _photo_buffers.pop(uid, None)
                                if not buf:
                                    return
                                n_photos = len(buf.photos)
                                logger.info(
                                    "photo_buffer[%s]: таймер сработал, обрабатываем %d фото",
                                    uid, n_photos,
                                )
                                # Промежуточное сообщение ДО тяжёлой работы —
                                # пользователь сразу видит, что бот получил фото и работает.
                                await vk.send_message(
                                    uid,
                                    f"Смотрю все {n_photos} фото — это может занять минуту-другую, не пропадай!",
                                )
                                async with async_session() as fresh_sess:
                                    from sqlalchemy import select as _sel
                                    from models import Pet as _Pet, User as _User
                                    fresh_pet = (await fresh_sess.execute(
                                        _sel(_Pet).where(_Pet.id == active_pet_id)
                                    )).scalar_one_or_none()
                                    if not fresh_pet:
                                        return
                                    _, _st = await get_user_state_data(fresh_sess, user_id)
                                    _scope = _st.get("scope") if _st else None
                                    _all_pets = (await fresh_sess.execute(
                                        _sel(_Pet).where(_Pet.user_id == user_id)
                                    )).scalars().all()
                                    _other_names = [p.name for p in _all_pets if p.id != active_pet_id]
                                    _other_hint = (
                                        f"У пользователя есть ещё питомцы: {', '.join(_other_names)}. "
                                        "Если он упоминает другого питомца — предложи переключиться (/pets)."
                                        if _other_names else ""
                                    )
                                    pf = await get_pet_facts_text(fresh_sess, fresh_pet)
                                    hs = await get_chat_history_text(fresh_sess, user_id, fresh_pet.id)
                                    set_current_vk_id(uid)
                                    try:
                                        resp = await generate_response(
                                            user_text=buf.text,
                                            pet_name=fresh_pet.name,
                                            species=fresh_pet.species,
                                            pet_facts=pf,
                                            other_pets_hint=_other_hint,
                                            history=hs,
                                            media_items=buf.photos,
                                            scope_instruction=_scope,
                                            emergency_active=is_emergency_active(uid, fresh_pet.id),
                                        )
                                    except Exception as exc:
                                        logger.error("photo_buffer flush LLM error: %s", exc)
                                        await vk.send_message(uid, "Ошибка при обработке фото. Попробуй ещё раз.")
                                        return
                                    await vk.send_message(uid, resp)
                                    content = f"[фото: {len(buf.photos)} шт.]"
                                    if buf.text:
                                        content = f"{buf.text} [фото: {len(buf.photos)} шт.]"
                                    await save_message(fresh_sess, user_id, fresh_pet.id, "user", content)
                                    await save_message(fresh_sess, user_id, fresh_pet.id, "model", resp)

                            batch.task = asyncio.create_task(_flush())
                            return
                    else:
                        media_items = new_photos

                # ── Задача 1 + 3: аудио и документы (PDF) ─────────────────
                if not media_error and has_non_photo_media(msg):
                    # Проверяем голосовые сообщения отдельно
                    audio_att = next(
                        (a for a in msg.get("attachments", []) if a.get("type") == "audio_message"),
                        None,
                    )

                    if audio_att and is_groq_available():
                        # Задача 1: Groq Whisper — расшифровываем голосовое
                        logger.info("Задача 1: голосовое сообщение → Groq Whisper")
                        audio_bytes, audio_mime, audio_err = await get_audio_message(audio_att, http_session)

                        if audio_err:
                            await vk.send_message(vk_id, audio_err)
                            return

                        if audio_bytes and audio_mime:
                            transcript, transcribe_err = await transcribe_audio(
                                audio_bytes, audio_mime, current_vk_id=vk_id
                            )
                            if transcript:
                                logger.info(
                                    "Groq: расшифровано голосовое → %d символов: %r",
                                    len(transcript), transcript[:80],
                                )
                                # Добавляем транскрипт к тексту, как будто пользователь написал
                                text = (text + " " + transcript).strip() if text else transcript
                                voice_transcribed = True
                                await vk.send_message(vk_id, f"🎙 Распознал: «{transcript}»")
                            else:
                                # Groq не смог — fallback на Gemini (передаём аудио байтами)
                                logger.info("Groq: fallback на Gemini для аудио")
                                media_items.append((audio_bytes, audio_mime))

                    elif audio_att and not is_groq_available():
                        # Нет Groq — передаём аудио напрямую в Gemini
                        audio_bytes, audio_mime, audio_err = await get_audio_message(audio_att, http_session)
                        if audio_err:
                            await vk.send_message(vk_id, audio_err)
                            return
                        if audio_bytes and audio_mime:
                            media_items.append((audio_bytes, audio_mime))

                    else:
                        # Задача 3: PDF и другие документы — через extract_media → Gemini нативно
                        nb, nm, media_error = await extract_media(msg, http_session)
                        if nb and nm:
                            if nm == "application/pdf":
                                logger.info("Задача 3: PDF (%d байт) → Gemini нативно", len(nb))
                            media_items.append((nb, nm))

        if media_error:
            await vk.send_message(vk_id, media_error)
            return

        # ── Triage check ───────────────────────────────────────────────────
        if text:
            triage_resp = await analyze_triage(text, pet.name, pet.species, user.id, pet.id)
            if triage_resp:
                await vk.send_message(vk_id, triage_resp)
                await save_message(session, user.id, pet.id, "user", text)
                await save_message(session, user.id, pet.id, "model", triage_resp)
                return

        _, st_data = await get_user_state_data(session, user.id)
        scope_instr = st_data.get("scope") if st_data else None

        if is_emergency_active(user.id, pet.id) and should_send_followup(user.id, pet.id):
            await vk.send_message(vk_id, "Кстати, как сейчас дела у питомца? Добрались до врача?")

        # ── Prepare LLM context ────────────────────────────────────────────
        pet_facts = await get_pet_facts_text(session, pet)
        history = await get_chat_history_text(session, user.id, pet.id)

        other_names = [p.name for p in all_pets if p.id != pet.id]
        other_hint = ""
        if other_names:
            other_hint = f"У пользователя есть ещё питомцы: {', '.join(other_names)}. Если он упоминает другого питомца — предложи переключиться (/pets)."

        # ── Generate response ──────────────────────────────────────────────
        try:
            response_text = await generate_response(
                user_text=text,
                pet_name=pet.name,
                species=pet.species,
                pet_facts=pet_facts,
                other_pets_hint=other_hint,
                history=history,
                media_items=media_items or [],
                scope_instruction=scope_instr,
                emergency_active=is_emergency_active(user.id, pet.id),
                web_search_context=web_search_context,   # Задача 2: Tavily контекст
                toxicology_active=toxicology_active,      # Задача 5: токсикология engine
            )
        except Exception as e:
            logger.error("LLM generate error: %s", e)
            await vk.send_message(vk_id, "Произошла ошибка при обработке запроса. Попробуй ещё раз.")
            return

        await vk.send_message(vk_id, response_text)

        # ── Save messages ──────────────────────────────────────────────────
        user_content = text
        if voice_transcribed and text:
            user_content = f"[голосовое → текст]: {text}"
        elif media_items and not text:
            user_content = f"[медиафайл: {len(media_items)} шт.]"
        elif media_items:
            user_content = f"{text} [медиафайл: {len(media_items)} шт.]"

        await save_message(session, user.id, pet.id, "user", user_content)
        await save_message(session, user.id, pet.id, "model", response_text)

        # ── Extract and save facts — запускаем в фоне ─────────────────────
        quiet_mode = bool(st_data.get("quiet")) if st_data else False
        if text:
            asyncio.create_task(
                _extract_and_confirm_facts(
                    vk_id=vk_id,
                    user_id=user.id,
                    pet_id=pet.id,
                    text=text,
                    pet_name=pet.name,
                    species=pet.species,
                    quiet_mode=quiet_mode,
                )
            )


# ─── Background: fact extraction ─────────────────────────────────────────────

async def _extract_and_confirm_facts(
    vk_id: int,
    user_id: int,
    pet_id: int,
    text: str,
    pet_name: str,
    species: str,
    quiet_mode: bool,
) -> None:
    """Извлекает факты из сообщения и спрашивает подтверждение — в фоновом таске."""
    try:
        overloaded, _ = is_overloaded()
        if overloaded:
            return
        facts = await extract_facts(text, pet_name, species)
        if not facts:
            return
        from memory import confirm_pending, _is_age_key, _is_sex_key
        async with async_session() as session:
            pending = await process_extracted_facts(session, pet_id, facts)
            for p_fact in pending:
                key = p_fact["key"]
                # Возраст и пол — критические поля: всегда требуют подтверждения,
                # даже если включён тихий режим (quiet). Автосохранение этих полей
                # приводило к записи галлюцинаций модели без возможности отклонить.
                always_confirm = _is_age_key(key) or _is_sex_key(key)
                if quiet_mode and not always_confirm:
                    await confirm_pending(session, pet_id, p_fact["id"])
                else:
                    confirm_msg = (
                        f"📝 Сохранить: {key} = «{p_fact['new_value']}»?"
                    )
                    if p_fact.get("old_value"):
                        confirm_msg += f"\n(сейчас: «{p_fact['old_value']}»)"
                    await vk.send_message(
                        vk_id,
                        confirm_msg,
                        keyboard=yes_no_keyboard(p_fact["id"]),
                    )
    except Exception as e:
        logger.warning("Fact extraction (background) error: %s", e)


# ─── Callback Event Handler ───────────────────────────────────────────────────

async def handle_callback_event(obj: dict) -> None:
    vk_id = obj.get("user_id")
    payload_raw = obj.get("payload", "{}")
    event_id = obj.get("event_id", "")
    peer_id = obj.get("peer_id", 0)
    cmid = obj.get("conversation_message_id", 0)

    if not vk_id:
        return

    try:
        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
    except Exception:
        return

    async with async_session() as session:
        user = await ensure_user(session, vk_id)
        await handlers.handle_confirm_callback(
            vk, session, user, vk_id, payload, event_id,
            peer_id=peer_id, cmid=cmid,
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _route_mapped_command(vk, session, user, vk_id, cmd, original_text):
    cmd = cmd.lower()
    if "/addpet" in cmd:
        await handlers.handle_addpet(vk, session, user, vk_id)
    elif "/me" in cmd or "профиль" in cmd:
        await handlers.handle_me(vk, session, user, vk_id)
    elif "/pets" in cmd:
        await handlers.handle_pets(vk, session, user, vk_id)
    elif "/reset" in cmd or "удали" in original_text.lower():
        await handlers.handle_reset(vk, session, user, vk_id)
    elif "/reminders" in cmd:
        await handlers.handle_reminders(vk, session, user, vk_id)
    elif "/remind" in cmd:
        await handlers.handle_remind(vk, session, user, vk_id)
    else:
        pass


async def _handle_profile_query(vk, session, user, vk_id):
    """Handle 'what do you know about my pet' queries with direct DB lookup."""
    from sqlalchemy import select
    from models import Pet as PetModel

    if user.active_pet_id is None:
        await vk.send_message(vk_id, "Питомцев пока нет. /addpet чтобы добавить.")
        return

    result = await session.execute(select(PetModel).where(PetModel.id == user.active_pet_id))
    pet = result.scalar_one_or_none()
    if not pet:
        await vk.send_message(vk_id, "Питомец не найден. /addpet")
        return

    facts = await get_pet_facts_text(session, pet)
    await vk.send_message(vk_id, f"📋 Всё, что я знаю о {pet.name}:\n\n{facts}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
