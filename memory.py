"""Business logic for pet memory: UPSERT facts, pending confirmations, chat history."""
import json
import logging
import re
from datetime import datetime, date, timedelta
from typing import Optional

from sqlalchemy import select, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from models import Memory, PendingConfirmation, ChatMessage, Pet, User, UserState
from prompt import SENSITIVE_KEYS, CATEGORY_LABELS
from logging_setup import log_event

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 15   # messages per (user, pet) kept in context


def _format_age(birth_date: "date") -> str:
    """Return human-readable age like '2 г. 3 мес. 10 дн.' from birth_date."""
    today = date.today()
    total_days = (today - birth_date).days
    years = total_days // 365
    remaining = total_days - years * 365
    months = remaining // 30
    days = remaining - months * 30

    parts = []
    if years:
        parts.append(f"{years} г.")
    if months:
        parts.append(f"{months} мес.")
    if days and not years:
        parts.append(f"{days} дн.")
    return " ".join(parts) if parts else "менее 1 дня"


def _is_sensitive(key: str) -> bool:
    key_lower = key.lower()
    return any(sk in key_lower for sk in SENSITIVE_KEYS)


_AGE_KEY_SUBSTRINGS = ("возраст", "age", "дата рождения", "birth_date", "birth date", "лет", "год")
_SEX_KEY_SUBSTRINGS = ("пол", "sex", "gender", "самец", "самка", "мальчик", "девочка", "кобель", "сука")


def _is_age_key(key: str) -> bool:
    """True если ключ описывает возраст / дату рождения питомца (substring match)."""
    k = key.lower().strip()
    return any(ak in k for ak in _AGE_KEY_SUBSTRINGS)


def _is_sex_key(key: str) -> bool:
    """True если ключ описывает пол питомца (substring match)."""
    k = key.lower().strip()
    return any(sk in k for sk in _SEX_KEY_SUBSTRINGS)


def _parse_age_to_birth_date(text: str) -> Optional[date]:
    """Парсит текстовое значение возраста → дата рождения (приблизительно).

    Поддерживает: «4 года», «2 г 3 мес», «10 месяцев», «180 дней», «5»→5 лет.
    Возвращает None если не распознано.
    """
    t = text.lower().strip()
    years_m  = re.search(r"(\d+)\s*(?:лет|года?|год|г\.?)\b", t)
    months_m = re.search(r"(\d+)\s*(?:месяц(?:а|ев)?|мес\.?|мес\b)", t)
    days_m   = re.search(r"(\d+)\s*(?:дней|дня|день|дн\.?|дн\b)", t)
    if not years_m and not months_m and not days_m:
        num_m = re.match(r"^(\d+)$", t)
        if num_m:
            y = int(num_m.group(1))
            if 0 <= y <= 30:
                return date.today().replace(year=date.today().year - y)
        return None
    years  = int(years_m.group(1))  if years_m  else 0
    months = int(months_m.group(1)) if months_m else 0
    days   = int(days_m.group(1))   if days_m   else 0
    return date.today() - timedelta(days=years * 365 + months * 30 + days)


async def upsert_memory(session: AsyncSession, pet_id: int, category: str, key: str, value: str, confidence: float = 1.0) -> None:
    """Write or update a non-sensitive fact directly."""
    stmt = (
        pg_insert(Memory)
        .values(pet_id=pet_id, category=category, key=key, value=value, sensitive=False, confidence=confidence, updated=datetime.utcnow())
        .on_conflict_do_update(
            index_elements=["pet_id", "category", "key"],
            set_={"value": value, "confidence": confidence, "updated": datetime.utcnow()},
        )
    )
    await session.execute(stmt)
    await session.commit()


async def queue_sensitive_confirmation(session: AsyncSession, pet_id: int, category: str, key: str, new_value: str) -> Optional[str]:
    """
    Store fact in pending_confirmations.
    Returns the old value if exists (for display), or None.
    Returns sentinel "SAME" (str) if new_value == old_value — caller should skip sending a button.
    """
    # Check current value
    result = await session.execute(
        select(Memory).where(and_(Memory.pet_id == pet_id, Memory.category == category, Memory.key == key))
    )
    existing = result.scalar_one_or_none()
    old_value = existing.value if existing else None

    # Если новое значение совпадает с уже сохранённым — кнопка не нужна.
    # Сравниваем без учёта регистра и пробелов.
    if old_value is not None and old_value.strip().lower() == new_value.strip().lower():
        logger.debug(
            "queue_sensitive_confirmation: пропускаем (значение не изменилось): key=%s val=%s",
            key, new_value,
        )
        return "SAME"

    # Remove any existing pending for same key
    await session.execute(
        delete(PendingConfirmation).where(
            and_(PendingConfirmation.pet_id == pet_id, PendingConfirmation.key == key)
        )
    )
    conf = PendingConfirmation(
        pet_id=pet_id, category=category, key=key,
        old_value=old_value, new_value=new_value, created=datetime.utcnow(),
    )
    session.add(conf)
    await session.commit()
    return old_value


async def confirm_pending(session: AsyncSession, pet_id: int, confirm_id: int) -> bool:
    """Apply a pending confirmation to memories."""
    result = await session.execute(
        select(PendingConfirmation).where(
            and_(PendingConfirmation.id == confirm_id, PendingConfirmation.pet_id == pet_id)
        )
    )
    conf = result.scalar_one_or_none()
    if not conf:
        return False

    stmt = (
        pg_insert(Memory)
        .values(pet_id=pet_id, category=conf.category, key=conf.key, value=conf.new_value,
                sensitive=True, confidence=1.0, updated=datetime.utcnow())
        .on_conflict_do_update(
            index_elements=["pet_id", "category", "key"],
            set_={"value": conf.new_value, "sensitive": True, "updated": datetime.utcnow()},
        )
    )
    await session.execute(stmt)
    await session.execute(delete(PendingConfirmation).where(PendingConfirmation.id == confirm_id))

    log_event(
        logger, logging.INFO, "fact_confirmed",
        pet_id=pet_id, key=conf.key, value=conf.new_value, sensitive=True,
    )

    # Синхронизируем Pet-поля при подтверждении чувствительных фактов.
    # Пол и возраст хранятся ТОЛЬКО в Pet (не дублируются в Memory) —
    # иначе в /me они выводятся дважды: из шапки Pet и из блока фактов Memory.
    if _is_sex_key(conf.key):
        # Пол → Pet.sex; строку из Memory удаляем
        pet_result = await session.execute(select(Pet).where(Pet.id == pet_id))
        pet_obj = pet_result.scalar_one_or_none()
        if pet_obj is not None:
            pet_obj.sex = conf.new_value
        # Удаляем все sex-записи из Memory для этого питомца
        all_mem = await session.execute(select(Memory).where(Memory.pet_id == pet_id))
        for mem_row in all_mem.scalars().all():
            if _is_sex_key(mem_row.key):
                await session.delete(mem_row)
        await session.execute(
            delete(Memory).where(
                and_(Memory.pet_id == pet_id, Memory.key == conf.key)
            )
        )

    elif _is_age_key(conf.key):
        # Возраст → Pet.birth_date; строку из Memory удаляем
        parsed_bd = _parse_age_to_birth_date(conf.new_value)
        if parsed_bd is not None:
            pet_result = await session.execute(select(Pet).where(Pet.id == pet_id))
            pet_obj = pet_result.scalar_one_or_none()
            if pet_obj is not None:
                pet_obj.birth_date = parsed_bd
                logger.info(
                    "confirm_pending: Pet %d birth_date → %s (from «%s»)",
                    pet_id, parsed_bd, conf.new_value,
                )
        else:
            logger.warning(
                "confirm_pending: не удалось распарсить возраст «%s» для Pet %d",
                conf.new_value, pet_id,
            )
        # Удаляем все age-записи из Memory для этого питомца
        all_mem = await session.execute(select(Memory).where(Memory.pet_id == pet_id))
        for mem_row in all_mem.scalars().all():
            if _is_age_key(mem_row.key):
                await session.delete(mem_row)
        await session.execute(
            delete(Memory).where(
                and_(Memory.pet_id == pet_id, Memory.key == conf.key)
            )
        )

    await session.commit()
    return True


async def reject_pending(session: AsyncSession, pet_id: int, confirm_id: int) -> bool:
    """Discard a pending confirmation."""
    result = await session.execute(
        delete(PendingConfirmation).where(
            and_(PendingConfirmation.id == confirm_id, PendingConfirmation.pet_id == pet_id)
        )
    )
    await session.commit()
    return result.rowcount > 0


async def process_extracted_facts(
    session: AsyncSession,
    pet_id: int,
    facts: list[dict],
) -> list[dict]:
    """
    Process list of extracted facts.
    Returns list of pending confirmations that need user approval.
    Each pending: {"id": int, "key": str, "old_value": str|None, "new_value": str}
    """
    pending_out = []
    for fact in facts:
        category = fact.get("category", "basic")
        key = fact.get("key", "").strip()
        value = fact.get("value", "").strip()
        confidence = float(fact.get("confidence", 1.0))
        sensitive = fact.get("sensitive", False) or _is_sensitive(key)

        if not key or not value:
            continue

        if sensitive:
            old = await queue_sensitive_confirmation(session, pet_id, category, key, value)
            # "SAME" = значение не изменилось, кнопка не нужна
            if old == "SAME":
                logger.debug("process_extracted_facts: пропускаем %s = %r (уже сохранено)", key, value)
                continue
            # Get the new pending ID
            result = await session.execute(
                select(PendingConfirmation).where(
                    and_(PendingConfirmation.pet_id == pet_id, PendingConfirmation.key == key)
                )
            )
            pc = result.scalar_one_or_none()
            if pc:
                pending_out.append({"id": pc.id, "key": key, "old_value": old, "new_value": value})
        else:
            await upsert_memory(session, pet_id, category, key, value, confidence)

    return pending_out


async def get_pet_facts_text(session: AsyncSession, pet: Pet) -> str:
    """Format all pet memories as text block for system prompt."""
    result = await session.execute(
        select(Memory).where(Memory.pet_id == pet.id).order_by(Memory.category, Memory.key)
    )
    memories = result.scalars().all()

    if not memories:
        base = f"Имя: {pet.name}\nВид: {pet.species}"
        if pet.breed:
            base += f"\nПорода: {pet.breed}"
        if pet.sex:
            base += f"\nПол: {pet.sex}"
        if pet.birth_date:
            base += f"\nВозраст: {_format_age(pet.birth_date)} (д.р. {pet.birth_date.strftime('%d.%m.%Y')})"
        return base

    by_cat: dict[str, list[Memory]] = {}
    for m in memories:
        by_cat.setdefault(m.category, []).append(m)

    lines = [f"Имя: {pet.name}", f"Вид: {pet.species}"]
    if pet.breed:
        lines.append(f"Порода: {pet.breed}")
    if pet.sex:
        lines.append(f"Пол: {pet.sex}")
    if pet.birth_date:
        lines.append(f"Возраст: {_format_age(pet.birth_date)} (д.р. {pet.birth_date.strftime('%d.%m.%Y')})")

    for cat, items in by_cat.items():
        # Пропускаем поля которые уже выведены в шапке профиля из Pet-модели:
        # - возраст (Pet.birth_date) → строка «Возраст: X г. (д.р. ДД.ММ.ГГГГ)»
        # - пол (Pet.sex) → строка «Пол: …»
        # Отображать их из Memory — дубль.
        visible = [
            m for m in items
            if not (pet.birth_date and _is_age_key(m.key))
            and not (pet.sex and _is_sex_key(m.key))
        ]
        if not visible:
            continue
        label = CATEGORY_LABELS.get(cat, cat)
        lines.append(f"\n{label}:")
        for m in visible:
            lines.append(f"  • {m.key}: {m.value}")

    return "\n".join(lines)


async def get_chat_history_text(session: AsyncSession, user_id: int, pet_id: Optional[int]) -> str:
    """Get last N messages as formatted history string."""
    if pet_id is None:
        return ""

    result = await session.execute(
        select(ChatMessage)
        .where(and_(ChatMessage.user_id == user_id, ChatMessage.pet_id == pet_id))
        .order_by(ChatMessage.created.desc())
        .limit(HISTORY_LIMIT)
    )
    messages = list(reversed(result.scalars().all()))

    lines = []
    for msg in messages:
        role = "Пользователь" if msg.role == "user" else "Ева"
        lines.append(f"{role}: {msg.content[:500]}")

    return "\n".join(lines) if lines else "(нет истории)"


async def save_message(session: AsyncSession, user_id: int, pet_id: Optional[int], role: str, content: str) -> None:
    """Append a message to chat_messages and trim old ones."""
    msg = ChatMessage(user_id=user_id, pet_id=pet_id, role=role, content=content, created=datetime.utcnow())
    session.add(msg)
    await session.commit()

    # Trim to keep only HISTORY_LIMIT * 2 messages per (user, pet)
    if pet_id:
        result = await session.execute(
            select(ChatMessage.id)
            .where(and_(ChatMessage.user_id == user_id, ChatMessage.pet_id == pet_id))
            .order_by(ChatMessage.created.desc())
            .offset(HISTORY_LIMIT * 2)
        )
        old_ids = [row[0] for row in result.all()]
        if old_ids:
            await session.execute(delete(ChatMessage).where(ChatMessage.id.in_(old_ids)))
            await session.commit()


async def get_or_create_user_state(session: AsyncSession, user_id: int) -> UserState:
    result = await session.execute(select(UserState).where(UserState.user_id == user_id))
    state = result.scalar_one_or_none()
    if not state:
        state = UserState(user_id=user_id, state=None, state_data=None)
        session.add(state)
        await session.commit()
    return state


async def set_user_state(session: AsyncSession, user_id: int, state: Optional[str], data: Optional[dict] = None) -> None:
    us = await get_or_create_user_state(session, user_id)
    from_state = us.state
    us.state = state
    us.state_data = json.dumps(data, ensure_ascii=False) if data else None
    us.updated = datetime.utcnow()
    await session.commit()

    if from_state != state:
        log_event(
            logger, logging.INFO, "state_changed",
            from_state=from_state, to_state=state,
        )


async def get_user_state_data(session: AsyncSession, user_id: int) -> tuple[Optional[str], dict]:
    us = await get_or_create_user_state(session, user_id)
    data = {}
    if us.state_data:
        try:
            data = json.loads(us.state_data)
        except Exception:
            data = {}
    return us.state, data
