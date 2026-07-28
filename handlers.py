"""Command and FSM handlers for ZOO Ментор VK bot."""
import json
import logging
import re
from datetime import datetime, date
from zoneinfo import ZoneInfo

_MSK = ZoneInfo("Europe/Moscow")


def _now_msk() -> datetime:
    """Current naive datetime in Moscow time."""
    return datetime.now(_MSK).replace(tzinfo=None)
from typing import Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from models import User, Pet, Memory, Reminder, UserState, ChatMessage, PendingConfirmation
from memory import (
    set_user_state, get_user_state_data,
    get_pet_facts_text, get_pet_facts_text,
)
from billing import get_subscription_info
from prompt import SPECIES_ALLOWLIST, CATEGORY_LABELS
from vk_client import VKClient, pets_keyboard, yes_no_keyboard, main_menu_keyboard, species_keyboard
from logging_setup import log_event

logger = logging.getLogger(__name__)

MAX_PETS = 5

KNOWN_SPECIES_NORMALIZED = {
    "кошка": "кошка", "кот": "кошка", "котик": "кошка", "кошечка": "кошка", "кис": "кошка",
    "собака": "собака", "пёс": "собака", "пес": "собака", "собачка": "собака",
    "попугай": "попугай", "волнистый": "попугай", "волнистик": "попугай",
    "кролик": "кролик", "кроль": "кролик",
    "морская свинка": "морская свинка", "свинка": "морская свинка",
    "хомяк": "хомяк", "хомячок": "хомяк",
    "крыса": "крыса", "крысёнок": "крыса", "мышь": "мышь", "мышка": "мышь",
    "черепаха": "черепаха", "рыба": "рыба", "рыбка": "рыба",
    "хорёк": "хорёк", "хорек": "хорёк",
    "шиншилла": "шиншилла", "ёж": "ёж", "еж": "ёж",
    "ящерица": "ящерица", "геккон": "геккон", "игуана": "игуана",
    "змея": "змея", "другое": "другое",
}


def _normalize_species(text: str) -> Optional[str]:
    """Return normalized species name or None if not recognized."""
    t = text.strip().lower()
    # Direct match
    if t in KNOWN_SPECIES_NORMALIZED:
        return KNOWN_SPECIES_NORMALIZED[t]
    # Partial match
    for k, v in KNOWN_SPECIES_NORMALIZED.items():
        if k in t or t in k:
            return v
    # Allow "другое" / "другой" / "other" as fallback
    if any(x in t for x in ("друг", "other", "иной", "прочее", "экзот")):
        return "другое"
    return None


def _format_age(birth_date: date) -> str:
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
    if days and not years:  # show days only when less than a year old
        parts.append(f"{days} дн.")
    return " ".join(parts) if parts else "менее 1 дня"


def _parse_age_to_birth_date(text: str) -> Optional[date]:
    """Try to extract age from text and convert to approximate birth_date.

    Supports formats like:
      "5 лет", "2 года", "1 год", "2 г", "2г"
      "3 месяца", "5 месяцев", "3 мес", "3м"
      "10 дней", "10 дн", "10 д"
      Combinations: "2 г 3 мес и 10 дней", "1 год 2 месяца 5 дней"
      Plain number: "3" → 3 years
    """
    from datetime import timedelta

    text = text.lower()

    # Years: "год", "года", "лет", "г" (abbreviation, with optional dot)
    years_match = re.search(r"(\d+)\s*(?:лет|года?|год|г\.?)\b", text)
    # Months: "месяц", "месяца", "месяцев", "мес", "м" (careful: only standalone "м")
    months_match = re.search(r"(\d+)\s*(?:месяц(?:а|ев)?|мес\.?|мес\b)", text)
    # Days: "день", "дня", "дней", "дн", "д"
    days_match = re.search(r"(\d+)\s*(?:дней|дня|день|дн\.?|дн\b)", text)

    if not years_match and not months_match and not days_match:
        # Try plain number → treat as years
        num_match = re.match(r"^\s*(\d+)\s*$", text.strip())
        if num_match:
            years = int(num_match.group(1))
            if 0 <= years <= 30:
                return date.today().replace(year=date.today().year - years)
        return None

    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0
    days = int(days_match.group(1)) if days_match else 0
    total_days = years * 365 + months * 30 + days
    return date.today() - timedelta(days=total_days)


def _looks_like_age(text: str) -> bool:
    """Возвращает True если текст выглядит как возраст, а не имя питомца.

    Задача 3 (регрессия): защита от бага когда пользователь вводит «2 года»
    или «1 год 3 месяца» в ответ на вопрос «Как зовут питомца?».
    Такое имя ломает промпт, склеивая имя с возрастом.
    """
    t = text.strip().lower()
    # Паттерн: только цифры + единицы времени (с или без пробела)
    age_only_pattern = re.compile(
        r"^\d+\s*(?:лет|года?|год|г\.?|месяц(?:а|ев)?|мес\.?|мес|дней|дня|день|дн\.?|дн)\b"
        r"(?:\s+\d+\s*(?:месяц(?:а|ев)?|мес\.?|мес|дней|дня|день|дн\.?|дн)\b)?$"
    )
    return bool(age_only_pattern.match(t))


# ─────────────────────────────────────────────────────────────────────────────
# /start
# ─────────────────────────────────────────────────────────────────────────────

async def handle_start(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    await set_user_state(session, user.id, None)
    text = (
        "Привет! Я Ева — ветеринарный ИИ-помощник 🐾\n\n"
        "Я помогаю заботиться о питомцах: разбираю анализы и рентгены, "
        "отвечаю на вопросы о питании, здоровье и поведении, запоминаю важные факты.\n\n"
        "🎁 Первые 7 дней — полный доступ бесплатно.\n\n"
        "Чтобы начать — добавь питомца: /addpet\n\n"
        "Список команд: /help"
    )
    await vk.send_message(vk_id, text, keyboard=main_menu_keyboard())


# ─────────────────────────────────────────────────────────────────────────────
# /addpet  FSM
# ─────────────────────────────────────────────────────────────────────────────

async def handle_addpet(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    # Count existing pets
    result = await session.execute(select(Pet).where(Pet.user_id == user.id))
    pets = result.scalars().all()
    if len(pets) >= MAX_PETS:
        await vk.send_message(
            vk_id,
            f"У тебя уже {len(pets)} питомцев — максимум {MAX_PETS}.\n"
            "Чтобы удалить одного: /reset"
        )
        return
    await set_user_state(session, user.id, "awaiting_pet_species")
    await vk.send_message(
        vk_id,
        "Кто твой питомец? Выбери ниже или напиши вид: кошка, собака, попугай, кролик и т.д.",
        keyboard=species_keyboard(),
    )


async def fsm_awaiting_pet_species(
    vk: VKClient, session: AsyncSession, user: User, vk_id: int, text: str
) -> None:
    if text.lower().startswith("/cancel"):
        await set_user_state(session, user.id, None)
        await vk.send_message(vk_id, "Добавление питомца отменено.")
        return

    species = _normalize_species(text)
    if not species:
        await vk.send_message(
            vk_id,
            f"Не узнал вид 🤔 Напиши, например: кошка, собака, попугай, хомяк, кролик.\n"
            f"Если вид нестандартный — напиши «другое».\n/cancel — отмена"
        )
        return

    await set_user_state(session, user.id, "awaiting_pet_name", {"species": species})
    await vk.send_message(vk_id, f"Отлично! Как зовут питомца?")


async def fsm_awaiting_pet_name(
    vk: VKClient, session: AsyncSession, user: User, vk_id: int,
    text: str, state_data: dict
) -> None:
    if text.lower().startswith("/cancel"):
        await set_user_state(session, user.id, None)
        await vk.send_message(vk_id, "Добавление питомца отменено.")
        return

    name = text.strip()
    if not name or len(name) > 50:
        await vk.send_message(vk_id, "Имя не должно быть пустым или слишком длинным. Попробуй ещё раз.")
        return

    # Фикс: проверка на возраст в имени.
    # Два случая:
    # 1. Строка целиком — возраст («2 года», «3 месяца») — _looks_like_age()
    # 2. Строка содержит возраст вместе с именем («Васька и ему 3 года») — regex по маркерам
    import re as _re
    _age_markers = r"\b\d+\s*(год|года|лет|месяц|месяца|месяцев|день|дня|дней)\b"
    if _looks_like_age(name) or _re.search(_age_markers, name, flags=_re.IGNORECASE):
        await vk.send_message(
            vk_id,
            f"Кажется, в сообщении «{name}» есть ещё и возраст — "
            f"я спрошу его отдельным вопросом чуть позже. Напиши, пожалуйста, только имя питомца."
        )
        return

    data = dict(state_data)
    data["name"] = name
    await set_user_state(session, user.id, "awaiting_pet_age", data)
    await vk.send_message(vk_id, f"Сколько лет {name}? (Напиши число или, например, «2 года 3 месяца»)")


async def fsm_awaiting_pet_age(
    vk: VKClient, session: AsyncSession, user: User, vk_id: int,
    text: str, state_data: dict
) -> None:
    if text.lower().startswith("/cancel"):
        await set_user_state(session, user.id, None)
        await vk.send_message(vk_id, "Добавление питомца отменено.")
        return

    birth_date = _parse_age_to_birth_date(text)
    name = state_data.get("name", "Питомец")
    species = state_data.get("species", "другое")

    pet = Pet(
        user_id=user.id,
        name=name,
        species=species,
        birth_date=birth_date,
        created=datetime.utcnow(),
    )
    session.add(pet)
    await session.flush()   # get pet.id

    # Make this the active pet
    user.active_pet_id = pet.id
    await session.commit()

    await set_user_state(session, user.id, None)

    log_event(
        logger, logging.INFO, "pet_created",
        pet_id=pet.id, pet_name=name, species=species,
        birth_date=birth_date.isoformat() if birth_date else None,
    )

    age_str = ""
    if birth_date:
        age_str = f", {_format_age(birth_date)}"

    await vk.send_message(
        vk_id,
        f"✅ Профиль создан: {name} ({species}{age_str}).\n\n"
        f"Теперь можешь рассказывать мне о {name} — я буду запоминать важные факты.\n"
        f"Посмотреть профиль: /me"
    )


# ─────────────────────────────────────────────────────────────────────────────
# /pets, /switch
# ─────────────────────────────────────────────────────────────────────────────

async def handle_pets(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    result = await session.execute(select(Pet).where(Pet.user_id == user.id))
    pets = result.scalars().all()

    if not pets:
        await vk.send_message(vk_id, "Питомцев пока нет. /addpet — добавить.")
        return

    if len(pets) == 1:
        pet = pets[0]
        user.active_pet_id = pet.id
        await session.commit()
        await vk.send_message(vk_id, f"Активный питомец: {pet.name} ({pet.species}).\nПрофиль: /me")
        return

    lines = ["Твои питомцы — нажми чтобы переключиться:"]
    for p in pets:
        mark = "✅" if p.id == user.active_pet_id else "○"
        lines.append(f"{mark} {p.name} ({p.species})")

    # Keep text fallback state for users who type instead of clicking
    await set_user_state(session, user.id, "awaiting_pet_switch")
    await vk.send_message(
        vk_id,
        "\n".join(lines),
        keyboard=pets_keyboard(pets),  # inline callback keyboard
    )


async def fsm_awaiting_pet_switch(
    vk: VKClient, session: AsyncSession, user: User, vk_id: int, text: str
) -> None:
    if text.lower().startswith("/cancel"):
        await set_user_state(session, user.id, None)
        await vk.send_message(vk_id, "Переключение отменено.")
        return

    result = await session.execute(select(Pet).where(Pet.user_id == user.id))
    pets = result.scalars().all()

    # Try to match by name in button label
    text_clean = text.strip().lower()
    matched = None
    for p in pets:
        if p.name.lower() in text_clean or text_clean in p.name.lower():
            matched = p
            break

    if not matched:
        await vk.send_message(vk_id, "Питомец не найден. Выбери из списка или /cancel.")
        return

    user.active_pet_id = matched.id
    await session.commit()
    await set_user_state(session, user.id, None)
    await vk.send_message(vk_id, f"Переключился на {matched.name} ({matched.species}). /me — профиль.")


# ─────────────────────────────────────────────────────────────────────────────
# /me
# ─────────────────────────────────────────────────────────────────────────────

async def handle_me(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    if not user.active_pet_id:
        result = await session.execute(select(Pet).where(Pet.user_id == user.id))
        pets = result.scalars().all()
        if not pets:
            await vk.send_message(vk_id, "Питомцев нет. /addpet — добавить.")
            return
        user.active_pet_id = pets[0].id
        await session.commit()

    result = await session.execute(select(Pet).where(Pet.id == user.active_pet_id))
    pet = result.scalar_one_or_none()
    if not pet:
        await vk.send_message(vk_id, "Питомец не найден. /addpet")
        return

    facts = await get_pet_facts_text(session, pet)
    await vk.send_message(vk_id, f"📋 Профиль {pet.name}:\n\n{facts}")


# ─────────────────────────────────────────────────────────────────────────────
# /reset  FSM
# ─────────────────────────────────────────────────────────────────────────────

async def handle_reset(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    result = await session.execute(select(Pet).where(Pet.user_id == user.id))
    pets = result.scalars().all()

    if not pets:
        await vk.send_message(vk_id, "Питомцев нет — нечего удалять.")
        return

    if len(pets) == 1:
        pet = pets[0]
        await set_user_state(session, user.id, "awaiting_confirm_reset", {"pet_id": pet.id, "what": "pet"})
        await vk.send_message(
            vk_id,
            f"Удалить {pet.name} и всю историю? Напиши «да» для подтверждения или «нет» для отмены."
        )
    else:
        # Multiple pets — ask what to delete
        lines = ["Что удалить?\n"]
        for p in pets:
            lines.append(f"• {p.name} ({p.species}) — напиши имя питомца")
        lines.append("• всё — удалить всех питомцев и весь профиль")
        lines.append("\n/cancel — отмена")
        await set_user_state(session, user.id, "awaiting_confirm_reset", {"what": "choose", "pets": [{"id": p.id, "name": p.name} for p in pets]})
        await vk.send_message(vk_id, "\n".join(lines))


async def fsm_awaiting_confirm_reset(
    vk: VKClient, session: AsyncSession, user: User, vk_id: int,
    text: str, state_data: dict
) -> None:
    t = text.strip().lower()

    if t in ("нет", "отмена", "/cancel", "cancel", "no"):
        await set_user_state(session, user.id, None)
        await vk.send_message(vk_id, "Отменено. Ничего не удалено.")
        return

    what = state_data.get("what")

    if what == "choose":
        # User typed pet name or "всё"
        if t in ("всё", "все", "all", "всех"):
            # Delete all pets
            pets_data = state_data.get("pets", [])
            for pd in pets_data:
                await _delete_pet(session, pd["id"], user)
            await session.commit()
            await set_user_state(session, user.id, None)
            await vk.send_message(vk_id, "Все питомцы и история удалены. /addpet — добавить нового.")
            return

        # Match pet name
        pets_data = state_data.get("pets", [])
        matched_id = None
        matched_name = None
        for pd in pets_data:
            if pd["name"].lower() in t or t in pd["name"].lower():
                matched_id = pd["id"]
                matched_name = pd["name"]
                break

        if not matched_id:
            await vk.send_message(vk_id, "Не нашёл такого питомца. Напиши имя точнее или «всё», или /cancel.")
            return

        await set_user_state(session, user.id, "awaiting_confirm_reset", {"pet_id": matched_id, "what": "pet"})
        await vk.send_message(vk_id, f"Удалить {matched_name} и всю историю? Напиши «да» или «нет».")
        return

    if what == "pet":
        if t in ("да", "yes", "ок", "ok", "подтверждаю"):
            pet_id = state_data.get("pet_id")
            result = await session.execute(select(Pet).where(Pet.id == pet_id, Pet.user_id == user.id))
            pet = result.scalar_one_or_none()
            if pet:
                pet_name = pet.name
                await _delete_pet(session, pet_id, user)
                await session.commit()
                await set_user_state(session, user.id, None)
                await vk.send_message(vk_id, f"✅ {pet_name} удалён вместе с историей. /addpet — добавить нового.")
            else:
                await set_user_state(session, user.id, None)
                await vk.send_message(vk_id, "Питомец уже удалён.")
        else:
            await vk.send_message(vk_id, "Напиши «да» для подтверждения или «нет» для отмены.")


async def _delete_pet(session: AsyncSession, pet_id: int, user: User) -> None:
    """Delete pet + all related data."""
    await session.execute(delete(ChatMessage).where(ChatMessage.pet_id == pet_id))
    await session.execute(delete(Memory).where(Memory.pet_id == pet_id))
    await session.execute(delete(PendingConfirmation).where(PendingConfirmation.pet_id == pet_id))
    await session.execute(delete(Reminder).where(Reminder.pet_id == pet_id))
    await session.execute(delete(Pet).where(Pet.id == pet_id))
    if user.active_pet_id == pet_id:
        # Switch to another pet if exists
        result = await session.execute(select(Pet).where(Pet.user_id == user.id, Pet.id != pet_id))
        other = result.scalars().first()
        user.active_pet_id = other.id if other else None


# ─────────────────────────────────────────────────────────────────────────────
# /reminders, /remind, /delremind
# ─────────────────────────────────────────────────────────────────────────────

async def handle_reminders(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    if not user.active_pet_id:
        await vk.send_message(vk_id, "Сначала добавь питомца: /addpet")
        return

    result = await session.execute(
        select(Reminder).where(Reminder.pet_id == user.active_pet_id, Reminder.active == True)
    )
    reminders = result.scalars().all()

    if not reminders:
        await vk.send_message(
            vk_id,
            "Активных напоминаний нет.\n/remind — добавить напоминание"
        )
        return

    lines = ["🔔 Активные напоминания:\n"]
    for r in reminders:
        fire_str = r.next_fire.strftime("%d.%m.%Y %H:%M")
        repeat = {"once": "однократно", "monthly": "ежемесячно", "yearly": "ежегодно"}.get(r.repeat_rule or "once", "однократно")
        lines.append(f"#{r.id} — {r.text} ({fire_str}, {repeat})\n   Удалить: /delremind {r.id}")

    await vk.send_message(vk_id, "\n".join(lines))


async def handle_remind(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    if not user.active_pet_id:
        await vk.send_message(vk_id, "Сначала добавь питомца: /addpet")
        return

    await set_user_state(session, user.id, "awaiting_remind_text")
    await vk.send_message(
        vk_id,
        "Что напомнить? Напиши текст напоминания (например: «Глистогонное»)."
    )


async def fsm_awaiting_remind_text(
    vk: VKClient, session: AsyncSession, user: User, vk_id: int,
    text: str, state_data: dict
) -> None:
    if text.lower().startswith("/cancel"):
        await set_user_state(session, user.id, None)
        await vk.send_message(vk_id, "Отменено.")
        return

    await set_user_state(session, user.id, "awaiting_remind_time", {"text": text})
    await vk.send_message(
        vk_id,
        "Когда напомнить? Напиши дату и время, например:\n"
        "«25.07.2026 10:00» — однократно\n"
        "«25.07 10:00 ежемесячно» — каждый месяц\n"
        "«25.07 10:00 ежегодно» — каждый год"
    )


async def fsm_awaiting_remind_time(
    vk: VKClient, session: AsyncSession, user: User, vk_id: int,
    text: str, state_data: dict
) -> None:
    if text.lower().startswith("/cancel"):
        await set_user_state(session, user.id, None)
        await vk.send_message(vk_id, "Отменено.")
        return

    remind_text = state_data.get("text", "Напоминание")
    next_fire, repeat_rule = _parse_remind_time(text)

    if not next_fire:
        await vk.send_message(
            vk_id,
            "Не понял дату 🤔 Напиши в формате «25.07.2026 10:00» или /cancel для отмены."
        )
        return

    reminder = Reminder(
        pet_id=user.active_pet_id,
        text=remind_text,
        next_fire=next_fire,
        repeat_rule=repeat_rule,
        active=True,
    )
    session.add(reminder)
    await session.commit()
    await session.refresh(reminder)
    await set_user_state(session, user.id, None)

    log_event(
        logger, logging.INFO, "reminder_created",
        reminder_id=reminder.id, pet_id=user.active_pet_id,
        text=remind_text[:200], repeat_rule=repeat_rule or "once",
    )

    fire_str = next_fire.strftime("%d.%m.%Y %H:%M")
    repeat_label = {"once": "", "monthly": " (ежемесячно)", "yearly": " (ежегодно)"}.get(repeat_rule or "once", "")
    await vk.send_message(
        vk_id,
        f"✅ Напоминание сохранено: «{remind_text}» — {fire_str}{repeat_label}\n\n"
        "Посмотреть все: /reminders"
    )


def _parse_remind_time(text: str):
    """Parse date/time from user text. Returns (datetime, repeat_rule) or (None, None)."""
    text = text.strip()
    
    repeat_rule = "once"
    if "ежемесячно" in text or "месяц" in text:
        repeat_rule = "monthly"
        text = text.replace("ежемесячно", "").replace("каждый месяц", "").strip()
    elif "ежегодно" in text or "год" in text.lower() and "лет" not in text:
        repeat_rule = "yearly"
        text = text.replace("ежегодно", "").replace("каждый год", "").strip()

    # Strip decorative quotes and trailing labels the user might copy from the example
    text = re.sub(r"[«»\"']", "", text)
    text = re.sub(r"—\s*(однократно|ежемесячно|ежегодно|каждый\s+\S+)", "", text)
    text = text.strip()

    # Try dd.mm.yyyy H:MM or HH:MM  (single-digit hour allowed)
    patterns = [
        r"(\d{1,2})\.(\d{2})\.(\d{4})\s+(\d{1,2}):(\d{2})",   # with year
        r"(\d{1,2})\.(\d{2})\s+(\d{1,2}):(\d{2})",             # without year → current
    ]

    for i, pat in enumerate(patterns):
        m = re.search(pat, text)
        if m:
            try:
                g = m.groups()
                if i == 0:
                    dt = datetime(int(g[2]), int(g[1]), int(g[0]), int(g[3]), int(g[4]))
                else:
                    dt = datetime(_now_msk().year, int(g[1]), int(g[0]), int(g[2]), int(g[3]))
                return dt, repeat_rule
            except Exception:
                continue

    return None, None


async def handle_delremind(vk: VKClient, session: AsyncSession, user: User, vk_id: int, args: str) -> None:
    try:
        rid = int(args.strip())
    except ValueError:
        await vk.send_message(vk_id, "Укажи номер напоминания, например: /delremind 3\nСписок: /reminders")
        return

    if not user.active_pet_id:
        await vk.send_message(vk_id, "Сначала добавь питомца: /addpet")
        return

    result = await session.execute(
        select(Reminder).where(Reminder.id == rid, Reminder.pet_id == user.active_pet_id)
    )
    reminder = result.scalar_one_or_none()
    if not reminder:
        await vk.send_message(vk_id, "Напоминание не найдено.")
        return

    reminder.active = False
    await session.commit()
    await vk.send_message(vk_id, f"✅ Напоминание #{rid} удалено.")


# ─────────────────────────────────────────────────────────────────────────────
# /subscription, /pay
# ─────────────────────────────────────────────────────────────────────────────

async def handle_subscription(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    info = get_subscription_info(user)
    await vk.send_message(vk_id, f"📊 Статус подписки:\n{info}")


async def handle_pay(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    if user.plan == "paid":
        await vk.send_message(vk_id, "✅ Подписка уже активна. Спасибо!")
        return
    await vk.send_message(
        vk_id,
        "💳 Подписка ZOO Ментор — 490 ₽/мес.\n\n"
        "Для оформления свяжись с администратором сообщества.\n"
        "Подписка даёт неограниченный доступ: сообщения, фото, голосовые без лимитов."
    )


# ─────────────────────────────────────────────────────────────────────────────
# /cancel
# ─────────────────────────────────────────────────────────────────────────────

async def handle_cancel(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    await set_user_state(session, user.id, None)
    await vk.send_message(vk_id, "Отменено. Чем могу помочь?")


# ─────────────────────────────────────────────────────────────────────────────
# /referral
# ─────────────────────────────────────────────────────────────────────────────

async def handle_referral(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    import os
    group_id = os.environ.get("VK_GROUP_ID", "")
    ref_link = f"https://vk.me/public{group_id}?ref={user.vk_id}" if group_id else "(настрой группу)"
    await vk.send_message(
        vk_id,
        f"🔗 Поделись ботом с друзьями у которых есть питомцы:\n\n"
        f"{ref_link}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# /help, /skills
# ─────────────────────────────────────────────────────────────────────────────

async def handle_help(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    text = (
        "📖 Что я умею:\n\n"
        "🐾 Питомцы:\n"
        "/addpet — добавить питомца (до 5)\n"
        "/pets — список питомцев / переключить активного\n"
        "/me — карточка активного питомца\n"
        "/reset — удалить питомца или профиль\n\n"
        "🔔 Напоминания:\n"
        "/remind — добавить напоминание\n"
        "/reminders — список напоминаний\n"
        "/delremind <N> — удалить напоминание\n\n"
        "💳 Подписка:\n"
        "/subscription — статус\n"
        "/pay — оформить подписку\n\n"
        "⚙ Прочее:\n"
        "/referral — поделиться ботом\n"
        "/cancel — отменить текущее действие\n"
        "/quiet — не спрашивать подтверждение при сохранении фактов\n"
        "/loud — снова спрашивать подтверждение (по умолчанию)\n\n"
        "Просто пиши — я помогаю с едой, здоровьем, поведением.\n"
        "Присылай фото, анализы, рентгены — разберём вместе!"
    )
    await vk.send_message(vk_id, text, keyboard=main_menu_keyboard())


# ─────────────────────────────────────────────────────────────────────────────
# /quiet, /loud
# ─────────────────────────────────────────────────────────────────────────────

async def handle_quiet(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    _, data = await get_user_state_data(session, user.id)
    data = data or {}
    data["quiet"] = True
    await set_user_state(session, user.id, None, data)
    await vk.send_message(
        vk_id,
        "🔕 Тихий режим включён — буду запоминать факты о питомце автоматически, без вопросов.\n"
        "/loud — вернуть подтверждения."
    )


async def handle_loud(vk: VKClient, session: AsyncSession, user: User, vk_id: int) -> None:
    _, data = await get_user_state_data(session, user.id)
    data = data or {}
    data["quiet"] = False
    await set_user_state(session, user.id, None, data)
    await vk.send_message(
        vk_id,
        "🔔 Подтверждения включены — буду уточнять перед сохранением важных фактов."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callback button handler (Да/Нет confirm flow)
# ─────────────────────────────────────────────────────────────────────────────

async def handle_confirm_callback(
    vk: VKClient,
    session: AsyncSession,
    user: User,
    vk_id: int,
    payload: dict,
    event_id: str,
    peer_id: int = 0,
    cmid: int = 0,
) -> None:
    action = payload.get("action")

    log_event(
        logger, logging.INFO, "button_pressed",
        peer_id=vk_id, callback_data=action,
        # payload может содержать pet_id/confirm id — не PII, но текст факта не логируем здесь
        payload_keys=list(payload.keys()) if payload else None,
    )

    # Always acknowledge the callback to dismiss VK's loading spinner
    await vk.send_event_answer(event_id=event_id, user_id=vk_id, peer_id=peer_id or vk_id)

    # ── Species selection from /addpet inline keyboard ────────────────────
    if action == "select_species":
        species = payload.get("species", "")
        state, _ = await get_user_state_data(session, user.id)
        if state == "awaiting_pet_species" and species:
            await set_user_state(session, user.id, "awaiting_pet_name", {"species": species})
            await vk.send_message(vk_id, f"Отлично! Как зовут питомца?")
        return

    # ── Pet switch from /pets inline keyboard ─────────────────────────────
    if action == "switch_pet":
        pet_id = payload.get("pet_id")
        if pet_id:
            result = await session.execute(
                select(Pet).where(Pet.id == pet_id, Pet.user_id == user.id)
            )
            pet = result.scalar_one_or_none()
            if pet:
                user.active_pet_id = pet.id
                await session.commit()
                await set_user_state(session, user.id, None)
                await vk.send_message(vk_id, f"Переключился на {pet.name} ({pet.species}). /me — профиль.")
            else:
                await vk.send_message(vk_id, "Питомец не найден.")
        return

    # ── Да/Нет confirm flow for sensitive fact changes ────────────────────
    confirm_id = payload.get("id")

    if not user.active_pet_id:
        return

    if action == "confirm_yes" and confirm_id:
        from models import PendingConfirmation as PC
        from memory import confirm_pending

        # Читаем факт ДО удаления, чтобы показать явный текст результата
        pc_result = await session.execute(
            select(PC).where(PC.id == confirm_id, PC.pet_id == user.active_pet_id)
        )
        pc = pc_result.scalar_one_or_none()

        ok = await confirm_pending(session, user.active_pet_id, confirm_id)
        if ok and pc:
            result_text = f"✅ Сохранено: {pc.key} = «{pc.new_value}»."
            # Деактивируем кнопки в исходном сообщении (убираем Да/Нет)
            if peer_id and cmid:
                orig_question = f"📝 Сохранить: {pc.key} = «{pc.new_value}»?"
                if pc.old_value:
                    orig_question += f"\n(сейчас: «{pc.old_value}»)"
                orig_question += "\n→ ✅ Да"
                await vk.edit_message_keyboard(peer_id, cmid, orig_question)
        elif ok:
            result_text = "✅ Сохранено."
        else:
            result_text = "Это уже обработано."
        await vk.send_message(vk_id, result_text)

    elif action == "confirm_no" and confirm_id:
        from models import PendingConfirmation as PC
        from memory import reject_pending

        # Читаем факт ДО удаления
        pc_result = await session.execute(
            select(PC).where(PC.id == confirm_id, PC.pet_id == user.active_pet_id)
        )
        pc = pc_result.scalar_one_or_none()

        rejected = await reject_pending(session, user.active_pet_id, confirm_id)
        if rejected and pc:
            result_text = f"❌ Не сохранено: {pc.key} = «{pc.new_value}»."
            # Деактивируем кнопки в исходном сообщении
            if peer_id and cmid:
                orig_question = f"📝 Сохранить: {pc.key} = «{pc.new_value}»?"
                if pc.old_value:
                    orig_question += f"\n(сейчас: «{pc.old_value}»)"
                orig_question += "\n→ ❌ Нет"
                await vk.edit_message_keyboard(peer_id, cmid, orig_question)
        elif rejected:
            result_text = "❌ Не сохранено."
        else:
            result_text = "Это уже обработано."
        await vk.send_message(vk_id, result_text)
