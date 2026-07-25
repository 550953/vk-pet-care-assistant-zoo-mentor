"""Emergency triage detector and state tracking."""
import logging
from datetime import datetime, timedelta
from typing import Optional

from llm import classify_triage

# Быстрый keyword-фильтр — LLM вызывается только если хотя бы одно слово совпало.
# Для обычных вопросов ("разбери состав", "привет") LLM-вызов полностью пропускается.
_TRIAGE_KEYWORDS = (
    "умира", "не дыш", "задыха", "судорог", "припадок", "обморок",
    "без сознани", "кровь", "крови", "кровотеч", "рвот", "рвёт", "рвет",
    "не ест", "не пьёт", "не пьет", "отказыва", "отравил", "яд", "токсич",
    "сломал", "перелом", "травм", "упал", "ударил", "не встаёт", "не встает",
    "хрипит", "задыхается", "синеет", "распухл", "опухол", "опух",
    "укусил", "ужалил", "ожог", "срочно", "экстренно", "помогите", "помоги",
    "скорую", "ветеринар", "клиник", "вызов", "скорая",
)

def _needs_triage_llm(text: str) -> bool:
    """True если текст содержит тревожные слова — тогда нужен LLM-триаж."""
    t = text.lower()
    return any(kw in t for kw in _TRIAGE_KEYWORDS)

logger = logging.getLogger(__name__)

# In-memory emergency state per (user_id, pet_id).
# Stored as {(user_id, pet_id): {"active": bool, "started": datetime, "followup_sent": bool}}
_emergency_states: dict[tuple[int, int], dict] = {}

EMERGENCY_TIMEOUT = timedelta(hours=4)


def is_emergency_active(user_id: int, pet_id: int) -> bool:
    key = (user_id, pet_id)
    state = _emergency_states.get(key)
    if not state:
        return False
    if datetime.utcnow() - state["started"] > EMERGENCY_TIMEOUT:
        _emergency_states.pop(key, None)
        return False
    return state["active"]


def set_emergency(user_id: int, pet_id: int, active: bool) -> None:
    key = (user_id, pet_id)
    if active:
        _emergency_states[key] = {
            "active": True,
            "started": datetime.utcnow(),
            "followup_sent": False,
        }
    else:
        _emergency_states.pop(key, None)


def should_send_followup(user_id: int, pet_id: int) -> bool:
    """Return True only once per emergency episode."""
    key = (user_id, pet_id)
    state = _emergency_states.get(key)
    if not state:
        return False
    if not state["followup_sent"]:
        state["followup_sent"] = True
        return True
    return False


EMERGENCY_RESPONSE_TEMPLATE = """🚨 ЭКСТРЕННАЯ СИТУАЦИЯ!

{brief_action}

📍 Немедленно свяжитесь с круглосуточной ветеринарной клиникой!
Не ждите — каждая минута важна."""

URGENT_RESPONSE_TEMPLATE = """⚠️ {brief_action}

Рекомендую обратиться к ветеринару в течение сегодняшнего дня."""


async def analyze_triage(
    message: str,
    pet_name: str,
    species: str,
    user_id: int,
    pet_id: int,
) -> Optional[str]:
    """
    Run triage analysis. Returns a special response string if emergency/urgent,
    or None if observation (normal dialog flow).
    Also handles topic_changed to exit emergency mode.
    """
    emergency_active = is_emergency_active(user_id, pet_id)

    # Пропускаем LLM-вызов если нет ни тревожных слов, ни активного режима экстренной ситуации
    if not emergency_active and not _needs_triage_llm(message):
        return None

    result = await classify_triage(message, pet_name, species, emergency_active)

    level = result.get("level", "observation")
    topic_changed = result.get("topic_changed", False)
    brief_action = result.get("brief_action", "")

    # If was in emergency and user changed topic — exit emergency mode
    if emergency_active and topic_changed:
        set_emergency(user_id, pet_id, False)
        return None   # proceed with normal dialog

    if level == "emergency":
        set_emergency(user_id, pet_id, True)
        return EMERGENCY_RESPONSE_TEMPLATE.format(brief_action=brief_action)

    if level == "urgent":
        return URGENT_RESPONSE_TEMPLATE.format(brief_action=brief_action)

    # Observation — normal flow, but if emergency is active, return None to let
    # the main handler decide (it will add a single followup if not sent yet)
    return None
