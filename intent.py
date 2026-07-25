"""Intent classification router."""
import logging
from typing import Optional

from llm import classify_intent

logger = logging.getLogger(__name__)

# Simple keyword-based pre-filter to avoid Gemini call for obvious commands
COMMAND_PREFIXES = [
    "/start", "/addpet", "/pets", "/switch", "/me", "/reset",
    "/reminders", "/remind", "/delremind", "/subscription", "/pay",
    "/cancel_subscription", "/referral", "/skills", "/help",
    "/quiet", "/loud", "/cancel",
]


def _quick_command_check(text: str) -> Optional[str]:
    stripped = text.strip().lower()
    for cmd in COMMAND_PREFIXES:
        if stripped.startswith(cmd):
            return stripped.split()[0]
    return None


PROFILE_KEYWORDS = (
    "что знаешь", "что помнишь", "расскажи про", "профиль", "мой питомец",
    "знаешь о", "помнишь о", "что ты знаешь", "покажи профиль",
)

SCOPE_KEYWORDS = (
    "отвечай только", "не говори", "забудь про", "игнорируй", "только о",
    "не упоминай", "сосредоточься", "фокусируйся",
)

# Ключевые слова для живого веб-поиска через Tavily.
# Поиск нужен только когда пользователь явно просит найти внешние данные.
WEB_SEARCH_KEYWORDS = (
    "найди отзывы",
    "поищи отзывы",
    "есть ли отзывы",
    "отзывы на корм",
    "отзывы о корме",
    "что пишут о",
    "что говорят о",
    "поищи в интернете",
    "найди в интернете",
    "поискать в интернете",
    "актуальная информация",
    "актуальные данные",
    "можно купить",
    "где купить",
    "есть ли в продаже",
    "есть ли в наличии",
    "доступен ли",
    # Поиск цен и наличия (Баг: не срабатывал на эти формулировки)
    "найди самый дешёвый",
    "найди самый дешевый",
    "найди дешевле",
    "где дешевле",
    "где дешевле взять",
    "а где дешевле",
    "сравни цены",
    "сколько стоит",
    "сколько стоит этот",
    "почём стоит",
    "почем стоит",
    "выгоднее купить",
    "выгоднее взять",
    "дешевле купить",
    "дешевле взять",
    "самый выгодный",
    "самый дешёвый",
    "самый дешевый",
    "цены на",
    "цена на",
    "где заказать",
    "где приобрести",
)

# ─── Токсикологические ключевые слова — для автоматического триггера Engine ───
# При совпадении: активируем TOXICOLOGY_RULES в промпте + автоматический Tavily поиск.
# Независим от WEB_SEARCH_KEYWORDS — не полагаемся на общий intent.
TOXICOLOGY_KEYWORDS = (
    "отравил",
    "отравилась",
    "отравился",
    "отравление",
    "яд",
    "токсич",
    "съел шоколад",
    "съела шоколад",
    "шоколад",
    "виноград",
    "изюм",
    "лук",
    "чеснок",
    "ксилит",
    "ибупрофен",
    "парацетамол",
    "аспирин",
    "таблетк",
    "лекарств",
    "проглотил таблетку",
    "проглотила таблетку",
    "выпил таблетку",
    "выпила таблетку",
    "авокадо",
    "кофеин",
    "кофе",
    "алкоголь",
    "антифриз",
    "крысиный яд",
    "средство от крыс",
    "инсектицид",
    "пестицид",
)


def _quick_general_check(text: str) -> bool:
    """True если текст явно не команда, не профильный запрос и не scope — LLM не нужен."""
    t = text.lower()
    if any(kw in t for kw in PROFILE_KEYWORDS):
        return False
    if any(kw in t for kw in SCOPE_KEYWORDS):
        return False
    return True


def _quick_web_search_check(text: str) -> bool:
    """True если пользователь явно просит найти внешние данные."""
    t = text.lower()
    return any(kw in t for kw in WEB_SEARCH_KEYWORDS)


def is_toxicology_message(text: str) -> bool:
    """True если сообщение содержит маркеры отравления/токсикологии.

    Используется для:
    1. Активации TOXICOLOGY_RULES engine в промпте (флаг toxicology_active=True).
    2. Автоматического запуска Tavily-поиска по токсикологическому запросу
       (независимо от общего intent = web_search_needed).
    """
    t = text.lower()
    return any(kw in t for kw in TOXICOLOGY_KEYWORDS)


def extract_toxin_from_message(text: str, species: str) -> str:
    """Формирует предметный запрос для Tavily из текста сообщения.

    Возвращает строку вида «шоколад токсичность кошка ветеринария».
    """
    t = text.lower()

    # Ищем упомянутое вещество — берём первое совпадение по полному слову
    SUBSTANCES = [
        "шоколад", "виноград", "изюм", "лук", "чеснок", "ксилит",
        "ибупрофен", "парацетамол", "аспирин", "авокадо", "кофеин",
        "кофе", "алкоголь", "антифриз", "крысиный яд", "инсектицид", "пестицид",
    ]
    found_substance = None
    for sub in SUBSTANCES:
        if sub in t:
            found_substance = sub
            break

    if not found_substance:
        # Общий запрос если вещество не распознано
        found_substance = "токсин отравление"

    return f"{found_substance} токсичность {species} ветеринария"


async def classify(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Returns (intent, mapped_command, scope_restriction).
    intent: command | profile_query | scope_instruction | web_search_needed | general
    """
    if not text:
        return "general", None, None

    # Быстрая проверка на явные /команды
    quick_cmd = _quick_command_check(text)
    if quick_cmd:
        return "command", quick_cmd, None

    # Проверяем запрос на веб-поиск (до LLM — экономим квоту)
    if _quick_web_search_check(text):
        return "web_search_needed", None, None

    # Если нет ключевых слов профиля или scope → сразу "general", без LLM
    if _quick_general_check(text):
        return "general", None, None

    # LLM только для профильных запросов и scope-инструкций
    try:
        result = await classify_intent(text)
        intent = result.get("intent", "general")
        mapped_command = result.get("mapped_command")
        scope_restriction = result.get("scope_restriction")
        return intent, mapped_command, scope_restriction
    except Exception as e:
        logger.warning("Intent classification failed: %s", e)
        return "general", None, None
