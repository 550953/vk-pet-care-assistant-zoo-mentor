"""
Tavily Web Search — живой веб-поиск для актуальных данных.

Используется в двух режимах:
1. При явном намерении web_search_needed (отзывы, наличие товара).
2. Автоматически при токсикологических сообщениях (отравление, яд, токсины) —
   независимый триггер, не через общий intent.

НЕ вызывается на каждый запрос.
"""
import logging
import re
from typing import Optional

import aiohttp

from key_pool import KeyPool

logger = logging.getLogger(__name__)

_pool = KeyPool("tavily", blackout_min=30.0)

TAVILY_URL = "https://api.tavily.com/search"

# ─── Фильтр качества источников для токсикологии ─────────────────────────────
# Используем ТОЛЬКО ветеринарные и научные ресурсы.
# Не цитируем: Reddit, YouTube, блоги, магазины, соцсети.
_TRUSTED_VET_DOMAINS = (
    "aspca.org",
    "petmd.com",
    "vcahospitals.com",
    "avma.org",
    "merckvetmanual.com",
    "vet.cornell.edu",
    "banfield.com",
    "vetstreet.com",
    "bluepearlvet.com",
    "animalpoisoncontrol.aspca.org",
    "poison.org",
)

_TRUSTED_URL_MARKERS = (
    "vet", "clinic", "клиник", "ветеринар", "animal", "pet", "hospital",
    "аспка", "aspca", "avma", "merck",
)

_UNTRUSTED_DOMAINS = (
    "reddit.com", "youtube.com", "youtu.be", "instagram.com",
    "vk.com", "t.me", "telegram", "facebook.com", "twitter.com",
    "pinterest.com", "tiktok.com", "wildberries.ru", "ozon.ru",
    "aliexpress.com", "amazon.com",
)


def _is_trusted_vet_source(url: str, title: str) -> bool:
    """Возвращает True если источник достаточно авторитетен для токсикологии."""
    url_lower = url.lower()
    title_lower = title.lower()

    # Явно ненадёжные — блокируем
    for bad in _UNTRUSTED_DOMAINS:
        if bad in url_lower:
            return False

    # Прямо в списке доверенных доменов
    for trusted in _TRUSTED_VET_DOMAINS:
        if trusted in url_lower:
            return True

    # Маркеры ветеринарной тематики в URL или заголовке
    combined = url_lower + " " + title_lower
    matches = sum(1 for marker in _TRUSTED_URL_MARKERS if marker in combined)
    return matches >= 2  # требуем хотя бы 2 совпадения для неочевидных доменов


def init_tavily_pool(secrets: dict) -> None:
    """Инициализирует пул Tavily ключей из словаря секретов Infisical."""
    _pool.init(secrets, "TAVILY_API_KEY_")


def is_tavily_available() -> bool:
    return not _pool.is_empty()


async def search(
    query: str,
    max_results: int = 5,
    current_vk_id: Optional[int] = None,
    toxicology_mode: bool = False,
) -> Optional[str]:
    """
    Выполняет веб-поиск через Tavily и возвращает форматированный текст-контекст.

    toxicology_mode=True включает фильтр качества источников:
    цитируются только ветеринарные клиники, ASPCA, научные базы.
    Если ни одного подходящего источника нет — возвращает None (не цитируем ничего).

    Возвращает None при ошибке или если Tavily не настроен.
    Результат вставляется в системный промпт как дополнительный контекст.
    """
    if _pool.is_empty():
        logger.debug("Tavily: пул пуст — поиск пропускаем")
        return None

    overloaded, _ = _pool.is_overloaded()
    if overloaded:
        logger.debug("Tavily: blackout активен")
        return None

    n = len(_pool.keys)
    for _ in range(n):
        idx, api_key = _pool.get_next_key()
        if api_key is None:
            break

        try:
            payload = {
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    TAVILY_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 401:
                        _pool.mark_error(
                            idx, "error_auth",
                            "Invalid Tavily API key",
                            cooldown=3600.0,
                            current_vk_id=current_vk_id,
                        )
                        continue

                    if resp.status == 429:
                        _pool.mark_error(
                            idx, "error_429",
                            "Tavily rate limit exceeded",
                            cooldown=60.0,
                            current_vk_id=current_vk_id,
                        )
                        continue

                    if resp.status >= 500:
                        body = await resp.text()
                        _pool.mark_error(
                            idx, "error_server",
                            f"Tavily server {resp.status}: {body[:200]}",
                            cooldown=30.0,
                            current_vk_id=current_vk_id,
                        )
                        continue

                    resp.raise_for_status()
                    data = await resp.json()

            answer = (data.get("answer") or "").strip()
            results = data.get("results", [])

            if toxicology_mode:
                # ── Фильтр качества источников для токсикологии ───────────────
                # Используем только ветеринарные / научные источники.
                # Если ни одного подходящего — не возвращаем ничего.
                trusted_results = [
                    r for r in results
                    if _is_trusted_vet_source(
                        r.get("url", ""),
                        r.get("title", ""),
                    )
                ]

                if not trusted_results:
                    logger.info(
                        "Tavily токсикология: запрос «%s...» → нет доверенных источников "
                        "среди %d результатов — не цитируем", query[:50], len(results),
                    )
                    return None  # Лучше без источника, чем с сомнительным

                parts: list[str] = []
                # Краткий ответ Tavily — если есть
                if answer:
                    parts.append(f"Краткий ответ из поиска: {answer}")

                for r in trusted_results[:2]:  # не больше 2 проверенных источников
                    title = (r.get("title") or "").strip()
                    content = (r.get("content") or "").strip()[:400]
                    url = (r.get("url") or "").strip()
                    if title or content:
                        line = f"• {title}: {content}"
                        if url:
                            line += f"\n  Источник: {url}"
                        parts.append(line)

                if not parts:
                    return None

                result_text = "\n\n".join(parts)
                logger.info(
                    "Tavily токсикология: запрос «%s...» → %d доверенных источников из %d (ключ %s)",
                    query[:50], len(trusted_results), len(results), _pool.key_names[idx],
                )
                return result_text

            else:
                # ── Стандартный режим (не токсикология) — без фильтра ─────────
                parts: list[str] = []
                if answer:
                    parts.append(f"Краткий ответ из поиска: {answer}")

                for r in results[:3]:
                    title = (r.get("title") or "").strip()
                    content = (r.get("content") or "").strip()[:400]
                    url = (r.get("url") or "").strip()
                    if title or content:
                        line = f"• {title}: {content}"
                        if url:
                            line += f"\n  Источник: {url}"
                        parts.append(line)

                if not parts:
                    return None

                result_text = "\n\n".join(parts)
                logger.info(
                    "Tavily: запрос «%s...» → %d результат(ов) (ключ %s)",
                    query[:50], len(results), _pool.key_names[idx],
                )
                return result_text

        except aiohttp.ClientError as e:
            logger.warning("Tavily: сетевая ошибка (ключ #%d): %s", idx, e)
            _pool.mark_error(idx, "error_network", str(e), cooldown=15.0)
        except Exception as e:
            logger.error("Tavily: неожиданная ошибка (ключ #%d): %s", idx, e)
            _pool.mark_error(idx, "error_unknown", str(e), cooldown=10.0)

    logger.warning("Tavily: все ключи перебраны — поиск недоступен")
    return None
