"""
Universal key pool — round-robin, per-key cooldown, failover notifications, DB logging.
Используется Gemini, Groq и Tavily для единообразной обработки ошибок ключей.
"""
import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable, Any

from logging_setup import log_event

logger = logging.getLogger(__name__)

# ─── Глобальные колбэки, устанавливаются из main.py после старта ─────────────

# async (vk_id: int, service: str) -> None
_on_failover_cb: Optional[Callable[..., Awaitable[None]]] = None
# async (service: str) -> None
_on_exhausted_cb: Optional[Callable[..., Awaitable[None]]] = None
# async (service: str, key_name: str, status: str, error_text: str) -> None
_db_log_cb: Optional[Callable[..., Awaitable[None]]] = None


def set_callbacks(
    on_failover: Optional[Callable] = None,
    on_exhausted: Optional[Callable] = None,
    db_log: Optional[Callable] = None,
) -> None:
    """Вызывается из main.py после инициализации VK и БД."""
    global _on_failover_cb, _on_exhausted_cb, _db_log_cb
    if on_failover is not None:
        _on_failover_cb = on_failover
    if on_exhausted is not None:
        _on_exhausted_cb = on_exhausted
    if db_log is not None:
        _db_log_cb = db_log


def _fire(coro_or_none: Any) -> None:
    """Безопасно запускает корутину в фоне, если есть цикл событий."""
    if coro_or_none is None:
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro_or_none)
    except Exception as e:
        logger.debug("_fire error: %s", e)


class KeyPool:
    """
    Round-robin пул API-ключей с автоматическим failover.

    Паттерн совпадает с Gemini pool в llm.py:
    - per-key cooldown (_blocked[idx]) при ошибке конкретного ключа
    - глобальный blackout (blackout_until) когда все ключи заблокированы
    - при переключении на резерв → мягкое сообщение пользователю
    - при исчерпании всех ключей → CRITICAL лог + уведомление администратору
    """

    def __init__(self, service: str, blackout_min: float = 30.0):
        self.service = service
        self.blackout_min = blackout_min
        self.keys: list[str] = []
        self.key_names: list[str] = []      # имена секретов (GROQ_API_KEY_550953 и т.п.)
        self.blocked: list[float] = []      # monotonic time до которого ключ на cooldown
        self.blackout_until: float = 0.0
        self.rr_index: int = 0

    def init(self, secrets: dict, prefix: str) -> None:
        """Инициализирует пул из словаря секретов по заданному префиксу."""
        seen: set[str] = set()
        keys: list[str] = []
        names: list[str] = []
        for k, v in secrets.items():
            if k.startswith(prefix) and isinstance(v, str):
                val = v.strip()
                if val and val not in seen:
                    seen.add(val)
                    keys.append(val)
                    names.append(k)
                    logger.info("%s pool: %s — добавлен ✓", self.service, k)
        self.keys = keys
        self.key_names = names
        self.blocked = [0.0] * len(keys)
        self.blackout_until = 0.0
        self.rr_index = 0
        logger.info("%s pool: инициализирован, %d ключ(ей)", self.service, len(keys))

    def is_empty(self) -> bool:
        return not self.keys

    def is_overloaded(self) -> tuple[bool, float]:
        now = time.monotonic()
        if now < self.blackout_until:
            return True, self.blackout_until - now
        return False, 0.0

    def get_next_key(self) -> tuple[Optional[int], Optional[str]]:
        """
        Возвращает (idx, api_key) следующего доступного ключа.
        Возвращает (None, None) если все ключи на cooldown.
        """
        if not self.keys:
            return None, None
        n = len(self.keys)
        now = time.monotonic()
        for _ in range(n):
            idx = self.rr_index % n
            self.rr_index += 1
            if now >= self.blocked[idx]:
                return idx, self.keys[idx]
        return None, None

    def mark_error(
        self,
        idx: int,
        error_type: str,
        error_text: str,
        cooldown: float = 0.0,
        current_vk_id: Optional[int] = None,
    ) -> bool:
        """
        Фиксирует ошибку ключа idx.

        Возвращает True если есть резервный ключ, False если все исчерпаны.
        Запускает фоновое логирование в БД и уведомление пользователя/администратора.
        """
        key_name = self.key_names[idx] if idx < len(self.key_names) else f"key_{idx}"

        # Логируем событие в БД (фоново, не блокируем)
        if _db_log_cb:
            _fire(_db_log_cb(self.service, key_name, error_type, error_text[:500]))

        if cooldown > 0:
            self.blocked[idx] = time.monotonic() + cooldown
            log_event(
                logger, logging.WARNING, "key_marked_error",
                service=self.service, key_name=key_name, error_type=error_type,
                cooldown_s=round(cooldown, 0),
            )

        # Проверяем: есть ли хоть один незаблокированный ключ?
        now = time.monotonic()
        available = [i for i in range(len(self.keys)) if now >= self.blocked[i]]

        if available:
            # Есть резервный — уведомляем пользователя мягко
            log_event(
                logger, logging.WARNING, "key_failover",
                service=self.service, old_key=key_name,
                new_key=self.key_names[available[0]] if available[0] < len(self.key_names) else None,
                remaining_keys=len(available),
            )
            if _on_failover_cb and current_vk_id:
                _fire(_on_failover_cb(current_vk_id, self.service))
            return True
        else:
            # Все ключи исчерпаны — критическая ситуация
            new_blackout = max(self.blocked) if self.blocked else 0.0
            if new_blackout > self.blackout_until:
                self.blackout_until = new_blackout
            log_event(
                logger, logging.CRITICAL, "keys_exhausted",
                service=self.service, total_keys=len(self.keys),
                blackout_s=round(max(0.0, self.blackout_until - now), 0),
            )
            # Логируем критическое событие
            if _db_log_cb:
                _fire(_db_log_cb(
                    self.service, "ALL", "all_keys_exhausted",
                    f"All {len(self.keys)} key(s) are blocked for service {self.service}",
                ))
            # Уведомляем администратора
            if _on_exhausted_cb:
                _fire(_on_exhausted_cb(self.service))
            return False
