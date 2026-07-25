"""APScheduler: fire reminders, check VK message_allow window."""
import logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import Callable, Awaitable

MSK = ZoneInfo("Europe/Moscow")

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from models import Reminder, Pet, User

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None
_send_fn: Callable[[int, str], Awaitable[None]] | None = None
_session_factory = None


def setup_scheduler(session_factory, send_fn: Callable[[int, str], Awaitable[None]]) -> AsyncIOScheduler:
    global _scheduler, _send_fn, _session_factory
    _session_factory = session_factory
    _send_fn = send_fn
    _scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
    _scheduler.add_job(
        _fire_reminders,
        trigger="interval",
        minutes=1,
        id="fire_reminders",
        replace_existing=True,
    )
    return _scheduler


async def _fire_reminders() -> None:
    if _session_factory is None or _send_fn is None:
        return
    async with _session_factory() as session:
        now = datetime.now(MSK).replace(tzinfo=None)  # naive MSK — matches how parser stores times
        result = await session.execute(
            select(Reminder)
            .where(and_(Reminder.active == True, Reminder.next_fire <= now))
        )
        reminders = result.scalars().all()

        for rem in reminders:
            pet_result = await session.execute(select(Pet).where(Pet.id == rem.pet_id))
            pet = pet_result.scalar_one_or_none()
            if not pet:
                rem.active = False
                continue

            user_result = await session.execute(select(User).where(User.id == pet.user_id))
            user = user_result.scalar_one_or_none()
            if not user:
                rem.active = False
                continue

            text = f"🔔 Напоминание для {pet.name}: {rem.text}"
            try:
                await _send_fn(user.vk_id, text)
            except Exception as e:
                logger.warning("Failed to send reminder to vk_id=%s: %s", user.vk_id, e)

            # Schedule next fire or deactivate
            if rem.repeat_rule == "monthly":
                rem.next_fire = rem.next_fire + timedelta(days=30)
            elif rem.repeat_rule == "yearly":
                rem.next_fire = rem.next_fire + timedelta(days=365)
            else:
                rem.active = False

        await session.commit()
