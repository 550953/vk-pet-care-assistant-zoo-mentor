"""Usage limits and subscription checks."""
import logging
from datetime import datetime, date, timedelta

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from models import User, UsageLog

logger = logging.getLogger(__name__)

# Free / trial limits per day
FREE_MESSAGES_PER_DAY = 30
FREE_MEDIA_PER_DAY = 10
TRIAL_DAYS = 7


async def ensure_user(session: AsyncSession, vk_id: int) -> User:
    """Get or create user by VK ID."""
    result = await session.execute(select(User).where(User.vk_id == vk_id))
    user = result.scalar_one_or_none()
    if not user:
        trial_until = datetime.utcnow() + timedelta(days=TRIAL_DAYS)
        user = User(vk_id=vk_id, plan="trial", trial_until=trial_until, created=datetime.utcnow())
        session.add(user)
        await session.commit()
        await session.refresh(user)
    return user


async def check_and_increment(session: AsyncSession, user: User, is_media: bool = False) -> tuple[bool, str]:
    """
    Check if user can send a message/media.
    Returns (allowed: bool, reason: str).
    Increments usage if allowed.
    """
    if user.plan == "paid":
        await _increment_usage(session, user.id, is_media)
        return True, ""

    # Trial check
    if user.plan == "trial":
        if user.trial_until and datetime.utcnow() < user.trial_until:
            await _increment_usage(session, user.id, is_media)
            return True, ""
        else:
            # Trial expired → downgrade to free
            user.plan = "free"
            await session.commit()

    # Free plan: check daily limits
    today = date.today()
    result = await session.execute(
        select(UsageLog).where(and_(UsageLog.user_id == user.id, UsageLog.date == today))
    )
    log = result.scalar_one_or_none()

    if log:
        if is_media and log.media_count >= FREE_MEDIA_PER_DAY:
            return False, (
                f"⚠️ Лимит медиафайлов на сегодня исчерпан ({FREE_MEDIA_PER_DAY}/день на бесплатном тарифе).\n"
                "Подключи подписку: /subscription"
            )
        if not is_media and log.messages_count >= FREE_MESSAGES_PER_DAY:
            return False, (
                f"⚠️ Лимит сообщений на сегодня исчерпан ({FREE_MESSAGES_PER_DAY}/день на бесплатном тарифе).\n"
                "Подключи подписку: /subscription"
            )

    await _increment_usage(session, user.id, is_media)
    return True, ""


async def _increment_usage(session: AsyncSession, user_id: int, is_media: bool) -> None:
    today = date.today()
    stmt = (
        pg_insert(UsageLog)
        .values(user_id=user_id, date=today, messages_count=0, media_count=0)
        .on_conflict_do_update(
            index_elements=["user_id", "date"],
            set_={
                "messages_count": UsageLog.messages_count + (0 if is_media else 1),
                "media_count": UsageLog.media_count + (1 if is_media else 0),
            },
        )
    )
    await session.execute(stmt)
    await session.commit()


def get_subscription_info(user: User) -> str:
    if user.plan == "paid":
        return "✅ Подписка активна. Безлимитный доступ."
    if user.plan == "trial":
        if user.trial_until and datetime.utcnow() < user.trial_until:
            days_left = (user.trial_until - datetime.utcnow()).days + 1
            return (
                f"🎁 Пробный период: осталось ~{days_left} дн.\n"
                f"После него: {FREE_MESSAGES_PER_DAY} сообщений и {FREE_MEDIA_PER_DAY} медиафайлов в день бесплатно.\n"
                "Подписка 490 ₽/мес: /pay"
            )
        return f"Пробный период истёк. Бесплатно: {FREE_MESSAGES_PER_DAY} сообщений/день. Подписка: /pay"
    return f"Бесплатный тариф: {FREE_MESSAGES_PER_DAY} сообщений и {FREE_MEDIA_PER_DAY} медиафайлов в день.\nПодписка 490 ₽/мес: /pay"
