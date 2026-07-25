"""Async SQLAlchemy engine and session factory for PostgreSQL (Supabase)."""
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from models import Base

_engine = None
_factory = None


def async_session():
    """Return a new AsyncSession context manager."""
    if _factory is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _factory()


async def init_db() -> None:
    """Initialise engine from DATABASE_URL env var and create tables if missing."""
    global _engine, _factory

    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    # SQLAlchemy asyncpg requires postgresql+asyncpg:// scheme
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    _engine = create_async_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
        # Отключаем кеш prepared statements — требуется для Supabase Pooler
        connect_args={"prepared_statement_cache_size": 0},
    )
    _factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
