"""Async SQLAlchemy engine and session factory with WAL mode."""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import event, text
from models import Base

DATABASE_URL = "sqlite+aiosqlite:///zoo_mentor.db"

engine = create_async_engine(
    DATABASE_URL,
    connect_args={"timeout": 30},
    echo=False,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create tables and set SQLite WAL mode + synchronous=NORMAL."""
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
