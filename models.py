"""SQLAlchemy ORM models."""
from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Boolean, Float, DateTime, Date,
    ForeignKey, Text, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    vk_id = Column(Integer, unique=True, nullable=False)
    active_pet_id = Column(Integer, ForeignKey("pets.id", use_alter=True, name="fk_active_pet"), nullable=True)
    plan = Column(String, default="trial")          # trial | free | paid
    trial_until = Column(DateTime, nullable=True)
    created = Column(DateTime, default=datetime.utcnow)

    pets = relationship("Pet", foreign_keys="Pet.user_id", back_populates="user", lazy="selectin")
    active_pet = relationship("Pet", foreign_keys=[active_pet_id], post_update=True, lazy="selectin")
    state = relationship("UserState", back_populates="user", uselist=False, lazy="selectin")
    usage_logs = relationship("UsageLog", back_populates="user", lazy="selectin")
    chat_messages = relationship("ChatMessage", back_populates="user", lazy="selectin")


class UserState(Base):
    __tablename__ = "user_states"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    state = Column(String, nullable=True)       # awaiting_pet_name | awaiting_pet_species |
                                                # awaiting_confirm_reset | awaiting_remind_text |
                                                # awaiting_remind_time | awaiting_confirm_fact
    state_data = Column(Text, nullable=True)    # JSON blob
    updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="state")


class Pet(Base):
    __tablename__ = "pets"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    species = Column(String, nullable=False)
    breed = Column(String, nullable=True)
    sex = Column(String, nullable=True)
    birth_date = Column(Date, nullable=True)
    created = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", foreign_keys=[user_id], back_populates="pets")
    memories = relationship("Memory", back_populates="pet", cascade="all, delete-orphan", lazy="selectin")
    chat_messages = relationship("ChatMessage", back_populates="pet", cascade="all, delete-orphan", lazy="selectin")
    pending_confirmations = relationship("PendingConfirmation", back_populates="pet", cascade="all, delete-orphan", lazy="selectin")
    reminders = relationship("Reminder", back_populates="pet", cascade="all, delete-orphan", lazy="selectin")


class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True)
    pet_id = Column(Integer, ForeignKey("pets.id"), nullable=False)
    category = Column(String, nullable=False)   # basic/health/food/behavior/toys/home/documents/appearance/history
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    sensitive = Column(Boolean, default=False)  # True for chip_id, allergy, weight, sex
    confidence = Column(Float, default=1.0)
    updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    pet = relationship("Pet", back_populates="memories")

    __table_args__ = (UniqueConstraint("pet_id", "category", "key", name="uq_memory_pet_cat_key"),)


class PendingConfirmation(Base):
    __tablename__ = "pending_confirmations"

    id = Column(Integer, primary_key=True)
    pet_id = Column(Integer, ForeignKey("pets.id"), nullable=False)
    category = Column(String, nullable=False)
    key = Column(String, nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=False)
    created = Column(DateTime, default=datetime.utcnow)

    pet = relationship("Pet", back_populates="pending_confirmations")


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True)
    pet_id = Column(Integer, ForeignKey("pets.id"), nullable=False)
    text = Column(String, nullable=False)
    next_fire = Column(DateTime, nullable=False)
    repeat_rule = Column(String, nullable=True)   # once | monthly | yearly
    active = Column(Boolean, default=True)

    pet = relationship("Pet", back_populates="reminders")


class UsageLog(Base):
    __tablename__ = "usage_log"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(Date, nullable=False)
    messages_count = Column(Integer, default=0)
    media_count = Column(Integer, default=0)

    user = relationship("User", back_populates="usage_logs")

    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_usage_user_date"),)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pet_id = Column(Integer, ForeignKey("pets.id"), nullable=True)
    role = Column(String, nullable=False)   # user | model
    content = Column(Text, nullable=False)
    created = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chat_messages")
    pet = relationship("Pet", back_populates="chat_messages")


class ApiKeyEvent(Base):
    """Лог событий API-ключей: ошибки, failover, исчерпание пула."""
    __tablename__ = "api_key_events"

    id = Column(Integer, primary_key=True)
    service = Column(String, nullable=False)    # gemini / groq / tavily / gigachat
    key_name = Column(String, nullable=False)   # напр. GROQ_API_KEY_550953 или ALL
    status = Column(String, nullable=False)     # error_429 / error_auth / error_server /
                                                # error_network / error_unknown / all_keys_exhausted
    error_text = Column(Text, nullable=True)
    created = Column(DateTime, default=datetime.utcnow)


class GigaChatUsage(Base):
    """
    Учёт использования GigaChat (fallback LLM) — годовой freemium-лимит
    физлиц не самовозобновляемый помесячно, поэтому важно знать остаток
    заранее, а не по факту отказа API.
    """
    __tablename__ = "gigachat_usage"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    model = Column(String, nullable=False)          # GigaChat-2 / GigaChat-2-Pro и т.д.
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    requests = Column(Integer, nullable=False, default=0)
    created = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("date", "model", name="uq_gigachat_usage_date_model"),)
