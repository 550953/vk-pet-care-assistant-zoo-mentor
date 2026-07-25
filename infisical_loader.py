"""
Загрузка секретов из Infisical (Universal Auth).
Используется при старте бота если INFISICAL_CLIENT_ID + INFISICAL_CLIENT_SECRET заданы.
При отсутствии — падает обратно на локальные переменные окружения.
"""
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

INFISICAL_URL = "https://app.infisical.com"
PROJECT_ID = "555e71be-4c53-4b3e-9409-0d9838aea8b6"
ENVIRONMENT = "dev"           # имя окружения в Infisical
SECRET_PATH = "/"             # корневой путь секретов


async def _get_access_token(session: aiohttp.ClientSession, client_id: str, client_secret: str) -> str:
    """Universal-auth login → возвращает accessToken."""
    url = f"{INFISICAL_URL}/api/v1/auth/universal-auth/login"
    async with session.post(url, json={"clientId": client_id, "clientSecret": client_secret}) as resp:
        resp.raise_for_status()
        data = await resp.json()
        token = data.get("accessToken") or data.get("access_token")
        if not token:
            raise RuntimeError(f"Infisical: нет accessToken в ответе: {data}")
        return token


async def _list_secrets(session: aiohttp.ClientSession, token: str) -> dict[str, str]:
    """Получает все секреты проекта → {key: value}."""
    url = f"{INFISICAL_URL}/api/v3/secrets/raw"
    params = {
        "workspaceId": PROJECT_ID,
        "environment": ENVIRONMENT,
        "secretPath": SECRET_PATH,
    }
    headers = {"Authorization": f"Bearer {token}"}
    async with session.get(url, params=params, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    secrets = data.get("secrets", [])
    return {s["secretKey"]: s["secretValue"] for s in secrets if s.get("secretValue")}


async def load_infisical_secrets() -> Optional[dict[str, str]]:
    """
    Загружает все секреты из Infisical.
    Возвращает dict {key: value} или None если учётные данные не заданы / ошибка.
    """
    client_id = os.environ.get("INFISICAL_CLIENT_ID", "")
    client_secret = os.environ.get("INFISICAL_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        logger.info("Infisical: INFISICAL_CLIENT_ID/SECRET не заданы — используем локальные env")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            token = await _get_access_token(session, client_id, client_secret)
            secrets = await _list_secrets(session, token)
        logger.info("Infisical: загружено %d секретов", len(secrets))
        return secrets
    except Exception as e:
        logger.error("Infisical: ошибка загрузки секретов: %s", e)
        return None


def extract_gemini_keys(secrets: dict[str, str]) -> list[str]:
    """Вытаскивает все GEMINI_API_KEY_* из словаря секретов."""
    keys = [v for k, v in secrets.items() if k.startswith("GEMINI_API_KEY_") and v]
    # дедупликация значений
    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def extract_groq_keys(secrets: dict[str, str]) -> list[tuple[str, str]]:
    """
    Вытаскивает все GROQ_API_KEY_* из словаря секретов.
    Возвращает [(key_name, key_value)] — имя нужно для логирования ошибок.
    """
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for k, v in secrets.items():
        if k.startswith("GROQ_API_KEY_") and v:
            val = v.strip()
            if val and val not in seen:
                seen.add(val)
                result.append((k, val))
    return result


def extract_tavily_keys(secrets: dict[str, str]) -> list[tuple[str, str]]:
    """
    Вытаскивает все TAVILY_API_KEY_* из словаря секретов.
    Возвращает [(key_name, key_value)] — имя нужно для логирования ошибок.
    """
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for k, v in secrets.items():
        if k.startswith("TAVILY_API_KEY_") and v:
            val = v.strip()
            if val and val not in seen:
                seen.add(val)
                result.append((k, val))
    return result
