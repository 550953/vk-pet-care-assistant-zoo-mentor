"""
Groq Whisper — транскрибирование голосовых сообщений из VK.

Модель: whisper-large-v3-turbo (скорость > точность)
Лимиты free tier: 20 req/min, 2000/day, 7200 сек аудио/час, 28800 сек/day
"""
import logging
from typing import Optional, Tuple

import aiohttp

from key_pool import KeyPool

logger = logging.getLogger(__name__)

_pool = KeyPool("groq", blackout_min=60.0)  # 20 req/min → cooldown 60 с при 429

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3-turbo"


def init_groq_pool(secrets: dict) -> None:
    """Инициализирует пул Groq ключей из словаря секретов Infisical."""
    _pool.init(secrets, "GROQ_API_KEY_")


def is_groq_available() -> bool:
    return not _pool.is_empty()


async def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
    current_vk_id: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Расшифровывает аудио через Groq Whisper.

    Возвращает (transcript, error_message).
    При успехе error_message = None.
    При ошибке transcript = None, error_message содержит причину.
    """
    if _pool.is_empty():
        logger.debug("Groq: пул пуст — пропускаем транскрибирование")
        return None, "Groq не настроен"

    overloaded, wait_secs = _pool.is_overloaded()
    if overloaded:
        logger.debug("Groq: blackout активен (%.0f с)", wait_secs)
        return None, f"Groq временно недоступен"

    # Расширение файла по MIME
    if "ogg" in mime_type:
        ext, ct = "ogg", "audio/ogg"
    elif "mpeg" in mime_type or "mp3" in mime_type:
        ext, ct = "mp3", "audio/mpeg"
    elif "mp4" in mime_type or "m4a" in mime_type:
        ext, ct = "m4a", "audio/mp4"
    elif "wav" in mime_type:
        ext, ct = "wav", "audio/wav"
    else:
        ext, ct = "ogg", "audio/ogg"  # VK default

    n = len(_pool.keys)
    for _ in range(n):
        idx, api_key = _pool.get_next_key()
        if api_key is None:
            break

        try:
            form = aiohttp.FormData()
            form.add_field("model", WHISPER_MODEL)
            form.add_field("language", "ru")
            form.add_field("response_format", "text")
            form.add_field(
                "file",
                audio_bytes,
                filename=f"voice.{ext}",
                content_type=ct,
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_TRANSCRIPTION_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429:
                        _pool.mark_error(
                            idx, "error_429",
                            "Groq rate limit exceeded",
                            cooldown=65.0,
                            current_vk_id=current_vk_id,
                        )
                        continue

                    if resp.status == 401:
                        _pool.mark_error(
                            idx, "error_auth",
                            "Invalid Groq API key",
                            cooldown=3600.0,
                            current_vk_id=current_vk_id,
                        )
                        continue

                    if resp.status >= 500:
                        error_body = await resp.text()
                        _pool.mark_error(
                            idx, "error_server",
                            f"Groq server error {resp.status}: {error_body[:200]}",
                            cooldown=30.0,
                            current_vk_id=current_vk_id,
                        )
                        continue

                    resp.raise_for_status()
                    text = (await resp.text()).strip()

                    if not text:
                        return None, "Голосовое сообщение пустое или не распознано"

                    logger.info(
                        "Groq: транскрибировано %d байт → %d символов (ключ %s)",
                        len(audio_bytes), len(text), _pool.key_names[idx],
                    )
                    return text, None

        except aiohttp.ClientError as e:
            logger.warning("Groq: сетевая ошибка (ключ #%d): %s", idx, e)
            _pool.mark_error(
                idx, "error_network", str(e),
                cooldown=15.0,
                current_vk_id=current_vk_id,
            )
        except Exception as e:
            logger.error("Groq: неожиданная ошибка (ключ #%d): %s", idx, e)
            _pool.mark_error(
                idx, "error_unknown", str(e),
                cooldown=10.0,
                current_vk_id=current_vk_id,
            )

    logger.warning("Groq: все ключи перебраны — транскрибирование недоступно")
    return None, None  # тихий fallback: передаём аудио в Gemini
