"""Download VK attachments and validate size limits."""
import logging
from typing import Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# Size limits before passing to Gemini
MAX_PHOTO_BYTES = 5 * 1024 * 1024       # 5 MB
MAX_AUDIO_BYTES = 20 * 1024 * 1024      # ~20 MB (~2 min OGG/Opus)
MAX_DOC_BYTES = 2 * 1024 * 1024         # 2 MB

AUDIO_MIME = "audio/ogg"
PHOTO_MIME = "image/jpeg"

# VK MIME type mapping
DOC_MIME_MAP = {
    "pdf": "application/pdf",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
}


async def _download(url: str, max_bytes: int, session: aiohttp.ClientSession) -> Optional[bytes]:
    """Download URL and return bytes if within size limit."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                return None  # too large
            data = await resp.read()
            if len(data) > max_bytes:
                return None  # too large
            return data
    except Exception as e:
        logger.warning("Download error %s: %s", url, e)
        return None


async def get_photo(attachment: dict, session: aiohttp.ClientSession) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Extract an appropriately-sized photo URL from a VK attachment and download it.
    Returns (bytes, mime_type, error_message).

    Size strategy: Gemini tiles images at 768×768px (≈258 tokens/tile).
    A 4000×3000px image needs ~20 tiles (5 160 tokens); a 1080×810px image
    needs only 2-3 tiles (516-774 tokens). We cap at ~1280px wide — enough
    to read labels/text clearly, without blowing up the input token budget.
    """
    photo = attachment.get("photo", {})
    sizes = photo.get("sizes", [])
    if not sizes:
        return None, None, "Не удалось получить фото."

    # Sort ascending by width
    sorted_sizes = sorted(sizes, key=lambda s: s.get("width", 0))

    # Pick the smallest size whose width is ≥ 1024px (readable text on labels).
    # If none is that large, fall back to the largest available.
    MAX_WIDTH = 1280
    MIN_READABLE = 1024
    best = None
    for s in sorted_sizes:
        w = s.get("width", 0)
        if w >= MIN_READABLE:
            best = s
            break  # first one ≥ 1024px — smallest acceptable
    if best is None:
        best = sorted_sizes[-1]  # largest available (all < 1024px)
    # Never go above MAX_WIDTH — if the chosen size is too large, pick the
    # largest one that is still ≤ MAX_WIDTH (or keep current if none smaller).
    candidates_in_range = [s for s in sorted_sizes if s.get("width", 0) <= MAX_WIDTH]
    if candidates_in_range:
        candidate = candidates_in_range[-1]  # largest ≤ MAX_WIDTH
        if candidate.get("width", 0) >= MIN_READABLE:
            best = candidate

    url = best.get("url")
    if not url:
        return None, None, "Не удалось получить ссылку на фото."

    data = await _download(url, MAX_PHOTO_BYTES, session)
    if data is None:
        return None, None, "⚠️ Фото слишком большое (максимум 5 МБ). Попробуй прислать меньший файл."

    logger.debug("Фото загружено: %dx%d px, %.1f KB",
                 best.get("width", 0), best.get("height", 0), len(data) / 1024)
    return data, PHOTO_MIME, None


async def get_audio_message(attachment: dict, session: aiohttp.ClientSession) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Download voice message from VK.
    Returns (bytes, mime_type, error_message).
    """
    audio_msg = attachment.get("audio_message", {})
    url = audio_msg.get("link_ogg") or audio_msg.get("link_mp3")
    if not url:
        return None, None, "Не удалось получить голосовое сообщение."

    mime = AUDIO_MIME if "ogg" in url else "audio/mpeg"
    data = await _download(url, MAX_AUDIO_BYTES, session)
    if data is None:
        return None, None, "⚠️ Голосовое слишком длинное (максимум ~2 минуты). Попробуй записать короче."

    return data, mime, None


async def get_document(attachment: dict, session: aiohttp.ClientSession) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Download document (PDF, image, etc.) from VK.
    Returns (bytes, mime_type, error_message).
    """
    doc = attachment.get("doc", {})
    url = doc.get("url")
    ext = (doc.get("ext") or "").lower()
    title = doc.get("title", "документ")

    if not url:
        return None, None, "Не удалось получить документ."

    mime = DOC_MIME_MAP.get(ext, "application/octet-stream")
    if mime == "application/octet-stream":
        return None, None, f"⚠️ Формат файла «{ext}» не поддерживается. Пришли PDF или изображение."

    data = await _download(url, MAX_DOC_BYTES, session)
    if data is None:
        return None, None, f"⚠️ Документ «{title}» слишком большой (максимум 2 МБ)."

    return data, mime, None


async def extract_media(message: dict, session: aiohttp.ClientSession) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Extract first usable media from VK message attachments.
    Returns (bytes, mime_type, error_message).
    Skips photos — use extract_all_photos() for photo batches.
    """
    attachments = message.get("attachments", [])
    for att in attachments:
        att_type = att.get("type")
        if att_type == "audio_message":
            return await get_audio_message(att, session)
        elif att_type == "doc":
            return await get_document(att, session)
    return None, None, None


_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}


async def extract_all_photos(
    message: dict, session: aiohttp.ClientSession
) -> Tuple[list, Optional[str]]:
    """
    Download ALL photo/image attachments from a VK message in arrival order.
    Returns (items, error_message) where items = list of (bytes, mime_type).

    Handles two VK attachment types:
    - type='photo'  — standard photo (sent via VK gallery / clipboard)
    - type='doc'    — file dragged from OS folder; treated as photo when
                      its extension is an image format (jpg/jpeg/png/gif/webp/bmp)

    При ошибке скачивания одного фото — пропускаем его (warning в лог),
    остальные фото продолжаем обрабатывать.  Возвращаем ошибку только если
    вообще ни одно фото не скачалось.
    """
    attachments = message.get("attachments", [])

    # Collect photo-type and image-doc-type attachments in order
    candidate_atts = []
    for att in attachments:
        t = att.get("type")
        if t == "photo":
            candidate_atts.append(("photo", att))
        elif t == "doc":
            ext = (att.get("doc", {}).get("ext") or "").lower()
            if ext in _IMAGE_EXTS:
                candidate_atts.append(("doc", att))

    logger.info(
        "extract_all_photos: найдено %d изображений в сообщении (%d photo + %d image-doc)",
        len(candidate_atts),
        sum(1 for k, _ in candidate_atts if k == "photo"),
        sum(1 for k, _ in candidate_atts if k == "doc"),
    )

    if not candidate_atts:
        return [], None

    items: list = []
    last_err: Optional[str] = None
    for idx, (kind, att) in enumerate(candidate_atts, start=1):
        if kind == "photo":
            data, mime, err = await get_photo(att, session)
        else:
            data, mime, err = await get_document(att, session)

        if err:
            logger.warning(
                "extract_all_photos: изображение %d/%d (%s) — ошибка: %s",
                idx, len(candidate_atts), kind, err,
            )
            last_err = err
            continue  # пропускаем, не прерываем весь батч
        if data and mime:
            logger.info(
                "extract_all_photos: изображение %d/%d (%s) — скачано %.1f KB",
                idx, len(candidate_atts), kind, len(data) / 1024,
            )
            items.append((data, mime))

    if not items and last_err:
        # Все фото упали — только тогда возвращаем ошибку
        return [], last_err

    logger.info(
        "extract_all_photos: итого успешно скачано %d из %d изображений",
        len(items), len(candidate_atts),
    )
    return items, None


def has_photos(message: dict) -> bool:
    """Return True if the message contains at least one photo or image-doc attachment."""
    for att in message.get("attachments", []):
        t = att.get("type")
        if t == "photo":
            return True
        if t == "doc":
            ext = (att.get("doc", {}).get("ext") or "").lower()
            if ext in _IMAGE_EXTS:
                return True
    return False


def has_non_photo_media(message: dict) -> bool:
    """Return True if the message contains audio_message or non-image doc attachments."""
    for att in message.get("attachments", []):
        t = att.get("type")
        if t == "audio_message":
            return True
        if t == "doc":
            ext = (att.get("doc", {}).get("ext") or "").lower()
            if ext not in _IMAGE_EXTS:  # image-docs handled by extract_all_photos
                return True
    return False
