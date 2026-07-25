"""VK Long Poll client: connection, message sending, keyboards, chunking."""
import asyncio
import json
import logging
import os
import random
import time
from typing import Optional, Any

import aiohttp

logger = logging.getLogger(__name__)

VK_API_VERSION = "5.199"
VK_API_BASE = "https://api.vk.com/method"
CHUNK_SIZE = 3800   # VK limit is ~4096, leave buffer

# Keyboard payloads
KBD_YES = "yes"
KBD_NO = "no"

# Dedup TTL (seconds): ignore events with the same ID seen within this window
_DEDUP_TTL = 60


def strip_vk_markdown(text: str) -> str:
    """
    Remove Markdown formatting that VK displays as raw asterisks/hashes.
    VK does NOT render Markdown — it shows **, *, ### literally.
    Applied to every outgoing message as a safety net.
    """
    import re

    # 1. Bullet "* item" at line start → "• item"  (must come before bold/italic stripping)
    text = re.sub(r"(?m)^\*\s+", "• ", text)

    # 2. Bold **text** → text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)

    # 3. Italic *text* → text
    text = re.sub(r"\*([^\*\n]+?)\*", r"\1", text)

    # 4. ATX headers ### text → text
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)

    # 5. Dash bullets "- item" at line start → "• item"
    #    Only matches "- " at the very start of a line (not mid-line dashes like "5-7").
    text = re.sub(r"(?m)^-\s+", "• ", text)

    # 6. Horizontal rules --- or ___ → clean divider
    text = re.sub(r"(?m)^[-_]{3,}\s*$", "─────────────────────", text)

    # 7. Remove any remaining stray asterisks (leftover from malformed markdown)
    text = text.replace("*", "")

    return text


def _chunk_text(text: str, max_len: int = CHUNK_SIZE) -> list[str]:
    """Split text at sentence/paragraph boundaries, never mid-word."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while len(text) > max_len:
        # Try to split at paragraph break
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            # Try sentence end
            for punct in (".\n", ". ", "!\n", "! ", "?\n", "? ", "\n"):
                split_at = text.rfind(punct, 0, max_len)
                if split_at != -1:
                    split_at += len(punct)
                    break
        if split_at <= 0:
            split_at = max_len  # fallback: hard cut (rare)
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()

    if text:
        chunks.append(text)
    return chunks


# ─── Keyboard builders ────────────────────────────────────────────────────────

def main_menu_keyboard() -> dict:
    """
    Permanent reply keyboard (3 rows, one_time=False).
    Attaches to the input field and stays until explicitly removed.
    """
    return {
        "one_time": False,
        "inline": False,
        "buttons": [
            # Row 1
            [
                {
                    "action": {"type": "text", "label": "🐾 Мой питомец"},
                    "color": "primary",
                },
                {
                    "action": {"type": "text", "label": "➕ Добавить питомца"},
                    "color": "default",
                },
            ],
            # Row 2
            [
                {
                    "action": {"type": "text", "label": "⏰ Напоминания"},
                    "color": "default",
                },
                {
                    "action": {"type": "text", "label": "🔄 Сменить питомца"},
                    "color": "default",
                },
            ],
            # Row 3
            [
                {
                    "action": {"type": "text", "label": "💎 Подписка"},
                    "color": "default",
                },
                {
                    "action": {"type": "text", "label": "❓ Помощь"},
                    "color": "secondary",
                },
            ],
        ],
    }


def species_keyboard() -> dict:
    """Inline keyboard for /addpet species selection step."""
    return {
        "inline": True,
        "buttons": [
            [
                {
                    "action": {
                        "type": "callback",
                        "label": "🐱 Кот",
                        "payload": json.dumps({"action": "select_species", "species": "кошка"}),
                    },
                    "color": "default",
                },
                {
                    "action": {
                        "type": "callback",
                        "label": "🐶 Собака",
                        "payload": json.dumps({"action": "select_species", "species": "собака"}),
                    },
                    "color": "default",
                },
                {
                    "action": {
                        "type": "callback",
                        "label": "🦎 Другое",
                        "payload": json.dumps({"action": "select_species", "species": "другое"}),
                    },
                    "color": "default",
                },
            ]
        ],
    }


def yes_no_keyboard(confirm_id: Optional[int] = None) -> dict:
    """Build VK inline keyboard with Да / Нет buttons (callback type)."""
    yes_payload = json.dumps({"action": "confirm_yes", "id": confirm_id})
    no_payload = json.dumps({"action": "confirm_no", "id": confirm_id})
    return {
        "inline": True,   # must be True for callback buttons; one_time not allowed with inline
        "buttons": [
            [
                {
                    "action": {"type": "callback", "label": "✅ Да", "payload": yes_payload},
                    "color": "positive",
                },
                {
                    "action": {"type": "callback", "label": "❌ Нет", "payload": no_payload},
                    "color": "negative",
                },
            ]
        ],
    }


def pets_keyboard(pets: list[Any]) -> dict:
    """Build inline callback keyboard for pet selection."""
    buttons = []
    for pet in pets:
        payload = json.dumps({"action": "switch_pet", "pet_id": pet.id})
        row = [{
            "action": {
                "type": "callback",
                "label": f"🐾 {pet.name} ({pet.species})"[:40],
                "payload": payload,
            },
            "color": "primary",
        }]
        buttons.append(row)
    return {"inline": True, "buttons": buttons}  # one_time not allowed with inline


def remove_keyboard() -> str:
    return json.dumps({"one_time": True, "buttons": []})


# ─── VKClient ─────────────────────────────────────────────────────────────────

class VKClient:
    def __init__(self, token: str, group_id: str):
        self.token = token
        self.group_id = int(group_id)
        self._session: Optional[aiohttp.ClientSession] = None
        # In-memory dedup cache: event_key -> monotonic timestamp
        self._seen_ids: dict[str, float] = {}

    def _is_duplicate(self, event_key: str) -> bool:
        """Return True if this event was already seen within _DEDUP_TTL seconds."""
        now = time.monotonic()
        # Evict expired entries
        expired = [k for k, t in self._seen_ids.items() if now - t > _DEDUP_TTL]
        for k in expired:
            del self._seen_ids[k]
        if event_key in self._seen_ids:
            return True
        self._seen_ids[event_key] = now
        return False

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    async def _api(self, method: str, **params) -> dict:
        if not self._session:
            raise RuntimeError("VKClient not started")
        params["access_token"] = self.token
        params["v"] = VK_API_VERSION
        async with self._session.post(
            f"{VK_API_BASE}/{method}",
            data=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            data = await resp.json()
            if "error" in data:
                raise RuntimeError(f"VK API error: {data['error']}")
            return data.get("response", {})

    async def send_message(
        self,
        peer_id: int,
        text: str,
        keyboard: Optional[dict] = None,
    ) -> None:
        """Send a (possibly long) message, chunked if needed.

        If `keyboard` is not provided, the current keyboard in the chat is
        preserved (no auto-clear). Pass an empty dict explicitly to clear.
        """
        if not text and keyboard is None:
            return

        # Strip Markdown artifacts — VK renders **, *, ### as raw symbols
        if text:
            text = strip_vk_markdown(text)

        chunks = _chunk_text(text) if text else [""]
        for i, chunk in enumerate(chunks):
            kbd_param = {}
            # Only attach keyboard to the last chunk
            if i == len(chunks) - 1 and keyboard is not None:
                kbd_param["keyboard"] = json.dumps(keyboard, ensure_ascii=False)

            logger.info(
                "send_message → peer=%s chunk=%d/%d len=%d",
                peer_id, i + 1, len(chunks), len(chunk),
            )
            try:
                result = await self._api(
                    "messages.send",
                    peer_id=peer_id,
                    message=chunk,
                    random_id=random.randint(0, 2**31),
                    **kbd_param,
                )
                logger.info("send_message ✅ peer=%s msg_id=%s", peer_id, result)
            except Exception as e:
                logger.error("send_message ❌ peer=%s: %s", peer_id, e)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)  # avoid spam flag

    async def send_message_plain(self, vk_id: int, text: str) -> None:
        """Simple send without keyboard (for reminders etc)."""
        await self.send_message(peer_id=vk_id, text=text)

    async def send_event_answer(self, event_id: str, user_id: int, peer_id: int) -> None:
        """Acknowledge a callback button press (dismisses loading spinner in VK)."""
        try:
            await self._api(
                "messages.sendMessageEventAnswer",
                event_id=event_id,
                user_id=user_id,
                peer_id=peer_id,
            )
        except Exception as e:
            logger.warning("sendMessageEventAnswer error: %s", e)

    async def edit_message_keyboard(
        self,
        peer_id: int,
        cmid: int,
        new_text: str,
    ) -> None:
        """Edit a previously sent message — remove inline keyboard and update text.

        Used after a Да/Нет callback is handled to deactivate the buttons so
        repeated presses cannot retrigger the action.
        Failures are silently swallowed so the main flow is never interrupted.
        """
        try:
            # Build an empty inline keyboard (removes buttons from the message)
            empty_kbd = json.dumps({"inline": True, "buttons": []}, ensure_ascii=False)
            await self._api(
                "messages.edit",
                peer_id=peer_id,
                conversation_message_id=cmid,
                message=new_text,
                keyboard=empty_kbd,
            )
        except Exception as e:
            logger.debug("edit_message_keyboard: ignored error (peer=%s cmid=%s): %s", peer_id, cmid, e)

    async def get_full_message(self, peer_id: int, cmid: int) -> Optional[dict]:
        """
        Fetch the complete message object via messages.getByConversationMessageId.

        Used when Long Poll delivers a message with is_cropped=True — in that case
        VK truncates attachments to 1 item and the bot must re-fetch to get them all.
        """
        try:
            resp = await self._api(
                "messages.getByConversationMessageId",
                peer_id=peer_id,
                conversation_message_ids=cmid,
            )
            items = resp.get("items", [])
            return items[0] if items else None
        except Exception as e:
            logger.warning("get_full_message(peer=%s, cmid=%s) failed: %s", peer_id, cmid, e)
            return None

    async def get_long_poll_server(self) -> dict:
        return await self._api("groups.getLongPollServer", group_id=self.group_id)

    async def get_updates(self, server: str, key: str, ts: str) -> dict:
        if not self._session:
            raise RuntimeError("VKClient not started")
        url = f"{server}?act=a_check&key={key}&ts={ts}&wait=25"
        async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=35)) as resp:
            return await resp.json()

    async def run_long_poll(self, handler) -> None:
        """
        Main Long Poll loop with exponential backoff reconnect.
        handler(event: dict) is called for each incoming event.
        Duplicate events (same message_id / event_id within _DEDUP_TTL) are silently dropped.
        """
        backoff = 1
        while True:
            try:
                lp = await self.get_long_poll_server()
                server = lp["server"]
                key = lp["key"]
                ts = lp["ts"]
                backoff = 1   # reset on successful connect
                logger.info("Long Poll connected, ts=%s", ts)

                while True:
                    try:
                        data = await self.get_updates(server, key, ts)
                    except asyncio.TimeoutError:
                        logger.debug("Long Poll timeout, re-polling")
                        continue

                    failed = data.get("failed")
                    if failed == 1:
                        ts = data["ts"]
                        continue
                    if failed in (2, 3):
                        break  # re-get server info

                    ts = data.get("ts", ts)
                    for update in data.get("updates", []):
                        # ── Deduplication ─────────────────────────────────
                        dedup_key = _extract_dedup_key(update)
                        if dedup_key and self._is_duplicate(dedup_key):
                            logger.debug("Dropping duplicate event: %s", dedup_key)
                            continue
                        # ──────────────────────────────────────────────────
                        try:
                            await handler(update)
                        except Exception as e:
                            logger.exception("Handler error: %s", e)

            except Exception as e:
                logger.error("Long Poll error: %s — reconnect in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


def _extract_dedup_key(update: dict) -> Optional[str]:
    """Extract a unique dedup key from a VK Long Poll event."""
    update_type = update.get("type")
    if update_type == "message_new":
        msg_id = update.get("object", {}).get("message", {}).get("id")
        if msg_id:
            return f"msg_{msg_id}"
    elif update_type == "message_event":
        event_id = update.get("object", {}).get("event_id")
        if event_id:
            return f"evt_{event_id}"
    return None
