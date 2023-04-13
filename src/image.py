from enum import Enum
from dataclasses import dataclass
import openai
from typing import Optional
import discord
from src.moderation import (
    send_moderation_flagged_message,
    send_moderation_blocked_message,
)
from src.constants import (
    BOT_INSTRUCTIONS,
    BOT_NAME,
    EXAMPLE_CONVOS,
)

MY_BOT_NAME = BOT_NAME
MY_BOT_EXAMPLE_CONVOS = EXAMPLE_CONVOS


class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    INVALID_REQUEST = 2
    OTHER_ERROR = 3
    MODERATION_FLAGGED = 4
    MODERATION_BLOCKED = 5


@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: Optional[str]
    status_text: Optional[str]


async def generate_image_response(message: str) -> Optional[str]:
    try:
        reply = openai.Image.create(
            prompt=message,
            n=1,
            size="1024x1024",
        )
        return reply['data'][0]['url']
    except Exception as e:
        logger.exception(e)
        return None
