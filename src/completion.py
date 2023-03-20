async def generate_completion_response(
    messages: List[Message], user: str
) -> CompletionData:
    try:
        chat_history = []
        for message in messages:
            chat_history.append({"role": "system" if message.user == "System" else "user", "content": message.text})
        chat_history.append({"role": "assistant", "content": f"{MY_BOT_NAME}:"})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            temperature=1.0,
            top_p=0.9,
            max_tokens=512,
            stop=None,
        )
        reply = response.choices[0].message['content'].strip()
        if reply:
            flagged_str, blocked_str = moderate_message(
                message=(reply)[-500:], user=user
            )
            if len(blocked_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_BLOCKED,
                    reply_text=reply,
                    status_text=f"from_response:{blocked_str}",
                )

            if len(flagged_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_FLAGGED,
                    reply_text=reply,
                    status_text=f"from_response:{flagged_str}",
                )

        return CompletionData(
            status=CompletionResult.OK, reply_text=reply, status_text=None
        )
    except openai.error.InvalidRequestError as e:
        if "This model's maximum context length" in e.user_message:
            return CompletionData(
                status=CompletionResult.TOO_LONG, reply_text=None, status_text=str(e)
            )
        else:
            logger.exception(e)
            return CompletionData(
                status=CompletionResult.INVALID_REQUEST,
                reply_text=None,
                status_text=str(e),
            )
    except Exception as e:
        logger.exception(e)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(e)
        )
