import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
import re
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
import yaml

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("chat-latest", "claude", "gemini", "gemma", "gpt-4", "gpt-5", "gpt-latest", "grok-4", "llama", "vision", "vl")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def resolve_env(node: Any) -> Any:
    if isinstance(node, dict):
        return {key.removesuffix("_env"): os.environ.get(value) if key.endswith("_env") else resolve_env(value) for key, value in node.items()}
    return node


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return resolve_env(yaml.safe_load(file))


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
channel_locks: dict[int, asyncio.Lock] = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


async def fetch_attachment(att: discord.Attachment) -> Optional[httpx.Response]:
    """Download an attachment, returning None if it is expired or can't be fetched.

    Discord CDN links expire over time and return 404 — embedding those responses
    as images makes the whole API request fail, so we skip them instead.
    """
    try:
        resp = await httpx_client.get(att.url)
        if resp.status_code != 200:
            logging.info(f"Skipping attachment (HTTP {resp.status_code}, likely expired): {att.filename}")
            return None
        return resp
    except httpx.HTTPError:
        logging.info(f"Skipping attachment (download failed): {att.filename}")
        return None


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    elif model not in config["models"]:
        output = f"Unknown model `{model}` — check the autocomplete list for valid names."
    elif user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
        curr_model = model
        output = f"Model switched to: `{model}`"
        logging.info(output)
    else:
        output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    is_dm = new_msg.channel.type == discord.ChannelType.private

    if new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    # --- MODIFIED: keyword triggers (after config is loaded, word-boundary matched) ---
    trigger_keywords = config.get("trigger_keywords", [])
    has_keyword = any(re.search(rf"\b{re.escape(kw.lower())}\b", new_msg.content.lower()) for kw in trigger_keywords)

    if not is_dm and discord_bot.user not in new_msg.mentions and not has_keyword:
        return

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    async def process_message() -> None:
        """Build the conversation from channel history and generate a reply.

        Runs under a per-channel lock so two triggers never overlap: the second
        handler waits, then reads a history that already includes the first reply.
        """
        global last_task_time

        user_warnings = set()

        # --- MODIFIED: build context from channel history instead of reply chains ---

        def format_history_msg(hist_msg, is_current=False):
            """Format a message with timestamp for clear temporal context."""
            ts = hist_msg.created_at.strftime("%b %d %H:%M")
            prefix = "CURRENT" if is_current else "history"
            role = "assistant" if hist_msg.author == discord_bot.user else "user"
            cleaned = hist_msg.content.removeprefix(discord_bot.user.mention).lstrip()
            if role == "user":
                text = f"[{prefix}] <@{hist_msg.author.id}> ({ts}): {cleaned}"
            else:
                text = f"[{prefix}] Meisho Doto ({ts}): {cleaned}"
            return text, role

        # Build messages in CORRECT API order: system prompt → history → current message
        messages = []

        # 1. System prompt FIRST
        if system_prompt := config.get("system_prompt"):
            now = datetime.now().astimezone()
            system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
            messages.append(dict(role="system", content=system_prompt))

        try:
            # Fetch channel history (oldest to newest). History is kept fully
            # paired (each user message followed by the bot's reply) so the model
            # knows which messages were already answered. The current message is
            # appended last, so the conversation never ends on an assistant turn.
            history_msgs = [msg async for msg in new_msg.channel.history(limit=max_messages, before=new_msg)]
            history_msgs.reverse()

            # Image budget for history: only the most recent images are kept
            # (newest messages win). Keeps the latest pics working without
            # re-sending every old image (and any expired links) on every reply.
            max_history_images = config.get("max_history_images", 10)
            images_allowed: dict[int, int] = {}
            dropped_history_images = 0
            budget_left = max_history_images
            for hist_msg in reversed(history_msgs):
                n_images = sum(1 for att in hist_msg.attachments if att.content_type and att.content_type.startswith("image"))
                if budget_left <= 0:
                    dropped_history_images += n_images
                    continue
                images_allowed[hist_msg.id] = min(n_images, budget_left)
                dropped_history_images += max(0, n_images - budget_left)
                budget_left -= n_images
            if dropped_history_images > 0:
                user_warnings.add(f"⚠️ {dropped_history_images} older image{'' if dropped_history_images == 1 else 's'} not included")

            # 2. Process history messages (oldest to newest)
            for hist_msg in history_msgs:
                msg_text, msg_role = format_history_msg(hist_msg, is_current=False)

                good_attachments = [att for att in hist_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[fetch_attachment(att) for att in good_attachments])

                # Skip expired/broken downloads (resp == None) and responses whose
                # body isn't actually an image, so dead links can't break the request.
                valid_image_pairs = [
                    (att, resp) for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image") and resp != None and resp.headers.get("content-type", "").startswith("image")
                ]
                valid_text_pairs = [
                    (att, resp) for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("text") and resp != None
                ]

                msg_images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in valid_image_pairs
                ]
                # Apply the per-message cap and the history-wide "recent images" budget.
                msg_images = msg_images[:max_images][:images_allowed.get(hist_msg.id, 0)]

                extra_text = "\n".join(
                    ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in hist_msg.embeds]
                    + [component.content for component in hist_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in valid_text_pairs]
                )
                if extra_text:
                    msg_text += "\n" + extra_text

                has_bad_attachments = len(hist_msg.attachments) > len(good_attachments)
                failed_attachments = len(good_attachments) - len(valid_image_pairs) - len(valid_text_pairs)

                if msg_images:
                    content = [dict(type="text", text=msg_text[:max_text])] + msg_images
                else:
                    content = msg_text[:max_text]

                if content != "":
                    messages.append(dict(content=content, role=msg_role))

                if len(msg_text) > max_text:
                    user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
                if len(valid_image_pairs) > max_images:
                    user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
                if has_bad_attachments:
                    user_warnings.add("⚠️ Unsupported attachments")
                if failed_attachments > 0:
                    user_warnings.add("⚠️ Some attachments couldn't be loaded (expired?)")

            # 3. Add the CURRENT (triggering) message LAST — this is what the bot responds to
            curr_text, _ = format_history_msg(new_msg, is_current=True)

            curr_attachments = [att for att in new_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
            curr_att_responses = await asyncio.gather(*[fetch_attachment(att) for att in curr_attachments])

            curr_images = [
                dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                for att, resp in zip(curr_attachments, curr_att_responses)
                if att.content_type.startswith("image") and resp != None and resp.headers.get("content-type", "").startswith("image")
            ]

            curr_extra_text = "\n".join(
                ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in new_msg.embeds]
                + [component.content for component in new_msg.components if component.type == discord.ComponentType.text_display]
                + [resp.text for att, resp in zip(curr_attachments, curr_att_responses) if att.content_type.startswith("text") and resp != None]
            )
            if curr_extra_text:
                curr_text += "\n" + curr_extra_text

            if curr_images[:max_images]:
                curr_content_final = [dict(type="text", text=curr_text[:max_text])] + curr_images[:max_images]
            else:
                curr_content_final = curr_text[:max_text]

            if curr_content_final != "":
                messages.append(dict(content=curr_content_final, role="user"))

        except (discord.NotFound, discord.HTTPException):
            logging.exception("Error fetching channel history")

        num_user_msgs = len(messages) - 1  # exclude system prompt
        if num_user_msgs < max_messages:
            user_warnings.add(f"⚠️ Only using last {num_user_msgs} message{'' if num_user_msgs == 1 else 's'}")

        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {num_user_msgs}):\n{new_msg.content}")

        # --- SAFETY: ensure conversation never ends on an assistant turn (Gemini requirement) ---
        # Only remove the LAST message if it's an assistant turn (preserves all other bot context)
        if messages and messages[-1].get("role") == "assistant":
            messages.pop()

        # Log final message roles for debugging
        final_roles = [m.get("role", "?") for m in messages]
        logging.info(f"Sending {len(messages)} messages to API. Order: {final_roles}")

        # Generate and send response message(s) (can be multiple if response is long)
        curr_content = finish_reason = None
        response_msgs = []
        response_contents = []

        openai_kwargs = dict(model=model, messages=messages, stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

        if use_plain_responses := config.get("use_plain_responses", False):
            max_message_length = 4000
        else:
            max_message_length = 4096 - len(STREAMING_INDICATOR)
            embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

        async def reply_helper(**reply_kwargs) -> None:
            reply_target = new_msg if not response_msgs else response_msgs[-1]
            response_msg = await reply_target.reply(**reply_kwargs)
            response_msgs.append(response_msg)

            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
            await msg_nodes[response_msg.id].lock.acquire()

        generation_failed = False

        try:
            async with new_msg.channel.typing():
                async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                    if finish_reason != None:
                        break

                    if not (choice := chunk.choices[0] if chunk.choices else None):
                        continue

                    finish_reason = choice.finish_reason

                    prev_content = curr_content or ""
                    curr_content = choice.delta.content or ""

                    new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        time_delta = datetime.now().timestamp() - last_task_time

                        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit = finish_reason != None or msg_split_incoming
                        is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                        if start_next_msg or ready_to_edit or is_final_edit:
                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            if start_next_msg:
                                await reply_helper(embed=embed, silent=True)
                            else:
                                await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                                await response_msgs[-1].edit(embed=embed)

                            last_task_time = datetime.now().timestamp()

                if use_plain_responses:
                    for content in response_contents:
                        await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

        except Exception:
            logging.exception("Error while generating response")
            generation_failed = True

        full_response = "".join(response_contents)

        # Finalize the last embed even if the stream died without a finish signal,
        # so replies never stay stuck with the streaming indicator.
        if response_msgs and not use_plain_responses:
            embed.description = (response_contents[-1] if response_contents else "") + (" ⚠️ (response interrupted)" if generation_failed else "")
            embed.color = EMBED_COLOR_COMPLETE if not generation_failed else EMBED_COLOR_INCOMPLETE
            try:
                await response_msgs[-1].edit(embed=embed)
            except discord.HTTPException:
                logging.exception("Failed to finalize response embed")

        # Never leave the user hanging: if the model produced nothing at all,
        # say so instead of silently not replying.
        if not full_response:
            if generation_failed:
                await reply_helper(content="⚠️ Something went wrong while I was thinking — could you try sending that again?")
            else:
                await reply_helper(content="⚠️ Hmm, I couldn't come up with a response to that — could you try rephrasing?")

        for response_msg in response_msgs:
            msg_nodes[response_msg.id].text = full_response
            msg_nodes[response_msg.id].lock.release()

        # Delete oldest MsgNodes (lowest message IDs) from the cache
        if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
            for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
                async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                    msg_nodes.pop(msg_id, None)

    # One reply at a time per channel: if another message triggers the bot while
    # it is still answering, that handler waits and then reads a history that
    # already includes the first reply (prevents out-of-order replies and
    # re-answering messages the bot already answered).
    async with channel_locks.setdefault(new_msg.channel.id, asyncio.Lock()):
        await process_message()


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
