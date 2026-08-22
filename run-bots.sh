#!/bin/sh
# Runs one llmcord bot process per config file listed in BOT_CONFIGS
# (space-separated), restarting each bot automatically if it crashes.
#
# Examples:
#   BOT_CONFIGS="config.yaml"                             -> one bot (default)
#   BOT_CONFIGS="config.yaml config-kant.yaml"            -> two bots in one container
#   BOT_CONFIGS="config.yaml config-kant.yaml config-bot3.yaml" -> three bots
#
# Give each extra bot its own namespaced environment variables so they don't
# collide, e.g. BOT2_DISCORD_TOKEN, BOT2_NAME, BOT2_SYSTEM_PROMPT (see
# config-2-example.yaml).

set -u

: "${BOT_CONFIGS:=config.yaml}"

for cfg in $BOT_CONFIGS; do
    (
        while true; do
            echo "[run-bots] starting bot for $cfg"
            python llmcord.py "$cfg"
            echo "[run-bots] bot for $cfg exited, restarting in 5 seconds..."
            sleep 5
        done
    ) &
done

wait
