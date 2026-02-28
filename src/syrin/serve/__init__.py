"""Agent serving layer — HTTP, CLI, STDIO protocols."""

from syrin.serve.config import ServeConfig
from syrin.serve.discovery import (
    AGENT_CARD_PATH,
    AgentCard,
    AgentCardAuth,
    AgentCardProvider,
    build_agent_card_json,
)
from syrin.serve.http import build_router, create_http_app
from syrin.serve.playground import add_playground_static_mount
from syrin.serve.router import AgentRouter
from syrin.serve.servable import Servable

__all__ = [
    "AGENT_CARD_PATH",
    "Servable",
    "add_playground_static_mount",
    "create_http_app",
    "AgentCard",
    "AgentCardAuth",
    "AgentCardProvider",
    "AgentRouter",
    "build_agent_card_json",
    "build_router",
    "ServeConfig",
]
