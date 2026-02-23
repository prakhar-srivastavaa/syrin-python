"""Memory module — first-class persistent memory for agents.

Supports four memory types (Core, Episodic, Semantic, Procedural),
pluggable backends, automatic extraction, forgetting curves, budget
integration, and position-aware context injection.
"""

from .backends import (
    BACKENDS,
    ChromaBackend,
    InMemoryBackend,
    PostgresBackend,
    QdrantBackend,
    RedisBackend,
    SQLiteBackend,
    get_backend,
)
from .config import (
    Consolidation,
    Decay,
    Memory,
    MemoryBudget,
    MemoryEntry,
)
from .conversation import BufferMemory, ConversationMemory, WindowMemory
from .store import MemoryStore
from .types import (
    CoreMemory,
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
    create_memory,
)

__all__ = [
    "Memory",
    "Decay",
    "MemoryBudget",
    "Consolidation",
    "MemoryEntry",
    "ConversationMemory",
    "BufferMemory",
    "WindowMemory",
    # Backends
    "InMemoryBackend",
    "SQLiteBackend",
    "QdrantBackend",
    "ChromaBackend",
    "RedisBackend",
    "PostgresBackend",
    "get_backend",
    "BACKENDS",
    # Storage
    "MemoryStore",
    # Memory types
    "CoreMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "create_memory",
]
