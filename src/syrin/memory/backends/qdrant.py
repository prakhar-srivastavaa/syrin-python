"""Qdrant backend for persistent memory storage with vector embeddings."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from syrin.enums import MemoryScope, MemoryType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    PointStruct = None
    VectorParams = None

from syrin.memory.config import MemoryEntry


class QdrantBackend:
    """Qdrant-based storage for memories with semantic search (vector embeddings).

    Requires: pip install qdrant-client

    Features:
    - Semantic search using embeddings
    - Persistent storage
    - Fast similarity search
    """

    def __init__(
        self,
        path: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "syrin_memory",
        vector_size: int = 384,
    ) -> None:
        """Initialize Qdrant backend.

        Args:
            path: Local path for embedded Qdrant; if set, host/port ignored.
            host: Qdrant server host.
            port: Qdrant server port.
            collection: Collection name for memories.
            vector_size: Embedding dimension (default 384).
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Install with: pip install qdrant-client"
            )

        self._host = host
        self._port = port
        self._collection = collection
        self._vector_size = vector_size

        if path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(host=host, port=port)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self._collection not in collection_names:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text. Override for custom embeddings."""
        import hashlib

        # Simple hash-based pseudo-embedding for demo
        # In production, use actual embeddings (sentence-transformers, etc.)
        hash_val = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_val[: self._vector_size]]

    def _entry_to_payload(self, entry: MemoryEntry) -> dict[str, Any]:
        """Convert MemoryEntry to Qdrant payload."""
        return {
            "id": entry.id,
            "content": entry.content,
            "type": entry.type.value,
            "importance": entry.importance,
            "scope": entry.scope.value,
            "source": entry.source,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "keywords": entry.keywords,
            "metadata": entry.metadata,
        }

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory to Qdrant."""
        vector = self._get_embedding(memory.content)

        point = PointStruct(
            id=memory.id,
            vector=vector,
            payload=self._entry_to_payload(memory),
        )

        self._client.upsert(
            collection_name=self._collection,
            points=[point],
        )

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=[memory_id],
        )

        if not results:
            return None

        return self._payload_to_entry(results[0].payload)

    def _payload_to_entry(self, payload: dict[str, Any]) -> MemoryEntry:
        """Convert Qdrant payload to MemoryEntry."""
        return MemoryEntry(
            id=payload["id"],
            content=payload["content"],
            type=MemoryType(payload["type"]),
            importance=payload.get("importance", 1.0),
            scope=MemoryScope(payload.get("scope", "user")),
            source=payload.get("source"),
            created_at=datetime.fromisoformat(payload["created_at"])
            if payload.get("created_at")
            else datetime.now(),
            keywords=payload.get("keywords", []),
            metadata=payload.get("metadata", {}),
        )

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        """Search memories by semantic similarity."""
        query_vector = self._get_embedding(query)

        filter_condition = None
        if memory_type:
            from qdrant_client.models import FieldCondition, MatchValue

            filter_condition = FieldCondition(
                key="type",
                match=MatchValue(value=memory_type.value),
            )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_condition,
        )

        return [self._payload_to_entry(r.payload) for r in results]

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories."""
        filter_conditions = []

        if memory_type:
            from qdrant_client.models import FieldCondition, MatchValue

            filter_conditions.append(
                FieldCondition(
                    key="type",
                    match=MatchValue(value=memory_type.value),
                )
            )

        if scope:
            from qdrant_client.models import FieldCondition, MatchValue

            filter_conditions.append(
                FieldCondition(
                    key="scope",
                    match=MatchValue(value=scope.value),
                )
            )

        from qdrant_client.models import Filter

        filter_obj = Filter(must=filter_conditions) if filter_conditions else None

        results = self._client.scroll(
            collection_name=self._collection,
            limit=limit,
            with_payload=True,
            scroll_filter=filter_obj,
        )

        return [self._payload_to_entry(p) for p in results[0]]

    def update(self, memory: MemoryEntry) -> None:
        """Update a memory."""
        self.add(memory)

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=[memory_id],
        )

    def clear(self) -> None:
        """Clear all memories."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=[{"match_all": True}],
        )

    def close(self) -> None:
        """Close the client connection."""
        # Qdrant client doesn't need explicit close


__all__ = ["QdrantBackend"]
