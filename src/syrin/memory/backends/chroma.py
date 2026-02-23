"""Chroma backend for persistent memory storage with vector embeddings."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from syrin.enums import MemoryScope, MemoryType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    Settings = None
    chromadb = None

from syrin.memory.config import MemoryEntry


class ChromaBackend:
    """Chroma-based storage for memories with semantic search (vector embeddings).

    Requires: pip install chromadb

    Features:
    - Semantic search using embeddings
    - Persistent storage
    - Fast similarity search
    - Lightweight
    """

    def __init__(
        self,
        path: str | None = None,
        collection_name: str = "syrin_memory",
    ) -> None:
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb is not installed. Install with: pip install chromadb")

        self._collection_name = collection_name

        # Initialize Chroma client
        if path:
            self._client = chromadb.Client(
                Settings(
                    persist_directory=path,
                    anonymized_telemetry=False,
                )
            )
        else:
            self._client = chromadb.Client(
                Settings(
                    anonymized_telemetry=False,
                )
            )

        # Get or create collection
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"description": "Syrin memory store"},
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text. Override for custom embeddings."""
        import hashlib

        # Simple hash-based pseudo-embedding for demo
        # In production, use actual embeddings (sentence-transformers, etc.)
        hash_val = hashlib.md5(text.encode()).digest()
        # Pad to typical embedding size
        embedding = [float(b) / 255.0 for b in hash_val]
        # Pad to 384 dimensions
        embedding.extend([0.0] * (384 - len(embedding)))
        return embedding

    def _entry_to_metadata(self, entry: MemoryEntry) -> dict[str, Any]:
        """Convert MemoryEntry to Chroma metadata."""
        return {
            "id": entry.id,
            "content": entry.content,
            "type": entry.type.value,
            "importance": entry.importance,
            "scope": entry.scope.value,
            "source": entry.source,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "keywords": ",".join(entry.keywords),
        }

    def add(self, memory: MemoryEntry) -> None:
        """Add a memory to Chroma."""
        embedding = self._get_embedding(memory.content)
        metadata = self._entry_to_metadata(memory)

        self._collection.upsert(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[metadata],
        )

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        try:
            result = self._collection.get(ids=[memory_id])
            if not result["ids"]:
                return None
            return self._metadata_to_entry(result["metadatas"][0], result["documents"][0])
        except Exception:
            return None

    def _metadata_to_entry(self, metadata: dict[str, Any], content: str) -> MemoryEntry:
        """Convert Chroma metadata to MemoryEntry."""
        return MemoryEntry(
            id=metadata["id"],
            content=content,
            type=MemoryType(metadata["type"]),
            importance=metadata.get("importance", 1.0),
            scope=MemoryScope(metadata.get("scope", "user")),
            source=metadata.get("source"),
            created_at=datetime.fromisoformat(metadata["created_at"])
            if metadata.get("created_at")
            else datetime.now(),
            keywords=metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
        )

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        """Search memories by semantic similarity."""
        query_embedding = self._get_embedding(query)

        where = None
        if memory_type:
            where = {"type": memory_type.value}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        entries = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            entries.append(self._metadata_to_entry(metadata, doc))

        return entries

    def list(
        self,
        memory_type: MemoryType | None = None,
        scope: MemoryScope | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories."""
        where = None
        if memory_type:
            where = {"type": memory_type.value}
        elif scope:
            where = {"scope": scope.value}

        try:
            results = self._collection.get(
                where=where,
                limit=limit,
            )

            if not results["ids"]:
                return []

            entries = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                entries.append(self._metadata_to_entry(metadata, doc))

            return entries
        except Exception:
            return []

    def update(self, memory: MemoryEntry) -> None:
        """Update a memory."""
        self.add(memory)

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._collection.delete(ids=[memory_id])

    def clear(self) -> None:
        """Clear all memories."""
        self._collection.delete(where={"type": {"$exists": True}})

    def close(self) -> None:
        """Close the client connection."""
        # Chroma client doesn't need explicit close


__all__ = ["ChromaBackend"]
