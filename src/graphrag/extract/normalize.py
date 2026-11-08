from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from graphrag.graph import GraphStore, NodeType, make_entity_node


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    name: str
    entity_type: str
    canonical_name: str
    is_new: bool


class EntityNormalizer:
    """Handles entity deduplication and canonicalization."""

    def __init__(self, store: GraphStore):
        self.store = store
        self._canonical_cache: dict[str, str] = {}  # canonical_name -> entity_id
        self._name_cache: dict[str, str] = {}  # name -> entity_id
        self._build_caches()

    def _build_caches(self) -> None:
        """Build caches from existing entities in the graph."""
        for node in self.store.iter_nodes(NodeType.ENTITY):
            canonical = node.properties.get("canonical_name", "").lower()
            name = node.properties.get("name", "").lower()
            if canonical:
                self._canonical_cache[canonical] = node.id
            if name:
                self._name_cache[name] = node.id

    def normalize(
        self, name: str, entity_type: str, canonical_name: Optional[str] = None
    ) -> EntityRecord:
        """Normalize an entity and return its record (new or existing)."""
        if canonical_name is None:
            canonical_name = name

        canonical_lower = canonical_name.lower().strip()
        name_lower = name.lower().strip()

        # Check canonical cache first (exact canonical match)
        if canonical_lower in self._canonical_cache:
            return EntityRecord(
                entity_id=self._canonical_cache[canonical_lower],
                name=name,
                entity_type=entity_type,
                canonical_name=canonical_name,
                is_new=False,
            )

        # Check name cache (exact name match)
        if name_lower in self._name_cache:
            existing_id = self._name_cache[name_lower]
            existing_node = self.store.get_node(existing_id)
            if existing_node:
                # Update canonical name if provided is more complete
                existing_canonical = existing_node.properties.get("canonical_name", "")
                if len(canonical_name) > len(existing_canonical):
                    existing_node.properties["canonical_name"] = canonical_name
                return EntityRecord(
                    entity_id=existing_id,
                    name=name,
                    entity_type=entity_type,
                    canonical_name=existing_node.properties.get(
                        "canonical_name", canonical_name
                    ),
                    is_new=False,
                )

        # Create new entity
        entity_id = f"ent_{hashlib.sha1(canonical_name.encode()).hexdigest()[:12]}"
        self.store.upsert_node(
            make_entity_node(entity_id, name, entity_type, canonical_name)
        )

        # Update caches
        self._canonical_cache[canonical_lower] = entity_id
        self._name_cache[name_lower] = entity_id

        return EntityRecord(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            canonical_name=canonical_name,
            is_new=True,
        )

    def resolve_entity_id(self, name: str) -> Optional[str]:
        """Resolve a name to an existing entity ID."""
        name_lower = name.lower().strip()
        if name_lower in self._canonical_cache:
            return self._canonical_cache[name_lower]
        if name_lower in self._name_cache:
            return self._name_cache[name_lower]
        # Try fuzzy match via store
        node = self.store.get_entity_by_name(name)
        return node.id if node else None
