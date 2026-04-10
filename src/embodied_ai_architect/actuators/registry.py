"""Actuator registry with TF-IDF keyword matching (issue #62).

Loads actuator YAML files from data/actuators/, builds an inverted keyword
index, and provides ranked TF-IDF-style search. Mirrors the
SensorRegistry pattern from sensors/registry.py.

Usage:
    from embodied_ai_architect.actuators import ActuatorRegistry

    registry = ActuatorRegistry()
    results = registry.search("gripper for fragile objects", top_k=5)
    for r in results:
        print(r.actuator_id, r.score, r.actuator.name)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from embodied_ai_architect.search_utils import bigrams as _bigrams
from embodied_ai_architect.search_utils import tokenize as _tokenize

logger = logging.getLogger(__name__)

# Default data directory — resolved relative to the repo root
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "actuators"

# Actuator categories from taxonomy.yaml
ACTUATOR_CATEGORIES = [
    "motor",
    "hydraulic",
    "pneumatic",
    "gripper",
    "locomotion",
    "fluid",
    "display",
    "specialty",
]


class ActuatorDefinition(BaseModel):
    """An actuator loaded from the registry."""

    id: str
    name: str
    category: str = ""
    actuator_type: str = ""
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    keywords: dict[str, list[str]] = Field(default_factory=dict)
    classification: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    reference_products: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    def all_keywords(self) -> list[str]:
        """Flatten all keyword groups into a single list."""
        result: list[str] = []
        for group in self.keywords.values():
            if isinstance(group, list):
                result.extend(group)
        return result


@dataclass
class ActuatorMatchResult:
    """A scored match from the actuator registry search."""

    actuator_id: str
    score: float
    actuator: ActuatorDefinition
    matched_keywords: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ActuatorMatchResult({self.actuator_id!r}, score={self.score:.3f})"


class ActuatorRegistry:
    """Actuator registry with TF-IDF keyword matching.

    Loads actuator YAML files from data/actuators/<category>/*.yaml,
    builds an inverted keyword index, and provides ranked search.
    """

    def __init__(self, data_dir: Path | str | None = None):
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._actuators: dict[str, ActuatorDefinition] = {}
        self._keyword_index: dict[str, set[str]] = {}
        self._idf: dict[str, float] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all actuator YAML files from the data directory."""
        import yaml

        self._actuators.clear()
        self._keyword_index.clear()

        if not self._data_dir.exists():
            logger.warning("Actuator data directory not found: %s", self._data_dir)
            self._loaded = True
            return

        yaml_files = [
            f
            for f in self._data_dir.rglob("*.yaml")
            if f.name not in ("schema.yaml", "taxonomy.yaml") and not f.name.startswith("_")
        ]

        for fpath in yaml_files:
            try:
                with open(fpath, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict) or "id" not in data:
                    continue
                actuator = ActuatorDefinition(
                    id=data["id"],
                    name=data.get("name", data["id"]),
                    category=data.get("category", ""),
                    actuator_type=data.get("actuator_type", ""),
                    description=data.get("description", ""),
                    aliases=data.get("aliases", []),
                    keywords=data.get("keywords", {}),
                    classification=data.get("classification", {}),
                    attributes=data.get("attributes", {}),
                    reference_products=data.get("reference_products", []),
                    raw=data,
                )
                self._actuators[actuator.id] = actuator
            except Exception:
                logger.warning("Failed to load actuator file: %s", fpath, exc_info=True)

        self._build_index()
        self._loaded = True
        logger.info("Loaded %d actuators from %s", len(self._actuators), self._data_dir)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _build_index(self) -> None:
        """Build inverted keyword index and compute IDF values."""
        self._keyword_index.clear()
        self._idf.clear()

        for aid, actuator in self._actuators.items():
            seen: set[str] = set()
            for kw in actuator.all_keywords():
                kw_lower = kw.lower().strip()
                if not kw_lower or kw_lower in seen:
                    continue
                seen.add(kw_lower)
                self._keyword_index.setdefault(kw_lower, set()).add(aid)

            for alias in actuator.aliases:
                alias_lower = alias.lower().strip()
                if alias_lower and alias_lower not in seen:
                    seen.add(alias_lower)
                    self._keyword_index.setdefault(alias_lower, set()).add(aid)

            for text in [actuator.name, actuator.description]:
                for word in _tokenize(text):
                    self._keyword_index.setdefault(word, set()).add(aid)

        n = max(len(self._actuators), 1)
        for kw, aids in self._keyword_index.items():
            self._idf[kw] = math.log(n / len(aids)) + 1.0

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.01,
    ) -> list[ActuatorMatchResult]:
        """Search for actuators matching a free-text query.

        TF-IDF scoring with phrase, token, and bigram matching.
        Returns ranked list, best first.
        """
        self._ensure_loaded()

        if top_k <= 0 or not self._actuators:
            return []

        query_lower = query.lower().strip()
        query_tokens = _tokenize(query_lower)
        query_bigrams = _bigrams(query_tokens)

        scores: dict[str, float] = {}
        matched_kws: dict[str, list[str]] = {}

        # Phase 1: phrase matching (multi-token keys only to avoid double-counting)
        for kw_phrase, aids in self._keyword_index.items():
            if " " not in kw_phrase:
                continue
            if kw_phrase in query_lower:
                weight = self._idf.get(kw_phrase, 1.0)
                word_count = len(kw_phrase.split())
                phrase_bonus = 1.0 + (word_count - 1) * 0.5
                for aid in aids:
                    scores[aid] = scores.get(aid, 0.0) + weight * phrase_bonus
                    matched_kws.setdefault(aid, []).append(kw_phrase)

        # Phase 2: token matching
        for token in query_tokens:
            if token in self._keyword_index:
                weight = self._idf.get(token, 1.0) * 0.5
                for aid in self._keyword_index[token]:
                    scores[aid] = scores.get(aid, 0.0) + weight
                    kws = matched_kws.setdefault(aid, [])
                    if token not in kws:
                        kws.append(token)

        # Phase 3: bigram matching
        for bigram in query_bigrams:
            if bigram in self._keyword_index:
                weight = self._idf.get(bigram, 1.0) * 0.8
                for aid in self._keyword_index[bigram]:
                    scores[aid] = scores.get(aid, 0.0) + weight
                    kws = matched_kws.setdefault(aid, [])
                    if bigram not in kws:
                        kws.append(bigram)

        if not scores:
            return []

        max_score = max(scores.values())
        if max_score > 0:
            scores = {aid: s / max_score for aid, s in scores.items()}

        results = [
            ActuatorMatchResult(
                actuator_id=aid,
                score=round(score, 4),
                actuator=self._actuators[aid],
                matched_keywords=matched_kws.get(aid, []),
            )
            for aid, score in scores.items()
            if score >= min_score
        ]
        results.sort(key=lambda r: (-r.score, r.actuator_id))
        return results[:top_k]

    def get(self, actuator_id: str) -> Optional[ActuatorDefinition]:
        """Get an actuator by ID."""
        self._ensure_loaded()
        return self._actuators.get(actuator_id)

    def list_actuators(self, category: Optional[str] = None) -> list[ActuatorDefinition]:
        """List all actuators, optionally filtered by category."""
        self._ensure_loaded()
        actuators = list(self._actuators.values())
        if category:
            actuators = [a for a in actuators if a.category == category]
        return actuators

    def list_by_category(self, category: str) -> list[ActuatorDefinition]:
        """List actuators in a specific category."""
        return self.list_actuators(category=category)

    def list_categories(self) -> list[str]:
        """List available actuator categories."""
        return list(ACTUATOR_CATEGORIES)

    def categories(self) -> list[str]:
        """List available actuator categories."""
        return self.list_categories()

    @property
    def actuator_count(self) -> int:
        """Total number of loaded actuators."""
        self._ensure_loaded()
        return len(self._actuators)
