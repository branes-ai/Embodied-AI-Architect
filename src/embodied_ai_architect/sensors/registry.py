"""Sensor registry with TF-IDF keyword matching (issue #59).

Loads sensor YAML files from data/sensors/, builds an inverted keyword
index, and provides ranked TF-IDF-style search. Mirrors the
PlatformRegistry pattern from platforms/registry.py.

Usage:
    from embodied_ai_architect.sensors import SensorRegistry

    registry = SensorRegistry()
    results = registry.search("stereo camera for VIO", top_k=5)
    for r in results:
        print(r.sensor_id, r.score, r.sensor.name)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default data directory — resolved relative to the repo root
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sensors"

# Modality categories from taxonomy.yaml
MODALITIES = [
    "visual",
    "ranging",
    "inertial",
    "position",
    "environmental",
    "force",
    "audio",
    "biological",
]

# Shared stop-words (same set as platforms/registry.py)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "not",
        "but",
        "can",
        "do",
        "if",
        "no",
        "so",
        "up",
        "out",
        "all",
        "one",
        "two",
        "our",
        "new",
        "use",
        "also",
        "than",
        "very",
        "just",
        "about",
        "into",
        "over",
        "such",
        "only",
        "other",
        "some",
        "may",
        "each",
        "any",
        "most",
        "more",
        "both",
        "which",
        "their",
        "these",
        "this",
        "those",
        "been",
        "being",
        "would",
        "could",
        "should",
    }
)


def _tokenize(text: str) -> list[str]:
    """Split text into lowercased tokens, filtering stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in _STOP_WORDS]


def _bigrams(tokens: list[str]) -> list[str]:
    """Generate bigrams from a token list."""
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


class SensorDefinition(BaseModel):
    """A sensor loaded from the registry."""

    id: str
    name: str
    category: str = ""
    sensor_type: str = ""
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
class SensorMatchResult:
    """A scored match from the sensor registry search."""

    sensor_id: str
    score: float
    sensor: SensorDefinition
    matched_keywords: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"SensorMatchResult({self.sensor_id!r}, score={self.score:.3f})"


class SensorRegistry:
    """Sensor registry with TF-IDF keyword matching.

    Loads sensor YAML files from data/sensors/<category>/*.yaml,
    builds an inverted keyword index, and provides ranked search.
    """

    def __init__(self, data_dir: Path | str | None = None):
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._sensors: dict[str, SensorDefinition] = {}
        self._keyword_index: dict[str, set[str]] = {}
        self._idf: dict[str, float] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all sensor YAML files from the data directory."""
        import yaml

        self._sensors.clear()
        self._keyword_index.clear()

        if not self._data_dir.exists():
            logger.warning("Sensor data directory not found: %s", self._data_dir)
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
                if not data or "id" not in data:
                    continue
                sensor = SensorDefinition(
                    id=data["id"],
                    name=data.get("name", data["id"]),
                    category=data.get("category", ""),
                    sensor_type=data.get("sensor_type", ""),
                    description=data.get("description", ""),
                    aliases=data.get("aliases", []),
                    keywords=data.get("keywords", {}),
                    classification=data.get("classification", {}),
                    attributes=data.get("attributes", {}),
                    reference_products=data.get("reference_products", []),
                    raw=data,
                )
                self._sensors[sensor.id] = sensor
            except Exception:
                logger.warning("Failed to load sensor file: %s", fpath, exc_info=True)

        self._build_index()
        self._loaded = True
        logger.info("Loaded %d sensors from %s", len(self._sensors), self._data_dir)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _build_index(self) -> None:
        """Build inverted keyword index and compute IDF values."""
        self._keyword_index.clear()
        self._idf.clear()

        for sid, sensor in self._sensors.items():
            seen: set[str] = set()
            for kw in sensor.all_keywords():
                kw_lower = kw.lower().strip()
                if not kw_lower or kw_lower in seen:
                    continue
                seen.add(kw_lower)
                self._keyword_index.setdefault(kw_lower, set()).add(sid)

            for text in [sensor.name, sensor.description]:
                for word in _tokenize(text):
                    self._keyword_index.setdefault(word, set()).add(sid)

        n = max(len(self._sensors), 1)
        for kw, sids in self._keyword_index.items():
            self._idf[kw] = math.log(n / len(sids)) + 1.0

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.01,
    ) -> list[SensorMatchResult]:
        """Search for sensors matching a free-text query.

        TF-IDF scoring with phrase, token, and bigram matching.
        Returns ranked list, best first.
        """
        self._ensure_loaded()

        if top_k <= 0 or not self._sensors:
            return []

        query_lower = query.lower().strip()
        query_tokens = _tokenize(query_lower)
        query_bigrams = _bigrams(query_tokens)

        scores: dict[str, float] = {}
        matched_kws: dict[str, list[str]] = {}

        # Phase 1: phrase matching
        for kw_phrase, sids in self._keyword_index.items():
            if kw_phrase in query_lower:
                weight = self._idf.get(kw_phrase, 1.0)
                word_count = len(kw_phrase.split())
                phrase_bonus = 1.0 + (word_count - 1) * 0.5
                for sid in sids:
                    scores[sid] = scores.get(sid, 0.0) + weight * phrase_bonus
                    matched_kws.setdefault(sid, []).append(kw_phrase)

        # Phase 2: token matching
        for token in query_tokens:
            if token in self._keyword_index:
                weight = self._idf.get(token, 1.0) * 0.5
                for sid in self._keyword_index[token]:
                    scores[sid] = scores.get(sid, 0.0) + weight
                    kws = matched_kws.setdefault(sid, [])
                    if token not in kws:
                        kws.append(token)

        # Phase 3: bigram matching
        for bigram in query_bigrams:
            if bigram in self._keyword_index:
                weight = self._idf.get(bigram, 1.0) * 0.8
                for sid in self._keyword_index[bigram]:
                    scores[sid] = scores.get(sid, 0.0) + weight
                    kws = matched_kws.setdefault(sid, [])
                    if bigram not in kws:
                        kws.append(bigram)

        if not scores:
            return []

        max_score = max(scores.values())
        if max_score > 0:
            scores = {sid: s / max_score for sid, s in scores.items()}

        results = [
            SensorMatchResult(
                sensor_id=sid,
                score=round(score, 4),
                sensor=self._sensors[sid],
                matched_keywords=matched_kws.get(sid, []),
            )
            for sid, score in scores.items()
            if score >= min_score
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, sensor_id: str) -> Optional[SensorDefinition]:
        """Get a sensor by ID."""
        self._ensure_loaded()
        return self._sensors.get(sensor_id)

    def list_sensors(self, modality: Optional[str] = None) -> list[SensorDefinition]:
        """List all sensors, optionally filtered by category/modality."""
        self._ensure_loaded()
        sensors = list(self._sensors.values())
        if modality:
            sensors = [s for s in sensors if s.category == modality]
        return sensors

    def list_by_category(self, category: str) -> list[SensorDefinition]:
        """List sensors in a specific category."""
        return self.list_sensors(modality=category)

    def categories(self) -> list[str]:
        """List available sensor modality categories."""
        return list(MODALITIES)

    @property
    def sensor_count(self) -> int:
        """Total number of loaded sensors."""
        self._ensure_loaded()
        return len(self._sensors)
