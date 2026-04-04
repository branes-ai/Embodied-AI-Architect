"""Platform Registry: load, index, and search embodied AI platform definitions.

The registry loads YAML platform definitions from a configurable directory
(default: data/platforms/) and provides keyword-based matching against user
prompts. Each platform carries domain context that can be injected into the
qualification and design planning flows.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default data directory relative to the repo root
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "platforms"


@dataclass
class PlatformDefinition:
    """A platform definition loaded from YAML."""

    id: str
    name: str
    category: str
    description: str
    aliases: list[str] = field(default_factory=list)
    keywords: dict[str, list[str]] = field(default_factory=dict)
    classification: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    implications: dict[str, Any] = field(default_factory=dict)
    subsystems: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    qualification: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def all_keywords(self) -> list[str]:
        """Flatten all keyword lists into a single list."""
        result: list[str] = []
        result.extend(self.aliases)
        for group in self.keywords.values():
            if isinstance(group, list):
                result.extend(group)
        return [k.lower() for k in result]


@dataclass
class MatchResult:
    """A scored match from the registry search."""

    platform_id: str
    score: float
    platform: PlatformDefinition
    matched_keywords: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"MatchResult({self.platform_id!r}, score={self.score:.3f})"


class PlatformRegistry:
    """File-based platform registry with keyword matching.

    Loads platform YAML files from a directory, builds an inverted index
    of keywords to platform IDs, and provides TF-IDF-inspired scoring
    for free-text queries.
    """

    def __init__(self, data_dir: Path | str | None = None):
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._platforms: dict[str, PlatformDefinition] = {}
        # Inverted index: keyword_phrase -> set of platform IDs
        self._keyword_index: dict[str, set[str]] = {}
        # IDF values: log(N / df) for each keyword phrase
        self._idf: dict[str, float] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all platform YAML files from the data directory."""
        import yaml

        self._platforms.clear()
        self._keyword_index.clear()

        if not self._data_dir.exists():
            logger.warning("Platform data directory not found: %s", self._data_dir)
            self._loaded = True
            return

        yaml_files = list(self._data_dir.rglob("*.yaml"))
        yaml_files = [
            f
            for f in yaml_files
            if f.name not in ("schema.yaml", "taxonomy.yaml")
            and not f.name.startswith("_")
            and "configurations" not in f.parts
        ]

        for fpath in yaml_files:
            try:
                with open(fpath) as fh:
                    data = yaml.safe_load(fh)
                if not data or "id" not in data:
                    continue
                platform = PlatformDefinition(
                    id=data["id"],
                    name=data.get("name", data["id"]),
                    category=data.get("category", ""),
                    description=data.get("description", ""),
                    aliases=data.get("aliases", []),
                    keywords=data.get("keywords", {}),
                    classification=data.get("classification", {}),
                    attributes=data.get("attributes", {}),
                    implications=data.get("implications", {}),
                    subsystems=data.get("subsystems", {}),
                    context=data.get("context", {}),
                    qualification=data.get("qualification", {}),
                    raw=data,
                )
                self._platforms[platform.id] = platform
            except Exception:
                logger.warning("Failed to load platform file: %s", fpath, exc_info=True)

        self._build_index()
        self._loaded = True
        logger.info("Loaded %d platforms from %s", len(self._platforms), self._data_dir)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _build_index(self) -> None:
        """Build inverted keyword index and compute IDF values."""
        self._keyword_index.clear()
        self._idf.clear()

        for pid, platform in self._platforms.items():
            seen: set[str] = set()
            for kw in platform.all_keywords():
                kw_lower = kw.lower().strip()
                if not kw_lower or kw_lower in seen:
                    continue
                seen.add(kw_lower)
                if kw_lower not in self._keyword_index:
                    self._keyword_index[kw_lower] = set()
                self._keyword_index[kw_lower].add(pid)

            # Also index the description and name words
            for text in [platform.name, platform.description]:
                for word in _tokenize(text):
                    if word not in self._keyword_index:
                        self._keyword_index[word] = set()
                    self._keyword_index[word].add(pid)

        # Compute IDF: log(N / df) for each keyword
        n = max(len(self._platforms), 1)
        for kw, pids in self._keyword_index.items():
            self._idf[kw] = math.log(n / len(pids)) + 1.0

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.01,
    ) -> list[MatchResult]:
        """Search for platforms matching a free-text query.

        Uses a TF-IDF-inspired scoring:
        - Tokenize query into words and 2-grams
        - For each token, find platforms that have it as a keyword
        - Score each platform by sum of IDF-weighted matches
        - Bonus for phrase matches (multi-word keywords)

        Returns ranked list of MatchResult, best first.
        """
        self._ensure_loaded()

        if not self._platforms:
            return []

        query_lower = query.lower().strip()
        query_tokens = _tokenize(query_lower)
        query_bigrams = _bigrams(query_tokens)

        # Score accumulator per platform
        scores: dict[str, float] = {}
        matched_kws: dict[str, list[str]] = {}

        # Phase 1: exact phrase matching against full keyword phrases
        for kw_phrase, pids in self._keyword_index.items():
            if kw_phrase in query_lower:
                # Phrase found in query — strong signal
                weight = self._idf.get(kw_phrase, 1.0)
                # Bonus for longer phrase matches
                word_count = len(kw_phrase.split())
                phrase_bonus = 1.0 + (word_count - 1) * 0.5
                for pid in pids:
                    scores[pid] = scores.get(pid, 0.0) + weight * phrase_bonus
                    if pid not in matched_kws:
                        matched_kws[pid] = []
                    matched_kws[pid].append(kw_phrase)

        # Phase 2: token matching (individual words)
        for token in query_tokens:
            if token in self._keyword_index:
                weight = self._idf.get(token, 1.0) * 0.5  # Lower weight than phrases
                for pid in self._keyword_index[token]:
                    scores[pid] = scores.get(pid, 0.0) + weight
                    if pid not in matched_kws:
                        matched_kws[pid] = []
                    if token not in matched_kws[pid]:
                        matched_kws[pid].append(token)

        # Phase 3: bigram matching
        for bigram in query_bigrams:
            if bigram in self._keyword_index:
                weight = self._idf.get(bigram, 1.0) * 0.8
                for pid in self._keyword_index[bigram]:
                    scores[pid] = scores.get(pid, 0.0) + weight
                    if pid not in matched_kws:
                        matched_kws[pid] = []
                    if bigram not in matched_kws[pid]:
                        matched_kws[pid].append(bigram)

        if not scores:
            return []

        # Normalize scores to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {pid: s / max_score for pid, s in scores.items()}

        # Build results
        results = []
        for pid, score in scores.items():
            if score < min_score:
                continue
            platform = self._platforms[pid]
            results.append(
                MatchResult(
                    platform_id=pid,
                    score=round(score, 4),
                    platform=platform,
                    matched_keywords=matched_kws.get(pid, []),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, platform_id: str) -> Optional[PlatformDefinition]:
        """Get a platform by ID."""
        self._ensure_loaded()
        return self._platforms.get(platform_id)

    def list_platforms(self) -> list[PlatformDefinition]:
        """Return all loaded platforms, sorted by ID."""
        self._ensure_loaded()
        return sorted(self._platforms.values(), key=lambda p: p.id)

    def list_categories(self) -> list[str]:
        """Return sorted list of unique categories."""
        self._ensure_loaded()
        return sorted({p.category for p in self._platforms.values()})

    def list_by_category(self, category: str) -> list[PlatformDefinition]:
        """Return platforms in a given category."""
        self._ensure_loaded()
        return sorted(
            [p for p in self._platforms.values() if p.category == category],
            key=lambda p: p.id,
        )

    @property
    def platform_count(self) -> int:
        """Number of loaded platforms."""
        self._ensure_loaded()
        return len(self._platforms)

    @property
    def data_dir(self) -> Path:
        return self._data_dir


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return [w for w in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower()) if len(w) > 1]


def _bigrams(tokens: list[str]) -> list[str]:
    """Generate space-joined bigrams from token list."""
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def _trigrams(tokens: list[str]) -> list[str]:
    """Generate space-joined trigrams from token list."""
    return [f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}" for i in range(len(tokens) - 2)]
