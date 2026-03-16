"""Tag-based research library for LLM context enrichment.

Loads curated markdown documents with structured frontmatter, retrieves
them by tag/domain/relevance matching, and formats them as context blocks
for injection into LLM prompts.

Not a vector database — intentionally simple, auditable, and deterministic.

Usage:
    from embodied_ai_architect.research.library import ResearchLibrary

    lib = ResearchLibrary()  # auto-discovers default library path
    docs = lib.retrieve(tags=["drone", "edge"], relevance="mission_decomposition")
    context = lib.build_context_block(docs, max_tokens=8000)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default library location relative to package root
_DEFAULT_LIBRARY_PATH = Path(__file__).resolve().parents[3] / "data" / "research_library"


class Document(BaseModel):
    """A single research library document with metadata and content."""

    path: str = Field(..., description="Relative path within the library")
    title: str = Field(default="", description="Document title")
    domain: str = Field(default="", description="Domain category")
    tags: list[str] = Field(default_factory=list, description="Tags for retrieval")
    relevance: list[str] = Field(
        default_factory=list,
        description="Relevance contexts (e.g. 'mission_decomposition')",
    )
    frontmatter: dict[str, Any] = Field(
        default_factory=dict,
        description="Full frontmatter including quantitative fields",
    )
    content: str = Field(default="", description="Markdown body (without frontmatter)")

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        return len(self.content) // 4


class ResearchLibrary:
    """Curated HW accelerator research for LLM context enrichment.

    Loads documents from a directory of markdown files with YAML frontmatter.
    Retrieval is tag-based, not vector-similarity — deterministic and auditable.
    """

    def __init__(self, library_path: Path | str | None = None):
        self.library_path = Path(library_path) if library_path else _DEFAULT_LIBRARY_PATH
        self._documents: list[Document] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load documents on first access."""
        if self._loaded:
            return
        self._load_all()
        self._loaded = True

    def _load_all(self) -> None:
        """Load all documents from the library directory."""
        index_path = self.library_path / "index.yaml"
        if index_path.exists():
            self._load_from_index(index_path)
        else:
            self._load_from_glob()

    def _load_from_index(self, index_path: Path) -> None:
        """Load documents listed in index.yaml."""
        with open(index_path) as f:
            index = yaml.safe_load(f) or {}

        for entry in index.get("documents", []):
            doc_path = self.library_path / entry["path"]
            if doc_path.exists():
                doc = self._parse_document(doc_path, entry["path"])
                if doc:
                    self._documents.append(doc)
            else:
                logger.warning("Document not found: %s", doc_path)

    def _load_from_glob(self) -> None:
        """Fallback: load all .md files in the library directory."""
        for md_file in sorted(self.library_path.rglob("*.md")):
            rel_path = str(md_file.relative_to(self.library_path))
            doc = self._parse_document(md_file, rel_path)
            if doc:
                self._documents.append(doc)

    def _parse_document(self, file_path: Path, rel_path: str) -> Optional[Document]:
        """Parse a markdown file with YAML frontmatter into a Document."""
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s: %s", file_path, e)
            return None

        # Split frontmatter from content
        frontmatter: dict[str, Any] = {}
        content = text

        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if fm_match:
            try:
                frontmatter = yaml.safe_load(fm_match.group(1)) or {}
            except yaml.YAMLError as e:
                logger.warning("Bad frontmatter in %s: %s", file_path, e)
            content = fm_match.group(2)

        return Document(
            path=rel_path,
            title=frontmatter.get("title", ""),
            domain=frontmatter.get("domain", ""),
            tags=frontmatter.get("tags", []),
            relevance=frontmatter.get("relevance", []),
            frontmatter=frontmatter,
            content=content.strip(),
        )

    @property
    def documents(self) -> list[Document]:
        """All loaded documents."""
        self._ensure_loaded()
        return list(self._documents)

    @property
    def count(self) -> int:
        """Number of documents in the library."""
        self._ensure_loaded()
        return len(self._documents)

    def retrieve(
        self,
        tags: list[str] | None = None,
        domain: str | None = None,
        relevance: str | None = None,
        mission_type: str | None = None,
        max_results: int = 10,
    ) -> list[Document]:
        """Retrieve documents matching criteria, ranked by relevance.

        Scoring: each matching tag = 1 point, domain match = 2 points,
        relevance match = 3 points, mission_type tag match = 2 points.
        Results sorted by score descending.

        Args:
            tags: Tags to match (any overlap counts).
            domain: Domain filter (exact match gets bonus).
            relevance: Relevance context (e.g. 'mission_decomposition').
            mission_type: Mission type to match against tags (e.g. 'drone').
            max_results: Maximum documents to return.

        Returns:
            Matching documents sorted by relevance score.
        """
        self._ensure_loaded()

        tag_set = set(t.lower() for t in tags) if tags else set()
        scored: list[tuple[float, Document]] = []

        for doc in self._documents:
            score = 0.0
            doc_tags = set(t.lower() for t in doc.tags)

            # Tag overlap
            if tag_set:
                overlap = len(tag_set & doc_tags)
                score += overlap

            # Domain match
            if domain and doc.domain.lower() == domain.lower():
                score += 2.0

            # Relevance match
            if relevance and relevance in doc.relevance:
                score += 3.0

            # Mission type match (check if mission_type appears in tags)
            if mission_type:
                mt_lower = mission_type.lower()
                if mt_lower in doc_tags:
                    score += 2.0
                # Also check partial matches (e.g., "drone" matches "drone")
                for tag in doc_tags:
                    if mt_lower in tag or tag in mt_lower:
                        score += 0.5
                        break

            if score > 0:
                scored.append((score, doc))

        # Sort by score descending, then by title for stability
        scored.sort(key=lambda x: (-x[0], x[1].title))
        return [doc for _, doc in scored[:max_results]]

    def build_context_block(
        self,
        documents: list[Document],
        max_tokens: int = 8000,
        include_frontmatter: bool = False,
    ) -> str:
        """Format retrieved documents as a context block for LLM prompts.

        Fits documents within the token budget, prioritizing earlier (higher-
        ranked) documents.

        Args:
            documents: Documents to include (already sorted by relevance).
            max_tokens: Approximate token budget.
            include_frontmatter: Include quantitative frontmatter summary.

        Returns:
            Formatted context string ready for prompt injection.
        """
        if not documents:
            return ""

        parts = ["<research_context>"]
        tokens_used = 10  # overhead for tags

        for doc in documents:
            # Build document block
            header = f"\n### {doc.title}"
            if include_frontmatter:
                fm_parts = []
                for key in ("peak_tops", "compute_density_tops_w", "process_nm"):
                    if key in doc.frontmatter:
                        fm_parts.append(f"{key}: {doc.frontmatter[key]}")
                if fm_parts:
                    header += f"\n*{', '.join(fm_parts)}*"

            block = f"{header}\n\n{doc.content}"
            block_tokens = len(block) // 4

            if tokens_used + block_tokens > max_tokens:
                # Try truncating the content to fit
                remaining_tokens = max_tokens - tokens_used - 20
                if remaining_tokens > 200:
                    truncated_chars = remaining_tokens * 4
                    block = f"{header}\n\n{doc.content[:truncated_chars]}\n\n[... truncated]"
                    parts.append(block)
                break

            parts.append(block)
            tokens_used += block_tokens

        parts.append("\n</research_context>")
        return "\n".join(parts)

    def get_by_path(self, path: str) -> Optional[Document]:
        """Get a specific document by its relative path."""
        self._ensure_loaded()
        for doc in self._documents:
            if doc.path == path:
                return doc
        return None

    def list_tags(self) -> list[str]:
        """List all unique tags across all documents."""
        self._ensure_loaded()
        tags: set[str] = set()
        for doc in self._documents:
            tags.update(doc.tags)
        return sorted(tags)

    def list_domains(self) -> list[str]:
        """List all unique domains."""
        self._ensure_loaded()
        return sorted({doc.domain for doc in self._documents if doc.domain})
