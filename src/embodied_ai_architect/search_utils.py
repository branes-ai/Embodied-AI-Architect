"""Shared search utilities for TF-IDF keyword matching.

Used by both the platform and sensor registries to tokenize text,
generate n-grams, and filter stop words consistently.
"""

from __future__ import annotations

import re

# Shared stop-words — union of platform and sensor registry sets.
STOP_WORDS = frozenset(
    {
        "a",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "back",
        "be",
        "been",
        "being",
        "both",
        "but",
        "by",
        "can",
        "come",
        "could",
        "did",
        "do",
        "each",
        "find",
        "for",
        "from",
        "get",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "know",
        "let",
        "like",
        "look",
        "made",
        "make",
        "man",
        "many",
        "may",
        "more",
        "most",
        "much",
        "must",
        "my",
        "need",
        "new",
        "next",
        "no",
        "not",
        "now",
        "of",
        "old",
        "on",
        "one",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "run",
        "say",
        "see",
        "set",
        "she",
        "should",
        "so",
        "some",
        "such",
        "take",
        "than",
        "that",
        "the",
        "them",
        "then",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "try",
        "two",
        "up",
        "us",
        "use",
        "very",
        "via",
        "was",
        "way",
        "we",
        "well",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
        "about",
        "also",
        "all",
    }
)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, filtering stop words.

    Handles hyphenated and underscore-joined compound words as single tokens.
    """
    return [
        w
        for w in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
        if len(w) > 1 and w not in STOP_WORDS
    ]


def bigrams(tokens: list[str]) -> list[str]:
    """Generate space-joined bigrams from token list."""
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def trigrams(tokens: list[str]) -> list[str]:
    """Generate space-joined trigrams from token list."""
    return [f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}" for i in range(len(tokens) - 2)]
