"""LLM client wrapper for Claude API with tool use support."""

import os
from dataclasses import dataclass, field
from typing import Any

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """Wrapper for Claude API with tool use support."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        """Initialize the LLM client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use
            max_tokens: Maximum tokens in response
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. " "Install with: pip install anthropic"
            )

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages and get response, possibly with tool calls.

        Args:
            messages: Conversation history
            tools: Available tools in Anthropic format
            system: System prompt

        Returns:
            LLMResponse with text and/or tool calls
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        # Parse response
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        args=block.input,
                    )
                )

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
        )

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ):
        """Stream a chat response.

        Yields text chunks as they arrive. Tool calls are collected and
        returned at the end.

        Args:
            messages: Conversation history
            tools: Available tools
            system: System prompt

        Yields:
            str: Text chunks as they stream
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        if system:
            kwargs["system"] = system

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
