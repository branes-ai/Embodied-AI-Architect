"""Base agent class for the Embodied AI Architect system."""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel


class AgentResult(BaseModel):
    """Standard result format for agent execution."""

    success: bool
    data: Dict[str, Any]
    error: str | None = None
    metadata: Dict[str, Any] = {}


class BaseAgent(ABC):
    """Base class for all agents in the system.

    Each agent implements the execute() method which processes input data
    and returns an AgentResult.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute the agent's primary function.

        Args:
            input_data: Dictionary containing input parameters for the agent

        Returns:
            AgentResult with success status, data, and optional error message
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
