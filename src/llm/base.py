"""Abstract LLM interface — all backends implement this."""

from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 2048, temperature: float = 0.7, stop: list[str] | None = None) -> str:
        """Generate text from a prompt. Returns the generated string."""
        ...

    @abstractmethod
    def generate_code(self, prompt: str, *, max_tokens: int = 4096, temperature: float = 0.2) -> str:
        """Generate code with lower temperature. Returns raw code string."""
        ...
