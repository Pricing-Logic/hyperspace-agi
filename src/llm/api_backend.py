"""Anthropic Claude API backend — useful for prototyping before switching to local."""

import os
from .base import LLMInterface


class AnthropicBackend(LLMInterface):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, prompt: str, *, max_tokens: int = 2048, temperature: float = 0.7, stop: list[str] | None = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_code(self, prompt: str, *, max_tokens: int = 4096, temperature: float = 0.2) -> str:
        code_prompt = f"Write Python code only. No markdown, no explanations. Just the code.\n\n{prompt}"
        result = self.generate(code_prompt, max_tokens=max_tokens, temperature=temperature)
        if "```python" in result:
            result = result.split("```python")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        return result.strip()
