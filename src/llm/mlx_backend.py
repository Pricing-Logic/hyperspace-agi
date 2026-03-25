"""MLX backend for local Apple Silicon inference."""

from .base import LLMInterface


class MLXBackend(LLMInterface):
    def __init__(self, model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit", adapter_path: str | None = None):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            from mlx_lm import load
            kwargs = {}
            if self.adapter_path:
                kwargs["adapter_path"] = self.adapter_path
            self._model, self._tokenizer = load(self.model_name, **kwargs)

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply the model's chat template to a raw prompt string."""
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            return formatted
        except Exception:
            # Fallback for models without chat template
            return prompt

    def generate(self, prompt: str, *, max_tokens: int = 2048, temperature: float = 0.7, stop: list[str] | None = None) -> str:
        self._load()
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        formatted = self._apply_chat_template(prompt)
        sampler = make_sampler(temp=temperature)
        response = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        return response

    def generate_code(self, prompt: str, *, max_tokens: int = 4096, temperature: float = 0.2) -> str:
        code_prompt = f"Write Python code only. No explanations.\n\n{prompt}"
        result = self.generate(code_prompt, max_tokens=max_tokens, temperature=temperature)
        if "```python" in result:
            result = result.split("```python")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        return result.strip()

    def unload(self):
        """Unload model from memory (for freeing RAM before fine-tuning)."""
        self._model = None
        self._tokenizer = None
        try:
            import gc
            import mlx.core as mx
            gc.collect()
            mx.metal.clear_cache()
        except Exception:
            pass
