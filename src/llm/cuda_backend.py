"""PyTorch/Transformers backend for CUDA GPUs (vast.ai, Modal, etc.)."""

import os
from .base import LLMInterface


class CUDABackend(LLMInterface):
    """HuggingFace Transformers backend for NVIDIA GPUs.

    Supports 4-bit quantization via bitsandbytes to fit larger models
    on consumer GPUs (e.g., 7B on 24GB 4090).
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", adapter_path: str | None = None, load_in_4bit: bool = False):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [CUDA] Loading {self.model_name} on {self._device}...", flush=True)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        load_kwargs = {
            "trust_remote_code": True,
        }

        if self.load_in_4bit and self._device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        elif self._device == "cuda":
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            print(f"  [CUDA] Loaded LoRA adapter from {self.adapter_path}", flush=True)

        self._model.eval()
        print(f"  [CUDA] Model ready on {self._device}", flush=True)

    def generate(self, prompt: str, *, max_tokens: int = 2048, temperature: float = 0.7, stop: list[str] | None = None) -> str:
        self._load()
        import torch

        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
                top_p=0.9,
            )

        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
        """Free GPU memory properly."""
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
        self._model = None
        self._tokenizer = None
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
