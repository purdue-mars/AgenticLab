import importlib
import logging
from typing import Dict

from .base import BaseLLMClient

logger = logging.getLogger(__name__)

_ADAPTER_MODULES = {
    "openai": ("agenticlab.core.llm.adapters.openai_adapter", "OpenAIAdapter"),
    "gemini": ("agenticlab.core.llm.adapters.gemini_adapter", "GeminiAdapter"),
    "claude": ("agenticlab.core.llm.adapters.claude_adapter", "ClaudeAdapter"),
    "qwen": ("agenticlab.core.llm.adapters.qwen_adapter", "QwenAdapter"),
    "qwen_local": (
        "agenticlab.core.llm.adapters.qwen_local_adapter",
        "QwenLocalAdapter",
    ),
    "molmo_local": (
        "agenticlab.core.llm.adapters.molmo_local_adapter",
        "MolmoLocalAdapter",
    ),
}


class LLMClientFactory:
    @classmethod
    def create(cls, provider: str, config: Dict) -> BaseLLMClient:
        provider = provider.lower()
        if provider not in _ADAPTER_MODULES:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {list(_ADAPTER_MODULES.keys())}"
            )
        module_path, class_name = _ADAPTER_MODULES[provider]
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import adapter for '{provider}'. "
                f"Make sure its dependencies are installed.\n  {e}"
            ) from e
        adapter_class = getattr(module, class_name)
        logger.info(f"Creating {provider} client")
        return adapter_class(config)

    @classmethod
    def register(cls, provider: str, module_path: str, class_name: str) -> None:
        _ADAPTER_MODULES[provider.lower()] = (module_path, class_name)
        logger.info(f"Registered new adapter: {provider}")
