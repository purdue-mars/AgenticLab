def test_import_agenticlab():
    import agenticlab

    assert agenticlab.__version__ == "0.1.0"


def test_import_core_llm():
    from agenticlab.core.llm import BaseLLMClient  # noqa: F401


def test_import_utils():
    from agenticlab.utils import log_utils  # noqa: F401


def test_import_llm_factory():
    from agenticlab.core.llm import LLMClientFactory  # noqa: F401


def test_factory_rejects_unknown_provider():
    import pytest

    from agenticlab.core.llm import LLMClientFactory

    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMClientFactory.create("nonexistent", {})
