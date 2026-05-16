import pytest


def test_import_qwen_backend():
    pytest.importorskip("torch")
    from agenticlab.core.perception.vlm_backends.qwen_local import (  # noqa: F401
        QwenLocalBackend,
    )


def test_import_molmo_backend():
    pytest.importorskip("torch")
    from agenticlab.core.perception.vlm_backends.molmo_local import (  # noqa: F401
        MolmoLocalBackend,
    )
