from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pytest
from PIL import Image

from agenticlab.core.llm import BaseLLMClient
from agenticlab.core.planning.grasp_planner import GraspPlanner


class _MockLLMClient(BaseLLMClient):
    def __init__(self, response: Dict[str, Any]):
        super().__init__({"model": "mock"})
        self._response = response
        self.last_prompt: Optional[str] = None

    def call(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, list]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.last_prompt = prompt
        return dict(self._response)

    def _build_messages(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, list]] = None,
    ) -> list:
        return [{"role": "user", "content": prompt}]


def _make_planner(llm, tmp_path, **overrides):
    cfg = {
        "GraspPlanner": {
            "output_dir": str(tmp_path / "grasp_planner"),
            "evaluation_prompt": ("target={target_object}\ntask={task_description}\n"),
            "use_vlm_for_grasp_evaluation": True,
            **overrides,
        }
    }
    return GraspPlanner(llm_client=llm, cfg=cfg)


def test_evaluate_grasp_with_mock_llm(tmp_path):
    llm = _MockLLMClient({"grasp_quality": "good", "score": 0.9})
    planner = _make_planner(llm, tmp_path)

    img_path = tmp_path / "grasp_viz.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_path)

    result = planner.evaluate_grasp(
        grasp_image_path=str(img_path),
        target_object="red cube",
        task_description="pick up the red cube",
    )

    assert result["grasp_quality"] == "good"
    assert "elapsed_time" in result
    assert "target=red cube" in llm.last_prompt
    assert "task=pick up the red cube" in llm.last_prompt


def test_graspnet_backend_unavailable_raises(tmp_path):
    llm = _MockLLMClient({})
    planner = _make_planner(llm, tmp_path)
    with pytest.raises((ImportError, RuntimeError)):
        planner._get_graspnet_wrapper("Kinect")
