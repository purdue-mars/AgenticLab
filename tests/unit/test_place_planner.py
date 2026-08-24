from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest
from PIL import Image

from agenticlab.core.llm import BaseLLMClient
from agenticlab.core.planning.place_planner import PlacePlanner


class _MockLLMClient(BaseLLMClient):
    def __init__(self, responses: List[Dict[str, Any]]):
        super().__init__({"model": "mock"})
        self._responses = list(responses)
        self.call_count = 0

    def call(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, list]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return dict(self._responses[idx])

    def _build_messages(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, list]] = None,
    ) -> list:
        return [{"role": "user", "content": prompt}]


def _make_planner(llm, tmp_path, **overrides):
    cfg = {
        "PlacePlanner": {
            "output_dir": str(tmp_path / "place_planner"),
            "reflection_prompt": (
                "target={target_object}\ntask={task_description}\n"
                "point={current_point}\niter={iteration}\n"
            ),
            "oriented_pose_prompt": "task={task_description}\n",
            "use_reflection": True,
            "max_iterations": 3,
            **overrides,
        }
    }
    return PlacePlanner(llm_client=llm, cfg=cfg)


def test_plan_placing_point_no_reflection(tmp_path):
    llm = _MockLLMClient([{}])
    planner = _make_planner(llm, tmp_path, use_reflection=False)
    planner.camera_cfg = None  # no intrinsics available

    image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    depth = np.zeros((64, 64), dtype=np.uint16)
    point = np.array([32, 32])

    placing_point = planner.plan_placing_point(
        image=image,
        depth=depth,
        initial_point=point,
        target_place="plate",
        task_description="place the cube on the plate",
    )

    assert placing_point.point == [32, 32]
    assert placing_point.pose_3d is None  # no intrinsics, so no projection
    assert placing_point.visualization_path is not None
    assert llm.call_count == 0


def test_plan_placing_point_uses_default_projection(tmp_path):
    llm = _MockLLMClient([{}])
    planner = _make_planner(llm, tmp_path, use_reflection=False)
    planner.camera_cfg = {
        "Kinect": {
            "intrinsic_matrix": [
                [100.0, 0.0, 32.0],
                [0.0, 100.0, 32.0],
                [0.0, 0.0, 1.0],
            ],
            "factor_depth": 1000.0,
        }
    }

    depth = np.full((64, 64), 2000, dtype=np.uint16)  # 2.0 m at factor 1000

    placing_point = planner.plan_placing_point(
        image=Image.new("RGB", (64, 64)),
        depth=depth,
        initial_point=np.array([42, 32]),
        target_place="plate",
        task_description="place the cube on the plate",
    )

    # (42 - 32) * 2.0 / 100 = 0.2 along x; the pixel sits on the principal
    # row, so y stays 0.
    assert placing_point.pose_3d == pytest.approx([0.2, 0.0, 2.0])


def test_plan_placing_point_reflection_adjust_then_success(tmp_path):
    llm = _MockLLMClient(
        [
            {"action": "NEED_ADJUSTMENT", "adjusted_point": [20, 40]},  # [y, x]
            {"action": "SUCCESS"},
        ]
    )
    planner = _make_planner(llm, tmp_path)

    image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    depth = np.zeros((64, 64), dtype=np.uint16)

    placing_point = planner.plan_placing_point(
        image=image,
        depth=depth,
        initial_point=np.array([32, 32]),
        target_place="plate",
        task_description="place the cube on the plate",
    )

    assert llm.call_count == 2
    assert placing_point.point == [40, 20]  # converted from [y, x] to [x, y]
    assert len(placing_point.evaluation) == 2


def test_plan_placing_point_with_injected_pixel_to_3d(tmp_path):
    llm = _MockLLMClient([{}])

    def fake_pixel_to_3d(point, depth, intrinsics):
        return np.array([0.1, 0.2, 0.3])

    cfg = {
        "PlacePlanner": {
            "output_dir": str(tmp_path / "place_planner"),
            "reflection_prompt": "x",
            "oriented_pose_prompt": "x",
            "use_reflection": False,
        }
    }
    planner = PlacePlanner(llm_client=llm, cfg=cfg, pixel_to_3d_fn=fake_pixel_to_3d)
    planner.camera_cfg = {"Kinect": {"intrinsic_matrix": np.eye(3).tolist()}}

    placing_point = planner.plan_placing_point(
        image=Image.new("RGB", (64, 64)),
        depth=np.zeros((64, 64), dtype=np.uint16),
        initial_point=np.array([32, 32]),
        target_place="plate",
        task_description="place",
    )

    assert placing_point.pose_3d == [0.1, 0.2, 0.3]


def test_plan_oriented_place_pose(tmp_path):
    llm = _MockLLMClient(
        [
            {
                "yaw_rotation": 90.0,
                "roll_rotation": 15.0,
                "place_offset": 0.08,
                "reasoning": "rotate to upright",
            }
        ]
    )
    planner = _make_planner(llm, tmp_path)

    from agenticlab.core.data_types import PlacingPoint

    placing_point = PlacingPoint(point=[32, 32])
    updated = planner.plan_oriented_place_pose(
        image=Image.new("RGB", (64, 64)),
        task_description="place the bottle upright",
        placing_point=placing_point,
    )

    assert updated.yaw_rotation == 90.0
    assert updated.roll_rotation == 15.0
    assert updated.place_offset == 0.08
    assert updated.orientation_reasoning == "rotate to upright"
