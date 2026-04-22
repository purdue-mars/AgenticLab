import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses/containers to JSON-friendly primitives."""
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


@dataclass
class GraspPose:
    translation: List[float]  # [x, y, z]
    rotation: List[List[float]]  # 3x3
    width: float
    evaluation: Optional[Dict[str, Any]] = None
    visualization_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class DetectionResult:
    success: bool
    objects: List[Dict[str, Any]]
    image_shape: Tuple[int, int]  # (H, W) 和旧代码一致
    raw_output: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class CheckResult:
    """Result of action checking operations."""

    success: bool
    reasoning: str
    timestamp: str
    elapsed_time: float
    failed_conditions: Optional[List[str]] = None
    detailed_analysis: Optional[List[Dict[str, Any]]] = None
    token_usage: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class PlacingPoint:
    """Represents a placing point with evaluation results"""

    point: List[float]  # [x, y] pixel coordinates
    pose_3d: Optional[List[float]] = None  # 3D pose in camera frame
    evaluation: Optional[List[Dict[str, Any]]] = None  # Per-iteration results
    visualization_path: Optional[str] = None
    yaw_rotation: float = 0.0  # Rotation around Z-axis in degrees
    roll_rotation: float = 0.0  # Rotation around X-axis in degrees
    place_offset: float = 0.05  # Vertical offset in meters
    orientation_reasoning: Optional[str] = None  # Reasoning for orientation

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)


@dataclass
class TaskPlan:
    task_description: str
    objects_identified: List[str]
    reasoning: str
    action_sequence: List[str]
    elapsed_time: float
    timestamp: str
    updated_domain: Optional[str] = None
    problem_pddl: Optional[str] = None
    action_details: Optional[List[Dict[str, Any]]] = None
    goal_conditions: Optional[List[str]] = None
    token_usage: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)
