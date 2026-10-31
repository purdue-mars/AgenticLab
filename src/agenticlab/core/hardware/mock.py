"""SDK-free mock hardware for tests and examples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from agenticlab.core.hardware.base import BaseCamera, BaseGripper, BaseRobotArm


class MockCamera(BaseCamera):
    """Replay one recorded RGB-D frame."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray):
        rgb_array = np.asarray(rgb)
        depth_array = np.asarray(depth)
        if rgb_array.ndim != 3 or rgb_array.shape[2] not in (3, 4):
            raise ValueError("RGB image must have shape (H, W, 3) or (H, W, 4).")
        if depth_array.ndim != 2:
            raise ValueError("Depth image must have shape (H, W).")
        if rgb_array.shape[:2] != depth_array.shape:
            raise ValueError("RGB and depth images must have matching dimensions.")

        self._rgb = rgb_array[:, :, :3].copy()
        self._depth = depth_array.copy()
        self.closed = False
        self.capture_count = 0

    @classmethod
    def from_files(cls, rgb_path: str | Path, depth_path: str | Path) -> "MockCamera":
        """Load a recorded frame from image files."""
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.asarray(Image.open(depth_path))
        return cls(rgb=rgb, depth=depth)

    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.closed:
            raise RuntimeError("Cannot capture from a closed camera.")
        self.capture_count += 1
        return self._rgb.copy(), self._depth.copy()

    def close(self) -> None:
        self.closed = True


class MockRobotArm(BaseRobotArm):
    """Record robot commands while maintaining a simulated TCP pose."""

    def __init__(
        self,
        initial_pose: Optional[np.ndarray] = None,
        home_pose: Optional[np.ndarray] = None,
        *,
        connected: bool = True,
        event_log: Optional[List[Dict]] = None,
    ):
        self.home_pose = self._validate_pose(
            np.eye(4) if home_pose is None else home_pose
        )
        self._pose = self._validate_pose(
            self.home_pose if initial_pose is None else initial_pose
        )
        self.connected = connected
        self.commands: List[Dict] = []
        self.event_log = event_log

    @staticmethod
    def _validate_pose(T_world_tcp: np.ndarray) -> np.ndarray:
        pose = np.asarray(T_world_tcp, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("Robot pose must have shape (4, 4).")
        return pose.copy()

    def connect(self) -> bool:
        self.connected = True
        self._record({"device": "robot", "action": "connect"})
        return True

    def disconnect(self) -> None:
        self.connected = False
        self._record({"device": "robot", "action": "disconnect"})

    def _record(self, command: Dict) -> None:
        self.commands.append(command)
        if self.event_log is not None:
            self.event_log.append(command)

    def move_pose(
        self,
        T_world_tcp: np.ndarray,
        *,
        speed: Optional[float] = None,
        acceleration: Optional[float] = None,
    ) -> bool:
        if not self.connected:
            raise ConnectionError("Mock robot is not connected.")
        pose = self._validate_pose(T_world_tcp)
        self._pose = pose
        self._record(
            {
                "device": "robot",
                "action": "move_pose",
                "T_world_tcp": pose.copy(),
                "speed": speed,
                "acceleration": acceleration,
            }
        )
        return True

    def get_pose(self) -> np.ndarray:
        if not self.connected:
            raise ConnectionError("Mock robot is not connected.")
        return self._pose.copy()

    def move_home(self) -> bool:
        if not self.connected:
            raise ConnectionError("Mock robot is not connected.")
        self._pose = self.home_pose.copy()
        self._record({"device": "robot", "action": "move_home"})
        return True


class MockGripper(BaseGripper):
    """Record binary gripper commands."""

    def __init__(self, *, event_log: Optional[List[Dict]] = None):
        self.is_open = True
        self.disconnected = False
        self.commands: List[Dict] = []
        self.event_log = event_log

    def _record(self, command: Dict) -> None:
        self.commands.append(command)
        if self.event_log is not None:
            self.event_log.append(command)

    def open(self, *, slight_open: bool = False) -> bool:
        if self.disconnected:
            raise ConnectionError("Mock gripper is disconnected.")
        self.is_open = True
        self._record(
            {
                "device": "gripper",
                "action": "open",
                "slight_open": slight_open,
            }
        )
        return True

    def close(self) -> bool:
        if self.disconnected:
            raise ConnectionError("Mock gripper is disconnected.")
        self.is_open = False
        self._record({"device": "gripper", "action": "close"})
        return True

    def disconnect(self) -> None:
        self.disconnected = True
        self._record({"device": "gripper", "action": "disconnect"})
