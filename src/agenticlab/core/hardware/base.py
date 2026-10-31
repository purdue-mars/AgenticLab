"""SDK-free interfaces for hardware used by AgenticLab.

The interfaces are intentionally derived from the existing pipeline and
``action_wrapper`` call sites. Concrete UR RTDE, Dynamixel, Kinect, and
RealSense adapters translate these generic operations to their SDKs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseCamera(ABC):
    """RGB-D source used by the perception and planning pipeline."""

    @abstractmethod
    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return an RGB image and aligned depth image."""

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""

    def __enter__(self) -> "BaseCamera":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class BaseRobotArm(ABC):
    """Robot-arm operations required by action execution.

    Poses use 4x4 ``T_world_tcp`` matrices. Hardware-specific representations,
    such as the UR ``[x, y, z, rx, ry, rz]`` rotvec pose, belong in adapters.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot and return whether the connection succeeded."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""

    @abstractmethod
    def move_pose(
        self,
        T_world_tcp: np.ndarray,
        *,
        speed: Optional[float] = None,
        acceleration: Optional[float] = None,
    ) -> bool:
        """Move the TCP to ``T_world_tcp``."""

    @abstractmethod
    def get_pose(self) -> np.ndarray:
        """Return the current TCP pose as ``T_world_tcp``."""

    @abstractmethod
    def move_home(self) -> bool:
        """Move to the adapter-configured home pose."""


class BaseGripper(ABC):
    """Binary gripper operations required by action execution."""

    @abstractmethod
    def open(self, *, slight_open: bool = False) -> bool:
        """Open the gripper, optionally only part way."""

    @abstractmethod
    def close(self) -> bool:
        """Close the gripper."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release gripper resources."""
