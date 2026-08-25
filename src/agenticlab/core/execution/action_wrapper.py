"""Hardware-independent pick, place, and move orchestration."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from agenticlab.core.hardware import BaseGripper, BaseRobotArm
from agenticlab.utils.geometry_transforms import (
    apply_local_rpy,
    camera_pose_from_wrist,
    compose_transforms,
    make_transform,
    offset_along_local_axis,
    pose6d_to_transform,
    tcp_pose_from_camera_grasp,
    transform_to_pose6d,
)

logger = logging.getLogger(__name__)


class ActionWrapper:
    """Execute manipulation primitives through injected hardware interfaces.

    All internal robot poses are 4x4 transforms named with the
    ``T_target_source`` convention. UR rotvec conversion belongs in the UR
    adapter; the compatibility conversion methods at the bottom contain no
    hardware calls.
    """

    def __init__(
        self,
        robot_arm: BaseRobotArm,
        gripper: BaseGripper,
        cfg: Dict,
        camera_cfg: Dict,
        *,
        connect_on_init: bool = True,
        reset_on_init: bool = True,
    ):
        self.robot_arm = robot_arm
        self.gripper = gripper
        self.config = cfg.get("ActionWrapper", cfg)
        self.camera_cfg = camera_cfg

        self.robot_speed = float(self.config.get("robot_speed", 0.05))
        self.robot_acceleration = float(self.config.get("robot_acceleration", 0.1))
        self.approach_distance = float(self.config.get("approach_distance", 0.1))
        self.clearance_height = float(self.config.get("clearance_height", 0.18))

        T_grasp_tcp_raw = self.config.get("T_grasp_tcp")
        if T_grasp_tcp_raw is None:
            raise ValueError("ActionWrapper config must define T_grasp_tcp.")
        self.T_grasp_tcp = np.asarray(T_grasp_tcp_raw, dtype=float)

        orientation_rotvec = np.asarray(
            self.config.get(
                "default_place_orientation_rotvec",
                [0.0, 0.0, 0.0],
            ),
            dtype=float,
        )
        if orientation_rotvec.shape != (3,):
            raise ValueError("default_place_orientation_rotvec must have shape (3,).")
        self.default_place_rotation = Rotation.from_rotvec(
            orientation_rotvec
        ).as_matrix()

        self.T_world_camera = self._camera_extrinsic("Kinect", "T_world_camera")
        self.T_tcp_camera = self._camera_extrinsic("RealSense", "T_tcp_camera")

        self.is_connected = True
        if connect_on_init:
            self.is_connected = self.robot_arm.connect()
            if not self.is_connected:
                raise ConnectionError("Failed to connect to the robot arm.")
        if reset_on_init:
            self.reset()

    def _camera_extrinsic(self, camera_name: str, transform_name: str) -> np.ndarray:
        camera = self.camera_cfg.get(camera_name)
        if not isinstance(camera, dict):
            raise ValueError(f"Camera config is missing '{camera_name}'.")
        calibration = camera.get("handeye_calibration")
        if not isinstance(calibration, dict):
            raise ValueError(
                f"{camera_name} config is missing handeye_calibration "
                f"for {transform_name}."
            )
        return make_transform(
            calibration.get("rotation"),
            calibration.get("translation"),
        )

    def _require_connection(self) -> None:
        if not self.is_connected:
            raise ConnectionError("Robot arm is not connected.")

    def _move_robot(self, T_world_tcp: np.ndarray) -> None:
        succeeded = self.robot_arm.move_pose(
            T_world_tcp,
            speed=self.robot_speed,
            acceleration=self.robot_acceleration,
        )
        if not succeeded:
            raise RuntimeError("Robot failed to execute move_pose.")

    def _world_camera(
        self, T_world_tcp_current: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if T_world_tcp_current is None:
            return self.T_world_camera.copy()
        return camera_pose_from_wrist(
            T_world_tcp_current,
            self.T_tcp_camera,
        )

    def camera_grasp_to_world_tcp(
        self,
        T_camera_grasp: np.ndarray,
        T_world_tcp_current: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert a camera-frame grasp to a world-frame TCP target."""
        T_world_camera = self._world_camera(T_world_tcp_current)
        return tcp_pose_from_camera_grasp(
            T_world_camera,
            T_camera_grasp,
            self.T_grasp_tcp,
        )

    def _camera_target_to_world(
        self,
        T_camera_target: np.ndarray,
        T_world_tcp_current: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert a generic camera-frame target without grasp/TCP alignment."""
        return compose_transforms(
            self._world_camera(T_world_tcp_current),
            T_camera_target,
        )

    def _generate_approach_pose(
        self,
        T_camera_grasp: np.ndarray,
        T_world_tcp_current: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        T_camera_approach = offset_along_local_axis(
            T_camera_grasp,
            distance=-self.approach_distance,
            axis=0,
        )
        return self.camera_grasp_to_world_tcp(
            T_camera_approach,
            T_world_tcp_current,
        )

    def pick(
        self,
        T_camera_grasp: np.ndarray,
        T_world_tcp_current: Optional[np.ndarray] = None,
        *,
        retreat: bool = True,
        grasp_offset: Optional[Sequence[float]] = None,
    ) -> None:
        """Execute open → approach → grasp → close → optional retreat."""
        self._require_connection()
        T_world_tcp_grasp = self.camera_grasp_to_world_tcp(
            T_camera_grasp,
            T_world_tcp_current,
        )
        if grasp_offset is not None:
            offset = np.asarray(grasp_offset, dtype=float)
            if offset.shape != (3,):
                raise ValueError("grasp_offset must have shape (3,).")
            T_world_tcp_grasp[:3, 3] += offset

        T_world_tcp_approach = self._generate_approach_pose(
            T_camera_grasp,
            T_world_tcp_current,
        )

        self.gripper.open()
        self._move_robot(T_world_tcp_approach)
        self._move_robot(T_world_tcp_grasp)
        self.gripper.close()
        if retreat:
            self._move_robot(T_world_tcp_approach)

    def move(
        self,
        T_target: np.ndarray,
        *,
        frame: str = "world",
        offset: Optional[Sequence[float]] = None,
        default_orientation_rotvec: Optional[Sequence[float]] = None,
        pure_translation: bool = True,
    ) -> None:
        """Move to a world/base- or camera-frame target pose."""
        self._require_connection()
        if frame == "camera":
            T_world_tcp = self.camera_grasp_to_world_tcp(T_target)
        elif frame in ("world", "base"):
            T_world_tcp = np.asarray(T_target, dtype=float).copy()
        else:
            raise ValueError("frame must be 'world', 'base', or 'camera'.")
        if T_world_tcp.shape != (4, 4):
            raise ValueError("T_target must have shape (4, 4).")

        if offset is not None:
            offset_vector = np.asarray(offset, dtype=float)
            if offset_vector.shape != (3,):
                raise ValueError("offset must have shape (3,).")
            T_world_tcp[:3, 3] += offset_vector

        if pure_translation:
            T_world_tcp[:3, :3] = self.robot_arm.get_pose()[:3, :3]
        if default_orientation_rotvec is not None:
            rotation_vector = np.asarray(default_orientation_rotvec, dtype=float)
            if rotation_vector.shape != (3,):
                raise ValueError("default_orientation_rotvec must have shape (3,).")
            T_world_tcp[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()

        self._move_robot(T_world_tcp)

    def oriented_place(
        self,
        T_target_place: np.ndarray,
        *,
        frame: str = "world",
        place_offset: float = 0.05,
        yaw_rotation: float = 0.0,
        roll_rotation: float = 0.0,
        T_world_tcp_current: Optional[np.ndarray] = None,
    ) -> None:
        """Apply orientation adjustments and execute a generic place sequence."""
        if frame == "camera":
            T_world_place = self._camera_target_to_world(
                T_target_place,
                T_world_tcp_current,
            )
            T_world_place[:3, :3] = self.default_place_rotation
        elif frame in ("world", "base"):
            T_world_place = np.asarray(T_target_place, dtype=float).copy()
        else:
            raise ValueError("frame must be 'world', 'base', or 'camera'.")

        T_world_place[2, 3] += place_offset
        T_world_place = apply_local_rpy(
            T_world_place,
            roll=np.deg2rad(roll_rotation),
            yaw=np.deg2rad(yaw_rotation),
        )
        self.place(
            T_world_place,
            place_offset=(0.0, 0.0, 0.0),
            retract_distance=0.15,
        )

    def place(
        self,
        T_target_place: np.ndarray,
        *,
        frame: str = "world",
        place_offset: Sequence[float] = (0.0, 0.0, 0.065),
        T_world_tcp_current: Optional[np.ndarray] = None,
        slight_open: bool = False,
        retract_distance: float = 0.05,
    ) -> None:
        """Execute clearance → XY → down → release → retract → reset."""
        self._require_connection()
        if frame == "camera":
            T_world_place = self._camera_target_to_world(
                T_target_place,
                T_world_tcp_current,
            )
            T_world_place[:3, :3] = self.default_place_rotation
        elif frame in ("world", "base"):
            T_world_place = np.asarray(T_target_place, dtype=float).copy()
        else:
            raise ValueError("frame must be 'world', 'base', or 'camera'.")
        if T_world_place.shape != (4, 4):
            raise ValueError("T_target_place must have shape (4, 4).")

        offset = np.asarray(place_offset, dtype=float)
        if offset.shape != (3,):
            raise ValueError("place_offset must have shape (3,).")
        T_world_place[:3, 3] += offset

        clearance_z = T_world_place[2, 3] + self.clearance_height
        T_world_tcp_clearance = self.robot_arm.get_pose()
        T_world_tcp_clearance[2, 3] = clearance_z

        T_world_tcp_xy = T_world_place.copy()
        T_world_tcp_xy[2, 3] = clearance_z

        self._move_robot(T_world_tcp_clearance)
        self._move_robot(T_world_tcp_xy)
        self._move_robot(T_world_place)
        self.gripper.open(slight_open=slight_open)

        T_world_tcp_retreat = T_world_place.copy()
        T_world_tcp_retreat[2, 3] += retract_distance
        self._move_robot(T_world_tcp_retreat)
        self.reset()

    def get_eef_pose(self) -> np.ndarray:
        self._require_connection()
        return self.robot_arm.get_pose()

    def move_to_home(self) -> None:
        self._require_connection()
        if not self.robot_arm.move_home():
            raise RuntimeError("Robot failed to move home.")

    def reset(self) -> None:
        self.move_to_home()
        self.gripper.open()

    def shutdown(self) -> None:
        """Release injected hardware resources."""
        self.gripper.disconnect()
        self.robot_arm.disconnect()
        self.is_connected = False

    # Compatibility helpers for legacy pipelines. New code should import these
    # conversions directly from agenticlab.utils.geometry_transforms.
    @staticmethod
    def transfer_T_to_pose6d(T_target_source: np.ndarray) -> np.ndarray:
        return transform_to_pose6d(T_target_source)

    @staticmethod
    def transfer_pose6d_to_T(pose6d: Sequence[float]) -> np.ndarray:
        return pose6d_to_transform(pose6d)

    @staticmethod
    def transfer_pos_rot_to_T(
        position: Sequence[float], rotation: np.ndarray
    ) -> np.ndarray:
        return make_transform(rotation, position)
