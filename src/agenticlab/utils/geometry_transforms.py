"""Pure geometry helpers for rigid-body transformations.

Transform variables follow the mandatory ``T_target_source`` convention:
``T_target_source`` maps source-frame coordinates into target-frame
coordinates, so ``p_target = T_target_source @ p_source``.

This module deliberately has no robot, camera, or VLM dependencies.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

ArrayLike = Sequence[float] | np.ndarray


def _as_transform(T_target_source: ArrayLike) -> np.ndarray:
    """Return a validated copy of a homogeneous rigid transform."""
    transform = np.asarray(T_target_source, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Transform must have shape (4, 4), got {transform.shape}.")
    if not np.all(np.isfinite(transform)):
        raise ValueError("Transform must contain only finite values.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("Transform last row must be [0, 0, 0, 1].")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("Transform rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError("Transform rotation determinant must be +1.")
    return transform.copy()


def make_transform(rotation: ArrayLike, translation: ArrayLike) -> np.ndarray:
    """Construct ``T_target_source`` from a rotation and translation."""
    rotation_matrix = np.asarray(rotation, dtype=float)
    translation_vector = np.asarray(translation, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(
            f"Rotation must have shape (3, 3), got {rotation_matrix.shape}."
        )
    if translation_vector.shape != (3,):
        raise ValueError(
            f"Translation must have shape (3,), got {translation_vector.shape}."
        )

    T_target_source = np.eye(4)
    T_target_source[:3, :3] = rotation_matrix
    T_target_source[:3, 3] = translation_vector
    return _as_transform(T_target_source)


def invert_transform(T_target_source: ArrayLike) -> np.ndarray:
    """Invert ``T_target_source`` and return ``T_source_target``."""
    T_target_source = _as_transform(T_target_source)
    rotation = T_target_source[:3, :3]
    translation = T_target_source[:3, 3]

    T_source_target = np.eye(4)
    T_source_target[:3, :3] = rotation.T
    T_source_target[:3, 3] = -(rotation.T @ translation)
    return T_source_target


def compose_transforms(*transforms: ArrayLike) -> np.ndarray:
    """Compose compatible transforms in written left-to-right order.

    For example:
    ``compose_transforms(T_world_camera, T_camera_grasp)`` returns
    ``T_world_grasp``.
    """
    if not transforms:
        return np.eye(4)

    result = np.eye(4)
    for transform in transforms:
        result = result @ _as_transform(transform)
    return result


def transform_points(
    T_target_source: ArrayLike, points_source: ArrayLike
) -> np.ndarray:
    """Transform one ``(3,)`` point or an ``(N, 3)`` point array."""
    T_target_source = _as_transform(T_target_source)
    points = np.asarray(points_source, dtype=float)
    if points.ndim == 1:
        if points.shape != (3,):
            raise ValueError(f"Point must have shape (3,), got {points.shape}.")
        point_source_h = np.append(points, 1.0)
        return (T_target_source @ point_source_h)[:3]
    if points.ndim == 2 and points.shape[1] == 3:
        points_source_h = np.column_stack([points, np.ones(len(points))])
        return (T_target_source @ points_source_h.T).T[:, :3]
    raise ValueError(f"Points must have shape (3,) or (N, 3), got {points.shape}.")


def pose6d_to_transform(pose6d: ArrayLike) -> np.ndarray:
    """Convert ``[x, y, z, rx, ry, rz]`` rotvec pose to a transform.

    The final three values are an axis-angle rotation vector in radians, as
    used by UR RTDE TCP poses.
    """
    pose = np.asarray(pose6d, dtype=float)
    if pose.shape != (6,):
        raise ValueError(f"Pose must have shape (6,), got {pose.shape}.")
    rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
    return make_transform(rotation, pose[:3])


def transform_to_pose6d(T_target_source: ArrayLike) -> np.ndarray:
    """Convert a transform to ``[x, y, z, rx, ry, rz]`` rotvec pose."""
    T_target_source = _as_transform(T_target_source)
    rotation_vector = Rotation.from_matrix(T_target_source[:3, :3]).as_rotvec()
    return np.concatenate([T_target_source[:3, 3], rotation_vector])


def camera_pose_from_wrist(
    T_world_tcp: ArrayLike, T_tcp_camera: ArrayLike
) -> np.ndarray:
    """Return ``T_world_camera = T_world_tcp @ T_tcp_camera``."""
    return compose_transforms(T_world_tcp, T_tcp_camera)


def tcp_pose_from_camera_grasp(
    T_world_camera: ArrayLike,
    T_camera_grasp: ArrayLike,
    T_grasp_tcp: ArrayLike,
) -> np.ndarray:
    """Return the target TCP pose for a camera-frame grasp.

    ``T_world_tcp = T_world_camera @ T_camera_grasp @ T_grasp_tcp``.
    """
    return compose_transforms(
        T_world_camera,
        T_camera_grasp,
        T_grasp_tcp,
    )


def offset_along_local_axis(
    T_target_source: ArrayLike,
    distance: float,
    axis: int = 0,
) -> np.ndarray:
    """Translate along a source-frame axis while preserving orientation.

    ``axis`` is 0, 1, or 2 for local x, y, or z. A negative distance produces
    the legacy grasp approach behavior along local ``-x``.
    """
    if axis not in (0, 1, 2):
        raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z).")
    if not np.isfinite(distance):
        raise ValueError("Distance must be finite.")

    T_target_source = _as_transform(T_target_source)
    local_offset = np.zeros(3)
    local_offset[axis] = distance
    T_target_source[:3, 3] += T_target_source[:3, :3] @ local_offset
    return T_target_source


def apply_local_rpy(
    T_target_source: ArrayLike,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """Apply local roll, pitch, then yaw rotations, in radians.

    This preserves the legacy controller order:
    ``R_new = R_current @ R_x(roll) @ R_y(pitch) @ R_z(yaw)``.
    Translation is unchanged.
    """
    T_target_source = _as_transform(T_target_source)
    rotation_delta = (
        Rotation.from_rotvec([roll, 0.0, 0.0]).as_matrix()
        @ Rotation.from_rotvec([0.0, pitch, 0.0]).as_matrix()
        @ Rotation.from_rotvec([0.0, 0.0, yaw]).as_matrix()
    )
    T_target_source[:3, :3] = T_target_source[:3, :3] @ rotation_delta
    return T_target_source
