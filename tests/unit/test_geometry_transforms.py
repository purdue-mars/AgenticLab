import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from agenticlab.utils.geometry_transforms import (
    apply_local_rpy,
    camera_pose_from_wrist,
    compose_transforms,
    invert_transform,
    make_transform,
    offset_along_local_axis,
    pixel_to_camera_point,
    pose6d_to_transform,
    tcp_pose_from_camera_grasp,
    transform_points,
    transform_to_pose6d,
)


def test_pose6d_transform_round_trip():
    pose6d = np.array([0.4, -0.2, 0.7, 0.0, 0.0, np.pi / 2])

    T_world_tcp = pose6d_to_transform(pose6d)
    recovered_pose6d = transform_to_pose6d(T_world_tcp)

    np.testing.assert_allclose(recovered_pose6d, pose6d, atol=1e-8)


def test_pose6d_transform_round_trip_at_180_degrees():
    T_world_tcp = pose6d_to_transform([0.4, -0.2, 0.7, np.pi, 0.0, 0.0])

    recovered_T_world_tcp = pose6d_to_transform(transform_to_pose6d(T_world_tcp))

    np.testing.assert_allclose(
        recovered_T_world_tcp,
        T_world_tcp,
        atol=1e-8,
    )


def test_transform_inverse_and_composition():
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T_world_camera = make_transform(rotation, [1.0, 2.0, 3.0])
    T_camera_world = invert_transform(T_world_camera)

    np.testing.assert_allclose(
        compose_transforms(T_world_camera, T_camera_world),
        np.eye(4),
        atol=1e-8,
    )


def test_transform_point_uses_target_source_direction():
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T_world_camera = make_transform(rotation, [1.0, 2.0, 0.0])
    p_camera = np.array([1.0, 0.0, 0.0])

    p_world = transform_points(T_world_camera, p_camera)

    np.testing.assert_allclose(p_world, [1.0, 3.0, 0.0], atol=1e-8)


def test_transform_multiple_points():
    T_world_camera = make_transform(np.eye(3), [1.0, 2.0, 3.0])
    points_camera = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    points_world = transform_points(T_world_camera, points_camera)

    np.testing.assert_allclose(
        points_world,
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
    )


def test_wrist_camera_chain():
    T_world_tcp = make_transform(np.eye(3), [0.5, 0.0, 0.0])
    T_tcp_camera = make_transform(np.eye(3), [0.0, 0.1, 0.0])

    T_world_camera = camera_pose_from_wrist(T_world_tcp, T_tcp_camera)

    np.testing.assert_allclose(
        T_world_camera,
        make_transform(np.eye(3), [0.5, 0.1, 0.0]),
    )


def test_camera_grasp_to_world_tcp_chain():
    T_world_camera = make_transform(np.eye(3), [1.0, 0.0, 0.0])
    T_camera_grasp = make_transform(np.eye(3), [0.0, 2.0, 0.0])
    T_grasp_tcp = make_transform(np.eye(3), [0.0, 0.0, 3.0])

    T_world_tcp = tcp_pose_from_camera_grasp(
        T_world_camera,
        T_camera_grasp,
        T_grasp_tcp,
    )

    np.testing.assert_allclose(
        T_world_tcp,
        make_transform(np.eye(3), [1.0, 2.0, 3.0]),
    )


def test_local_axis_offset_respects_source_rotation():
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T_camera_grasp = make_transform(rotation, [1.0, 2.0, 3.0])

    T_camera_approach = offset_along_local_axis(
        T_camera_grasp,
        distance=-0.1,
        axis=0,
    )

    np.testing.assert_allclose(
        T_camera_approach[:3, 3],
        [1.0, 1.9, 3.0],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        T_camera_approach[:3, :3],
        rotation,
        atol=1e-8,
    )


def test_apply_local_rpy_preserves_translation_and_order():
    T_world_tcp = make_transform(np.eye(3), [1.0, 2.0, 3.0])

    T_world_tcp_rotated = apply_local_rpy(
        T_world_tcp,
        roll=np.pi / 2,
        yaw=np.pi / 2,
    )

    expected_rotation = (
        Rotation.from_euler("x", 90, degrees=True).as_matrix()
        @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
    )
    np.testing.assert_allclose(
        T_world_tcp_rotated[:3, :3],
        expected_rotation,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        T_world_tcp_rotated[:3, 3],
        [1.0, 2.0, 3.0],
    )


def _pinhole_intrinsics():
    return np.array(
        [
            [100.0, 0.0, 320.0],
            [0.0, 200.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_pixel_at_principal_point_projects_along_optical_axis():
    depth_image = np.full((480, 640), 1500, dtype=np.uint16)

    p_camera = pixel_to_camera_point(
        [320, 240],
        depth_image,
        _pinhole_intrinsics(),
        factor_depth=1000.0,
    )

    np.testing.assert_allclose(p_camera, [0.0, 0.0, 1.5])


def test_pixel_offset_scales_with_focal_length_and_depth():
    depth_image = np.full((480, 640), 2000, dtype=np.uint16)

    p_camera = pixel_to_camera_point(
        [420, 340],
        depth_image,
        _pinhole_intrinsics(),
        factor_depth=1000.0,
    )

    # x: (420 - 320) * 2.0 / 100, y: (340 - 240) * 2.0 / 200
    np.testing.assert_allclose(p_camera, [2.0, 1.0, 2.0])


def test_factor_depth_converts_raw_units():
    depth_image = np.full((480, 640), 20000, dtype=np.uint16)

    p_camera = pixel_to_camera_point(
        [320, 240],
        depth_image,
        _pinhole_intrinsics(),
        factor_depth=10000.0,
    )

    np.testing.assert_allclose(p_camera, [0.0, 0.0, 2.0])


def test_pixel_outside_depth_image_is_rejected():
    depth_image = np.zeros((480, 640), dtype=np.uint16)

    with pytest.raises(ValueError, match="outside"):
        pixel_to_camera_point([-1, 240], depth_image, _pinhole_intrinsics())


def test_non_positive_factor_depth_is_rejected():
    depth_image = np.zeros((480, 640), dtype=np.uint16)

    with pytest.raises(ValueError, match="factor_depth"):
        pixel_to_camera_point(
            [320, 240],
            depth_image,
            _pinhole_intrinsics(),
            factor_depth=0.0,
        )


def test_invalid_transform_is_rejected():
    invalid_transform = np.eye(4)
    invalid_transform[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthonormal"):
        invert_transform(invalid_transform)
