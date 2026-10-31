import numpy as np
import pytest
from PIL import Image

from agenticlab.core.hardware import MockCamera, MockGripper, MockRobotArm
from agenticlab.utils.geometry_transforms import make_transform


def test_mock_camera_replays_copies():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth = np.full((4, 5), 1000, dtype=np.uint16)
    camera = MockCamera(rgb, depth)

    captured_rgb, captured_depth = camera.capture()
    captured_rgb[0, 0] = 255
    captured_depth[0, 0] = 0
    replayed_rgb, replayed_depth = camera.capture()

    assert camera.capture_count == 2
    assert np.all(replayed_rgb[0, 0] == 0)
    assert replayed_depth[0, 0] == 1000


def test_mock_camera_loads_recorded_files(tmp_path):
    rgb_path = tmp_path / "color.png"
    depth_path = tmp_path / "depth.png"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.full((3, 4), 1200, dtype=np.uint16)).save(depth_path)

    camera = MockCamera.from_files(rgb_path, depth_path)
    rgb, depth = camera.capture()

    assert rgb.shape == (3, 4, 3)
    assert depth.shape == (3, 4)
    assert depth.dtype == np.uint16


def test_mock_camera_context_manager_closes_camera():
    camera = MockCamera(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint16),
    )

    with camera as opened_camera:
        opened_camera.capture()

    assert camera.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        camera.capture()


def test_mock_robot_records_generic_transform_commands():
    robot = MockRobotArm()
    T_world_tcp = make_transform(np.eye(3), [0.1, 0.2, 0.3])

    robot.move_pose(T_world_tcp, speed=0.05, acceleration=0.1)

    np.testing.assert_allclose(robot.get_pose(), T_world_tcp)
    command = robot.commands[-1]
    assert command["action"] == "move_pose"
    assert command["speed"] == 0.05
    assert command["acceleration"] == 0.1


def test_mock_robot_home_and_disconnect():
    T_world_home = make_transform(np.eye(3), [0.0, 0.0, 0.5])
    robot = MockRobotArm(home_pose=T_world_home)
    robot.move_pose(make_transform(np.eye(3), [1.0, 2.0, 3.0]))

    assert robot.move_home() is True
    np.testing.assert_allclose(robot.get_pose(), T_world_home)

    robot.disconnect()
    with pytest.raises(ConnectionError, match="not connected"):
        robot.get_pose()


def test_mock_gripper_records_open_close_sequence():
    gripper = MockGripper()

    gripper.close()
    gripper.open(slight_open=True)

    assert gripper.is_open is True
    assert gripper.commands == [
        {"device": "gripper", "action": "close"},
        {"device": "gripper", "action": "open", "slight_open": True},
    ]
