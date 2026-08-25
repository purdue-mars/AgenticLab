import numpy as np

from agenticlab.core.execution import ActionWrapper
from agenticlab.core.hardware import MockGripper, MockRobotArm
from agenticlab.utils.geometry_transforms import make_transform


def _config():
    return {
        "ActionWrapper": {
            "approach_distance": 0.1,
            "clearance_height": 0.2,
            "robot_speed": 0.05,
            "robot_acceleration": 0.1,
            "T_grasp_tcp": np.eye(4).tolist(),
            "default_place_orientation_rotvec": [0.0, 0.0, 0.0],
        }
    }


def _camera_config():
    return {
        "Kinect": {
            "handeye_calibration": {
                "rotation": np.eye(3).tolist(),
                "translation": [1.0, 0.0, 0.0],
            }
        },
        "RealSense": {
            "handeye_calibration": {
                "rotation": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.1],
            }
        },
    }


def _wrapper(initial_pose=None):
    event_log = []
    robot = MockRobotArm(initial_pose=initial_pose, event_log=event_log)
    gripper = MockGripper(event_log=event_log)
    wrapper = ActionWrapper(
        robot,
        gripper,
        _config(),
        _camera_config(),
        connect_on_init=False,
        reset_on_init=False,
    )
    return wrapper, robot, gripper, event_log


def _move_commands(robot):
    return [command for command in robot.commands if command["action"] == "move_pose"]


def test_static_camera_grasp_chain_uses_target_source_order():
    wrapper, _, _, _ = _wrapper()
    T_camera_grasp = make_transform(np.eye(3), [0.0, 2.0, 3.0])

    T_world_tcp = wrapper.camera_grasp_to_world_tcp(T_camera_grasp)

    np.testing.assert_allclose(
        T_world_tcp,
        make_transform(np.eye(3), [1.0, 2.0, 3.0]),
    )


def test_wrist_camera_grasp_chain_uses_current_tcp_pose():
    wrapper, _, _, _ = _wrapper()
    T_world_tcp_current = make_transform(np.eye(3), [1.0, 2.0, 3.0])
    T_camera_grasp = make_transform(np.eye(3), [0.0, 0.0, 0.2])

    T_world_tcp = wrapper.camera_grasp_to_world_tcp(
        T_camera_grasp,
        T_world_tcp_current,
    )

    # T_world_tcp_current @ T_tcp_camera @ T_camera_grasp
    np.testing.assert_allclose(
        T_world_tcp,
        make_transform(np.eye(3), [1.0, 2.0, 3.3]),
    )


def test_pick_sequence_and_poses():
    wrapper, robot, _, event_log = _wrapper()
    T_camera_grasp = make_transform(np.eye(3), [0.5, 0.2, 0.3])

    wrapper.pick(T_camera_grasp)

    assert [(event["device"], event["action"]) for event in event_log] == [
        ("gripper", "open"),
        ("robot", "move_pose"),
        ("robot", "move_pose"),
        ("gripper", "close"),
        ("robot", "move_pose"),
    ]
    moves = _move_commands(robot)
    np.testing.assert_allclose(
        moves[0]["T_world_tcp"][:3, 3],
        [1.4, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        moves[1]["T_world_tcp"][:3, 3],
        [1.5, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        moves[2]["T_world_tcp"],
        moves[0]["T_world_tcp"],
    )


def test_pick_can_skip_retreat_and_apply_world_offset():
    wrapper, robot, _, event_log = _wrapper()

    wrapper.pick(
        np.eye(4),
        retreat=False,
        grasp_offset=[0.0, 0.0, 0.05],
    )

    assert [event["action"] for event in event_log] == [
        "open",
        "move_pose",
        "move_pose",
        "close",
    ]
    moves = _move_commands(robot)
    np.testing.assert_allclose(
        moves[1]["T_world_tcp"][:3, 3],
        [1.0, 0.0, 0.05],
    )


def test_place_sequence_lifts_then_moves_xy_then_releases():
    initial_pose = make_transform(np.eye(3), [0.1, 0.2, 0.3])
    wrapper, robot, _, event_log = _wrapper(initial_pose=initial_pose)
    T_world_place = make_transform(np.eye(3), [0.8, 0.9, 0.4])

    wrapper.place(
        T_world_place,
        place_offset=(0.0, 0.0, 0.0),
        retract_distance=0.1,
    )

    assert [(event["device"], event["action"]) for event in event_log] == [
        ("robot", "move_pose"),
        ("robot", "move_pose"),
        ("robot", "move_pose"),
        ("gripper", "open"),
        ("robot", "move_pose"),
        ("robot", "move_home"),
        ("gripper", "open"),
    ]
    moves = _move_commands(robot)
    np.testing.assert_allclose(moves[0]["T_world_tcp"][:3, 3], [0.1, 0.2, 0.6])
    np.testing.assert_allclose(moves[1]["T_world_tcp"][:3, 3], [0.8, 0.9, 0.6])
    np.testing.assert_allclose(moves[2]["T_world_tcp"][:3, 3], [0.8, 0.9, 0.4])
    np.testing.assert_allclose(moves[3]["T_world_tcp"][:3, 3], [0.8, 0.9, 0.5])


def test_camera_frame_place_uses_camera_extrinsic_not_grasp_alignment():
    wrapper, robot, _, _ = _wrapper()
    T_camera_place = make_transform(np.eye(3), [0.2, 0.3, 0.4])

    wrapper.place(
        T_camera_place,
        frame="camera",
        place_offset=(0.0, 0.0, 0.0),
    )

    moves = _move_commands(robot)
    np.testing.assert_allclose(
        moves[2]["T_world_tcp"][:3, 3],
        [1.2, 0.3, 0.4],
    )


def test_move_pure_translation_preserves_current_orientation():
    current_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    wrapper, robot, _, _ = _wrapper(
        initial_pose=make_transform(current_rotation, [0.0, 0.0, 0.0])
    )

    wrapper.move(
        make_transform(np.eye(3), [0.5, 0.6, 0.7]),
        offset=[0.1, 0.0, 0.0],
    )

    T_world_tcp = _move_commands(robot)[0]["T_world_tcp"]
    np.testing.assert_allclose(T_world_tcp[:3, :3], current_rotation)
    np.testing.assert_allclose(T_world_tcp[:3, 3], [0.6, 0.6, 0.7])
