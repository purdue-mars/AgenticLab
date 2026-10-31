#!/usr/bin/env python3
import logging

import numpy as np
import yaml
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

logger = logging.getLogger(__name__)


class RTDEController:
    def __init__(self, config):
        self.config = config["UR5e"]
        self.ip_address = self.config.get("ip_address", "192.168.1.125")
        self.home_joints = np.array(
            self.config.get("home_joints", [0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
        )
        self.rtde_c = None
        self.rtde_r = None
        self.is_connected = False
        self.max_joint_speed = self.config.get("max_joint_speed", 1.0)  # rad/s
        self.max_joint_acceleration = self.config.get("max_joint_acceleration", 1.0)
        self.max_cartesian_speed = self.config.get("max_cartesian_speed", 0.5)  # m/s
        self.max_cartesian_acceleration = self.config.get(
            "max_cartesian_acceleration", 0.5
        )

        self.cartesian_limit = self.config.get(
            "cartesian_limits_m", {"x": [-1.0, 1.0], "y": [-2.0, 2.0], "z": [-0.3, 1.3]}
        )

        # Current state storage
        self.current_joint_positions = np.zeros(6)
        self.current_tcp_pose = np.zeros(6)

    def _configure_robot(self):
        config = self.config
        tcp_offset = config.get("tcp_offset", [0, 0, 0, 0, 0, 0])
        payload = config.get("payload_mass", 0.0)
        payload_cog = config.get("payload_cog", [0, 0, 0])
        self.rtde_c.setTcp(tcp_offset)
        self.rtde_c.setPayload(payload, payload_cog)

    def connect(self):
        try:
            self.rtde_c = RTDEControlInterface(self.ip_address)
            self.rtde_r = RTDEReceiveInterface(self.ip_address)

            self._configure_robot()
            self.is_connected = True
            logger.info(f"Connected to UR5e at {self.ip_address}")
            self.move_to_home()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to UR5e: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        try:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
            self.is_connected = False
            logger.info("Disconnected from UR5e")
        except Exception as e:
            logger.error(f"Error during disconnection: {e}")

    def move_to_home(self):
        if self.is_connected:
            success = self.rtde_c.moveJ(
                self.home_joints.tolist(),
                self.max_joint_speed,
                self.max_joint_acceleration,
            )
            if success:
                logger.info("Moved to home position")
            else:
                logger.warning("Failed to move to home position")
        else:
            logger.warning("Robot not connected. Cannot move to home position.")

    def move_joints(self, joints_num, angle):
        # angle is in degrees
        if not self.is_connected:
            logger.warning("Robot not connected. Cannot move to joints.")
            return False
        joints_angle = self.get_joint_positions()
        joints_angle[joints_num] = np.deg2rad(angle)
        success = self.rtde_c.moveJ(
            joints_angle.tolist(), self.max_joint_speed, self.max_joint_acceleration
        )
        if success:
            logger.info("Moved to joints successfully")
        else:
            logger.warning("Failed to move to joints")
        return success

    def move_cartesian(self, tcp_pose, speed=None, acceleration=None):
        if not self.is_connected:
            logger.warning("Robot not connected. Cannot move to Cartesian pose.")
            return False
        if speed is None:
            speed = self.max_cartesian_speed
        if acceleration is None:
            acceleration = self.max_cartesian_acceleration
        try:
            tcp_pose = np.array(tcp_pose)
            if tcp_pose.shape != (6,):
                logger.error(
                    "tcp_pose must be a list or array of 6 elements [x, y, z, rx, ry, rz]"
                )
                return False

            # Safety check
            pos = tcp_pose[:3]
            if (
                pos[0] < self.cartesian_limit["x"][0]
                or pos[0] > self.cartesian_limit["x"][1]
                or pos[1] < self.cartesian_limit["y"][0]
                or pos[1] > self.cartesian_limit["y"][1]
                or pos[2] < self.cartesian_limit["z"][0]
                or pos[2] > self.cartesian_limit["z"][1]
            ):
                logger.error("Target TCP pose is out of defined Cartesian limits.")
                return False

            success = self.rtde_c.moveL(tcp_pose.tolist(), speed, acceleration)
            if success:
                logger.info("Moved to Cartesian pose successfully")
            else:
                logger.warning("Failed to move to Cartesian pose")
            return success
        except Exception as e:
            logger.error(f"Error during Cartesian move: {e}")
            return False

    def set_free_drive(self, enable=True):
        if not self.is_connected:
            logger.warning("Robot not connected. Cannot set free drive mode.")
        if enable:
            self.rtde_c.teachMode()
            logger.info("Free drive mode enabled")
        else:
            self.rtde_c.endTeachMode()
            logger.info("Free drive mode disabled")

    def get_joint_positions(self):
        if self.is_connected:
            self.current_joint_positions = np.array(self.rtde_r.getActualQ())
            return self.current_joint_positions.copy()
        else:
            logger.warning("Robot not connected. Cannot get joint positions.")
            return None

    def get_tcp_pose(self):
        if self.is_connected:
            self.current_tcp_pose = np.array(self.rtde_r.getActualTCPPose())
            return self.current_tcp_pose.copy()
        else:
            logger.warning("Robot not connected. Cannot get TCP pose.")
            return None

    def apply_rpy_rotation(self, T, yaw_angle, roll_angle, pitch_angle):
        """
        Apply yaw (around z-axis) then pitch (around y-axis) rotation to SE(3) matrix T.
        Args:
            T (np.ndarray): 4x4 SE3 transformation matrix
            roll_angle (float): roll angle (radians), rotation around x-axis
            pitch_angle (float): pitch angle (radians), rotation around y-axis
            yaw_angle (float): yaw angle (radians), rotation around z-axis
        Returns:
            np.ndarray: new SE(3) matrix after applying rotations
        """

        # Rotation around X (roll)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll_angle), -np.sin(roll_angle)],
                [0, np.sin(roll_angle), np.cos(roll_angle)],
            ]
        )

        # Rotation around Y (pitch)
        Ry = np.array(
            [
                [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                [0, 1, 0],
                [-np.sin(pitch_angle), 0, np.cos(pitch_angle)],
            ]
        )

        # Rotation around Z (yaw)
        Rz = np.array(
            [
                [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
                [np.sin(yaw_angle), np.cos(yaw_angle), 0],
                [0, 0, 1],
            ]
        )

        T_new = np.eye(4)
        T_new[:3, :3] = T[:3, :3] @ Rx @ Ry @ Rz
        T_new[:3, 3] = T[:3, 3]
        return T_new


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open("configs/ur5_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    robot = RTDEController(config)

    print(
        "Commands: 'c'=connect, 'h'=home, 'j'=joints, 'p'=pose, "
        "'f'=freedrive, 'q'=quit, 'e'=example move, 'r'=test rotation"
    )

    try:
        pose = [
            0.80268735,
            -0.06083425,
            0.18924001,
            -2.20896584,
            2.16971269,
            -0.02183809,
        ]
        while True:
            cmd = input("> ").strip().lower()

            if cmd == "c":
                robot.connect()
            elif cmd == "h":
                robot.move_to_home()
            elif cmd == "j":
                joints = robot.get_joint_positions()
                if joints is not None:
                    print(f"Joints: {joints}")
            elif cmd == "p":
                pose = robot.get_tcp_pose()
                if pose is not None:
                    print(f"TCP: {pose}")
            elif cmd == "f":
                robot.set_free_drive(True)
                input("Press Enter to exit freedrive...")
                robot.set_free_drive(False)
            elif cmd == "e":
                robot.move_cartesian(pose, 0.4, 0.4)
            elif cmd == "r":
                # Test apply_yaw_pitch with actual robot movement
                current_pose = robot.get_tcp_pose()
                from scipy.spatial.transform import Rotation

                T = np.eye(4)
                T[:3, 3] = current_pose[:3]  # Position
                R = Rotation.from_rotvec(current_pose[3:]).as_matrix()
                T[:3, :3] = R  # Orientation
                # Get rotation angles from user
                roll_deg = float(input("Enter roll angle (degrees): "))
                pitch_deg = float(input("Enter pitch angle (degrees): "))
                yaw_deg = float(input("Enter yaw angle (degrees): "))
                roll_rad = np.deg2rad(roll_deg)
                pitch_rad = np.deg2rad(pitch_deg)
                yaw_rad = np.deg2rad(yaw_deg)
                # Apply rotation
                T_new = robot.apply_rpy_rotation(T, yaw_rad, roll_rad, pitch_rad)

                print(
                    f"\nTransformed T matrix (roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°):\n{T_new}"
                )
                # Convert back to axis-angle representation for robot
                R_new = T_new[:3, :3]
                rotvec_new = Rotation.from_matrix(R_new).as_rotvec()
                new_pose = np.concatenate([T_new[:3, 3], rotvec_new])
                confirm = input("Execute this movement? (y/n): ").strip().lower()
                if confirm == "y":
                    robot.move_cartesian(new_pose, 0.1, 0.1)
                    robot.move_joints(5, yaw_deg)
                else:
                    print("Movement cancelled.")
            elif cmd == "q":
                break
            else:
                print("Use 'c', 'h', 'j', 'p', 'f', 'e', 'r', or 'q'")

    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
