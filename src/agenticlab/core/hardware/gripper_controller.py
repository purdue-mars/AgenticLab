#!/usr/bin/env python3
import logging
import time

import numpy as np
import yaml

from agenticlab.core.hardware.dynamixel_client import DynamixelClient

logger = logging.getLogger(__name__)


class GripperController:
    def __init__(self, config):
        self.gripper_config = config["gripper"]
        self.max_angle = self.gripper_config.get("max_angle", 330.0)
        self.min_angle = self.gripper_config.get("min_angle", 245.0)
        motor_id = self.gripper_config.get("motor_id", 9)
        self.motors = [motor_id]
        self._initialize_connection(self.gripper_config)
        self._configure_motors(self.gripper_config)
        self._overcurrent_start_time = None  # Record overcurrent start time

    def _initialize_connection(self, config):
        try:
            self.dxl_client = DynamixelClient(
                motor_ids=self.motors, port=config["port"], baudrate=config["baudrate"]
            )
            self.dxl_client.connect()
            logger.info(f"Connected to Dynamixel on {config['port']}")
        except Exception as e:
            logger.error(f"Failed to connect to Dynamixel: {e}")
            raise

    def _configure_motors(self, config):
        try:
            self.dxl_client.set_torque_enabled(self.motors, False)
            self.dxl_client.sync_write(
                self.motors,
                np.ones(len(self.motors)) * config.get("driven_mode", 5),
                11,
                1,
            )  # Position Control
            self.dxl_client.set_torque_enabled(self.motors, True)
            # PID configuration
            # P
            self.dxl_client.sync_write(
                self.motors,
                np.ones(len(self.motors)) * config["pid"].get("kp", 600),
                84,
                2,
            )
            # I
            self.dxl_client.sync_write(
                self.motors,
                np.ones(len(self.motors)) * config["pid"].get("ki", 0),
                82,
                2,
            )
            # D
            self.dxl_client.sync_write(
                self.motors,
                np.ones(len(self.motors)) * config["pid"].get("kd", 200),
                80,
                2,
            )
            # Current limit
            self.dxl_client.sync_write(
                self.motors,
                np.ones(len(self.motors)) * config.get("curr_lim", 500),
                38,
                2,
            )

            logger.info("Motor configuration completed successfully")
        except Exception as e:
            logger.error(f"Failed to configure motors: {e}")
            raise

    def open_gripper(self, slight_open=False) -> bool:
        current_angle = np.rad2deg(self.dxl_client.read_pos())
        angle = (
            self.max_angle
            if not slight_open
            else min(current_angle + 40, self.max_angle)
        )
        return self.set_gripper_width(angle)

    def close_gripper(self) -> bool:
        return self.set_gripper_width(self.min_angle)

    def set_gripper_width(self, width: float) -> bool:
        if self.dxl_client:
            self.dxl_client.set_torque_enabled(self.motors, True)
            width_rad = np.array([np.deg2rad(width)] * len(self.motors))

            while True:
                # Current protection check
                if not self._current_protection():
                    logger.warning("Current protection activated, stopping gripper")
                    # Stop at current position
                    current_pos = self.dxl_client.read_pos()
                    self.dxl_client.write_desired_pos(self.motors, current_pos)
                else:
                    self.dxl_client.write_desired_pos(self.motors, width_rad)

                return True

    def _current_protection(
        self, current_threshold: int = 500, during_time: float = 1.0
    ) -> bool:
        current = self.dxl_client.read_cur()
        print(f"Current: {current}")
        if current > current_threshold:
            # Current exceeds threshold
            if self._overcurrent_start_time is None:
                # First time detecting overcurrent, record time
                self._overcurrent_start_time = time.time()
                logger.warning(
                    f"Current {current} exceeds threshold {current_threshold}"
                )
                return True  # First detection, no protection needed yet
            else:
                # Check if overcurrent persists
                elapsed = time.time() - self._overcurrent_start_time
                if elapsed >= during_time:
                    logger.warning(
                        f"Current protection triggered: {current} > {current_threshold} for {elapsed:.2f}s"
                    )
                    return False  # Trigger protection
                return True  # Still within tolerance time
        else:
            # Current is normal, reset timer
            if self._overcurrent_start_time is not None:
                logger.info("Current returned to normal range")
                self._overcurrent_start_time = None
            return True

    def disconnect(self):
        try:
            if self.dxl_client:
                self.dxl_client.disconnect()
                logger.info("Dynamixel client disconnected")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    with open("configs/gripper_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    gripper = GripperController(config)

    print("Commands: 'o'=open, 'c'=close, 'so'=slight open, 'q'=quit")

    try:
        while True:
            cmd = input("> ").strip().lower()

            if cmd == "o":
                gripper.open_gripper()
            elif cmd == "c":
                gripper.close_gripper()
            elif cmd == "so":  # slight open
                gripper.open_gripper(slight_open=True)
            elif cmd == "r":
                position = gripper.dxl_client.read_pos()
                print(f"Position: {position}")
            elif cmd == "q":
                break
            else:
                print("Use 'o', 'c', 'so', or 'q'")

    except KeyboardInterrupt:
        pass
    finally:
        gripper.disconnect()
