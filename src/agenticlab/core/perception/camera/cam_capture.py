import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from pyk4a import PyK4A


class CameraCapture:
    """Camera capture class that keeps camera open for multiple captures"""

    def __init__(self, which_cam: str = "Kinect"):
        """Initialize camera and keep it open
        Args:
            which_cam (str): Camera type, 'Kinect' or 'RealSense'
        """
        self.which_cam = which_cam

        if which_cam == "Kinect":
            self.k4a = PyK4A()
            self.k4a.start()
            self.pipeline = None
            self.align = None
            self.temporal_filter = None
        elif which_cam == "RealSense":
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.pipeline.start(config)

            # Setup depth filters combination
            # 1. Depth to Disparity
            self.depth_to_disparity = rs.disparity_transform(True)

            # 2. Spatial Filter
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 5)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

            # 3. Temporal Filter
            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

            # 4. Disparity to Depth
            self.disparity_to_depth = rs.disparity_transform(False)

            # Warm up the camera
            for _ in range(100):
                self.pipeline.wait_for_frames()

            # Setup alignment
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            self.k4a = None
        else:
            raise ValueError("Unsupported camera type. Use 'Kinect' or 'RealSense'.")

    def np_array_to_image(self, np_array):
        """Convert numpy array to PIL Image"""
        return Image.fromarray(np_array)

    def capture(self):
        """Capture a single frame from the opened camera
        Returns:
            tuple: (color_image, depth_image)
        """
        if self.which_cam == "Kinect":
            capture = self.k4a.get_capture()
            color_image = capture.color[:, :, :3]  # Remove alpha channel
            depth_image = capture.transformed_depth

        elif self.which_cam == "RealSense":
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise RuntimeError("Could not acquire depth or color frames.")

            # Apply combined depth filters for better quality
            filtered_depth = depth_frame
            filtered_depth = self.depth_to_disparity.process(filtered_depth)
            filtered_depth = self.spatial_filter.process(filtered_depth)
            filtered_depth = self.temporal_filter.process(filtered_depth)
            filtered_depth = self.disparity_to_depth.process(filtered_depth)

            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return color_image, depth_image

    def destroy(self):
        """Stop camera and release resources"""
        if self.which_cam == "Kinect" and self.k4a is not None:
            self.k4a.stop()
        elif self.which_cam == "RealSense" and self.pipeline is not None:
            self.pipeline.stop()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.destroy()


def get_capture(which_cam: str = "Kinect"):
    """Capture a single frame from Camera (standalone function for backward compatibility)
    Args:
        which_cam (str): Camera type, 'Kinect' or 'RealSense'
    Returns:
        tuple: (color_image, depth_image)
    """
    with CameraCapture(which_cam) as camera:
        return camera.capture()


def save_images(color_image, depth_image, save_dir):
    """Save color and depth images to specified directory"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    color_path = os.path.join(save_dir, f"color_{timestamp}.png")
    depth_path = os.path.join(save_dir, f"depth_{timestamp}.png")

    # Save color image
    Image.fromarray(color_image).save(color_path)

    # Save depth image (as 16-bit PNG)
    Image.fromarray(depth_image.astype(np.uint16)).save(depth_path)

    print(f"Saved color image to {color_path}")
    print(f"Saved depth image to {depth_path}")


def display_preview(color_image, transformed_depth):
    """Display real-time preview of color and aligned depth images"""
    # Create preview displays
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(transformed_depth, alpha=0.03), cv2.COLORMAP_JET
    )

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # Create side-by-side display
    display_color = cv2.resize(color_image, (640, 360))
    display_depth = cv2.resize(depth_colormap, (640, 360))
    combined = np.hstack([display_color, display_depth])

    cv2.imshow(
        "Kinect Preview: Color (left) | Aligned Depth (right) - Press 'q' to exit",
        combined,
    )

    # Check for capture trigger (key press)
    key = cv2.waitKey(30) & 0xFF  # 30ms delay for smoother display
    if key == ord("q"):
        return True
    else:
        return False


def pixel_to_camera_3d(
    point_2d, depth_image, camera_intrinsics, which_cam: str = "Kinect"
):
    """Convert a 2D pixel in RGB image to 3D camera coordinates using depth map and camera intrinsics."""
    x, y = int(point_2d[0]), int(point_2d[1])
    depth = depth_image[y, x]

    fx = camera_intrinsics[0][0]
    fy = camera_intrinsics[1][1]
    cx = camera_intrinsics[0][2]
    cy = camera_intrinsics[1][2]

    # Transfer depth from mm to meters
    if which_cam == "Kinect":
        depth = depth / 1000.0
    elif which_cam == "RealSense":
        depth = depth / 10000.0

    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z])


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="./configs")
    parser.add_argument(
        "--which_cam", default="Kinect", choices=["Kinect", "RealSense"]
    )
    args = parser.parse_args()

    # Load configs
    with open(os.path.join(args.config_dir, "camera_config.yaml"), "r") as f:
        camera_config = yaml.safe_load(f)

    # Get camera-specific configuration
    cam_config = camera_config.get(args.which_cam, None)
    if cam_config is None:
        raise ValueError(
            f"No configuration found for {args.which_cam} in camera_config.yaml"
        )

    cam_intrinsics = cam_config.get("intrinsic_matrix", None)

    if args.which_cam == "Kinect":
        save_directory = "output/kinect_captures"
    elif args.which_cam == "RealSense":
        save_directory = "output/realsense_captures"
    os.makedirs(save_directory, exist_ok=True)

    print(f"Starting {args.which_cam} preview. Press 'q' to exit.")

    # Use class for continuous capture
    camera = CameraCapture(args.which_cam)
    try:
        while True:
            color_img, depth_img = camera.capture()
            if display_preview(color_img, depth_img):
                sample_point2d = [462, 568]
                pose_3d = pixel_to_camera_3d(sample_point2d, depth_img, cam_intrinsics)
                print(f"3D coordinates of pixel {sample_point2d}: {pose_3d}")
                break

        cv2.destroyAllWindows()

        # Capture and save a final image before exiting
        color_img, depth_img = camera.capture()
        save_images(color_img, depth_img, save_directory)
        print("Capture complete. Exiting.")
    finally:
        camera.destroy()
