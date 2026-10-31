"""Place planning: iterative VLM reflection on placing points + orientation.

Pixels are back-projected with ``geometry_transforms.pixel_to_camera_point``
using the intrinsics in the camera config. A different projection can be
injected for cameras whose model does not fit the pinhole default.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
import yaml
from PIL import Image, ImageDraw

from agenticlab.core.data_types import PlacingPoint
from agenticlab.core.llm import BaseLLMClient
from agenticlab.core.paths import output_path, resolve_repo_path
from agenticlab.utils.geometry_transforms import pixel_to_camera_point

logger = logging.getLogger(__name__)

# Signature: (point_xy, depth, intrinsic_matrix) -> 3D point in camera frame
PixelTo3DFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# Raw depth units per meter, used when the camera config omits factor_depth.
DEFAULT_FACTOR_DEPTH = 1000.0


class PlacePlanner:
    """Plans placing points via iterative VLM reflection and orientation analysis."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        cfg: Optional[Dict] = None,
        cfg_path: Optional[os.PathLike] = None,
        pixel_to_3d_fn: Optional[PixelTo3DFn] = None,
    ):
        """
        Args:
            llm_client: Injected LLM client (use a mock client in tests).
            cfg: Dict config (either full YAML or PlacePlanner section).
            cfg_path: Optional path to YAML config. If cfg is None, defaults to
                repo_root/configs/place_planner_config.yaml
            pixel_to_3d_fn: Optional override for pixel back-projection.
                Defaults to geometry_transforms.pixel_to_camera_point.
        """
        self.llm_client = llm_client
        self.pixel_to_3d_fn = pixel_to_3d_fn
        raw_cfg = cfg if cfg is not None else self._load_default_cfg(cfg_path)
        self.cfg = raw_cfg.get("PlacePlanner", raw_cfg)

        self.max_iterations = self.cfg.get("max_iterations", 3)
        self.use_reflection = self.cfg.get("use_reflection", True)

        self.reflection_prompt_template = self._load_prompt(
            inline_key="reflection_prompt",
            path_key="reflection_prompt_path",
            default_path="configs/prompts/place_point_reflection_prompt.txt",
        )
        self.oriented_pose_prompt_template = self._load_prompt(
            inline_key="oriented_pose_prompt",
            path_key="oriented_pose_prompt_path",
            default_path="configs/prompts/place_orientated_pose_prompt.txt",
        )

        # Camera intrinsics config (optional; only needed for 3D projection)
        camera_cfg_path = resolve_repo_path(
            self.cfg.get("camera_cfg_path", "configs/camera_config.yaml"),
            must_exist=False,
        )
        if camera_cfg_path.exists():
            with camera_cfg_path.open("r", encoding="utf-8") as f:
                self.camera_cfg = yaml.safe_load(f) or {}
        else:
            logger.debug("Camera config not found: %s", camera_cfg_path)
            self.camera_cfg = None

        base_output_dir = self.cfg.get("output_dir", "output/place_planner")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_output_dir = str(
            output_path(os.path.join(base_output_dir, timestamp))
        )

        logger.info("PlacePlanner initialized.")

    @classmethod
    def _load_default_cfg(cls, cfg_path: Optional[os.PathLike]) -> Dict:
        if cfg_path is None:
            cfg_path = "configs/place_planner_config.yaml"
        cfg_file = resolve_repo_path(cfg_path, must_exist=True)
        with cfg_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_prompt(self, inline_key: str, path_key: str, default_path: str) -> str:
        if inline_key in self.cfg:
            return self.cfg[inline_key]
        fp = resolve_repo_path(self.cfg.get(path_key, default_path), must_exist=False)
        if fp.exists():
            return fp.read_text(encoding="utf-8")
        logger.warning("Prompt file not found (%s); using empty prompt.", fp)
        return ""

    def _compute_pose_3d(
        self, point: np.ndarray, depth: np.ndarray, which_camera: str
    ) -> Optional[np.ndarray]:
        """Back-project a pixel into the camera frame using camera intrinsics."""
        cam_cfg = (self.camera_cfg or {}).get(which_camera)
        intrinsics_raw = (
            cam_cfg.get("intrinsic_matrix") if isinstance(cam_cfg, dict) else None
        )
        if intrinsics_raw is None:
            logger.warning(
                "Camera config missing intrinsics for '%s'; skipping 3D pose.",
                which_camera,
            )
            return None

        intrinsics = np.array(intrinsics_raw)
        depth_map = np.array(depth)
        if self.pixel_to_3d_fn is not None:
            return self.pixel_to_3d_fn(point, depth_map, intrinsics)
        return pixel_to_camera_point(
            point,
            depth_map,
            intrinsics,
            factor_depth=cam_cfg.get("factor_depth", DEFAULT_FACTOR_DEPTH),
        )

    def plan_placing_point(
        self,
        image: Image.Image,
        depth: np.ndarray,
        initial_point: np.ndarray,  # [x, y] pixel coordinates
        target_place: str,
        task_description: str,
        which_camera: str = "Kinect",  # Options: "Kinect", "RealSense"
    ) -> PlacingPoint:
        if not self.use_reflection:
            # Simple mode: just visualize and return
            placing_point = PlacingPoint(point=self._as_list(initial_point))
            viz_path = self._save_visualization(
                image,
                initial_point,
                os.path.join(
                    self.session_output_dir, "placing_point_no_reflection.png"
                ),
            )
            placing_point.visualization_path = viz_path
            pose_3d = self._compute_pose_3d(initial_point, depth, which_camera)
            placing_point.pose_3d = self._as_list(pose_3d)
            return placing_point

        # Iterative reflection
        curr_point = initial_point
        iterations = []
        curr_point_viz = None

        for i in range(1, self.max_iterations + 1):
            logger.info(f"Iteration {i}/{self.max_iterations}")

            result = self._reflect_on_placing_point(
                image, curr_point, target_place, task_description, i
            )

            viz_path = os.path.join(self.session_output_dir, f"{i:02d}_iteration.png")
            curr_point_viz = self._save_visualization(image, curr_point, viz_path)
            iterations.append(
                {
                    "iteration": i,
                    "point_pixels": self._as_list(curr_point),
                    "evaluation": result,
                    "visualization": curr_point_viz,
                }
            )
            logger.info(f"Action: {result.get('action')}")
            if result.get("action", "ERROR") == "SUCCESS":
                break
            elif result.get("action", "ERROR") == "NEED_ADJUSTMENT":
                adjusted = result.get("adjusted_point", curr_point)
                # Convert [y, x] (VLM convention) to [x, y]
                curr_point = np.array([adjusted[1], adjusted[0]])
                logger.info(f"Adjusted to: {self._as_list(curr_point)}")

        placing_point = PlacingPoint(point=self._as_list(curr_point))
        pose_3d = self._compute_pose_3d(curr_point, depth, which_camera)
        placing_point.pose_3d = self._as_list(pose_3d)
        placing_point.evaluation = iterations
        placing_point.visualization_path = curr_point_viz
        return placing_point

    def plan_oriented_place_pose(
        self,
        image: Image.Image,
        task_description: str,
        placing_point: PlacingPoint,
    ) -> PlacingPoint:
        """
        Analyze the grasped object orientation and determine rotation parameters
        for upright placement.

        Args:
            image: RGB image showing the grasped object
            task_description: Description of the placement task
            placing_point: PlacingPoint with position already determined

        Returns:
            Updated PlacingPoint with orientation parameters
            (yaw_rotation, roll_rotation, place_offset)
        """
        logger.info("Planning oriented place pose...")

        prompt = self.oriented_pose_prompt_template.format(
            task_description=task_description
        )

        start = time.time()
        result = self.llm_client.call(prompt=prompt, image=image)
        elapsed_time = time.time() - start

        logger.info(f"VLM orientation analysis completed in {elapsed_time:.2f}s")
        logger.info(f"Result: {result}")

        placing_point.yaw_rotation = result.get(
            "yaw_rotation", placing_point.yaw_rotation
        )
        placing_point.roll_rotation = result.get(
            "roll_rotation", placing_point.roll_rotation
        )
        placing_point.place_offset = result.get(
            "place_offset", placing_point.place_offset
        )
        placing_point.orientation_reasoning = result.get(
            "reasoning", placing_point.orientation_reasoning or ""
        )
        return placing_point

    def _reflect_on_placing_point(
        self,
        image: Image.Image,
        point: np.ndarray,
        target_place: str,
        task_description: str,
        iteration: int,
    ) -> Dict:
        """Analyze placing point using VLM."""
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)
        x, y, r = int(point[0]), int(point[1]), 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill="red")

        prompt = self.reflection_prompt_template.format(
            target_object=target_place,
            task_description=task_description,
            current_point=f"[{point[1]}, {point[0]}]",  # [y, x] for the VLM
            iteration=iteration,
        )

        start = time.time()
        result = self.llm_client.call(prompt=prompt, image=annotated_img)
        result["elapsed_time"] = time.time() - start
        return result

    def _save_visualization(
        self, image: Image.Image, point: np.ndarray, save_path: str
    ) -> str:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        r = 10  # radius of the point marker
        draw.ellipse(
            (point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill="red"
        )
        img.save(save_path)
        return save_path

    def save_results(
        self, image: Image.Image, placing_point: PlacingPoint, output_dir: str
    ):
        os.makedirs(output_dir, exist_ok=True)

        final_viz_path = os.path.join(output_dir, "final_placing_point.png")
        self._save_visualization(image, np.array(placing_point.point), final_viz_path)

        result = {
            "placing_point": placing_point.to_dict(),
            "visualization_path": final_viz_path,
        }
        results_path = os.path.join(output_dir, "placing_point_result.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        logger.info(f"Placing point results saved to {results_path}")

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        return value.tolist() if isinstance(value, np.ndarray) else list(value)
