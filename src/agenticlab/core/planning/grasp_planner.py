"""Grasp planning: GraspNet candidate generation + optional VLM evaluation.

The GraspNet backend (and its torch/open3d dependencies) is loaded lazily,
so this module can be imported and the VLM-evaluation path exercised without
the heavy perception stack installed.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import yaml
from PIL import Image

from agenticlab.core.data_types import GraspPose
from agenticlab.core.llm import BaseLLMClient
from agenticlab.core.paths import output_path, resolve_repo_path

logger = logging.getLogger(__name__)


class GraspPlanner:
    """Plans grasp poses with GraspNet and evaluates them with a VLM."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        cfg: Optional[Dict] = None,
        cfg_path: Optional[os.PathLike] = None,
    ):
        """
        Args:
            llm_client: Injected LLM client (use a mock client in tests).
            cfg: Dict config (either full YAML or GraspPlanner section).
            cfg_path: Optional path to YAML config. If cfg is None, defaults to
                repo_root/configs/grasp_planner_config.yaml
        """
        self.llm_client = llm_client
        raw_cfg = cfg if cfg is not None else self._load_default_cfg(cfg_path)
        self.cfg = raw_cfg.get("GraspPlanner", raw_cfg)

        self.use_evaluation = self.cfg.get("use_vlm_for_grasp_evaluation", True)

        # Evaluation prompt: inline config takes precedence over file path
        if "evaluation_prompt" in self.cfg:
            self.evaluation_prompt_template = self.cfg["evaluation_prompt"]
        else:
            prompt_file = resolve_repo_path(
                self.cfg.get(
                    "evaluation_prompt_path",
                    "configs/prompts/grasp_pose_evaluation_prompt.txt",
                ),
                must_exist=False,
            )
            if prompt_file.exists():
                self.evaluation_prompt_template = prompt_file.read_text(
                    encoding="utf-8"
                )
            else:
                logger.warning(
                    "Evaluation prompt not found (%s); using empty prompt.",
                    prompt_file,
                )
                self.evaluation_prompt_template = ""

        # GraspNet / camera configs (optional until the backend is used)
        self.graspnet_cfg = self._load_optional_yaml(
            self.cfg.get("graspnet_cfg_path", "configs/graspnet_config.yaml")
        )
        self.camera_cfg = self._load_optional_yaml(
            self.cfg.get("camera_cfg_path", "configs/camera_config.yaml")
        )

        base_output_dir = self.cfg.get("output_dir", "output/grasp_planner")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_output_dir = str(
            output_path(os.path.join(base_output_dir, timestamp))
        )

        logger.info("GraspPlanner initialized.")

    @classmethod
    def _load_default_cfg(cls, cfg_path: Optional[os.PathLike]) -> Dict:
        if cfg_path is None:
            cfg_path = "configs/grasp_planner_config.yaml"
        cfg_file = resolve_repo_path(cfg_path, must_exist=True)
        with cfg_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_optional_yaml(path: os.PathLike) -> Optional[Dict]:
        fp = resolve_repo_path(path, must_exist=False)
        if not fp.exists():
            logger.debug("Optional config not found: %s", fp)
            return None
        with fp.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def plan_grasp(
        self,
        color: Image.Image,
        depth: np.ndarray,
        obj_mask: np.ndarray,
        target_object: str,
        task_description: str,
        which_camera: str = "Kinect",  # Options: "Kinect", "RealSense"
        cur_eef_pose=None,
        angle_range=None,
        grasp_max_width=None,
    ) -> GraspPose:
        """Generate a grasp candidate and (optionally) evaluate it with a VLM."""
        logger.info("Generating grasp candidate...")
        color = np.array(color)[:, :, :3]
        depth = depth.astype(np.uint16)
        if which_camera == "RealSense":
            logger.info("Using wrist camera view for grasp planning")
            grasp_pose = self._generate_grasp_candidate(
                color,
                depth,
                obj_mask,
                which_camera,
                cur_eef_pose,
                angle_range=angle_range,
                grasp_max_width=grasp_max_width,
            )
        else:
            logger.info("Using Kinect camera view for grasp planning")
            grasp_pose = self._generate_grasp_candidate(
                color,
                depth,
                obj_mask,
                which_camera,
                angle_range=angle_range,
                grasp_max_width=grasp_max_width,
            )
        if self.use_evaluation:
            logger.info("Evaluating grasp candidate with VLM...")
            grasp_pose.evaluation = self.evaluate_grasp(
                grasp_image_path=grasp_pose.visualization_path,
                target_object=target_object,
                task_description=task_description,
            )
        return grasp_pose

    def _get_graspnet_wrapper(self, which_camera: str, grasp_max_width=None):
        """Lazily import and construct the GraspNet backend.

        Kept lazy so that importing this module does not require torch/open3d.
        """
        try:
            from agenticlab.core.perception.graspnet_wrapper import (  # noqa: WPS433
                GraspNetWrapper,
            )
        except ImportError as e:
            raise ImportError(
                "GraspNet backend is not available. It has not been migrated "
                "yet or its dependencies (torch/open3d/graspnet) are missing."
            ) from e
        if self.graspnet_cfg is None or self.camera_cfg is None:
            raise RuntimeError(
                "graspnet_cfg_path / camera_cfg_path must point to existing "
                "YAML files to run grasp candidate generation."
            )
        return GraspNetWrapper(
            self.graspnet_cfg,
            self.camera_cfg,
            which_camera,
            self.session_output_dir,
            grasp_max_width=grasp_max_width,
        )

    def _generate_grasp_candidate(
        self,
        color,
        depth,
        obj_mask: np.ndarray,
        which_camera,
        cur_eef_pose=None,
        angle_range=None,
        grasp_max_width=None,
    ) -> GraspPose:
        """Use GraspNet to generate a grasp candidate."""
        visualize_grasp = self.cfg.get("visualize_grasp", True)
        graspnet_wrapper = self._get_graspnet_wrapper(
            which_camera, grasp_max_width=grasp_max_width
        )
        workspace_mask = graspnet_wrapper.generate_mask_from_bbox(
            obj_mask[:2], obj_mask[2:]
        )
        translation, rotation_matrix, width = graspnet_wrapper.run_grasp_inference(
            color,
            depth,
            workspace_mask,
            show_window=visualize_grasp,
            cur_eef_pose=cur_eef_pose,
            angle_range=angle_range,
        )
        result_visualization_path = graspnet_wrapper.get_grasp_result_path()
        return GraspPose(
            translation=(
                translation.tolist() if hasattr(translation, "tolist") else translation
            ),
            rotation=(
                rotation_matrix.tolist()
                if hasattr(rotation_matrix, "tolist")
                else rotation_matrix
            ),
            width=float(width),
            evaluation=None,
            visualization_path=result_visualization_path,
        )

    def evaluate_grasp(
        self,
        grasp_image_path: str,
        target_object: str,
        task_description: str,
    ) -> Dict:
        """Use the VLM to evaluate the grasp pose from its visualization image."""
        grasp_image = Image.open(grasp_image_path)

        prompt = self.evaluation_prompt_template.format(
            target_object=target_object,
            task_description=task_description,
        )

        start_time = time.time()
        result = self.llm_client.call(prompt=prompt, image=grasp_image)
        result["elapsed_time"] = time.time() - start_time
        return result

    def save_grasp_result(self, grasp_pose: GraspPose):
        result = {
            "timestamp": datetime.now().isoformat(),
            "grasp_pose": grasp_pose.to_dict() if grasp_pose else None,
        }
        result_path = os.path.join(
            self.session_output_dir, "grasp_planning_result.json"
        )
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        logger.info("Grasp planning result saved to %s", result_path)
