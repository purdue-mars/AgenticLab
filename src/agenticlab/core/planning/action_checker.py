import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import yaml
from PIL import Image

from agenticlab.core.data_types import CheckResult
from agenticlab.core.llm import BaseLLMClient
from agenticlab.core.paths import output_path, resolve_repo_path

logger = logging.getLogger(__name__)


class ActionChecker:
    """ActionChecker validates preconditions, effects, and task completion using VLM."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        cfg: Optional[Dict] = None,
        cfg_path: Optional[os.PathLike] = None,
    ):
        """
        Args:
            llm_client: Injected LLM client (use MockLLMClient in tests).
            cfg: Dict config (either full YAML or ActionChecker section).
            cfg_path: Optional path to YAML config. If cfg is None, defaults to
                repo_root/configs/action_checker_config.yaml
        """
        self.llm_client = llm_client
        raw_cfg = cfg if cfg is not None else self._load_default_cfg(cfg_path)
        self.cfg = raw_cfg.get("ActionChecker", raw_cfg)

        # Load prompt templates
        prompt_paths = {
            "precondition": self.cfg.get(
                "precondition_prompt_path",
                "configs/prompts/precondition_check_prompt.txt",
            ),
            "effect": self.cfg.get(
                "effect_prompt_path", "configs/prompts/effect_check_prompt.txt"
            ),
            "completion": self.cfg.get(
                "completion_prompt_path",
                "configs/prompts/task_completion_check_prompt.txt",
            ),
            "completion_multi_img": self.cfg.get(
                "completion_multi_img_prompt_path",
                "configs/prompts/task_completion_check_multi_img_prompt.txt",
            ),
        }

        self.prompts = {}
        for key, path in prompt_paths.items():
            inline_key = f"{key}_prompt"
            if inline_key in self.cfg:
                self.prompts[key] = self.cfg[inline_key]
                continue
            fp = resolve_repo_path(path, must_exist=False)
            if fp.exists():
                self.prompts[key] = fp.read_text(encoding="utf-8")
            else:
                logger.warning("Prompt file not found (%s); using empty prompt.", fp)
                self.prompts[key] = ""

        # Setup output directory
        base_output_dir = self.cfg.get("output_dir", "output/action_checker")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_output_dir = str(
            output_path(os.path.join(base_output_dir, timestamp))
        )

        self.is_check_precondition = self.cfg.get("is_check_precondition", True)
        self.is_check_effect = self.cfg.get("is_check_effect", True)
        self.is_check_task_completion = self.cfg.get("is_check_task_completion", True)

        logger.info("ActionChecker initialized.")

    @classmethod
    def _load_default_cfg(cls, cfg_path: Optional[os.PathLike]) -> Dict:
        if cfg_path is None:
            cfg_path = resolve_repo_path(
                "configs/action_checker_config.yaml", must_exist=True
            )
        cfg_file = resolve_repo_path(cfg_path, must_exist=True)
        with cfg_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _call_vlm(
        self, image: Union[Image.Image, List[Image.Image]], prompt: str
    ) -> CheckResult:
        """Call VLM for validation checks."""
        start_time = time.time()

        try:
            result_data = self.llm_client.call(prompt=prompt, image=image)
            elapsed = time.time() - start_time

            # Extract detailed analysis
            detailed_analysis = (
                result_data.get("conditions_analysis")
                or result_data.get("effects_analysis")
                or result_data.get("goals_analysis")
            )

            return CheckResult(
                success=result_data.get("success", False),
                reasoning=result_data.get("reasoning", ""),
                timestamp=datetime.now().isoformat(),
                elapsed_time=elapsed,
                failed_conditions=result_data.get("failed_conditions", []),
                detailed_analysis=detailed_analysis,
                token_usage=result_data.get("token_usage"),
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"VLM call error: {e}")
            return CheckResult(
                success=False,
                reasoning=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                elapsed_time=elapsed,
                failed_conditions=[],
                detailed_analysis=None,
            )

    def check_precondition(
        self,
        current_image: Union[Image.Image, List[Image.Image]],
        action: str,
        precondition: str,
    ) -> CheckResult:
        """Check if action preconditions are satisfied."""
        if not self.is_check_precondition:
            logger.info("Precondition check is disabled.")
            return CheckResult(
                success=True,
                reasoning="Precondition check is disabled.",
                timestamp=datetime.now().isoformat(),
                elapsed_time=0.0,
            )
        prompt = self.prompts["precondition"].format(
            action=action, precondition=precondition
        )
        result = self._call_vlm(current_image, prompt)
        logger.info(
            f"Precondition: {'✓' if result.success else '✗'} ({result.elapsed_time:.2f}s)"
        )
        return result

    def check_effect(
        self,
        current_image: Union[Image.Image, List[Image.Image]],
        action: str,
        effect: str,
    ) -> CheckResult:
        """Check if action effects are achieved."""
        if not self.is_check_effect:
            logger.info("Effect check is disabled.")
            return CheckResult(
                success=True,
                reasoning="Effect check is disabled.",
                timestamp=datetime.now().isoformat(),
                elapsed_time=0.0,
            )
        prompt = self.prompts["effect"].format(action=action, effect=effect)
        result = self._call_vlm(current_image, prompt)
        logger.info(
            f"Effect: {'✓' if result.success else '✗'} ({result.elapsed_time:.2f}s)"
        )
        return result

    def check_task_completion(
        self,
        current_image: Union[Image.Image, List[Image.Image]],
        goal_conditions: List[str],
    ) -> CheckResult:
        """Check if all goal conditions are satisfied."""
        if not self.is_check_task_completion:
            logger.info("Task completion check is disabled.")
            return CheckResult(
                success=True,
                reasoning="Task completion check is disabled.",
                timestamp=datetime.now().isoformat(),
                elapsed_time=0.0,
            )
        goal_str = "\n".join([f"- {goal}" for goal in goal_conditions])

        # Use multi-image prompt if input is a list of images, otherwise use single-image prompt
        if isinstance(current_image, list) and len(current_image) > 1:
            prompt = self.prompts["completion_multi_img"].format(
                goal_conditions=goal_str
            )
            logger.info(f"Using multi-image prompt for {len(current_image)} images")
        else:
            prompt = self.prompts["completion"].format(goal_conditions=goal_str)

        result = self._call_vlm(current_image, prompt)
        logger.info(
            f"Completion: {'✓' if result.success else '✗'} ({result.elapsed_time:.2f}s)"
        )
        return result

    def save_result(
        self,
        check_result: CheckResult,
        check_type: str,
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        action: Optional[str] = None,
    ):
        """Save check result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_action = (
            action.replace(" ", "_").replace("(", "").replace(")", "")[:50]
            if action
            else ""
        )
        filename = (
            f"{check_type}_{safe_action}_{timestamp}"
            if action
            else f"{check_type}_{timestamp}"
        )

        json_path = os.path.join(self.session_output_dir, f"{filename}.json")
        with open(json_path, "w") as f:
            json.dump(check_result.to_dict(), f, indent=4, ensure_ascii=False)

        if image:
            if isinstance(image, list):
                # Save multiple images
                for idx, img in enumerate(image):
                    if isinstance(img, Image.Image):
                        img.save(
                            os.path.join(
                                self.session_output_dir, f"{filename}_{idx}.png"
                            )
                        )
            elif isinstance(image, Image.Image):
                # Save single image
                image.save(os.path.join(self.session_output_dir, f"{filename}.png"))
