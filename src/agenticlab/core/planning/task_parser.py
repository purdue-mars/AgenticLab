import glob
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from PIL import Image

from agenticlab.core.data_types import TaskPlan
from agenticlab.core.llm import BaseLLMClient
from agenticlab.core.paths import output_path, resolve_repo_path

logger = logging.getLogger(__name__)


class TaskParser:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        cfg: Optional[Dict] = None,
        cfg_path: Optional[os.PathLike] = None,
    ):
        """
        Args:
            llm_client: Injected LLM client (use MockLLMClient in tests).
            cfg: Dict config (either full YAML or TaskParser section).
            cfg_path: Optional path to YAML config. If cfg is None, defaults to
                repo_root/configs/task_parser_config.yaml
        """
        self.llm_client = llm_client
        raw_cfg = cfg if cfg is not None else self._load_default_cfg(cfg_path)
        self.cfg = raw_cfg.get("TaskParser", raw_cfg)

        self.prompt_path = self.cfg.get(
            "prompt_path", "configs/prompts/task_parser_prompt.txt"
        )
        self.prompt_use_pddl_path = self.cfg.get(
            "prompt_use_pddl_path", "configs/prompts/task_parser_prompt_with_pddl.txt"
        )
        # optional: allow passing prompt template directly (useful for tests)
        self.prompt_template_inline = self.cfg.get("prompt_template")
        self.prompt_use_pddl_template_inline = self.cfg.get("prompt_use_pddl_template")

        self.use_pddl = self.cfg.get("use_pddl", False)
        self.domain_path = self.cfg.get("pddl_domain_path", "")
        self.example_nl_path = self.cfg.get("example_nl_path", "")
        self.example_pddl_path = self.cfg.get("example_pddl_path", "")

        self.pddl_cfg = self.cfg.get("pddl_cfg", {})
        self.fast_downward_path = self.pddl_cfg.get(
            "fast_downward_path", "third_party/downward/fast-downward.py"
        )
        self.fast_downward_alias = self.pddl_cfg.get(
            "fast_downward_alias", "seq-sat-lama-2011"
        )
        self.time_limit = self.pddl_cfg.get("time_limit", 20)

        base_output_dir = self.cfg.get("output_dir", "output/task_parser")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_output_dir = str(output_path(Path(base_output_dir) / timestamp))

        self.domain_content = ""
        if self.use_pddl and self.domain_path:
            domain_file = resolve_repo_path(self.domain_path, must_exist=True)
            self.domain_content = domain_file.read_text(encoding="utf-8")
            logger.info("Loaded domain file: %s", str(domain_file))

        if self.use_pddl:
            if self.prompt_use_pddl_template_inline:
                self.prompt_template = self.prompt_use_pddl_template_inline
            else:
                self.prompt_template = self._read_text_or_default(
                    self.prompt_use_pddl_path,
                    default_template="{task_description}",
                )
            self._load_pddl_files()
        else:
            if self.prompt_template_inline:
                self.prompt_template = self.prompt_template_inline
            else:
                self.prompt_template = self._read_text_or_default(
                    self.prompt_path,
                    default_template="{task_description}",
                )

        logger.info("TaskParser initialized. Use PDDL mode: %s", self.use_pddl)
        logger.info("Output directory: %s", self.session_output_dir)

    @classmethod
    def _load_default_cfg(cls, cfg_path: Optional[os.PathLike]) -> Dict:
        if cfg_path is None:
            cfg_path = resolve_repo_path(
                "configs/task_parser_config.yaml", must_exist=True
            )
        cfg_file = resolve_repo_path(cfg_path, must_exist=True)
        with cfg_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def _read_text_or_default(cls, p: str, default_template: str) -> str:
        fp = resolve_repo_path(p, must_exist=False)
        if fp.exists():
            return fp.read_text(encoding="utf-8")
        logger.warning("Prompt file not found (%s); using default template.", fp)
        return default_template

    def _get_action_info(self, action_str: str, domain_content: str) -> Dict:
        """Extract precondition and effect for a given action from domain content."""
        cleaned = action_str.strip().strip("()")
        parts = cleaned.split()

        if not parts:
            return {"action": action_str}

        action_name = parts[0]
        params = parts[1:] if len(parts) > 1 else []

        # Regex to find action definition
        pattern = rf":action\s+{action_name}.*?:precondition\s+(.*?):effect\s+(.*?)(?=\(:action|\Z)"
        action_match = re.search(pattern, domain_content, re.DOTALL)

        return {
            "action": action_str,
            "action_name": action_name,
            "parameters": params,
            "precondition": action_match.group(1).strip(),
            "effect": action_match.group(2).strip(),
        }

    def _get_goal_info(self, problem_content: str) -> List[str]:
        """Extract goal conditions from problem PDDL file."""
        if not problem_content:
            return []

        # Extract goal section using regex
        goal_pattern = r":goal\s*\((.*)\)\s*\)\s*\)"
        goal_match = re.search(goal_pattern, problem_content, re.DOTALL)

        if not goal_match:
            return []

        goal_content = goal_match.group(1).strip()

        # Parse goal conditions
        goal_conditions = []

        # Handle 'and' wrapper
        if goal_content.startswith("and"):
            # Remove 'and' keyword and extract conditions
            inner_content = goal_content[3:].strip()

            # Parse predicates by counting parentheses
            depth = 0
            current_pred = ""
            for char in inner_content:
                if char == "(":
                    depth += 1
                    current_pred += char
                elif char == ")":
                    depth -= 1
                    current_pred += char
                    if depth == 0 and current_pred.strip():
                        goal_conditions.append(current_pred.strip())
                        current_pred = ""
                elif depth > 0:
                    current_pred += char
        else:
            # Single condition
            goal_conditions = [goal_content]

        return goal_conditions

    def _load_pddl_files(self):
        nl_file = resolve_repo_path(self.example_nl_path, must_exist=False)
        pddl_file = resolve_repo_path(self.example_pddl_path, must_exist=False)
        self.example_nl = (
            nl_file.read_text(encoding="utf-8") if nl_file.exists() else ""
        )
        self.example_pddl = (
            pddl_file.read_text(encoding="utf-8") if pddl_file.exists() else ""
        )
        if self.example_nl_path:
            logger.info("Loaded example NL: %s", str(nl_file))
        if self.example_pddl_path:
            logger.info("Loaded example PDDL: %s", str(pddl_file))

    def parse_task(self, task_description: str, scene_image: Image.Image) -> TaskPlan:
        logger.info(f"Parsing task: {task_description}")

        if self.use_pddl:
            prompt = self.prompt_template.format(
                task_description=task_description,
                domain_file=self.domain_content,
                example_nl=self.example_nl,
                example_pddl=self.example_pddl,
            )
        else:
            prompt = self.prompt_template.format(task_description=task_description)

        start_time = time.time()
        result = self.llm_client.call(prompt=prompt, image=scene_image)
        elapsed = time.time() - start_time
        action_sequence = result.get("action_sequence", [])
        if (
            self.use_pddl
            and result.get("updated_domain")
            and result.get("problem_pddl")
        ):
            logger.info("PDDL mode: solving with Fast Downward...")
            action_sequence = self._solve_with_fast_downward(
                result.get("updated_domain"), result.get("problem_pddl")
            )
            if not action_sequence:
                logger.warning(
                    "Fast Downward failed to find a plan, using original action sequence from VLM."
                )
            elapsed = time.time() - start_time

        # extract action details if available
        updated_domain = result.get("updated_domain")
        action_details = None
        domain = updated_domain or self.domain_content
        if action_sequence and domain:
            action_details = (
                [self._get_action_info(a, domain) for a in action_sequence]
                if self.use_pddl
                else None
            )

        # extract goal details if available
        goal_conditions = None
        problem_pddl = result.get("problem_pddl")
        if problem_pddl:
            goal_conditions = self._get_goal_info(problem_pddl)

        task_plan = TaskPlan(
            task_description=task_description,
            objects_identified=result.get("objects_identified", []),
            reasoning=result.get("reasoning", ""),
            action_sequence=action_sequence,
            elapsed_time=elapsed,
            timestamp=datetime.now().isoformat(),
            updated_domain=result.get("updated_domain"),
            action_details=action_details,
            goal_conditions=goal_conditions,
            problem_pddl=result.get("problem_pddl"),
            token_usage=result.get("token_usage"),
        )

        logger.info(f"Task parsed in {elapsed:.2f}s")
        logger.info(f"Objects identified: {task_plan.objects_identified}")
        logger.info(f"Action sequence: {len(task_plan.action_sequence)} actions")

        return task_plan

    def _solve_with_fast_downward(
        self, domain_content: str, problem_content: str
    ) -> List[str]:
        domain_file = os.path.join(self.session_output_dir, "domain.pddl")
        problem_file = os.path.join(self.session_output_dir, "problem.pddl")
        plan_file = os.path.join(self.session_output_dir, "plan.sol")

        with open(domain_file, "w") as f:
            f.write(domain_content)
        with open(problem_file, "w") as f:
            f.write(problem_content)

        logger.info("Running Fast Downward planner...")
        os.system(
            f"python {self.fast_downward_path} "
            f"--alias {self.fast_downward_alias} "
            f"--search-time-limit {self.time_limit} "
            f"--plan-file {plan_file} "
            f"{domain_file} {problem_file}"
        )

        best_plan = []
        best_cost = float("inf")
        for fn in glob.glob(f"{plan_file}.*"):
            with open(fn, "r") as f:
                lines = f.readlines()
                if lines:
                    actions = [
                        line.strip()
                        for line in lines[:-1]
                        if line.strip() and not line.startswith(";")
                    ]
                    try:
                        cost = (
                            int(lines[-1].split("=")[-1].strip())
                            if "cost" in lines[-1]
                            else len(actions)
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_plan = actions
                    except Exception:
                        if not best_plan:
                            best_plan = actions

        if best_plan:
            logger.info(f"Found plan with {len(best_plan)} actions (cost: {best_cost})")
        else:
            logger.warning("No plan found")

        return best_plan

    def save_results(
        self, task_plan: TaskPlan, save_image: Optional[Image.Image] = None
    ):
        json_path = os.path.join(self.session_output_dir, "task_plan.json")
        with open(json_path, "w") as f:
            json.dump(task_plan.to_dict(), f, indent=4, ensure_ascii=False)
        logger.info(f"Task plan metadata saved to {json_path}")

        txt_path = os.path.join(self.session_output_dir, "action_sequence.txt")
        task_plan.save_to_file(txt_path)

        if self.use_pddl:
            if task_plan.updated_domain:
                domain_path = os.path.join(self.session_output_dir, "domain.pddl")
                with open(domain_path, "w") as f:
                    f.write(task_plan.updated_domain)
                logger.info(f"Updated domain saved to {domain_path}")

            if task_plan.problem_pddl:
                pddl_path = os.path.join(self.session_output_dir, "problem.pddl")
                with open(pddl_path, "w") as f:
                    f.write(task_plan.problem_pddl)
                logger.info(f"Problem PDDL saved to {pddl_path}")

        if save_image:
            img_path = os.path.join(self.session_output_dir, "scene_image.png")
            save_image.save(img_path)
            logger.info(f"Scene image saved to {img_path}")

        logger.info(f"All results saved to {self.session_output_dir}")
