from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from agenticlab.core.llm import BaseLLMClient
from agenticlab.core.planning.task_parser import TaskParser


class _MockLLMClient(BaseLLMClient):
    def __init__(self, response: Dict[str, Any]):
        # Only "model" is required by BaseLLMClient.validate_config()
        super().__init__({"model": "mock"})
        self._response = response

    def call(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return dict(self._response)

    def _build_messages(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
    ) -> list:
        return [{"role": "user", "content": prompt}]


def test_task_parser_mock_smoke():
    llm = _MockLLMClient(
        {
            "objects_identified": ["cup"],
            "reasoning": "mock reasoning",
            "action_sequence": ["pick(cup)"],
            "token_usage": 1,
        }
    )
    cfg = {
        "TaskParser": {
            "use_pddl": False,
            "prompt_template": "{task_description}",
            "output_dir": "output/task_parser_test",
        }
    }
    parser = TaskParser(llm_client=llm, cfg=cfg)
    img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    plan = parser.parse_task("pick up the cup", img)
    assert plan.action_sequence == ["pick(cup)"]
    assert plan.objects_identified == ["cup"]
