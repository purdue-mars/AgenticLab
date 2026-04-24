from __future__ import annotations

from typing import Any, Dict, Optional, Union

from PIL import Image

from agenticlab.core.llm import BaseLLMClient
from agenticlab.core.planning.action_checker import ActionChecker


class _MockLLMClient(BaseLLMClient):
    def __init__(self, response: Dict[str, Any]):
        super().__init__({"model": "mock"})
        self._response = response

    def call(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, list]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return dict(self._response)

    def _build_messages(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, list]] = None,
    ) -> list:
        return [{"role": "user", "content": prompt}]


def test_action_checker_precondition_mock_smoke():
    llm = _MockLLMClient(
        {
            "success": True,
            "reasoning": "ok",
            "failed_conditions": [],
            "conditions_analysis": [{"cond": "hand_empty", "ok": True}],
            "token_usage": 1,
        }
    )
    cfg = {
        "ActionChecker": {
            "output_dir": "output/action_checker_test",
            "precondition_prompt": "action={action}\nprecondition={precondition}\n",
            "effect_prompt": "action={action}\neffect={effect}\n",
            "completion_prompt": "goals={goal_conditions}\n",
            "completion_multi_img_prompt": "goals={goal_conditions}\n",
        }
    }
    checker = ActionChecker(llm_client=llm, cfg=cfg)
    img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    result = checker.check_precondition(img, "(pick cup)", "(hand-empty)")
    assert result.success is True
    assert result.reasoning == "ok"
