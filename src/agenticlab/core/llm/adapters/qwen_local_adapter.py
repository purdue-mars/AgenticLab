import logging
from typing import Any, Dict, Optional

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..base import BaseLLMClient
from ..utils import parse_json_response

logger = logging.getLogger(__name__)


class QwenLocalAdapter(BaseLLMClient):
    """Adapter for Qwen LLMs"""

    def __init__(self, config: Dict):
        super().__init__(config)

        if not self.validate_config():
            raise ValueError("Invalid configuration for QwenAdapter")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def call(
        self, prompt: str, image: Optional[Image.Image] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Unified call method for OpenAI LLMs
        Args:
            prompt: The text prompt to send to the LLM
            image: Optional image input (PIL Image or image path)
            **kwargs: Additional provider-specific parameters
        Returns:
            Dict[str, Any]: Parsed response from the LLM
        """
        messages = self._build_messages(prompt, image)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            max_tokens = 1000
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

        # Extract generated text following the demo pattern
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        result = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        try:
            result = parse_json_response(result)
        except Exception as e:
            logger.warning(f"Response is not in JSON format, returning raw text: {e}")
            result = {"response": result}
        return result

    def _build_messages(self, prompt: str, image: Optional[Image.Image] = None) -> list:
        if image is not None:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            return [{"role": "user", "content": prompt}]
