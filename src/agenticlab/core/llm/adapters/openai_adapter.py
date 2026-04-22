import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from openai import OpenAI
from PIL import Image

from ..base import BaseLLMClient
from ..utils import encode_image, parse_json_response

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLMClient):
    """Adapter for OpenAI LLMs"""

    def __init__(self, config: Dict):
        super().__init__(config)

        if not self.validate_config():
            raise ValueError("Invalid configuration for OpenAIAdapter")

        self.client = OpenAI(api_key=self.api_key)

        # Set up token counter CSV path
        self.project_root = Path(__file__).resolve().parents[3]
        self.token_csv_path = (
            self.project_root / "output" / "token_counter" / "openai_total_token.csv"
        )

    def _log_token_count(self, total_tokens: int):
        """Log total token count to CSV file"""
        try:
            self.token_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_csv_path, "a") as f:
                f.write(f"{total_tokens}\n")
        except Exception as e:
            logger.warning(f"Failed to log token count: {e}")

    def call(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, list, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified call method for OpenAI LLMs
        Args:
            prompt: The text prompt to send to the LLM
            image: Optional image input (PIL Image, list of PIL Images, or image path)
            **kwargs: Additional provider-specific parameters
        Returns:
            Dict[str, Any]: Parsed response from the LLM
        """
        messages = self._build_messages(prompt, image)

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )

        # Extract and log token count
        total_tokens = response.usage.total_tokens
        logger.critical(f"total_token_count used: {total_tokens}")
        self._log_token_count(total_tokens)

        try:
            result = parse_json_response(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Response is not in JSON format, returning raw text: {e}")
            result = {"response": response.choices[0].message.content}

        result["token_usage"] = total_tokens

        return result

    def _build_messages(
        self, prompt: str, image: Optional[Union[Image.Image, list, Any]] = None
    ) -> list:
        if image is not None:
            # Normalize to a list of images
            images = image if isinstance(image, list) else [image]

            content = []
            for img in images:
                if isinstance(img, Image.Image):
                    base64_image = encode_image(img)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        }
                    )
                elif isinstance(img, str) and (
                    img.startswith("http") or len(img) > 100
                ):  # rudimentary check for URL or base64
                    content.append({"type": "image_url", "image_url": {"url": img}})

            # Add the text prompt
            content.append({"type": "text", "text": prompt})

            return [{"role": "user", "content": content}]
        else:
            return [{"role": "user", "content": prompt}]
