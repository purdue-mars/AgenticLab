import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from google import genai
from google.genai import types
from PIL import Image

from ..base import BaseLLMClient
from ..utils import encode_image, parse_json_response

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLMClient):
    """Adapter for Gemini LLMs"""

    def __init__(self, config: Dict):
        super().__init__(config)

        if not self.validate_config():
            raise ValueError("Invalid configuration for GeminiAdapter")

        self.client = genai.Client(api_key=self.api_key)

        # Set up token counter CSV path
        self.project_root = Path(__file__).resolve().parents[3]
        self.token_csv_path = (
            self.project_root / "output" / "token_counter" / "gemini_total_token.csv"
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
        Unified call method for Gemini LLMs
        Args:
            prompt: The text prompt to send to the LLM
            image: Optional image input (PIL Image, list of PIL Images, or compatible objects like uploaded files)
            **kwargs: Additional provider-specific parameters
        Returns:
            Dict[str, Any]: Parsed response from the LLM
        """
        messages = self._build_messages(prompt, image)

        response = self.client.models.generate_content(
            model=self.model_name, contents=messages, **kwargs
        )

        # Extract and log token count
        total_tokens = response.usage_metadata.total_token_count
        logger.critical(f"total_token_count used: {total_tokens}")
        self._log_token_count(total_tokens)

        # logger.info(f"Response: {response.text.strip()}")
        try:
            result = parse_json_response(response.text.strip())
        except Exception as e:
            logger.error(f"Response is not in JSON format: {response.text.strip()}")
            logger.warning(f"Response is not in JSON format, returning raw text: {e}")
            result = {"response": response.text.strip()}

        if isinstance(result, list):
            logger.info(f"Original result: {result}")
            result = result[0]
            logger.warning(f"Result is a list, returning the first element: {result}")

        result["token_usage"] = total_tokens

        return result

    def _build_messages(
        self, prompt: str, image: Optional[Union[Image.Image, list, Any]] = None
    ) -> list:
        # Start with the text prompt as the first part
        parts = [types.Part(text=prompt)]

        if image is not None:
            # Normalize to a list of images/parts
            images = image if isinstance(image, list) else [image]

            for img in images:
                if isinstance(img, Image.Image):
                    # Handle PIL Images by encoding them to base64
                    base64_image = encode_image(img)
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",  # encode_image uses PNG
                                data=base64_image,
                            )
                        )
                    )
                elif isinstance(img, (types.Part, types.Blob)):
                    # Directly append Gemini-specific types
                    parts.append(img)
                else:
                    # Try to append other types directly (e.g. uploaded file references)
                    parts.append(img)

        return [types.Content(parts=parts)]
