import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch  # type: ignore
from PIL import Image, ImageDraw, ImageFont
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)

from agenticlab.core.paths import output_path, resolve_repo_path

logger = logging.getLogger(__name__)


class MolmoLocalBackend:
    """
    Molmo backend for object detection using local inference with Hugging Face Transformers.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.vlm_config = config["vlm"]
        self.model_config = self.vlm_config["models"].get("molmo_local", {})

        # Configure model
        model_name = self.model_config.get("model", "allenai/Molmo-7B-D-0924")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )

        # Load the model with correct dtype
        dtype_config = self.model_config.get("torch_dtype", "auto")
        if dtype_config == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype_config == "float32":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )

        self.model_name = "molmo_detection_local"
        logger.info(
            "Molmo Object Detection Local Model Loaded with dtype=%s", dtype_config
        )

    def generate_object_detection_prompt(
        self, prompt_path: str, image: Image, query: str
    ):
        """
        Generate a prompt for object detection task.
        Args:
            prompt_path (str): Path to the prompt template file.
            image (Image): Input image as a PIL Image.
            query (str): Query describing what objects to detect.
        Returns:
            str: The formatted prompt text
        """
        prompt_template = resolve_repo_path(prompt_path, must_exist=True).read_text(
            encoding="utf-8"
        )

        # Format prompt with the object name
        if query == "all":
            prompt = prompt_template
        else:
            prompt = prompt_template.format(object=query)

        return prompt

    def detect_objects(self, image: Image, object_name, prompt_path):
        """Detect objects in an image and return their bounding boxes using local inference."""
        # Get model configuration
        molmo_cfg = self.vlm_config["models"].get("molmo_local", {})

        # Generate prompt text
        prompt = self.generate_object_detection_prompt(prompt_path, image, object_name)

        # Convert image to RGB
        orig_image = image.convert("RGB")

        # Process the image and text using Molmo's processor
        inputs = self.processor.process(images=[orig_image], text=prompt)

        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # Convert images to bfloat16 if using bfloat16 weights for memory efficiency
        if self.torch_dtype == torch.bfloat16 and "images" in inputs:
            inputs["images"] = inputs["images"].to(torch.bfloat16)

        # Generate output using Molmo's generation method
        max_tokens = molmo_cfg.get("max_tokens", 1000)
        stop_strings = molmo_cfg.get("stop_strings", "<|endoftext|>")

        # Use autocast for efficient inference as recommended by Molmo
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(
                        max_new_tokens=max_tokens, stop_strings=stop_strings
                    ),
                    tokenizer=self.processor.tokenizer,
                )

        # Only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        result = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        # Print raw model response
        logger.info("\n=== Molmo Object Detection Local Raw Response ===")
        logger.info(result)
        logger.info("=========================================\n")

        # Store the raw response for returning later
        raw_response = result

        # Extract JSON from the response
        try:
            # Find JSON content between triple backticks
            json_pattern_start = result.find("```json")
            if json_pattern_start != -1:
                # If found JSON code block format ```json ... ```
                json_content_start = result.find("\n", json_pattern_start) + 1
                json_pattern_end = result.find("```", json_content_start)
                if json_pattern_end != -1:
                    json_str = result[json_content_start:json_pattern_end].strip()
                    # Remove comments before parsing
                    json_str = self._remove_json_comments(json_str)
                    detection_data = json.loads(json_str)
                else:
                    # Fall back to searching for { } if ending ``` not found
                    start_idx = result.find("{")
                    end_idx = result.rfind("}") + 1
                    if start_idx >= 0 and end_idx > 0:
                        json_str = result[start_idx:end_idx]
                        # Remove comments before parsing
                        json_str = self._remove_json_comments(json_str)
                        detection_data = json.loads(json_str)
                    else:
                        # Remove comments before parsing
                        result = self._remove_json_comments(result)
                        detection_data = json.loads(result)
            else:
                # No JSON code block format, look for { }
                start_idx = result.find("{")
                end_idx = result.rfind("}") + 1
                if start_idx >= 0 and end_idx > 0:
                    json_str = result[start_idx:end_idx]
                    # Remove comments before parsing
                    json_str = self._remove_json_comments(json_str)
                    detection_data = json.loads(json_str)
                else:
                    # Remove comments before parsing
                    result = self._remove_json_comments(result)
                    detection_data = json.loads(result)

            return detection_data, raw_response
        except json.JSONDecodeError as e:
            logger.error(f"Error: Could not parse JSON from response: {str(e)}")
            error_data = {"error": "Failed to parse response", "message": str(e)}

            return error_data, raw_response

    def _remove_json_comments(self, json_str):
        """
        Remove comments from JSON string before parsing.

        Args:
            json_str (str): JSON string that may contain comments

        Returns:
            str: Clean JSON string without comments
        """
        import re

        # Remove single-line comments (starting with //)
        clean_json = re.sub(r"//.*$", "", json_str, flags=re.MULTILINE)
        # Remove multi-line comments (/* ... */)
        clean_json = re.sub(r"/\*.*?\*/", "", clean_json, flags=re.DOTALL)
        return clean_json

    def visualize_detections(self, image: Image, detection_data, output_path):
        """Draw bounding boxes or points for all objects in detection results."""
        if "error" in detection_data:
            logger.warning("No valid detection data provided.")
            return None

        img = image.copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            font = ImageFont.load_default()
        color = (255, 0, 0)
        # Support multiple objects
        objects = detection_data.get("objects", [])
        if not objects and "object" in detection_data:
            objects = [detection_data["object"]]
        for obj in objects:
            label = obj.get("label", "Object")
            # Prefer bounding_box, fallback to top_left/bottom_right, fallback to center_point
            if "bounding_box" in obj and len(obj["bounding_box"]) == 4:
                x1, y1, x2, y2 = obj["bounding_box"]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                text_size = draw.textbbox((0, 0), label, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                draw.rectangle(
                    [x1, y1 - text_height - 4, x1 + text_width, y1], fill=color
                )
                draw.text(
                    (x1, y1 - text_height - 4), label, fill=(255, 255, 255), font=font
                )
                logger.info(
                    f"Drew bounding box: ({x1}, {y1}) to ({x2}, {y2}) for {label}"
                )
            elif "top_left" in obj and "bottom_right" in obj:
                x1, y1 = obj["top_left"]
                x2, y2 = obj["bottom_right"]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                text_size = draw.textbbox((0, 0), label, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                draw.rectangle(
                    [x1, y1 - text_height - 4, x1 + text_width, y1], fill=color
                )
                draw.text(
                    (x1, y1 - text_height - 4), label, fill=(255, 255, 255), font=font
                )
                logger.info(
                    f"Drew bounding box: ({x1}, {y1}) to ({x2}, {y2}) for {label}"
                )
            if "center_point" in obj or "top_surface_point" in obj:
                cx, cy = (
                    obj["center_point"]
                    if "center_point" in obj
                    else obj["top_surface_point"]
                )
                radius = 8
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=color,
                    outline=(255, 255, 255),
                    width=2,
                )
                cross_size = 4
                draw.line(
                    [cx - cross_size, cy, cx + cross_size, cy],
                    fill=(255, 255, 255),
                    width=2,
                )
                draw.line(
                    [cx, cy - cross_size, cx, cy + cross_size],
                    fill=(255, 255, 255),
                    width=2,
                )
                text_size = draw.textbbox((0, 0), label, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                label_x = cx + radius + 5
                label_y = cy - text_height // 2
                if label_x + text_width > img.size[0]:
                    label_x = cx - text_width - radius - 5
                if label_y < 0:
                    label_y = 0
                elif label_y + text_height > img.size[1]:
                    label_y = img.size[1] - text_height
                draw.rectangle(
                    [
                        label_x - 2,
                        label_y - 2,
                        label_x + text_width + 2,
                        label_y + text_height + 2,
                    ],
                    fill=color,
                )
                draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
                logger.info(f"Drew center point: ({cx}, {cy}) for {label}")
            else:
                logger.warning(f"Object {label} has no valid coordinate format.")
        if output_path:
            img.save(output_path)
            return output_path
        else:
            return img

    def save_results(self, image: Image, detection_data, query):
        """Save detection results to a timestamped folder under output/detection/molmo_local/."""
        # Create timestamped folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = "".join([c if c.isalnum() else "_" for c in query[:30]])
        folder_name = f"{timestamp}_{clean_name}"

        results_dir = output_path(
            Path("output") / "detection" / self.model_name / folder_name
        )

        # Save detection data as JSON
        json_path = results_dir / "detection_result.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(detection_data, f, indent=2)

        # Save visualization with bounding boxes overlaid on the original image
        visualization_path = results_dir / "visualization.png"
        self.visualize_detections(image, detection_data, str(visualization_path))

        # Save a copy of the original image
        img = image.copy()
        img.save(results_dir / "original_img.png")

        logger.info("Results saved to %s", str(results_dir))
        return str(results_dir)
