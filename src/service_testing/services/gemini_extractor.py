"""Multimodal extraction service using Google GenAI SDK (gemini-2.5-flash)."""

import io
import json
import logging
from typing import Optional
from PIL import Image

from google import genai
from google.genai.types import HttpOptions, Part, GenerateContentConfig

from service_testing.config import settings
from service_testing.constants import UNIQUE_STANDARD_SERVICES
from service_testing.schemas import ExtractedMenuItem, GeminiExtractionResponse

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT_TEMPLATE = """
You are an expert AI assistant specialized in digitizing salon service menus and catalogues.
Your task is to analyze the provided salon price menu/catalogue image and extract every single service and service variation into structured JSON.

We have a STRICT list of standard service categories (approx 100). For each item on the menu, you MUST classify its broad category into one of these exact predefined service categories:
{allowed_services}

If an item does not cleanly match one, pick the closest semantic match (e.g., 'Creative Bob Haircut' -> 'haircut', 'Deep Cleansing Facial' -> 'facial', 'Chocolate Waxing' -> 'waxing', 'French Mani' -> 'manicure').

For each item, extract:
1. `service`: Broad service category from the allowed list above. (Lowercase string, matching one in the list).
2. `service_variation`: The exact variation or full service name as written on the menu (e.g. 'Creative Haircut - BOB By Senior Stylist', 'Full Arms Rika Wax', 'Hydra Facial Premium').
3. `price`: Numeric price (float/int), omit currency symbol (e.g. 850, 1500). If not available or range, use the starting price or null.
4. `price_raw`: The raw price text as written on the menu (e.g., '₹850', 'Starting @ 500', '$40').
5. `approx_duration`: Estimated or printed duration if mentioned (e.g., '45 mins', '1 hr', '30m').
6. `gender`: 'M' for Men, 'F' for Women, or 'U' for Unisex/General.
7. `person_type`: 'Adult' or 'Child'.

Return a JSON object with an "items" array containing all the detected services.
"""


class GeminiExtractorService:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model

        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured in settings or environment.")
            self.client = None
        else:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=HttpOptions(api_version="v1beta")
            )

    def extract_from_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> GeminiExtractionResponse:
        """Sends menu image to Gemini with structured schema output."""
        if not self.client:
            raise ValueError(
                "GEMINI_API_KEY is missing. Please set your GEMINI_API_KEY in .env or the environment."
            )

        # Open with PIL to validate image integrity
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Format image to RGB if needed
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image file uploaded: {e}")

        allowed_services_formatted = ", ".join(UNIQUE_STANDARD_SERVICES)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(allowed_services=allowed_services_formatted)

        # Prepare Part using google-genai SDK
        image_part = Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        try:
            logger.info(f"Invoking Gemini model {self.model_name} for image menu extraction...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image_part],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiExtractionResponse,
                    temperature=0.1,
                ),
            )

            response_text = response.text
            logger.info("Received structured extraction from Gemini.")

            if not response_text:
                return GeminiExtractionResponse(items=[])

            data = json.loads(response_text)
            return GeminiExtractionResponse.model_validate(data)

        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            # Try parsing with fallback if JSON output format was wrapped
            raise RuntimeError(f"Error extracting menu items from image with Gemini: {str(e)}")
