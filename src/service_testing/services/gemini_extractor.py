"""Multimodal extraction service using Google GenAI SDK (gemini-2.5-flash) via Vertex AI."""

import io
import json
import logging
from typing import Optional
from PIL import Image

from google import genai
from google.genai.types import Part, GenerateContentConfig

from service_testing.config import settings
from service_testing.schemas import ExtractedMenuItem, GeminiExtractionResponse

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT_TEMPLATE = """
You are an expert document-understanding and data-extraction system specialized in digitizing salon service menus, price lists, brochures, and catalogues from images.

Your task is to inspect the provided image and extract every distinct salon service, variation, and price into structured JSON.

The input image may be captured via mobile phone, tilted, rotated, skewed, blurry, low-resolution, unevenly illuminated, affected by shadows/glare, or printed in various fonts and styles. Maximize extraction completeness while preserving accuracy. Never invent information.

==================================================
1. IMAGE UNDERSTANDING & LAYOUT INTERPRETATION
==================================================
- Identify layout, sections, columns, headings, and subheadings.
- Read each column and section independently before determining reading order. Never combine text or prices from unrelated sections.
- When a service name spans multiple lines, reconstruct the complete name if they clearly belong together.
- Distinguish actual service entries from descriptions, contact details, or non-purchasable category headings.
- **CRITICAL**: Distinguish between the broad SERVICE CATEGORY (e.g., "Waxing", "Facial", "Haircut") and the specific PURCHASABLE VARIATION (e.g., "Full Leg Wax", "Basic Cleanup Facial", "Adult Haircut"). The service is the parent category; the variation is the specific item with price.

==================================================
2. EXTRACTION FIELDS & MAPPING RULES
==================================================

For each extracted offering, populate the following fields:

1. **service**: The broad, base service category extracted from the menu (e.g., "Waxing", "Facial", "Haircut"). This is the parent/grouping category, NOT the specific variation.

2. **variation_name**: The specific purchasable item including meaningful modifiers like packages, brands, body areas, hair lengths, or techniques (e.g., "Full Leg Wax", "Root Touchup - Inoa", "Basic Cleanup Facial"). Use "Base Service - Package Name" format with an exact " - " separator when applicable. This MUST be different from the service field when a specific variation exists.

3. **canonical_service_name**: Map the extracted service (field 1) to the most appropriate match from this strictly allowed whitelist, or set to `null` if it does not fit:
[
  "Basic Cleanup Facial", "Beard Colour", "Beard Shaving", "Beard Spa", "Beard Styling", 
  "Bleach", "Blow Dry", "Body Spa", "Cleansing", "Curls And Waves", "D Tan", 
  "Dandruff Treatment", "Deep Facial", "Face Wash", "Hair Colouring Entire Hair", 
  "Hair Colouring Root Touch Up", "Hair Conditioning", "Hair Detoxifying", "Hair Extensions", 
  "Hair Highlighting", "Hair Patch", "Hair Revive Treatment", "Hair Smoothing", "Hair Spa", 
  "Hair Straightening", "Hair Wash", "Haircut", "Hairstyle By Arrangement", 
  "Laser Hair Removal", "Makeup", "Manicure", "Mask", "Massage", "Moustache Colour", 
  "Nails Art", "Nails Extensions", "Nails Overlays", "Nails Polish", "Nails Removal", 
  "Pedicure", "Saree Draping", "Scrubbing", "Skin Treatment", "Threading", "Trimming", 
  "Waxing", "Nails Basic Care", "Mehendi", "Bond Care Treatment", "W One Nanoplastia", 
  "Protein Treatment", "Vegan Botox", "Curl Softening Treatment", "Balayage", 
  "Temporary Ironing", "Adult Haircut", "Child Haircut", "Beard Care", "Gel Overlay", 
  "Tattoo", "Piercing", "PMU", "Bridal", "Academy"
]
- Match the SERVICE category (field 1) to this whitelist, not the variation.
- If the menu uses a synonym (e.g., "Hair Cut" -> "Haircut", "Wax" -> "Waxing"), map to the canonical name.
- If no whitelist entry matches the service category, set to `null`.

4. **recommended_new_service_name**: If the service (field 1) does NOT fit neatly into any whitelisted category, set `canonical_service_name` to `null` and provide a clean, standardized candidate string here (e.g., "Scalp Treatment"). Otherwise, set to `null`.

5. **price**: Clean numeric value (integer/float) without currency symbols, or `null` if unreadable. For price ranges, use the lower bound. MUST be a number, not a string.

6. **price_raw**: Exact price string as it appears in the image (e.g., "₹500", "Starting @ 500", "500-800"). Preserve original formatting.

7. **price_type**: Must be `"fixed"` or `"starting"`. 
   - Use `"starting"` if prefixed with terms like "Starts from", "Onwards", "From", "@", or if it represents a price range/base tier (e.g., "500-800").
   - Use `"fixed"` for single prices without such prefixes (e.g., "₹500", "500").
   - Default to `"fixed"` if uncertain.

8. **approx_duration**: Printed duration (e.g., "45 mins", "1 hour", "30 min") or `null`.

9. **gender**: Exactly one of `"M"`, `"F"`, or `"U"` (unisex/general). Default `"U"` if not specified.

10. **person_type**: Exactly one of `"Adult"` or `"Child"`. Default `"Adult"` if not specified.

==================================================
3. VALIDATION RULES (MUST FOLLOW)
==================================================
- Every item MUST have all 10 fields. Use `null` (JSON null) for missing optional data.
- `service` and `variation_name` are REQUIRED strings (non-empty).
- `price` must be a number or null, never a string.
- `price_type` must be exactly "fixed" or "starting".
- `gender` must be exactly "M", "F", or "U".
- `person_type` must be exactly "Adult" or "Child".
- `canonical_service_name` must be from the whitelist or null.
- If `canonical_service_name` is null, `recommended_new_service_name` must be a string (not null).
- If `canonical_service_name` is not null, `recommended_new_service_name` must be null.

==================================================
4. OUTPUT FORMAT
==================================================
Return ONLY a valid JSON object containing an "items" array. No Markdown code block wrappers outside the JSON, no explanations, no introductory text.

Example format:
{
  "items": [
    {
      "service": "Waxing",
      "variation_name": "Full Leg Wax",
      "canonical_service_name": "Waxing",
      "recommended_new_service_name": null,
      "price": 500,
      "price_raw": "₹500",
      "price_type": "fixed",
      "approx_duration": null,
      "gender": "U",
      "person_type": "Adult"
    },
    {
      "service": "Facial",
      "variation_name": "Basic Cleanup Facial",
      "canonical_service_name": "Basic Cleanup Facial",
      "recommended_new_service_name": null,
      "price": 800,
      "price_raw": "₹800",
      "price_type": "fixed",
      "approx_duration": "45 mins",
      "gender": "F",
      "person_type": "Adult"
    },
    {
      "service": "Haircut",
      "variation_name": "Adult Haircut",
      "canonical_service_name": "Adult Haircut",
      "recommended_new_service_name": null,
      "price": 300,
      "price_raw": "Starting @ 300",
      "price_type": "starting",
      "approx_duration": "30 mins",
      "gender": "U",
      "person_type": "Adult"
    }
  ]
}

OUTPUT REQUIREMENTS:
- Return a JSON object with top-level key "items".
- Include all ten required fields for every item.
- Use JSON null for unavailable price, raw price, duration, or unmapped names. Never use the string "null".
- If no services can be reliably identified, return: {"items": []}
"""




class GeminiExtractorService:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.gemini_model

        self.client = genai.Client(
            enterprise=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )

    def extract_from_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> GeminiExtractionResponse:
        """Sends menu image to Gemini with structured schema output."""

        # Open with PIL to validate image integrity
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Format image to RGB if needed
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image file uploaded: {e}")

        # Prepare Part using google-genai SDK
        image_part = Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        try:
            logger.info(f"Invoking Gemini model {self.model_name} for image menu extraction...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[EXTRACTION_PROMPT_TEMPLATE, image_part],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiExtractionResponse,
                    temperature=0.1,
                ),
            )

            response_text = response.text
            print(response_text)
            logger.info("Received structured extraction from Gemini.")

            if not response_text:
                return GeminiExtractionResponse(items=[])

            data = json.loads(response_text)
            return GeminiExtractionResponse.model_validate(data)

        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            # Try parsing with fallback if JSON output format was wrapped
            raise RuntimeError(f"Error extracting menu items from image with Gemini: {str(e)}")
