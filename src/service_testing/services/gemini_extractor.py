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

# - Organized into multiple columns, tables, sections, or text blocks.
#C. PRICE-ASSOCIATION RULES

# 1. Associate prices using visual alignment, table rows, column
#    structure, spacing, borders, typography, and section context.

EXTRACTION_PROMPT_TEMPLATE = """
You are an expert document-understanding and data-extraction system
specialized in digitizing salon service menus, price lists, brochures,
and catalogues from images.

Your task is to inspect the provided image and extract every distinct,
detectable salon service and its variations into structured JSON.

The input image may be:
- A photograph captured using a mobile phone.
- Tilted, rotated, skewed, or captured at an angle.
- Blurry, noisy, low-resolution, or unevenly illuminated.
- Affected by shadows, glare, reflections, folds, or perspective
  distortion.
- Partially cropped, obstructed, or difficult to read.
- Printed, handwritten, laminated, or digitally designed.

- Written in different fonts, sizes, colors, or languages.

Maximize extraction completeness while preserving accuracy.
Never invent information to compensate for poor image quality.

==================================================
1. IMAGE UNDERSTANDING AND VISUAL INTERPRETATION
==================================================

Before extracting services, analyze the entire image.

1. Identify the image orientation, layout, sections, columns,
   headings, subheadings, and visual grouping.

2. Account for rotation, perspective distortion, skew, and uneven
   alignment when interpreting text. Read content according to its
   apparent intended orientation.

3. Use visual context to interpret text affected by blur, low contrast,
   shadows, glare, or imperfect focus. Use surrounding characters,
   repeated terminology, and visual patterns only as supporting evidence.

4. Inspect every visible region systematically. Do not extract only
   prominent services or the first section.

5. Read each column and section independently before determining the
   intended reading order. Never accidentally combine text or prices
   from adjacent columns or unrelated sections.

6. When a service name spans multiple lines, reconstruct the complete
   name if the lines clearly belong together.

7. Distinguish actual service entries from headings, descriptions,
   advertisements, disclaimers, contact details, and other content.

8. If part of the image is unreadable or cropped, extract the readable
   services from the remaining regions.

9. Use visual evidence to distinguish similar-looking characters,
   digits, currency symbols, and punctuation.

10. Never claim to have recovered information that is not supported
    by the visible image.

==================================================
2. SERVICE EXTRACTION AND COMPLETENESS
==================================================

Extract every distinct salon service and service variation that can
be identified from the image.

For each service:
- Preserve the original service name and meaningful wording.
- Preserve distinctions such as Basic, Premium, Deluxe, Advanced,
  Bridal, Express, Full Body, Half Body, and similar qualifiers.
- Preserve stylist levels, product brands, treatment names, body
  areas, hair lengths, quantities, and other meaningful modifiers.
- Do not merge distinct services merely because they are similar.
- Do not omit a service merely because its price is missing.
- Do not invent services based on typical salon offerings.
- Do not treat category headings as service entries unless the
  heading itself clearly represents a purchasable service.
- Avoid duplicate entries caused by overlapping text or repeated
  visual inspection.
- Retain genuinely distinct offerings even if their names are similar.

The menu may contain services from any salon-related category,
including hair, skin, makeup, waxing, nails, spa, massage, grooming,
beauty treatments, packages, and other services.

Do not restrict extraction to a predefined service taxonomy.

If a service name is partially readable, extract it only when the
remaining text can be interpreted with reasonable confidence.

Never silently replace unreadable text with an invented service name.

==================================================
3. SERVICE FIELD
==================================================

The "service" field represents the broad or base service name
extracted from the menu.

Rules:
- Preserve the service name as written wherever possible.
- Remove package-specific or option-specific wording from the
  "service" field only when it can be clearly separated from the
  base service name.
- Do not force the service into any predefined category.
- Do not restrict service names to a fixed list.
- Do not map the service to a database entry.
- Do not replace the original service with a guessed standardized name.
- Do not unnecessarily paraphrase, translate, or normalize the name.
- When the base service cannot be separated confidently from its
  variation, preserve the complete identifiable service name rather
  than guessing.

Examples:

Menu text: "Full Arms - Sugar"
service: "Full Arms"
service_variation: "Full Arms - Sugar"

Menu text: "Haircut - Senior Stylist"
service: "Haircut"
service_variation: "Haircut - Senior Stylist"

Menu text: "Hydra Facial Premium"
service: "Hydra Facial"
service_variation: "Hydra Facial - Premium"

Menu text: "Creative Haircut - BOB By Senior Stylist"
service: "Creative Haircut"
service_variation: "Creative Haircut - BOB By Senior Stylist"

These examples illustrate the intended separation. Preserve the
actual wording and meaning supported by the menu.

==================================================
4. SERVICE VARIATION AND ORIGINAL NAMING
==================================================

The "service_variation" field identifies the specific service
offering, including its meaningful modifiers and options.

Preserve:
- Package names and package levels.
- Product brands and product types.
- Treatment names and techniques.
- Body areas and body coverage.
- Hair lengths and styles.
- Stylist levels.
- Service intensity, duration labels, and other meaningful options.
- Any other explicitly stated distinction affecting the offering.

Do not unnecessarily translate, paraphrase, or normalize names.

Reconstruct line breaks only when they clearly belong to the same
service.

If the menu shows no distinct variation or package, use the
identifiable service name itself as "service_variation".

==================================================
5. PACKAGE-BASED AND PRICE-BASED VARIATIONS
==================================================

A single base service may have multiple packages, product types,
treatment options, service levels, or pricing variants.

You MUST preserve these distinctions in "service_variation".

CORE RULE:

Whenever the same base service is offered with different explicitly
named packages, options, product types, or other selectable variants,
create a separate item for each distinct service-option combination.

When a package or option name is present, use this naming format:

"Base Service - Package Name"

Use exactly " - " (space, hyphen, space) as the separator.

Examples:
- "Full Arms - Sugar"
- "Full Arms - Rica"
- "Full Arms - Chocolate Wax"
- "Haircut - Basic Package"
- "Haircut - Premium Package"
- "Facial - Gold Package"
- "Facial - Diamond Package"
- "Manicure - Classic"
- "Manicure - Deluxe"
- "Haircut - Senior Stylist"
- "Haircut - Junior Stylist"
- "Full Legs - Sugar"
- "Full Legs - Rica"

PACKAGE ASSOCIATION RULES:

1. If a service has multiple explicitly named packages with different
   prices, create one item per package and associate each item with
   its correct price.

2. If multiple product types, brands, materials, techniques, or
   treatment options are offered for the same service, include the
   distinguishing option in "service_variation".

3. If the package name appears in a heading, subheading, table column,
   nearby label, or other layout element, associate it with services
   only when the visual structure clearly establishes the relationship.

4. If one package applies to several services, include the package
   name in each applicable service_variation when the relationship
   is clearly established.

5. Preserve all meaningful distinctions when a service includes a
   base name, package name, and additional qualifiers.

   Example:
   "Full Arms - Sugar - Premium"

   Use this only when the menu supports all those distinctions.

6. If a service has different prices for body areas, hair lengths,
   stylist levels, quantities, or other selectable options, preserve
   those differences in separate items when appropriate.

7. If the same service appears with different prices under different
   explicitly identified packages, do not merge those entries.

8. Do not create duplicate entries merely because a shared package
   heading appears above multiple services. Create separate entries
   for distinct service-package combinations, not for headings.

9. Do not invent package names based on common salon practices.
   Use only names supported by the image.

10. If the package name is unreadable, do not guess it. Preserve the
    identifiable service name and any reliably associated price.

11. If the menu does not specify a package or distinct option, retain
    the original service name without appending an invented suffix.

12. Do not append the same package name multiple times to one
    service_variation.

13. Keep the base service in "service" and the complete selectable
    offering in "service_variation". Do not put package names into
    the base service field when they can be separated confidently.

14. The "price" and "price_raw" fields must correspond to the exact
    service-tier combination represented by the item.
    Use it only for services with tiers, not for all the services.
    Tier services has generally more than one price depending upon the Tier

15. If a menu provides a general service price and additional
    package-specific prices, preserve the general offering only when
    it is clearly a distinct selectable option. Do not automatically
    create an extra base-service entry if the general label is merely
    a heading for the listed packages.

==================================================
6. PRICE EXTRACTION AND SERVICE-PRICE ASSOCIATION
==================================================

Each item must contain both "price" and "price_raw".

A. PRICE FIELD

The "price" field must be a JSON number without currency symbols
or thousands separators, or JSON null when the price cannot be
reliably determined.

Rules:
- Extract the printed numeric price accurately.
- If a price range is printed, use its lower bound as the numeric
  starting price.
- If a price is explicitly marked as starting from a value, use
  that value.
- If the price is missing, illegible, or cannot be reliably
  associated with the service, return null.
- Never estimate a price using neighboring services or typical
  salon pricing.
- Never calculate a price that is not explicitly supported by
  the menu.

B. RAW PRICE FIELD

The "price_raw" field must preserve the price text as it appears
in the image, including currency symbols, range notation, prefixes,
and meaningful price wording.

Examples:
- "₹850" -> price: 850, price_raw: "₹850"
- "Starting @ 500" -> price: 500, price_raw: "Starting @ 500"
- "From ₹1,500" -> price: 1500, price_raw: "From ₹1,500"
- "500 - 1200" -> price: 500, price_raw: "500 - 1200"
- No visible price -> price: null, price_raw: null
- Price present but completely illegible -> price: null,
  price_raw: null

For a readable price range, preserve the complete original range
in "price_raw" and use the lower bound in "price".

C. PRICE-ASSOCIATION RULES



2. Never assign a price to a service solely because it is the
   nearest visible number.

3. In multi-column layouts, ensure that each price belongs to the
   correct service in the correct column.

4. When several prices are listed under labeled variations, map
   each price to its corresponding variation.

5. If one price clearly applies to multiple services, use it for
   each applicable service only when the layout supports this.

6. When different prices correspond to packages, product types,
   stylist levels, body areas, or other options, preserve the
   distinction in "service_variation" and assign the correct price.

7. If the relationship between a service and its price is ambiguous,
   use null rather than an unsupported association.

8. Do not confuse page numbers, phone numbers, discount percentages,
   quantities, dates, or durations with service prices.

==================================================
7. DURATION EXTRACTION
==================================================

The "approx_duration" field represents the duration of a service.

Rules:
- Extract duration when explicitly printed or clearly associated
  with a service.
- Preserve the original wording where possible.
- Examples: "45 mins", "1 hr", "30m", "1 hour 30 minutes".
- Return null when no duration is shown or when its association
  with the service is uncertain.
- Do not estimate typical salon service durations.
- Do not confuse prices, quantities, dates, or product sizes
  with durations.

==================================================
8. GENDER CLASSIFICATION
==================================================

The "gender" field must contain exactly one of:

"M" = explicitly intended for men.
"F" = explicitly intended for women.
"U" = unisex, general, or not explicitly gender-specific.

Rules:
- Prefer explicit section headings such as Men's Services,
  Ladies' Services, Gents, or Women's Salon.
- Apply section-level gender labels only when the layout clearly
  establishes that they apply to the relevant services.
- Use explicit labels in service names as additional evidence.
- Do not infer gender solely from stereotypes, service names,
  hair length, or common salon practices.
- If no gender restriction is established, use "U".

==================================================
9. PERSON TYPE CLASSIFICATION
==================================================

The "person_type" field must contain exactly one of:

"Adult"
"Child"

Rules:
- Use "Child" when the service is explicitly identified as a
  children's, kids', or child-specific service, or appears under
  a clearly applicable children's section.
- Use "Adult" for adult-specific services and ordinary salon
  offerings without a child-specific designation.
- Do not infer that a service is for children because it is
  inexpensive, simple, or appears suitable for children.
- Apply section-level labels only when their scope is clear.

==================================================
10. HANDLING UNCERTAINTY AND POOR IMAGE QUALITY
==================================================

Accuracy is more important than guessing.

- Extract every service supported by visible evidence.
- Do not fabricate missing service names, packages, prices,
  durations, or other details.
- Use null for unavailable or unreadable price, raw price,
  and duration values.
- Use "U" when gender is not established by the menu.
- Use "Adult" when no child-specific designation is evident.
- If a service is too illegible to identify reliably, do not
  invent its name.
- Do not discard an otherwise identifiable service merely because
  one of its fields is unreadable.
- Use surrounding context to resolve ambiguity only when it
  provides genuine visual or semantic evidence.
- If the image contains readable and unreadable sections, extract
  the readable sections rather than rejecting the entire image.

==================================================
11. OUTPUT FORMAT
==================================================

Return ONLY a valid JSON object containing an "items" array.

Every item must contain exactly these seven fields:

{
  "service": "Full Arms",
  "service_variation": "Full Arms - Sugar",
  "price": 500,
  "price_raw": "₹500",
  "approx_duration": null,
  "gender": "U",
  "person_type": "Adult"
}

This is an illustrative example only. Extract actual values from
the supplied image.

OUTPUT REQUIREMENTS:

1. Return a JSON object with the top-level key "items".
2. Return one object for every distinct, detectable service or
   service-option combination.
3. Include all seven required fields for every item.
4. Do not add additional fields.
5. Use JSON null for unavailable price, raw price, or duration.
6. Never use the string "null" in place of JSON null.
7. Use valid JSON double quotes and proper escaping.
8. Do not include Markdown fences, explanations, comments, or
   introductory text outside the JSON object.
9. Do not restrict the "service" field to any predefined category.
10. Preserve the service name and package distinctions accurately.
11. Every "gender" value must be "M", "F", or "U".
12. Every "person_type" value must be "Adult" or "Child".
13. Every non-null "price" value must be numeric.
14. If no services can be reliably identified, return:
    {"items": []}

==================================================
12. FINAL VALIDATION BEFORE RESPONDING
==================================================

Before producing the JSON, silently verify:

IMAGE COVERAGE:
- All visible sections, columns, tables, and readable regions
  have been inspected.
- No clearly detectable service has been overlooked.

SERVICE ACCURACY:
- Service names preserve the original meaningful wording.
- The "service" field contains the base service name when it can
  be confidently separated from its variation.
- No predefined category restrictions have been applied.
- Distinct services have not been incorrectly merged.
- Unreadable content has not been replaced with invented text.

PACKAGE AND VARIATION ACCURACY:
- Distinct package-based offerings have separate entries.
- Each applicable service_variation includes the package or option
  name using "Base Service - Package Name".
- Package names have not been invented.
- Each price belongs to the correct service-option combination.

DATA ACCURACY:
- Prices are associated with the correct services.
- Price ranges retain their complete original text in price_raw.
- Missing or uncertain prices and durations are null.
- Gender and person_type values follow the specified rules.
- Duplicate entries have been avoided unless they represent
  genuinely distinct offerings.

JSON VALIDITY:
- The output is valid JSON.
- Every item contains exactly the seven required fields.
- No text appears outside the JSON object.

Return the final JSON object now.
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
            logger.info("Received structured extraction from Gemini.")

            if not response_text:
                return GeminiExtractionResponse(items=[])

            data = json.loads(response_text)
            return GeminiExtractionResponse.model_validate(data)

        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            # Try parsing with fallback if JSON output format was wrapped
            raise RuntimeError(f"Error extracting menu items from image with Gemini: {str(e)}")
