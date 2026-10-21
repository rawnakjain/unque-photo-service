"""Multimodal menu extraction using the Google GenAI SDK (Gemini) via Vertex AI.

Gemini answers in a compact, line-oriented format (one record per line) rather than
verbose JSON. Output generation dominates end-to-end latency, and this format needs
~5x fewer output tokens. Lines are parsed while the response streams in, so callers
can start matching items before generation has finished.
"""

import io
import logging
import queue
import re
import threading
import time
from typing import Callable, Optional

from PIL import Image

from google import genai
from google.genai.types import (
    FinishReason,
    GenerateContentConfig,
    HttpOptions,
    HttpRetryOptions,
    MediaResolution,
    Part,
    ThinkingConfig,
)

from service_testing.config import settings
from service_testing.schemas import (
    ExtractedMenuItem,
    GeminiExtractionResponse,
    GenderEnum,
    PersonTypeEnum,
    PriceTypeEnum,
)

logger = logging.getLogger(__name__)

_MEDIA_RESOLUTION = {
    "LOW": MediaResolution.MEDIA_RESOLUTION_LOW,
    "MEDIUM": MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "HIGH": MediaResolution.MEDIA_RESOLUTION_HIGH,
}

# Transient upstream errors worth retrying at the HTTP layer.
_RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504]

# Event kinds posted by streaming attempts.
_CHUNK, _DONE, _ERROR = "chunk", "done", "error"

ALLOWED_SERVICE_NAMES: tuple[str, ...] = (
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
    "Tattoo", "Piercing", "PMU", "Bridal", "Academy",
)
_CANONICAL_BY_LOWER = {name.lower(): name for name in ALLOWED_SERVICE_NAMES}

EXTRACTION_PROMPT_TEMPLATE = """You are an expert document-understanding system that digitizes salon service menus, price lists and brochures from images. Photos may be tilted, skewed, blurry, low-resolution, unevenly lit or affected by glare. Extract EVERY purchasable service with its price, maximizing completeness while preserving accuracy. Never invent information.

READING RULES
- Identify layout, sections, headings, sub-headings and columns. Read each column and section independently and never pair a name with a price from another row, column or section.
- Reconstruct service names that wrap onto multiple lines.
- Price tables (row labels x column headers, e.g. body areas x wax types): output one item per priced cell, named "<row> - <column>" (e.g. "Half Arms - Rica Wax"). Whenever a row has prices under several column headers, always append the column header to the name.
- Output every item exactly once.
- Skip descriptions, contact details, slogans and non-purchasable category headings. Include listed services that have no printed price (leave price empty).

OUTPUT FORMAT: plain text, one record per line, fields separated by "|". No JSON, no markdown, no commentary.
G|service|canonical|new
I|name|price|raw|flags|duration

A G line starts a group; every following I line belongs to it. Start a new group whenever the broad service category changes.
- service: the broad, base service category (e.g. "Waxing", "Facial", "Haircut") - the parent grouping, NOT the specific variation. Use the most specific heading that names a service category: when a section title (e.g. "BEAUTY", "SKIN SERVICES") has sub-headings (e.g. "Threading", "Waxing"), use the sub-heading.
- canonical: the matching entry from this whitelist for the service category, copied exactly; map synonyms (e.g. "Hair Cut" -> "Haircut", "Wax" -> "Waxing"). Leave empty if nothing fits.
  Whitelist: """ + ", ".join(f'"{name}"' for name in ALLOWED_SERVICE_NAMES) + """
- new: only when canonical is empty - a clean, standardized candidate service name (e.g. "Scalp Treatment").
- name: the specific purchasable variation including meaningful modifiers like packages, brands, body areas, hair lengths or techniques (e.g. "Full Leg Wax", "Root Touchup - Inoa", "Basic Cleanup Facial"). Use "Base Service - Package Name" with an exact " - " separator when applicable.
- price: clean number without currency symbols or commas; lower bound for ranges. Empty if unreadable or not printed. List numbering or bullets (e.g. "1.", "2)") are never prices.
- raw: the price exactly as printed (e.g. "₹500", "500/-", "500-800").
- flags: concatenate any that apply, else leave empty: S = starting price (prefixed with "Starts from", "Onwards", "From", "@", or a range / base tier like "500-800"); M or F = male- or female-specific item or section; C = children's service.
- duration: printed duration (e.g. "45 mins"); empty if not printed.
Trailing empty fields may be dropped.

Example:
G|Waxing|Waxing|
I|Full Leg Wax|500|₹500
I|Half Arms - Rica Wax|170|₹170|S
G|Scalp Care||Scalp Treatment
I|Dandruff Control|1200|1200/-|F|45 mins"""

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_FLAGS_RE = re.compile(r"[SMFC]*")


def _clean(value: str) -> Optional[str]:
    value = value.strip()
    return value or None


class MenuLineParser:
    """Incrementally parses the G|/I| line format into ExtractedMenuItem objects.

    Malformed lines are skipped rather than failing the whole extraction.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._service: Optional[str] = None
        self._canonical: Optional[str] = None
        self._new_service: Optional[str] = None
        self.skipped_lines = 0

    def feed(self, text: str) -> list[ExtractedMenuItem]:
        self._buffer += text
        *lines, self._buffer = self._buffer.split("\n")
        return [item for item in map(self._parse_line, lines) if item is not None]

    def close(self, discard_partial: bool = False) -> list[ExtractedMenuItem]:
        """Parses whatever is left in the buffer. A truncated response's last line is
        incomplete, so callers pass discard_partial=True in that case."""
        remainder, self._buffer = self._buffer, ""
        if discard_partial:
            return []
        item = self._parse_line(remainder)
        return [item] if item is not None else []

    def _parse_line(self, line: str) -> Optional[ExtractedMenuItem]:
        line = line.strip()
        if "|" not in line:
            return None
        fields = [f.strip() for f in line.split("|")]
        tag = fields[0].upper()

        if tag == "G":
            fields += [""] * (4 - len(fields))
            self._service = _clean(fields[1])
            self._canonical = _clean(fields[2])
            self._new_service = _clean(fields[3])
            return None

        if tag != "I":
            self.skipped_lines += 1
            return None

        fields += [""] * (6 - len(fields))
        name = fields[1]
        if not name:
            self.skipped_lines += 1
            return None

        number = _NUMBER_RE.search(fields[2].replace(",", ""))
        flags = fields[4].upper().replace(" ", "")
        duration = fields[5]
        if not _FLAGS_RE.fullmatch(flags):
            # Shifted column (e.g. a duration in the flags slot): don't read its
            # letters as flags.
            duration = duration or fields[4]
            flags = ""
        has_m, has_f = "M" in flags, "F" in flags

        # Prompt instructions alone are not validation: enforce the canonical
        # whitelist here before downstream matching.
        service = self._service or name
        canonical = _CANONICAL_BY_LOWER.get((self._canonical or "").lower())
        new_service = None if canonical else (self._new_service or service)

        return ExtractedMenuItem(
            service=service,
            variation_name=name,
            canonical_service_name=canonical,
            recommended_new_service_name=new_service,
            price=float(number.group()) if number else None,
            price_raw=_clean(fields[3]),
            price_type=PriceTypeEnum.starting if "S" in flags else PriceTypeEnum.fixed,
            approx_duration=_clean(duration),
            gender=GenderEnum.M if has_m and not has_f else GenderEnum.F if has_f and not has_m else GenderEnum.U,
            person_type=PersonTypeEnum.Child if "C" in flags else PersonTypeEnum.Adult,
        )


class GeminiExtractorService:
    def __init__(self, model_name: Optional[str] = None, fallback_model_name: Optional[str] = None):
        self.model_name = model_name or settings.gemini_model
        self.fallback_model_name = fallback_model_name or settings.gemini_fallback_model or None

        self.client = genai.Client(
            enterprise=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
            http_options=HttpOptions(
                timeout=int(settings.gemini_timeout_s * 1000),
                retry_options=HttpRetryOptions(
                    attempts=settings.gemini_max_attempts,
                    initial_delay=0.5,
                    max_delay=4.0,
                    http_status_codes=_RETRYABLE_STATUS_CODES,
                ),
            ),
        )

    def extract_from_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        on_item: Optional[Callable[[ExtractedMenuItem], None]] = None,
    ) -> GeminiExtractionResponse:
        """Extracts menu items from an image, streaming the model output.

        on_item is called for each item as soon as its line has been generated. If the
        primary model fails, the fallback model is tried, so on_item may also see items
        from an abandoned attempt; only the returned response is authoritative.
        """
        image_part = self._prepare_image_part(image_bytes)

        models = [self.model_name]
        if self.fallback_model_name and self.fallback_model_name != self.model_name:
            models.append(self.fallback_model_name)

        errors: list[str] = []
        last_error: Optional[Exception] = None
        for model in models:
            try:
                return self._stream_extract(model, image_part, on_item)
            except Exception as e:
                last_error = e
                errors.append(f"{model}: {e}")
                logger.warning("Gemini extraction with %s failed: %s", model, e)

        logger.error(f"Gemini extraction failed: {'; '.join(errors)}")
        raise RuntimeError(
            f"Error extracting menu items from image with Gemini: {'; '.join(errors)}"
        ) from last_error

    def _prepare_image_part(self, image_bytes: bytes) -> Part:
        """Validates, normalizes color mode, downscales and serializes the image sent to Gemini."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as source_image:
                source_image.load()
                detected_format = (source_image.format or "").upper()
                image = source_image.copy()
                if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                    # Flatten onto white: dropping alpha would turn transparent
                    # backgrounds black and hide dark menu text.
                    rgba = image.convert("RGBA")
                    image = Image.new("RGB", rgba.size, (255, 255, 255))
                    image.paste(rgba, mask=rgba.getchannel("A"))
                elif image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")

            max_dim = settings.gemini_max_image_dim
            width, height = image.size
            if max_dim > 0 and max(width, height) > max_dim:
                scale = max_dim / float(max(width, height))
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                image = image.resize(new_size, Image.LANCZOS)
                logger.info("Downscaled image from %sx%s to %sx%s", width, height, *new_size)

            format_to_mime = {
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }
            # Normalize unsupported formats to JPEG so bytes and MIME always agree.
            normalized_mime = format_to_mime.get(detected_format, "image/jpeg")
            output_format = {
                "image/jpeg": "JPEG",
                "image/png": "PNG",
                "image/webp": "WEBP",
            }.get(normalized_mime, "JPEG")

            normalized_buffer = io.BytesIO()
            image.save(normalized_buffer, format=output_format)
            normalized_image_bytes = normalized_buffer.getvalue()
        except Exception as e:
            raise ValueError(f"Invalid image file uploaded: {e}") from e

        return Part.from_bytes(data=normalized_image_bytes, mime_type=normalized_mime)

    def _generation_config(self, model: str) -> GenerateContentConfig:
        # Gemini 2.x takes a thinking token budget (0 disables thinking); Gemini 3+
        # takes a thinking level instead.
        if model.startswith("gemini-2"):
            thinking = ThinkingConfig(thinking_budget=settings.gemini_thinking_budget)
        else:
            thinking = ThinkingConfig(thinking_level=settings.gemini_thinking_level.strip().upper())

        config = GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=settings.gemini_max_output_tokens,
            thinking_config=thinking,
        )
        media_resolution = _MEDIA_RESOLUTION.get(
            (settings.gemini_media_resolution or "").strip().upper()
        )
        if media_resolution is not None:
            config.media_resolution = media_resolution
        return config

    def _start_attempt(
        self,
        model: str,
        image_part: Part,
        attempt: int,
        events: queue.Queue,
        cancel: threading.Event,
    ) -> None:
        """Streams one request on a background thread, posting (attempt, kind, payload) events."""

        def run() -> None:
            try:
                stream = self.client.models.generate_content_stream(
                    model=model,
                    contents=[EXTRACTION_PROMPT_TEMPLATE, image_part],
                    config=self._generation_config(model),
                )
                for chunk in stream:
                    if cancel.is_set():
                        close = getattr(stream, "close", None)
                        if close is not None:
                            close()
                        return
                    events.put((attempt, _CHUNK, chunk))
                events.put((attempt, _DONE, None))
            except Exception as e:
                events.put((attempt, _ERROR, e))

        threading.Thread(target=run, name=f"gemini-attempt-{attempt}", daemon=True).start()

    def _stream_extract(
        self,
        model: str,
        image_part: Part,
        on_item: Optional[Callable[[ExtractedMenuItem], None]],
    ) -> GeminiExtractionResponse:
        logger.info("Invoking Gemini model %s for image menu extraction...", model)
        parser = MenuLineParser()
        items: list[ExtractedMenuItem] = []
        finish_reason: Optional[FinishReason] = None
        first_token_ms: Optional[float] = None
        start = time.perf_counter()

        def emit(new_items: list[ExtractedMenuItem]) -> None:
            for item in new_items:
                items.append(item)
                if on_item is not None:
                    on_item(item)

        # Hedging: occasionally a request queues upstream for 8-9s before its first
        # token (normally ~1-2s). If nothing has streamed after hedge_after_s, send an
        # identical request and keep whichever starts streaming first.
        hedge_after_s = settings.gemini_hedge_after_s
        events: queue.Queue = queue.Queue()
        cancels = [threading.Event()]
        self._start_attempt(model, image_part, 0, events, cancels[0])
        running = 1
        winner: Optional[int] = None
        last_error: Optional[Exception] = None

        while True:
            can_hedge = winner is None and len(cancels) == 1 and hedge_after_s > 0
            timeout = (
                max(0.0, hedge_after_s - (time.perf_counter() - start))
                if can_hedge
                else settings.gemini_timeout_s
            )
            try:
                attempt, kind, payload = events.get(timeout=timeout)
            except queue.Empty:
                if not can_hedge:
                    raise TimeoutError(f"Gemini produced no output for {settings.gemini_timeout_s}s")
                logger.warning("No output from %s after %.1fs; sending a hedged request.", model, hedge_after_s)
                cancels.append(threading.Event())
                self._start_attempt(model, image_part, 1, events, cancels[1])
                running += 1
                continue

            if winner is not None and attempt != winner:
                continue
            if kind == _ERROR:
                if winner is not None:
                    raise payload
                running -= 1
                last_error = payload
                if running == 0:
                    raise last_error
                continue
            if winner is None:
                winner = attempt
                for i, cancel in enumerate(cancels):
                    if i != winner:
                        cancel.set()
                if winner > 0:
                    logger.info("Hedged request for %s won.", model)
            if kind == _DONE:
                break

            chunk = payload
            if chunk.candidates and chunk.candidates[0].finish_reason:
                finish_reason = chunk.candidates[0].finish_reason
            text = chunk.text
            if not text:
                continue
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - start) * 1000
            emit(parser.feed(text))

        truncated = finish_reason == FinishReason.MAX_TOKENS
        emit(parser.close(discard_partial=truncated))

        if truncated:
            logger.warning("Gemini output hit max_output_tokens; returning %d items parsed so far.", len(items))
        elif finish_reason not in (None, FinishReason.STOP) and not items:
            raise RuntimeError(f"Gemini stopped with finish_reason={finish_reason} and no items")
        if parser.skipped_lines:
            logger.warning("Skipped %d malformed lines in Gemini output.", parser.skipped_lines)

        logger.info(
            "Received %d items from %s (first token %.0fms, total %.0fms).",
            len(items),
            model,
            first_token_ms or 0.0,
            (time.perf_counter() - start) * 1000,
        )
        return GeminiExtractionResponse(items=items, model_used=model, first_token_ms=first_token_ms)
