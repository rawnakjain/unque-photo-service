# Salon Menu AI Onboarding POC

A proof-of-concept for automating salon price menu onboarding using:
- **Google Gemini 2.5 Flash** (Vertex AI) for multimodal menu image extraction
- **MongoDB Atlas Fuzzy Search** (`$search` with fuzzy/phrase matching) for service & variation matching
- **FastAPI** backend with a **Vercel Design System** styled web UI

## Features

1. **Automated Menu Extraction**: Upload a salon price menu/image → Gemini extracts:
   - Base service category (mapped to a canonical whitelist of 60+ services)
   - Specific service variation name
   - Price, raw price, price type (`fixed` / `starting`), duration
   - Gender (`M`/`F`/`U`) and person type (`Adult`/`Child`)

2. **Fuzzy Service & Variation Matching**: For each extracted item, runs MongoDB Atlas Fuzzy Search:
   - Matches the service against the `salonservices` collection to obtain a `serviceId`
   - Matches the variation against the `salonservicevariations` collection (scoped to the matched service when possible)
   - Returns top-5 matches with normalized similarity percentages

3. **Interactive Web UI**:
   - Drag-and-drop image upload with instant preview
   - Real-time processing status
   - Editable extraction fields and dropdown selector for top-5 matches
   - Visual match quality indicators (High Match / Review Match)
   - Salon owner confirmation flow

## Architecture

```
┌─────────────────┐    ┌────────────────────────────┐    ┌──────────────────────┐
│   Menu Image    │───▶│ Gemini 2.5 Flash Extraction│───▶│ Extracted Items      │
│ (Upload via UI) │    │ (Structured JSON Output)   │    │ + Variation Names    │
└─────────────────┘    └────────────────────────────┘    └──────────────────────┘
                                                                     │
                                                                     ▼
                                                       ┌─────────────────────────────┐
                                                       │ MongoDB Atlas Fuzzy Search  │
                                                       │ 1. salonservices            │
                                                       │ 2. salonservicevariations   │
                                                       │    ($search + fuzzy, regex  │
                                                       │     fallback)               │
                                                       └─────────────────────────────┘
                                                                     │
                                                                     ▼
                                                       ┌─────────────────────────────┐
                                                       │ Top-5 Recommendations per   │
                                                       │ Extracted Item (normalized  │
                                                       │ similarity scores)          │
                                                       └─────────────────────────────┘
                                                                     │
                                                                     ▼
                                                       ┌─────────────────────────────┐
                                                       │ Salon Owner Review &        │
                                                       │ Selection (Web UI)          │
                                                       └─────────────────────────────┘
                                                                     │
                                                                     ▼
                                                       ┌─────────────────────────────┐
                                                       │ Confirmed Catalog           │
                                                       └─────────────────────────────┘
```

## Application Flow

### File Flow Order

| # | File | Purpose |
|---|------|---------|
| 1 | `main.py` | FastAPI entrypoint, runs Uvicorn on port 8000 |
| 2 | `src/service_testing/app.py` | Creates the FastAPI app, mounts static files, includes API routes |
| 3 | `src/service_testing/config.py` | Pydantic Settings: Gemini/Vertex AI, MongoDB URI, collection & index names |
| 4 | `src/service_testing/schemas.py` | Pydantic models for all request/response structures |
| 5 | `src/service_testing/constants.py` | Standard service categories used in the extraction prompt |
| 6 | `src/service_testing/services/gemini_extractor.py` | `GeminiExtractorService` – sends the menu image to Gemini with a structured prompt and returns `ExtractedMenuItem`s |
| 7 | `src/service_testing/services/fuzzy_search.py` | `FuzzySearchService` – Atlas `$search` with fuzzy/phrase clauses, regex fallback, score normalization, dedup and a semantic firewall |
| 8 | `src/service_testing/services/image_quality.py` | Blur/corruption validation before extraction |
| 9 | `src/service_testing/services/menu_processor.py` | `MenuProcessorService` – orchestrates extraction + matching |
| 10 | `src/service_testing/api/routes.py` | API endpoints |
| 11 | `src/service_testing/static/index.html` | Alpine.js single-page UI |

### Complete Request Flow

```
User uploads menu image
        │
        ▼
POST /api/menu/extract-and-match
        │
        ▼
Image quality validation (blur / corruption)
        │
        ▼
MenuProcessorService.process_menu_image()
        │
        ├──► GeminiExtractorService.extract_from_image()
        │       │
        │       ▼
        │   Gemini 2.5 Flash (Vertex AI), structured JSON output
        │       │
        │       ▼
        │   Returns: List[ExtractedMenuItem]
        │
        ▼
For each ExtractedMenuItem:
        │
        ├──► FuzzySearchService.recommend_services(service_lookup_name)
        │       │  lookup priority: canonical_service_name
        │       │                 > recommended_new_service_name
        │       │                 > service
        │       ▼
        │   Atlas $search on salonservices (index: salonservices_index)
        │       │
        │       ▼
        │   Returns top-5 services → select top match → serviceId
        │
        └──► FuzzySearchService.recommend_variations(variation_name, service_id=…)
                │
                ▼
            Atlas $search on salonservicevariations
            (index: salonservicevariations_index), scoped to serviceId
                │
                ▼
            Returns top-5 variations
        │
        ▼
Return MenuProcessingResponse
        │
        ▼
Frontend renders table with service + variation match columns
```

### Scoring & Ranking

MongoDB Atlas Search uses **BM25** relevance scoring.

**Search strategy** (in `fuzzy_search.py`):
- Short queries (≤ 4 chars) use strict `text` clauses with no fuzzy expansion to prevent drift.
- Longer queries add a `phrase` clause plus `text` clauses with `fuzzy` (`maxEdits: 2`, `prefixLength: 2`), boosting exact field matches.
- If Atlas Search is unavailable, the service falls back to a regex `$or` search over the same fields.

**Score normalization** (application level, since raw BM25 scores are unbounded):

```python
max_score = max((r.get("score", 0.0) for r in raw_results), default=1.0)
normalized_score = min(raw_score / max_score, 1.0)      # 0.0 – 1.0
similarity_percentage = int(round(normalized_score * 100))  # 0–100%
```

**Post-processing**: minimum 80% confidence threshold → deduplicate by ID and name (case-insensitive) → semantic firewall (drops unrelated fuzzy drift, e.g. generic category overlap) → top 5 per category.

| Threshold | Meaning |
|-----------|---------|
| **≥ 80%** | HIGH MATCH – shown with a blue badge |
| **< 80%** | REVIEW MATCH – owner attention required |
| **No results** | "No confident match" – manual selection required |

### Atlas Search Indexes Required

`salonservices_index` (collection `salonservices`):
```json
{
  "mappings": {
    "fields": {
      "salonServiceName": { "type": "string" },
      "serviceNameEnum": { "type": "string" }
    }
  }
}
```

`salonservicevariations_index` (collection `salonservicevariations`):
```json
{
  "mappings": {
    "fields": {
      "fullName": { "type": "string" },
      "salonServiceName": { "type": "string" },
      "comment": { "type": "string" }
    }
  }
}
```

## Setup

1. **Environment Variables** (copy `.env.example` to `.env` and fill in real values):
   ```bash
   cp .env.example .env
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

3. **Run the Application**:
   ```bash
   uvicorn main:app --reload
   ```
   Then visit: http://localhost:8000

## API Endpoints

- `POST /api/menu/extract-and-match` – Upload a menu image → extraction + matching, returns `MenuProcessingResponse`
- `POST /api/v1/media-service/image-2-text/photo` – Same pipeline, returns a nested `MediaServiceResponse` (`recommendedVariations` + `variationRecommendations`)

## Technology Stack

- **Backend**: FastAPI, Python 3.13
- **AI**: Google Gemini 2.5 Flash (Vertex AI) for multimodal extraction
- **Database**: MongoDB Atlas Search (fuzzy + phrase, BM25 scoring)
- **Frontend**: Vercel Design System (Geist fonts, monochrome aesthetic), Tailwind CSS + Alpine.js (no build step)
- **Key dependencies**: `fastapi`, `uvicorn`, `google-genai`, `pymongo`, `pillow`, `opencv-python`

## Project Structure

```
.
├── main.py                     # Uvicorn entrypoint
├── pyproject.toml
├── DESIGN.md                   # Vercel design-system token reference for the UI
└── src/service_testing/
    ├── app.py
    ├── config.py
    ├── constants.py
    ├── schemas.py
    ├── api/routes.py
    ├── services/
    │   ├── gemini_extractor.py
    │   ├── fuzzy_search.py
    │   ├── image_quality.py
    │   └── menu_processor.py
    └── static/index.html
```

## Verification

Start the server and upload a sample menu image via the web UI at http://localhost:8000.

## Future Enhancements

- [ ] Add support for multi-page menu PDFs
- [ ] Implement batch menu processing
- [ ] Add salon-specific custom service variations
- [ ] Export to CSV/JSON for POS system integration
- [ ] Add authentication and multi-salon tenant support
- [ ] Implement caching for frequent menu variations
- [ ] Add feedback loop to improve extraction accuracy

---
*Built as a POC to reduce manual salon onboarding work from hours to minutes.*
