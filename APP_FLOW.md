# Salon Menu AI Onboarding - Complete Flow Documentation

## Overview
This application automates salon price menu onboarding by:
1. Extracting services & variations from menu images using **Gemini 2.5 Flash Vision**
2. Matching extracted items to salon owner's database using **MongoDB Atlas Fuzzy Search**
3. Presenting confidence-scored recommendations for owner confirmation

---

## File Flow Order

### 1. Entry Point
| File | Purpose |
|------|---------|
| `main.py` | FastAPI app entrypoint, runs Uvicorn server on port 8000 |
| `src/service_testing/app.py` | Creates FastAPI app, mounts static files, includes API routes |

### 2. Configuration
| File | Purpose |
|------|---------|
| `src/service_testing/config.py` | Pydantic Settings: Gemini API key, MongoDB URI, collection names, Atlas Search index names |

### 3. Data Schemas
| File | Purpose |
|------|---------|
| `src/service_testing/schemas.py` | Pydantic models for all data structures:<br>- `ExtractedMenuItem` - Gemini output<br>- `VariationRecommendation` - DB match with score<br>- `ProcessedMenuItem` - Combined item + recommendations<br>- `MenuProcessingResponse` - Full API response |

### 4. Constants
| File | Purpose |
|------|---------|
| `src/service_testing/constants.py` | Predefined ~100 standard service categories (`UNIQUE_STANDARD_SERVICES`) used in Gemini prompt |

### 5. Services Layer

#### 5.1 Gemini Extraction
| File | Purpose |
|------|---------|
| `src/service_testing/services/gemini_extractor.py` | `GeminiExtractorService`<br>- Sends menu image to Gemini 2.5 Flash with structured prompt<br>- Returns `GeminiExtractionResponse` with list of `ExtractedMenuItem`<br>- Each item has: `service` (category), `service_variation` (full name), `price`, `gender`, `person_type` |

#### 5.2 MongoDB Atlas Fuzzy Search
| File | Purpose |
|------|---------|
| `src/service_testing/services/fuzzy_search.py` | `FuzzySearchService`<br>**Two search methods:**<br>1. `recommend_services(query)` - searches `salonservices` collection on `salonServiceName`, `serviceNameEnum`<br>2. `recommend_variations(query)` - searches `salonservicevariations` collection on `fullName`, `salonServiceName`, `comment`<br><br>**Pipeline:** Atlas `$search` with fuzzy (`maxEdits: 2`) → `$addFields` for score → `$match` score ≥ threshold → `$project` → `$limit`<br><br>**Fallback:** Regex search if Atlas index unavailable<br><br>**Deduplication:** By ID + name (case-insensitive)<br><br>**Score Normalization:** Atlas BM25 scores normalized to 0-100% using max score in result set |

#### 5.3 Menu Processor (Orchestrator)
| File | Purpose |
|------|---------|
| `src/service_testing/services/menu_processor.py` | `MenuProcessorService`<br>**Orchestrates full pipeline:**<br>1. Call Gemini to extract items from image<br>2. For each extracted item:<br>   a. Search **services** in `salonservices` using `item.service`<br>   b. Search **variations** in `salonservicevariations` using `item.service_variation`<br>3. Apply 80% confidence threshold to both<br>4. Return `ProcessedMenuItem` with separate `service_recommendations` and `variation_recommendations` lists |

### 6. API Routes
| File | Purpose |
|------|---------|
| `src/service_testing/api/routes.py` | FastAPI endpoints:<br>- `GET /api/health` - System status (DB, Gemini)<br>- `GET /api/services` - List 100 standard categories<br>- `POST /api/services/search` - Direct service fuzzy search<br>- `POST /api/variations/search` - Direct variation fuzzy search<br>- `POST /api/menu/extract-and-match` - **Main endpoint**: upload image → extract → match → return recommendations<br>- `POST /api/menu/confirm` - Owner confirms selections |

### 7. Frontend
| File | Purpose |
|------|---------|
| `src/service_testing/static/index.html` | Alpine.js single-page app:<br>- Drag-drop image upload<br>- Processing progress indicator<br>- Results table with 5 columns:<br>  1. Extracted Service (editable)<br>  2. Price/Duration<br>  3. Demographic (gender/person_type)<br>  4. **Matched Service** (salonservices dropdown)<br>  5. **Matched Variation** (salonservicevariations dropdown)<br>- HIGH MATCH / REVIEW MATCH badges per column<br>- Confirm catalog button |

---

## Complete Request Flow

```
User uploads menu image
        │
        ▼
POST /api/menu/extract-and-match
        │
        ▼
MenuProcessorService.process_menu_image()
        │
        ├──► GeminiExtractorService.extract_from_image()
        │       │
        │       ▼
        │   Gemini 2.5 Flash Vision API
        │       │
        │       ▼
        │   Returns: List[ExtractedMenuItem]
        │
        ▼
For each ExtractedMenuItem:
        │
        ├──► FuzzySearchService.recommend_services(item.service)
        │       │
        │       ▼
        │   Atlas Search on salonservices
        │   (index: salonservices_index)
        │       │
        │       ▼
        │   Returns: List[VariationRecommendation] 
        │   (serviceId, variationName, similarity_percentage)
        │
        └──► FuzzySearchService.recommend_variations(item.service_variation)
                │
                ▼
            Atlas Search on salonservicevariations
            (index: salonservicevariations_index)
                │
                ▼
            Returns: List[VariationRecommendation]
            (serviceId, variationId, variationName, similarity_percentage)
        │
        ▼
Combine & sort by similarity_percentage
Apply 80% threshold
Select top 5 each
        │
        ▼
Return MenuProcessingResponse
        │
        ▼
Frontend renders table with two dropdown columns
        │
        ▼
Owner selects service + variation per row
        │
        ▼
POST /api/menu/confirm → CatalogConfirmationResponse
```

---

## Scoring & Ranking System

### Atlas Search Scoring (BM25)
MongoDB Atlas Search uses **BM25** (Best Matching 25) algorithm for text relevance scoring.

**Factors affecting score:**
- **Term Frequency (TF)** - How often query terms appear in document
- **Inverse Document Frequency (IDF)** - Rarity of terms across collection
- **Field Length Normalization** - Shorter fields score higher
- **Fuzzy Matching** - `maxEdits: 2` allows ~2 character differences

### Score Normalization (Application Level)
Raw Atlas BM25 scores are **not bounded** (can be 0.5 to 5.0+). We normalize:

```python
# In recommend_services() and recommend_variations():
max_score = max(r.get("score", 0) for r in raw_results)
normalized_score = min(raw_score / max_score, 1.0)  # 0.0 to 1.0
similarity_percentage = int(round(normalized_score * 100))  # 0-100%
```

**Why normalize?**
- Top result always gets ~100%
- Others relative to top match
- Enables consistent 80% threshold across all queries

### Confidence Threshold
| Threshold | Meaning |
|-----------|---------|
| **≥ 80%** | HIGH MATCH - Auto-selected, shown with blue badge |
| **< 80%** | REVIEW MATCH - Requires owner attention, orange badge |
| **No results** | Shows "No confident match" - owner must manually select |

### Ranking Order
1. **Atlas Search relevance score** (BM25 + fuzzy)
2. **Normalized to 0-100%** per query
3. **Filtered** at 80% minimum
4. **Deduplicated** by ID + name
5. **Sorted** descending by similarity_percentage
6. **Top 5** returned per category (service + variation)

---

## Atlas Search Indexes Required

### salonservices_index (on `salonservices` collection)
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

### salonservicevariations_index (on `salonservicevariations` collection)
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

---

## Key Design Decisions

1. **Separate service vs variation search** - Service category from `salonservices`, specific variation from `salonservicevariations`
2. **No hybrid/vector search** - Pure fuzzy text search per requirement
3. **80% confidence floor** - Prevents recommending unrelated items
4. **Deduplication** - Avoids showing same service/variation multiple times
5. **Regex fallback** - Works even without Atlas Search indexes (slower, exact word matching)
6. **Frontend shows both columns** - Owner can match service category AND specific variation independently