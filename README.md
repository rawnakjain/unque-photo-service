# Salon Menu AI Onboarding & Hybrid RAG Variation Matcher POC

A proof-of-concept for automating salon price menu onboarding using:
- **Google Gemini 3.5 Flash** for multimodal menu image extraction
- **MongoDB Atlas Hybrid Search** ($rankFusion combining vector + BM25 search)
- **FastAPI** backend with **Vercel Design System** styled web UI

## Features

1. **Automated Menu Extraction**: Upload a salon price menu/image → Gemini 3.5 Flash extracts:
   - Service category (matched to 100+ predefined services)
   - Service variation name
   - Price & duration
   - Gender (M/F/U) and person type (Adult/Child)

2. **Hybrid RAG Variation Matching**: For each extracted variation, performs MongoDB Atlas Hybrid Search:
   - Vector search using BAAI/bge-m3 embeddings (1024-dim)
   - Full-text BM25 search on variation names
   - Returns top-5 database matches with similarity scores

3. **Interactive Web UI**:
   - Drag-and-drop image upload with instant preview
   - Real-time processing status (Vision → Embedding → Hybrid Search)
   - Editable extraction fields and dropdown selector for top-5 matches
   - Visual match quality indicators (High Match/Review Match)
   - Salon owner confirmation flow with JSON/CSV export

## Architecture

```
┌─────────────────┐    ┌────────────────────────────┐    ┌──────────────────────┐
│   Menu Image    │───▶│ Gemini 3.5 Flash Extraction│───▶│ Extracted Items      │
│ (Upload via UI) │    │ (Structured JSON Output)   │    │ + Variation Names    │
└─────────────────┘    └────────────────────────────┘    └──────────────────────┘
                                                                     │
                                                                     ▼
                                                              ┌─────────────────────┐
                                                              │ MongoDB Atlas       │
                                                              │ Hybrid Search       │
                                                              │ ($rankFusion)       │
                                                              │ - Vector Search     │
                                                              │ - BM25 Full Text    │
                                                              └─────────────────────┘
                                                                     │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │ Top-5 Recommendations per   │
                                                      │ Extracted Variation         │
                                                      │ (with similarity scores)  │
                                                      └─────────────────────────────┘
                                                                     │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │ Salon Owner Review &        │
                                                      │ Selection (Web UI)          │
                                                      │ - Edit extraction fields  │
                                                      │ - Choose from top-5 matches│
                                                      │ - Confirm catalog         │
                                                      └─────────────────────────────┘
                                                                     │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │ Confirmed Catalog           │
                                                      │ (Ready for Salon Onboarding)│
                                                      └─────────────────────────────┘
```

## Setup

1. **Environment Variables** (copy `.env.example` to `.env` and fill):
   ```bash
   cp .env.example .env
   # Edit .env to add your GEMINI_API_KEY
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

- `GET /api/health` - System health check
- `GET /api/services` - List of 100+ standard salon services
- `POST /api/variations/search` - Direct hybrid search (query: `{ "query": "Bob haircut", "top_k": 5 }`)
- `POST /api/menu/extract-and-match` - Upload menu image → extraction + matching
- `POST /api/menu/confirm` - Confirm salon owner selections

## Technology Stack

- **Backend**: FastAPI, Python 3.13
- **ML/AI**: 
  - Google Gemini 3.5 Flash (multimodal extraction)
  - BAAI/bge-m3 (Sentence Transformers) for embeddings
- **Database**: MongoDB Atlas with Hybrid Search ($rankFusion)
- **Frontend**: 
  - Vercel Design System (Geist fonts, monochrome aesthetic)
  - Tailwind CSS + Alpine.js (no build step required)
- **Dependencies**: 
  - `fastapi`, `uvicorn`, `google-genai`, `pymongo`, `sentence-transformers`, `pillow`

## Implementation Notes

- **Embedding Service**: Singleton pattern ensures BAAI/bge-m3 model loads once per server process
- **MongoDB Connection**: Reused MongoClient instance via service singleton
- **Gemini Client**: Configured with `HttpOptions(api_version="v1")` as per latest GenAI SDK
- **Structured Output**: Uses Gemini's `response_mime_type="application/json"` and `response_schema` for reliable parsing
- **Hybrid Search**: Implements exact pipeline from `Mongo-Integration.ipynb` with fallback to vector-only search
- **Web UI**: 
  - Vercel-inspired design (Geist Sans/Mono, hairline borders, monochrome palette)
  - Real-time processing steps with spinner
  - Interactive table with editable fields and match selectors
  - Visual match quality indicators (percentage-based similarity)

## Verification

Run the test suite:
```bash
python test_app.py          # Basic API health and services
python test_mongo_search.py # MongoDB hybrid search (requires Atlas connectivity)
python test_gemini_config.py# Gemini configuration and client initialization
```

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