"""FastAPI application factory and static file serving."""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from service_testing.api.routes import get_menu_processor, router as api_router

STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _warm_up() -> None:
    try:
        get_menu_processor().fuzzy_search.warm_up()
    except Exception:
        logger.exception("Startup warm-up failed; connections will be opened on first request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the MongoDB pool ahead of the first upload without delaying startup.
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Salon Menu AI Onboarding & Hybrid RAG",
        description="POC for automating salon price menu onboarding via Gemini Flash-Lite & MongoDB Atlas Hybrid Search",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router)

    # Ensure static directory exists
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {
            "message": "Salon Onboarding API is running. Web UI not found in static/."
        }

    return app


app = create_app()
