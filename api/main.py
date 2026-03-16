"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.routes import router
from config.settings import get_settings
from monitoring.logger import setup_logging, get_logger

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info("starting_researchgpt", host=settings.host, port=settings.port)

    from api.dependencies import get_app_state
    get_app_state()  # eager initialization
    logger.info("all_services_initialized")

    yield

    logger.info("shutting_down_researchgpt")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ResearchGPT",
        description="AI Research Paper Assistant — ask questions across thousands of papers",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        _html_content = index_html.read_text(encoding="utf-8")

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def serve_frontend():
            return HTMLResponse(content=_html_content)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
