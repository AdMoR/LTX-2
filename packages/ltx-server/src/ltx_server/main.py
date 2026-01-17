"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ltx_server import __version__
from ltx_server.config import get_settings
from ltx_server.routes import generation_router, health_router, jobs_router, pipelines_router
from ltx_server.services.job_manager import JobManager
from ltx_server.services.media import MediaService
from ltx_server.services.pipeline_registry import PipelineRegistry
from ltx_server.services.storage import StorageService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    settings = get_settings()
    logger.info(f"Starting LTX Server v{__version__}")

    # Initialize services
    logger.info("Initializing storage service...")
    storage = StorageService(settings.s3)
    storage.ensure_bucket_exists()

    logger.info("Initializing media service...")
    media_service = MediaService(settings.temp_dir)

    logger.info("Initializing pipeline registry...")
    pipeline_registry = PipelineRegistry(settings.pipeline)
    pipeline_registry.load_pipelines()

    logger.info("Initializing job manager...")
    job_manager = JobManager(pipeline_registry, storage)
    await job_manager.start()

    # Attach services to app state
    app.state.settings = settings
    app.state.storage = storage
    app.state.media_service = media_service
    app.state.pipeline_registry = pipeline_registry
    app.state.job_manager = job_manager

    logger.info(f"LTX Server ready. Loaded pipelines: {pipeline_registry.list_loaded()}")

    yield

    # Shutdown
    logger.info("Shutting down LTX Server...")
    await job_manager.stop()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LTX-2 Video Generation API",
        description="API server for LTX-2 video generation pipelines",
        version=__version__,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(pipelines_router)
    app.include_router(jobs_router)
    app.include_router(generation_router)

    return app


# Create app instance
app = create_app()


def run() -> None:
    """Run the server (entry point for console script)."""
    settings = get_settings()
    uvicorn.run(
        "ltx_server.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
