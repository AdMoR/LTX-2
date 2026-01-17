"""Health check endpoint."""

from fastapi import APIRouter, Request

from ltx_server import __version__
from ltx_server.models.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Check server health and status."""
    pipeline_registry = request.app.state.pipeline_registry

    return HealthResponse(
        status="healthy",
        version=__version__,
        pipelines_loaded=pipeline_registry.list_loaded(),
        gpu_available=pipeline_registry.gpu_available,
    )
