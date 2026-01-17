"""API route modules."""

from ltx_server.routes.generation import router as generation_router
from ltx_server.routes.health import router as health_router
from ltx_server.routes.jobs import router as jobs_router
from ltx_server.routes.pipelines import router as pipelines_router

__all__ = ["generation_router", "health_router", "jobs_router", "pipelines_router"]
