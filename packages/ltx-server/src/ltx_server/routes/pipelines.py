"""Pipeline listing endpoint."""

from fastapi import APIRouter, Request

from ltx_server.models.responses import PipelinesResponse

router = APIRouter(tags=["pipelines"])


@router.get("/pipelines", response_model=PipelinesResponse)
async def list_pipelines(request: Request) -> PipelinesResponse:
    """List all available pipelines."""
    pipeline_registry = request.app.state.pipeline_registry
    return PipelinesResponse(pipelines=pipeline_registry.list_available())
