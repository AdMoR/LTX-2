"""Job management endpoints."""

from fastapi import APIRouter, HTTPException, Query, Request

from ltx_server.models.responses import JobListResponse, JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=JobListResponse)
async def list_jobs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> JobListResponse:
    """List recent generation jobs."""
    job_manager = request.app.state.job_manager
    storage = request.app.state.storage

    jobs, total = job_manager.list_jobs(limit=limit, offset=offset)

    return JobListResponse(
        jobs=[job.to_response(storage) for job in jobs],
        total=total,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(request: Request, job_id: str) -> JobResponse:
    """Get status and results for a specific job."""
    job_manager = request.app.state.job_manager
    storage = request.app.state.storage

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return job.to_response(storage)


@router.delete("/{job_id}", response_model=JobResponse)
async def cancel_job(request: Request, job_id: str) -> JobResponse:
    """Cancel a pending job."""
    job_manager = request.app.state.job_manager
    storage = request.app.state.storage

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job_manager.cancel_job(job_id):
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be cancelled (status: {job.status.value})",
        )

    return job.to_response(storage)
