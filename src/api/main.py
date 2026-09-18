from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import app_state
from src.api.routers import fleet, health, predict, tires
from src.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.load()
    if app_state.load_errors:
        log.warning("Started with partial load errors: %s", app_state.load_errors)
    else:
        log.info("Application state loaded successfully.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="TireGuard AI API",
    description=(
        "Failure prediction, root-cause explanation, RUL estimation, and fleet "
        "monitoring for connected tires. Backed by models trained in Milestones "
        "3-8; data source is a temporary in-memory dataset until Milestone 11 "
        "adds a real database — see /health for current load status."
    ),
    version="0.10.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catches anything that isn't already an HTTPException (which
    FastAPI handles natively with a clean {"detail": ...} body) so a
    caller never sees a raw Python traceback — structured error
    responses, per this project's engineering requirements."""
    log.exception("Unhandled exception processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/", tags=["health"])
def root():
    return {"service": "TireGuard AI API", "docs": "/docs", "health": "/health"}


app.include_router(health.router)
app.include_router(tires.router)
app.include_router(predict.router)
app.include_router(fleet.router)