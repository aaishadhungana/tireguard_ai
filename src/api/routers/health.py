from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import AppState, get_app_state

router = APIRouter(tags=["health"])


@router.get("/health")
def health(state: AppState = Depends(get_app_state)) -> dict:
    components = {
        "dataset": state.dataset is not None,
        "failure_model": state.failure_predictor is not None,
        "rul_model": state.rul_model is not None,
    }
    all_ok = all(components.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "components": components,
        "load_errors": state.load_errors,
    }