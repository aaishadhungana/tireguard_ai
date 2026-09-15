from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.dependencies import AppState, get_app_state, get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(state: AppState = Depends(get_app_state), session: Session = Depends(get_db)) -> dict:
    db_ok = False
    db_error = None
    try:
        session.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_error = str(e)

    components = {
        "database": db_ok,
        "failure_model": state.failure_predictor is not None,
        "rul_model": state.rul_model is not None,
    }
    all_ok = all(components.values())

    load_errors = dict(state.load_errors)
    if db_error:
        load_errors["database"] = db_error

    return {
        "status": "ok" if all_ok else "degraded",
        "components": components,
        "load_errors": load_errors,
    }