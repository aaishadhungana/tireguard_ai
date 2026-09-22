from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import AppState, get_app_state, get_db
from src.api.schemas import CopilotAskRequest, CopilotAskResponse
from src.copilot.agent import CopilotError, ask_copilot

router = APIRouter(prefix="/copilot", tags=["copilot"])


@router.post("/ask", response_model=CopilotAskResponse)
def copilot_ask(
    request: CopilotAskRequest,
    state: AppState = Depends(get_app_state),
    session: Session = Depends(get_db),
):
    """Natural-language fleet questions, answered ONLY via tool calls
    against real data (src/copilot/tools.py) -- see src/copilot/agent.py
    for the constraint that the LLM never invents a fact."""
    try:
        return ask_copilot(request.question, session, state)
    except CopilotError as e:
        raise HTTPException(status_code=503, detail=str(e))