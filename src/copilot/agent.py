from __future__ import annotations

from google import genai
from google.genai import types

from src.config.settings import settings
from src.copilot.tools import TOOL_DEFINITIONS, execute_tool
from src.utils.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """You are TireGuard AI's fleet monitoring assistant.

You answer questions about tire fleet health using ONLY the tools provided.
You must call a tool to get any factual information -- tire status, risk
levels, failure probabilities, RUL estimates, root-cause explanations, or
fleet statistics. NEVER state a specific number, risk level, or fact about
a tire or the fleet unless it came from a tool result in this conversation.

If a tool returns an error (e.g. a tire ID doesn't exist), report that
plainly -- do not guess or make up a plausible-sounding answer instead.

When explaining WHY a tire is at risk, clearly distinguish model-derived
findings (SHAP contributions, tagged "model_contribution" in tool results)
from engineering-rule findings (tagged "engineering_rule") -- these are
different kinds of evidence and should not be blended into one claim.

Be concise. A fleet operator wants a direct answer, not a report."""

MAX_TOOL_ITERATIONS = 5


class CopilotError(Exception):
    pass


def _get_client():
    if not settings.llm_api_key:
        raise CopilotError(
            "LLM_API_KEY is not set. The AI Copilot requires a real Gemini API "
            "key (set LLM_API_KEY in your .env -- get a free one at "
            "https://aistudio.google.com/apikey) -- there is no functional "
            "fallback, since fabricating answers without one would violate "
            "this project's no-invented-data rule."
        )
    return genai.Client(api_key=settings.llm_api_key)


def _build_gemini_tools():
    declarations = [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
        for tool in TOOL_DEFINITIONS
    ]
    return [types.Tool(function_declarations=declarations)]


def ask_copilot(question, session, state):
    client = _get_client()
    gemini_tools = _build_gemini_tools()

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    tool_calls_made = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=gemini_tools,
            ),
        )

        candidate = response.candidates[0]
        function_calls = [
            part.function_call for part in candidate.content.parts if part.function_call is not None
        ]

        if not function_calls:
            final_text = "".join(
                part.text for part in candidate.content.parts if part.text
            )
            return {
                "answer": final_text,
                "tool_calls": tool_calls_made,
                "iterations": iteration + 1,
            }

        contents.append(candidate.content)

        response_parts = []
        for call in function_calls:
            tool_input = dict(call.args) if call.args else {}
            log.info("Copilot calling tool: %s(%s)", call.name, tool_input)
            result = execute_tool(call.name, tool_input, session, state)
            tool_calls_made.append({"tool": call.name, "input": tool_input, "result": result})

            response_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

        contents.append(types.Content(role="user", parts=response_parts))

    raise CopilotError(
        f"Exceeded {MAX_TOOL_ITERATIONS} tool-call iterations without a final answer -- "
        "stopping rather than looping indefinitely."
    )