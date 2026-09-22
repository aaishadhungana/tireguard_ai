# TireGuard AI : AI Copilot 

## Architecture

```text
User question
      |
      v
Gemini (decides which tool(s) to call)
      |
      v
src/copilot/tools.py  --calls-->  src/api/services.py (SAME functions
                                    the REST API itself uses)
      |
      v
Real database query / real trained model inference
      |
      v
Structured result returned to Gemini
      |
      v
Gemini phrases the final answer in natural language
```


## Setup

1. Get a free API key at https://aistudio.google.com/apikey 
2. Add to your `.env`:
   ```
   LLM_API_KEY=AIza...
   LLM_MODEL=gemini-2.5-flash
   ```

There is deliberately no functional fallback without a key -- the
alternative (answering without real tool calls) would violate the
project's no-invented-data rule. `POST /copilot/ask` returns a clear
503 error if `LLM_API_KEY` isn't set, not a fabricated answer. Verified
through the real HTTP stack (not just a unit test): a live server with
no key configured returns exactly this 503 with this message when
queried with `curl`.

## Usage

```bash
curl -X POST http://localhost:8000/copilot/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which tires need attention right now?"}'
```

Example questions this is designed to handle:
- "Why is tire V-0001-T0 at high risk?" -> calls `get_root_cause`
- "Which tires need attention?" -> calls `get_fleet_alerts`
- "What's the overall fleet failure rate?" -> calls `get_fleet_stats`
- "What's tire V-0002-T1's recent pressure trend?" -> calls `get_tire_history`

## Known Limitations

- No conversation memory across requests -- each `/copilot/ask` call is
  independent. Multi-turn follow-up questions would need session
  support, deferred to keep this milestone scoped to the core
  tool-calling architecture.
- `MAX_TOOL_ITERATIONS = 5` is a safety bound against a runaway loop,
  not tuned against real usage patterns.
- No streaming -- the full answer is returned in one response.
- Gemini's free tier has rate limits (~1,500 requests/day as of this
  writing) -- fine for development and demo use, not production scale.