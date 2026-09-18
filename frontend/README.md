# TireGuard AI — Fleet Dashboard (Milestone 12)

Next.js 16 + TypeScript + Tailwind CSS + Recharts, consuming the
FastAPI backend 

## Setup

```bash
cd frontend
npm install
```

Create `.env.local` (already gitignored) if it doesn't exist:
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Running

The backend must be running first (from the project root):
```bash
uvicorn src.api.main:app --reload
```

Then, in `frontend/`:
```bash
npm run dev
```

Visit http://localhost:3000.

## Pages

- **`/`** — Fleet Overview: vehicle/tire counts, healthy/warning/critical
  breakdown, full tire status table
- **`/tires/[tireId]`** — Tire Detail: current readings, RUL estimate,
  telemetry history chart, SHAP model contributions, triggered
  engineering rules
- **`/intelligence`** — Fleet Intelligence: highest-risk tires ranked,
  failure-type distribution chart

## Design Decisions

- **Server Components fetch directly from the API** — no client-side
  loading spinners needed for the initial page render, since Next.js
  renders the page on the server after the data is already in hand.
  Charts (Recharts) are the one exception, marked `"use client"` since
  they need the browser to render SVG interactively.
- **Dark, data-dense theme** — an operations dashboard, not a generic
  bright admin template. Colors are plain Tailwind utility classes
  (one fixed theme, no light/dark toggle), not CSS custom properties.
- **`force-dynamic` on every page** — fleet status changes as new
  telemetry/predictions arrive; a cached, stale dashboard would be
  actively misleading for this use case.
- **No client-side auto-refresh polling** — deliberately deferred.
  Milestone 15 (production hardening) is the more appropriate place to
  add WebSocket/polling-based live updates, once auth exists to gate
  who can see live fleet data.

## Known Limitations

- "Active Anomalies" on the Fleet Overview page shows a placeholder —
  Milestones 7-8 (anomaly detection, sensor integrity) exist as
  standalone trained models but aren't yet exposed as API endpoints.
  Wiring them in is a natural next step, not done here to keep this
  milestone scoped to the dashboard itself.
- No per-tire alert acknowledgment UI yet (the `Alert.acknowledged`
  database field exists from Milestone 11 but has no API endpoint or
  UI to set it).
- No authentication — anyone who can reach the dashboard sees all fleet
  data (Milestone 15 scope).