import type {
  AlertsResponse,
  FleetStatsResponse,
  FleetTiresResponse,
  RootCauseResponse,
  RULResponse,
  TireHistoryResponse,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, revalidateSeconds = 0): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    cache: revalidateSeconds > 0 ? "force-cache" : "no-store",
    next: revalidateSeconds > 0 ? { revalidate: revalidateSeconds } : undefined,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `API request to ${path} failed: ${res.status} ${res.statusText} ${body}`
    );
  }

  return res.json() as Promise<T>;
}

export const api = {
  fleetStats: () => apiFetch<FleetStatsResponse>("/fleet/stats"),
  fleetTires: () => apiFetch<FleetTiresResponse>("/fleet/tires"),
  alerts: (riskThreshold: "LOW" | "MEDIUM" | "HIGH" = "MEDIUM") =>
    apiFetch<AlertsResponse>(`/fleet/alerts?risk_threshold=${riskThreshold}`),
  tireHistory: (tireId: string, limit = 100) =>
    apiFetch<TireHistoryResponse>(
      `/tires/${encodeURIComponent(tireId)}/history?limit=${limit}`
    ),
  tireRUL: (tireId: string) =>
    apiFetch<RULResponse>(`/tires/${encodeURIComponent(tireId)}/rul`),
  tireRootCause: (tireId: string) =>
    apiFetch<RootCauseResponse>(
      `/tires/${encodeURIComponent(tireId)}/root-cause`
    ),
};

export function riskColor(risk: "LOW" | "MEDIUM" | "HIGH"): string {
  switch (risk) {
    case "HIGH":
      return "text-red-600 bg-red-50 border-red-200";
    case "MEDIUM":
      return "text-amber-600 bg-amber-50 border-amber-200";
    default:
      return "text-emerald-600 bg-emerald-50 border-emerald-200";
  }
}
