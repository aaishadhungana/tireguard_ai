export interface TireSummary {
  tire_id: string;
  vehicle_id: string;
  pressure: number;
  temperature: number;
  tread_depth: number;
  failure_probability: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  timestamp: string;
}

export interface FleetTiresResponse {
  tire_count: number;
  tires: TireSummary[];
}

export interface FleetStatsResponse {
  total_tires: number;
  total_vehicles: number;
  total_rows: number;
  failure_rate_pct: number;
  failure_type_counts: Record<string, number>;
  data_source_note: string;
}

export interface AlertItem {
  tire_id: string;
  vehicle_id: string;
  failure_probability: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  timestamp: string;
}

export interface AlertsResponse {
  alert_count: number;
  alerts: AlertItem[];
}

export interface TireHistoryRow {
  timestamp: string;
  pressure: number;
  temperature: number;
  tread_depth: number;
  speed: number;
  load: number;
  failure: number;
}

export interface TireHistoryResponse {
  tire_id: string;
  row_count: number;
  rows: TireHistoryRow[];
}

export interface RULResponse {
  tire_id: string;
  estimated_rul_km: number;
  model_type: string;
  note: string;
}

export interface RootCauseContribution {
  feature: string;
  value: number | string;
  contribution: number;
  direction: string;
  source: "model_contribution" | "engineering_rule";
}

export interface RootCauseResponse {
  tire_id: string;
  failure_probability: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  top_model_contributions: RootCauseContribution[];
  triggered_engineering_rules: RootCauseContribution[];
  caveat: string;
}
