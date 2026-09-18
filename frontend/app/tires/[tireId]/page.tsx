import { notFound } from "next/navigation";
import { api } from "@/lib/api";
import { RiskBadge } from "@/components/RiskBadge";
import { StatCard } from "@/components/StatCard";
import { TireHistoryChart } from "@/components/TireHistoryChart";

export const dynamic = "force-dynamic";

export default async function TireDetailPage({
  params,
}: {
  params: Promise<{ tireId: string }>;
}) {
  const { tireId } = await params;

  let history;
  try {
    history = await api.tireHistory(tireId, 200);
  } catch {
    notFound();
  }

  const [rulResult, rootCauseResult] = await Promise.allSettled([
    api.tireRUL(tireId),
    api.tireRootCause(tireId),
  ]);

  const rul = rulResult.status === "fulfilled" ? rulResult.value : null;
  const rootCause =
    rootCauseResult.status === "fulfilled" ? rootCauseResult.value : null;

  const latest = history.rows[history.rows.length - 1];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-mono text-2xl font-semibold text-slate-100">
          {tireId}
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          {history.row_count} telemetry readings loaded
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <StatCard label="Pressure" value={`${latest.pressure.toFixed(1)} psi`} />
        <StatCard label="Temperature" value={`${latest.temperature.toFixed(1)}°C`} />
        <StatCard label="Tread Depth" value={`${latest.tread_depth.toFixed(1)} mm`} />
        <StatCard
          label="Est. RUL"
          value={rul ? `${rul.estimated_rul_km.toLocaleString()} km` : "N/A"}
          sublabel={rul ? undefined : "RUL model not loaded"}
        />
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
            Risk Level
          </div>
          <div className="mt-3">
            {rootCause ? (
              <RiskBadge risk={rootCause.risk_level} />
            ) : (
              <span className="text-sm text-slate-500">Unavailable</span>
            )}
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
        <h2 className="mb-4 text-sm font-semibold text-slate-200">
          Telemetry History
        </h2>
        <TireHistoryChart rows={history.rows} />
      </div>

      {rootCause && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
            <h2 className="mb-1 text-sm font-semibold text-slate-200">
              Top Model Contributions
            </h2>
            <p className="mb-4 text-xs text-slate-500">
              SHAP values — how the model weighted each feature for this
              prediction. Consistent with, not proof of, physical cause.
            </p>
            <ul className="space-y-2">
              {rootCause.top_model_contributions.map((c) => (
                <li
                  key={c.feature}
                  className="flex items-center justify-between rounded border border-slate-800 bg-slate-950 px-3 py-2 text-sm"
                >
                  <span className="font-mono text-xs text-slate-400">
                    {c.feature}
                  </span>
                  <span
                    className={
                      c.direction === "increases_risk"
                        ? "font-mono text-xs text-red-400"
                        : "font-mono text-xs text-emerald-400"
                    }
                  >
                    {c.contribution > 0 ? "+" : ""}
                    {c.contribution.toFixed(4)}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
            <h2 className="mb-1 text-sm font-semibold text-slate-200">
              Triggered Engineering Rules
            </h2>
            <p className="mb-4 text-xs text-slate-500">
              Independent, hardcoded domain thresholds — not something the
              model learned.
            </p>
            {rootCause.triggered_engineering_rules.length === 0 ? (
              <p className="text-sm text-slate-500">
                No hardcoded thresholds crossed for this reading.
              </p>
            ) : (
              <ul className="space-y-2">
                {rootCause.triggered_engineering_rules.map((c) => (
                  <li
                    key={c.feature}
                    className="flex items-center justify-between rounded border border-amber-900/40 bg-amber-950/20 px-3 py-2 text-sm"
                  >
                    <span className="font-mono text-xs text-amber-200">
                      {c.feature}
                    </span>
                    <span className="font-mono text-xs text-amber-400">
                      {String(c.value)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
            <p className="mt-4 text-xs italic text-slate-600">
              {rootCause.caveat}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}