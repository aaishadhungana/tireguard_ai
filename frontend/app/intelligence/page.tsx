import Link from "next/link";
import { api } from "@/lib/api";
import { RiskBadge } from "@/components/RiskBadge";
import { FailureTypeChart } from "@/components/FailureTypeChart";

export const dynamic = "force-dynamic";

export default async function FleetIntelligencePage() {
  const [stats, fleetTires] = await Promise.all([
    api.fleetStats(),
    api.fleetTires(),
  ]);

  const highestRisk = fleetTires.tires.slice(0, 10);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-slate-100">
          Fleet Intelligence
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Highest-risk tires and fleet-wide failure pattern analysis.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-200">
            Highest-Risk Tires
          </h2>
          <ul className="space-y-2">
            {highestRisk.map((tire, i) => (
              <li key={tire.tire_id}>
                <Link
                  href={`/tires/${encodeURIComponent(tire.tire_id)}`}
                  className="flex items-center justify-between rounded border border-slate-800 bg-slate-950 px-3 py-2.5 text-sm hover:bg-slate-800/60"
                >
                  <span className="flex items-center gap-3">
                    <span className="w-5 text-right font-mono text-xs text-slate-600">
                      {i + 1}
                    </span>
                    <span className="font-mono text-xs text-slate-300">
                      {tire.tire_id}
                    </span>
                  </span>
                  <span className="flex items-center gap-3">
                    <span className="font-mono text-xs text-slate-400">
                      {(tire.failure_probability * 100).toFixed(1)}%
                    </span>
                    <RiskBadge risk={tire.risk_level} />
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="mb-1 text-sm font-semibold text-slate-200">
            Failure Type Distribution
          </h2>
          <p className="mb-4 text-xs text-slate-500">
            Across all recorded failures in the loaded dataset.
          </p>
          {Object.keys(stats.failure_type_counts).length > 0 ? (
            <FailureTypeChart counts={stats.failure_type_counts} />
          ) : (
            <p className="text-sm text-slate-500">
              No failures recorded in the loaded dataset.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}