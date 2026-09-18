import Link from "next/link";
import { api } from "@/lib/api";
import { StatCard } from "@/components/StatCard";
import { RiskBadge } from "@/components/RiskBadge";

export const dynamic = "force-dynamic"; // always fetch fresh fleet status, never a stale build-time snapshot

export default async function FleetOverviewPage() {
  const [stats, fleetTires] = await Promise.all([
    api.fleetStats(),
    api.fleetTires(),
  ]);

  const healthy = fleetTires.tires.filter((t) => t.risk_level === "LOW").length;
  const warning = fleetTires.tires.filter((t) => t.risk_level === "MEDIUM").length;
  const critical = fleetTires.tires.filter((t) => t.risk_level === "HIGH").length;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-slate-100">Fleet Overview</h1>
        <p className="mt-1 text-sm text-slate-500">{stats.data_source_note}</p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Total Vehicles" value={stats.total_vehicles} />
        <StatCard label="Total Tires" value={stats.total_tires} />
        <StatCard label="Healthy" value={healthy} accent="emerald" />
        <StatCard label="Warning" value={warning} accent="amber" />
        <StatCard label="Critical" value={critical} accent="red" />
        <StatCard
          label="Active Anomalies"
          value="—"
          sublabel="Not yet wired to API (Milestones 7-8 exist as standalone models, not exposed as endpoints)"
        />
      </div>

      <div className="rounded-lg border border-slate-800 bg-slate-900">
        <div className="border-b border-slate-800 px-5 py-4">
          <h2 className="text-sm font-semibold text-slate-200">
            Fleet — All Tires ({fleetTires.tire_count})
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs uppercase tracking-wider text-slate-500">
                <th className="px-5 py-3 font-medium">Tire ID</th>
                <th className="px-5 py-3 font-medium">Vehicle</th>
                <th className="px-5 py-3 font-medium">Pressure</th>
                <th className="px-5 py-3 font-medium">Temp</th>
                <th className="px-5 py-3 font-medium">Tread</th>
                <th className="px-5 py-3 font-medium">Failure Prob.</th>
                <th className="px-5 py-3 font-medium">Risk</th>
              </tr>
            </thead>
            <tbody>
              {fleetTires.tires.map((tire) => (
                <tr
                  key={tire.tire_id}
                  className="border-b border-slate-800/60 last:border-0 hover:bg-slate-800/40"
                >
                  <td className="px-5 py-3 font-mono text-xs text-slate-300">
                    <Link
                      href={`/tires/${encodeURIComponent(tire.tire_id)}`}
                      className="hover:text-emerald-400 hover:underline"
                    >
                      {tire.tire_id}
                    </Link>
                  </td>
                  <td className="px-5 py-3 font-mono text-xs text-slate-400">
                    {tire.vehicle_id}
                  </td>
                  <td className="px-5 py-3 tabular-nums text-slate-300">
                    {tire.pressure.toFixed(1)} psi
                  </td>
                  <td className="px-5 py-3 tabular-nums text-slate-300">
                    {tire.temperature.toFixed(1)}°C
                  </td>
                  <td className="px-5 py-3 tabular-nums text-slate-300">
                    {tire.tread_depth.toFixed(1)} mm
                  </td>
                  <td className="px-5 py-3 tabular-nums text-slate-300">
                    {(tire.failure_probability * 100).toFixed(1)}%
                  </td>
                  <td className="px-5 py-3">
                    <RiskBadge risk={tire.risk_level} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}