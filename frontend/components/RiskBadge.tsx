import { riskColor } from "@/lib/api";

export function RiskBadge({ risk }: { risk: "LOW" | "MEDIUM" | "HIGH" }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold tracking-wide ${riskColor(
        risk
      )}`}
    >
      {risk}
    </span>
  );
}