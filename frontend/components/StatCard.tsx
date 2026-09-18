export function StatCard({
  label,
  value,
  sublabel,
  accent,
}: {
  label: string;
  value: string | number;
  sublabel?: string;
  accent?: "default" | "red" | "amber" | "emerald";
}) {
  const accentClass = {
    default: "text-slate-100",
    red: "text-red-400",
    amber: "text-amber-400",
    emerald: "text-emerald-400",
  }[accent ?? "default"];

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="text-xs font-medium uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className={`mt-2 text-3xl font-semibold tabular-nums ${accentClass}`}>
        {value}
      </div>
      {sublabel && (
        <div className="mt-1 text-xs text-slate-500">{sublabel}</div>
      )}
    </div>
  );
}