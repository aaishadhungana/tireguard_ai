"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TireHistoryRow } from "@/lib/types";

export function TireHistoryChart({ rows }: { rows: TireHistoryRow[] }) {
  const data = rows.map((r) => ({
    ...r,
    label: new Date(r.timestamp).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }),
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="label"
          stroke="#64748b"
          fontSize={11}
          minTickGap={40}
        />
        <YAxis yAxisId="left" stroke="#64748b" fontSize={11} />
        <YAxis yAxisId="right" orientation="right" stroke="#64748b" fontSize={11} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0f172a",
            border: "1px solid #1e293b",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="pressure"
          name="Pressure (psi)"
          stroke="#34d399"
          dot={false}
          strokeWidth={2}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="temperature"
          name="Temperature (°C)"
          stroke="#f59e0b"
          dot={false}
          strokeWidth={2}
        />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="tread_depth"
          name="Tread Depth (mm)"
          stroke="#60a5fa"
          dot={false}
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}