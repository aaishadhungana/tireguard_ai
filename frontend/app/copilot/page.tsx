"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { CopilotAskResponse } from "@/lib/types";

const EXAMPLE_QUESTIONS = [
  "Which tires need attention right now?",
  "What's the overall fleet failure rate?",
];

export default function CopilotPage() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CopilotAskResponse | null>(null);

  async function ask(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.copilotAsk(q);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-slate-100">AI Copilot</h1>
        <p className="mt-1 text-sm text-slate-500">
          Ask a question about the fleet. Every answer is backed by real
          tool calls against the live database and models — the AI never
          invents a number.
        </p>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
        className="flex gap-2"
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. Why is tire V-0001-T0 at high risk?"
          className="flex-1 rounded-lg border border-slate-800 bg-slate-900 px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:border-emerald-600 focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
        >
          {loading ? "Asking..." : "Ask"}
        </button>
      </form>

      <div className="flex flex-wrap gap-2">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => {
              setQuestion(q);
              ask(q);
            }}
            className="rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-400 hover:border-slate-700 hover:text-slate-200"
          >
            {q}
          </button>
        ))}
      </div>

      {error && (
        <div className="rounded-lg border border-red-900/40 bg-red-950/20 p-4 text-sm text-red-300">
          <p className="font-medium">Copilot unavailable</p>
          <p className="mt-1 text-red-400/80">{error}</p>
          <p className="mt-2 text-xs text-red-400/60">
            This usually means LLM_API_KEY isn&apos;t set on the backend —
            see docs/copilot_architecture.md.
          </p>
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">
              {result.answer}
            </p>
          </div>

          {result.tool_calls.length > 0 && (
            <div className="rounded-lg border border-slate-800 bg-slate-950 p-4">
              <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-600">
                Tool calls used ({result.tool_calls.length})
              </p>
              <ul className="space-y-1">
                {result.tool_calls.map((call, i) => (
                  <li
                    key={i}
                    className="font-mono text-xs text-slate-500"
                  >
                    {call.tool}({JSON.stringify(call.input)})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}