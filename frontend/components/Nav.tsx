import Link from "next/link";

export function Nav() {
  return (
    <header className="border-b border-slate-800 bg-slate-950">
      <div className="mx-auto flex max-w-7xl items-center gap-8 px-6 py-4">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-lg font-bold tracking-tight text-slate-100">
            TireGuard<span className="text-emerald-400">AI</span>
          </span>
        </Link>
        <nav className="flex gap-6 text-sm font-medium text-slate-400">
          <Link href="/" className="hover:text-slate-100">
            Fleet Overview
          </Link>
          <Link href="/intelligence" className="hover:text-slate-100">
            Fleet Intelligence
          </Link>
        </nav>
      </div>
    </header>
  );
}