import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Moon, Sun } from "lucide-react";
import { USE_FIXTURES } from "../lib/api";
import { Button } from "./ui";

export function AppHeader({ children }: { children?: React.ReactNode }) {
  return (
    <header className="flex h-10 shrink-0 items-center gap-3 border-b border-border bg-surface px-3">
      <Link to="/" className="text-[13px] font-semibold tracking-tight">
        Dubbing Studio
      </Link>
      {USE_FIXTURES ? (
        <span
          className="rounded border border-border px-1.5 py-px text-[10px] font-semibold uppercase tracking-[0.12em] text-muted"
          title="VITE_USE_FIXTURES=1 — sample data, no server, no models"
        >
          fixtures
        </span>
      ) : null}
      {children}
      <Link
        to="/setup"
        className="ml-auto text-[12px] text-muted transition-colors hover:text-primary"
        title="What this machine has installed"
      >
        Setup
      </Link>
      <ThemeToggle />
    </header>
  );
}

/**
 * Stamps data-theme on <html>. The CSS declares the dark steps under that
 * scope as well as prefers-color-scheme, so the toggle wins in both directions.
 */
function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <Button
      variant="ghost"
      className="ml-auto px-1.5"
      aria-label={theme === "dark" ? "Switch to light" : "Switch to dark"}
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}
