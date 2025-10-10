"use client";

import { MoonIcon, SunIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useEffect } from "react";

export function Header() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    // Check system preference
    const isDarkMode = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setIsDark(isDarkMode);
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    }
  }, []);

  const toggleDarkMode = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle("dark");
  };

  return (
    <header className="flex-shrink-0 border-b bg-card">
      <div className="flex items-center justify-between h-16 px-6">
        <div className="flex items-center">
          <h2 className="text-lg font-semibold">
            Glass Organisms Dashboard
          </h2>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-2 w-2 rounded-full bg-green-500" />
            <span>System Healthy</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleDarkMode}
            aria-label="Toggle dark mode"
          >
            {isDark ? (
              <SunIcon className="h-5 w-5" />
            ) : (
              <MoonIcon className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>
    </header>
  );
}
