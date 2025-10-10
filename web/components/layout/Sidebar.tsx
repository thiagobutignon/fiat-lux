"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  HomeIcon,
  FlaskConicalIcon,
  BugIcon,
  SettingsIcon,
  ActivityIcon,
  NetworkIcon
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/", icon: HomeIcon },
  { name: "System Status", href: "/status", icon: NetworkIcon },
  { name: "Organisms", href: "/organisms", icon: FlaskConicalIcon },
  { name: "Debug Tools", href: "/debug", icon: BugIcon },
  { name: "Activity", href: "/activity", icon: ActivityIcon },
  { name: "Settings", href: "/settings", icon: SettingsIcon },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
      <div className="flex-1 flex flex-col min-h-0 bg-card border-r">
        <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
          <div className="flex items-center flex-shrink-0 px-4">
            <h1 className="text-2xl font-bold">
              🟡 Chomsky DevTools
            </h1>
          </div>
          <nav className="mt-8 flex-1 px-2 space-y-1">
            {navigation.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  <item.icon
                    className={cn(
                      "mr-3 flex-shrink-0 h-5 w-5",
                      isActive ? "text-primary-foreground" : "text-muted-foreground"
                    )}
                    aria-hidden="true"
                  />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>
        <div className="flex-shrink-0 flex border-t p-4">
          <div className="flex items-center">
            <div>
              <p className="text-xs font-medium text-muted-foreground">
                Version 1.0.0
              </p>
              <p className="text-xs text-muted-foreground">
                Nó AMARELO (DevTools)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
