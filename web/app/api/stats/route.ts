import { NextResponse } from "next/server";
import { SystemStats } from "@/lib/types";
import * as fs from "fs/promises";
import * as path from "path";

const ORGANISMS_DIR = path.join(process.cwd(), "..", "organisms");

export async function GET() {
  try {
    // Get all organisms
    const files = await fs.readdir(ORGANISMS_DIR).catch(() => []);
    const glassFiles = files.filter((f) => f.endsWith(".glass"));

    let totalCost = 0;
    let totalQueries = 0;

    for (const file of glassFiles) {
      try {
        const content = await fs.readFile(
          path.join(ORGANISMS_DIR, file),
          "utf-8"
        );
        const organism = JSON.parse(content);
        totalCost += organism.stats?.total_cost || 0;
        totalQueries += organism.stats?.queries_count || 0;
      } catch (error) {
        console.error(`Failed to parse ${file}:`, error);
      }
    }

    const stats: SystemStats = {
      total_organisms: glassFiles.length,
      total_queries: totalQueries,
      total_cost: totalCost,
      budget_limit: 100, // $100 default
      health: totalCost > 90 ? "critical" : totalCost > 70 ? "warning" : "healthy",
      uptime: process.uptime() * 1000, // ms
    };

    return NextResponse.json(stats);
  } catch (error) {
    console.error("Failed to get stats:", error);
    return NextResponse.json(
      { error: "Failed to get stats" },
      { status: 500 }
    );
  }
}
