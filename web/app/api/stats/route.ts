import { NextResponse } from "next/server";
import { SystemStats } from "@/lib/types";
import { getAllOrganisms } from "@/lib/integrations/sqlo";

/**
 * GET /api/stats - Get system statistics
 *
 * INTEGRATION: ✅ Connected to LARANJA (getAllOrganisms)
 */
export async function GET() {
  try {
    // Get all organisms from LARANJA
    const organisms = await getAllOrganisms();

    let totalCost = 0;
    let totalQueries = 0;

    for (const organism of organisms) {
      totalCost += organism.stats?.total_cost || 0;
      totalQueries += organism.stats?.queries_count || 0;
    }

    const stats: SystemStats = {
      total_organisms: organisms.length,
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
      { error: "Failed to get stats", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
