import { NextRequest, NextResponse } from "next/server";
import { executeQuery } from "@/lib/integrations/glass";
import { storeEpisodicMemory } from "@/lib/integrations/sqlo";

/**
 * POST /api/query - Execute query against organism
 *
 * INTEGRATION: ✅ Connected to ROXO (executeQuery) + LARANJA (storeEpisodicMemory)
 */
export async function POST(request: NextRequest) {
  try {
    const { organismId, query, userId } = await request.json();

    if (!organismId || !query) {
      return NextResponse.json(
        { error: "Missing organismId or query" },
        { status: 400 }
      );
    }

    // Execute query using ROXO GlassRuntime
    const result = await executeQuery(organismId, query);

    // Store in episodic memory using LARANJA
    if (userId) {
      try {
        await storeEpisodicMemory({
          organism_id: organismId,
          query,
          result,
          user_id: userId,
          timestamp: new Date().toISOString(),
          session_id: `session_${Date.now()}`,
        });
      } catch (error) {
        console.error("Failed to store episodic memory:", error);
        // Continue even if storage fails
      }
    }

    return NextResponse.json(result);
  } catch (error) {
    console.error("Failed to execute query:", error);
    return NextResponse.json(
      { error: "Failed to execute query", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
