import { NextRequest, NextResponse } from "next/server";
import { getOrganism, deleteOrganism } from "@/lib/integrations/sqlo";

/**
 * GET /api/organisms/:id - Get organism details
 *
 * INTEGRATION: ✅ Connected to LARANJA (getOrganism)
 */
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const organism = await getOrganism(params.id);
    return NextResponse.json(organism);
  } catch (error) {
    console.error("Failed to get organism:", error);

    // Check if organism not found
    if (error instanceof Error && error.message.includes("not found")) {
      return NextResponse.json(
        { error: "Organism not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(
      { error: "Failed to get organism", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/organisms/:id - Delete organism
 *
 * INTEGRATION: ✅ Connected to LARANJA (deleteOrganism)
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    await deleteOrganism(params.id);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete organism:", error);
    return NextResponse.json(
      { error: "Failed to delete organism", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
