import { NextRequest, NextResponse } from "next/server";
import { GlassOrganism } from "@/lib/types";
import { getAllOrganisms, storeOrganism } from "@/lib/integrations/sqlo";

/**
 * GET /api/organisms - List all organisms
 *
 * INTEGRATION: ✅ Connected to LARANJA (getAllOrganisms)
 */
export async function GET() {
  try {
    const organisms = await getAllOrganisms();
    return NextResponse.json(organisms);
  } catch (error) {
    console.error("Failed to list organisms:", error);
    return NextResponse.json(
      { error: "Failed to list organisms", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/organisms - Upload .glass file
 *
 * INTEGRATION: ✅ Connected to LARANJA (storeOrganism)
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json(
        { error: "No file provided" },
        { status: 400 }
      );
    }

    if (!file.name.endsWith(".glass")) {
      return NextResponse.json(
        { error: "File must be .glass format" },
        { status: 400 }
      );
    }

    const buffer = await file.arrayBuffer();
    const content = Buffer.from(buffer).toString("utf-8");

    // Parse and validate
    let organism: GlassOrganism;
    try {
      organism = JSON.parse(content);
    } catch (error) {
      return NextResponse.json(
        { error: "Invalid .glass format" },
        { status: 400 }
      );
    }

    // Validate required fields
    if (!organism.metadata?.name) {
      return NextResponse.json(
        { error: "Missing organism.metadata.name" },
        { status: 400 }
      );
    }

    // Generate ID if not present
    if (!organism.id) {
      organism.id = organism.metadata.name.toLowerCase().replace(/\s+/g, "-");
    }

    // Store using LARANJA
    await storeOrganism(organism);

    return NextResponse.json(organism);
  } catch (error) {
    console.error("Failed to upload organism:", error);
    return NextResponse.json(
      { error: "Failed to upload organism", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
