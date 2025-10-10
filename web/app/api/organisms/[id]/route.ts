import { NextRequest, NextResponse } from "next/server";
import { GlassOrganism } from "@/lib/types";
import * as fs from "fs/promises";
import * as path from "path";

const ORGANISMS_DIR = path.join(process.cwd(), "..", "organisms");

// GET /api/organisms/:id - Get organism details
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const filePath = path.join(ORGANISMS_DIR, `${params.id}.glass`);

    try {
      const content = await fs.readFile(filePath, "utf-8");
      const organism = JSON.parse(content) as GlassOrganism;
      return NextResponse.json(organism);
    } catch (error) {
      return NextResponse.json(
        { error: "Organism not found" },
        { status: 404 }
      );
    }
  } catch (error) {
    console.error("Failed to get organism:", error);
    return NextResponse.json(
      { error: "Failed to get organism" },
      { status: 500 }
    );
  }
}

// DELETE /api/organisms/:id - Delete organism
export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const filePath = path.join(ORGANISMS_DIR, `${params.id}.glass`);

    try {
      await fs.unlink(filePath);
      return NextResponse.json({ success: true });
    } catch (error) {
      return NextResponse.json(
        { error: "Organism not found" },
        { status: 404 }
      );
    }
  } catch (error) {
    console.error("Failed to delete organism:", error);
    return NextResponse.json(
      { error: "Failed to delete organism" },
      { status: 500 }
    );
  }
}
