import { NextRequest, NextResponse } from "next/server";
import { GlassOrganism } from "@/lib/types";
import * as fs from "fs/promises";
import * as path from "path";

// Temporary in-memory storage (replace with .sqlo in Sprint 2)
const ORGANISMS_DIR = path.join(process.cwd(), "..", "organisms");

// Ensure organisms directory exists
async function ensureOrganismsDir() {
  try {
    await fs.access(ORGANISMS_DIR);
  } catch {
    await fs.mkdir(ORGANISMS_DIR, { recursive: true });
  }
}

// GET /api/organisms - List all organisms
export async function GET() {
  try {
    await ensureOrganismsDir();
    const files = await fs.readdir(ORGANISMS_DIR);
    const glassFiles = files.filter((f) => f.endsWith(".glass"));

    const organisms: GlassOrganism[] = [];

    for (const file of glassFiles) {
      try {
        const content = await fs.readFile(
          path.join(ORGANISMS_DIR, file),
          "utf-8"
        );
        const organism = JSON.parse(content) as GlassOrganism;
        organisms.push(organism);
      } catch (error) {
        console.error(`Failed to parse ${file}:`, error);
      }
    }

    return NextResponse.json(organisms);
  } catch (error) {
    console.error("Failed to list organisms:", error);
    return NextResponse.json(
      { error: "Failed to list organisms" },
      { status: 500 }
    );
  }
}

// POST /api/organisms - Upload .glass file
export async function POST(request: NextRequest) {
  try {
    await ensureOrganismsDir();

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

    // Save to file
    const filePath = path.join(ORGANISMS_DIR, `${organism.id}.glass`);
    await fs.writeFile(filePath, JSON.stringify(organism, null, 2));

    return NextResponse.json(organism);
  } catch (error) {
    console.error("Failed to upload organism:", error);
    return NextResponse.json(
      { error: "Failed to upload organism" },
      { status: 500 }
    );
  }
}
