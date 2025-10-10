import { NextRequest, NextResponse } from "next/server";
import { QueryResult } from "@/lib/types";
import * as fs from "fs/promises";
import * as path from "path";

const ORGANISMS_DIR = path.join(process.cwd(), "..", "organisms");

// POST /api/query - Execute query against organism
export async function POST(request: NextRequest) {
  try {
    const { organismId, query } = await request.json();

    if (!organismId || !query) {
      return NextResponse.json(
        { error: "Missing organismId or query" },
        { status: 400 }
      );
    }

    // Load organism
    const filePath = path.join(ORGANISMS_DIR, `${organismId}.glass`);
    let organism;

    try {
      const content = await fs.readFile(filePath, "utf-8");
      organism = JSON.parse(content);
    } catch (error) {
      return NextResponse.json(
        { error: "Organism not found" },
        { status: 404 }
      );
    }

    // TODO: Integrate with ROXO GlassRuntime in DIA 5
    // For now, simulate a query execution
    const startTime = Date.now();

    // Simulate LLM processing time
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Simulate query result
    const result: QueryResult = {
      answer: `This is a simulated answer for organism "${organism.metadata.name}".\n\nYour query: "${query}"\n\nThe organism has ${organism.code.functions.length} emerged functions and ${organism.knowledge.papers} knowledge sources available. In a full implementation, this would execute the GlassRuntime from ROXO and return real results.`,
      confidence: 0.85,
      functions_used: organism.code.functions.map((f: any) => f.name).slice(0, 2),
      constitutional: "pass",
      cost: 0.05,
      time_ms: Date.now() - startTime,
      sources: [
        {
          id: "source-1",
          title: "Example Research Paper 1",
          type: "paper",
          relevance: 0.9,
        },
        {
          id: "source-2",
          title: "Example Clinical Trial",
          type: "trial",
          relevance: 0.85,
        },
        {
          id: "source-3",
          title: "Example Documentation",
          type: "document",
          relevance: 0.75,
        },
      ],
      attention: organism.knowledge.patterns?.slice(0, 5).map((p: any, i: number) => ({
        source_id: `knowledge_${i + 1}`,
        weight: Math.max(0.1, 1 / (i + 1) * 0.3),
      })) || [],
      reasoning: [
        {
          step: 1,
          description: "Analyzed query intent",
          confidence: 0.95,
          time_ms: 200,
        },
        {
          step: 2,
          description: "Selected relevant functions",
          confidence: 0.90,
          time_ms: 150,
        },
        {
          step: 3,
          description: "Accessed knowledge base",
          confidence: 0.88,
          time_ms: 400,
        },
        {
          step: 4,
          description: "Synthesized answer",
          confidence: 0.85,
          time_ms: Date.now() - startTime - 750,
        },
      ],
    };

    // Update organism stats
    organism.stats.queries_count = (organism.stats.queries_count || 0) + 1;
    organism.stats.total_cost = (organism.stats.total_cost || 0) + result.cost;
    organism.stats.avg_query_time_ms =
      ((organism.stats.avg_query_time_ms || 0) * (organism.stats.queries_count - 1) + result.time_ms) /
      organism.stats.queries_count;
    organism.stats.last_query_at = new Date().toISOString();

    // Save updated organism
    await fs.writeFile(filePath, JSON.stringify(organism, null, 2));

    return NextResponse.json(result);
  } catch (error) {
    console.error("Failed to execute query:", error);
    return NextResponse.json(
      { error: "Failed to execute query" },
      { status: 500 }
    );
  }
}
