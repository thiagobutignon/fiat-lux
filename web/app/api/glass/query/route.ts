/**
 * ROXO Query Execution API
 *
 * POST /api/glass/query
 *
 * Executes queries against .glass organisms using GlassRuntime
 */

import { NextRequest, NextResponse } from 'next/server';
import { executeQuery } from '@/lib/integrations/glass';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { organismId, query } = body;

    if (!organismId) {
      return NextResponse.json(
        { error: 'Missing required parameter: organismId' },
        { status: 400 }
      );
    }

    if (!query) {
      return NextResponse.json(
        { error: 'Missing required parameter: query' },
        { status: 400 }
      );
    }

    const result = await executeQuery(organismId, query);

    return NextResponse.json({
      success: true,
      data: result,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/glass/query error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}

// GET endpoint for testing
export async function GET() {
  return NextResponse.json({
    endpoint: '/api/glass/query',
    method: 'POST',
    description: 'Execute query against a .glass organism using GlassRuntime',
    parameters: {
      organismId: 'string (required) - Organism ID',
      query: 'string (required) - Query to execute',
    },
    example: {
      organismId: 'cancer-research-1.0.0',
      query: 'What are the latest treatments for lung cancer?',
    },
  });
}
