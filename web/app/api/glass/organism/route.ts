/**
 * ROXO Organism Management API
 *
 * GET /api/glass/organism?organismId=... - Get organism metadata
 * POST /api/glass/organism - Get patterns, emerged functions, etc.
 *
 * Manages .glass organism data and code emergence
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  loadOrganism,
  getPatterns,
  getEmergedFunctions,
} from '@/lib/integrations/glass';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const organismId = searchParams.get('organismId');

    if (!organismId) {
      return NextResponse.json(
        { error: 'Missing required parameter: organismId' },
        { status: 400 }
      );
    }

    const organism = await loadOrganism(organismId);

    return NextResponse.json({
      success: true,
      data: organism,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/glass/organism GET error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { organismId, action } = body;

    if (!organismId) {
      return NextResponse.json(
        { error: 'Missing required parameter: organismId' },
        { status: 400 }
      );
    }

    if (!action) {
      return NextResponse.json(
        { error: 'Missing required parameter: action' },
        { status: 400 }
      );
    }

    let result;

    switch (action) {
      case 'patterns':
        result = await getPatterns(organismId);
        break;

      case 'functions':
        result = await getEmergedFunctions(organismId);
        break;

      case 'full':
        const organism = await loadOrganism(organismId);
        const patterns = await getPatterns(organismId);
        const functions = await getEmergedFunctions(organismId);

        result = {
          organism,
          patterns,
          functions,
        };
        break;

      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }

    return NextResponse.json({
      success: true,
      data: result,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/glass/organism POST error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}
