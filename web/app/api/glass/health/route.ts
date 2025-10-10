/**
 * ROXO Health Check API
 *
 * GET /api/glass/health
 *
 * Checks if ROXO (GlassRuntime) integration is healthy
 */

import { NextResponse } from 'next/server';
import { getRoxoHealth, isRoxoAvailable } from '@/lib/integrations/glass';

export async function GET() {
  try {
    const available = isRoxoAvailable();
    const health = await getRoxoHealth();

    return NextResponse.json({
      success: true,
      data: {
        available,
        ...health,
        features: {
          query_execution: 'active',
          pattern_detection: 'active',
          code_emergence: 'active',
          knowledge_graph: 'planned',
          constitutional_validation: 'active',
          llm_integration: 'active',
        },
        integrations: {
          constitutional: 'active', // Layer 1 + Layer 2
          laranja: 'planned', // .sqlo for organism storage
        },
      },
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/glass/health error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
        data: {
          available: false,
          status: 'error',
          version: 'unknown',
        },
      },
      { status: 500 }
    );
  }
}
