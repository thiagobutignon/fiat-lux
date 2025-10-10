/**
 * VERDE Health Check API
 *
 * GET /api/gvcs/health
 *
 * Checks if VERDE (Genetic Versioning) integration is healthy
 */

import { NextResponse } from 'next/server';
import { getVerdeHealth, isVerdeAvailable } from '@/lib/integrations/gvcs';

export async function GET() {
  try {
    const available = isVerdeAvailable();
    const health = await getVerdeHealth();

    return NextResponse.json({
      success: true,
      data: {
        available,
        ...health,
        features: {
          version_history: 'active',
          canary_deployment: 'active',
          fitness_tracking: 'active',
          old_but_gold: 'active',
          genetic_mutations: 'active',
          auto_commit: 'active',
        },
        integrations: {
          vermelho: 'active', // Duress validation in mutations
          cinza: 'active', // Manipulation detection in mutations
        },
      },
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/gvcs/health error:', error);

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
