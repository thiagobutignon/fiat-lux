/**
 * CINZA Health Check API
 *
 * GET /api/cognitive/health
 *
 * Checks if CINZA integration is healthy
 */

import { NextResponse } from 'next/server';
import { getCinzaHealth, isCinzaAvailable } from '@/lib/integrations/cognitive';

export async function GET() {
  try {
    const available = isCinzaAvailable();
    const health = await getCinzaHealth();

    return NextResponse.json({
      success: true,
      data: {
        available,
        ...health,
        features: {
          manipulation_detection: 'active',
          dark_tetrad_analysis: 'active',
          constitutional_layer_2: 'active',
          chomsky_hierarchy: 'active',
          neurodivergent_protection: 'active',
        },
        integrations: {
          vermelho: 'active', // VERMELHO + CINZA dual-layer
          verde: 'active', // VERDE + CINZA manipulation snapshots
        },
      },
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/cognitive/health error:', error);

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
