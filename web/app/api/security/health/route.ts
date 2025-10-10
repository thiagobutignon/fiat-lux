/**
 * VERMELHO Health Check API
 *
 * GET /api/security/health
 *
 * Checks if VERMELHO integration is healthy
 */

import { NextResponse } from 'next/server';
import { getVermelhoHealth, isVermelhoAvailable } from '@/lib/integrations/security';

export async function GET() {
  try {
    const available = isVermelhoAvailable();
    const health = await getVermelhoHealth();

    return NextResponse.json({
      success: true,
      data: {
        available,
        ...health,
        integrations: {
          cinza: 'active', // CINZA (cognitive) integration
          verde: 'active', // VERDE (Git security) integration
          laranja: 'active', // LARANJA (storage) integration
        },
      },
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/security/health error:', error);

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
