import { NextResponse } from 'next/server';
import { checkAllNodesHealth, getIntegrationStatus } from '@/lib/integrations';

/**
 * GET /api/health
 *
 * Returns health status of all 5 Chomsky nodes and integration progress
 *
 * Response:
 * {
 *   nodes: {
 *     roxo: { available: boolean, status: string, version: string },
 *     verde: { available: boolean, status: string, version: string },
 *     vermelho: { available: boolean, status: string, version: string },
 *     cinza: { available: boolean, status: string, version: string },
 *     laranja: { available: boolean, status: string, version: string, performance_us?: number }
 *   },
 *   integration: {
 *     nodes: Array<{ name: string, color: string, available: boolean }>,
 *     available_count: number,
 *     total_count: number,
 *     progress_percent: number,
 *     ready: boolean
 *   },
 *   timestamp: string
 * }
 */
export async function GET() {
  try {
    const [nodesHealth, integrationStatus] = await Promise.all([
      checkAllNodesHealth(),
      Promise.resolve(getIntegrationStatus())
    ]);

    return NextResponse.json({
      nodes: nodesHealth,
      integration: integrationStatus,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Health check error:', error);

    return NextResponse.json(
      {
        error: 'Failed to check health',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
