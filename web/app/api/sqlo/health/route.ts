/**
 * LARANJA Health Check API
 *
 * GET /api/sqlo/health
 *
 * Checks if LARANJA (.sqlo database) integration is healthy
 * Returns performance metrics and storage statistics
 */

import { NextResponse } from 'next/server';
import {
  getLaranjaHealth,
  isLaranjaAvailable,
  getSQLOMetrics,
  getConsolidationStatus,
} from '@/lib/integrations/sqlo';
import { getLaranjaAdapter } from '@/lib/integrations/laranja-adapter';

export async function GET() {
  try {
    const available = isLaranjaAvailable();
    const health = await getLaranjaHealth();
    const metrics = await getSQLOMetrics();
    const consolidation = await getConsolidationStatus();

    // Get storage stats from adapter
    let storageStats = null;
    try {
      const adapter = getLaranjaAdapter();
      storageStats = adapter.getStorageStats();
    } catch {
      // Ignore if adapter not available
    }

    return NextResponse.json({
      success: true,
      data: {
        available,
        ...health,
        metrics: {
          avg_query_time_us: metrics.avg_query_time_us,
          total_queries: metrics.total_queries,
          cache_hit_rate: metrics.cache_hit_rate,
        },
        consolidation,
        storage: storageStats,
        features: {
          organism_storage: 'active',
          episodic_memory: 'active',
          constitutional_logs: 'active',
          llm_call_logging: 'active',
          rbac: 'active',
          consolidation_optimizer: 'active',
        },
        performance_targets: {
          queries: '<1ms',
          inserts: '<500μs',
          permission_checks: '<100μs',
        },
      },
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[API] /api/sqlo/health error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
        data: {
          available: false,
          status: 'error',
          version: 'unknown',
          performance_us: 0,
        },
      },
      { status: 500 }
    );
  }
}
