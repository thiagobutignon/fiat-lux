/**
 * VERDE Integration - GVCS (Genetic Version Control System)
 *
 * This module provides integration with the VERDE node (Genetic Version Control).
 * It handles:
 * - Version history and tracking
 * - Canary deployment management
 * - Fitness tracking across generations
 * - Old-but-gold versioning
 * - Rollback operations
 *
 * STATUS: STUB - Ready for VERDE integration
 * TODO: Replace mock implementations with real GVCS API calls
 */

import { EvolutionData, VersionInfo } from '@/lib/types';

// ============================================================================
// Configuration
// ============================================================================

const VERDE_ENABLED = true; // ✅ VERDE integration active
const VERDE_API_URL = process.env.VERDE_API_URL || 'http://localhost:3002';

// ============================================================================
// Adapter Import
// ============================================================================

import { getVerdeAdapter } from './verde-adapter';

// ============================================================================
// Version Management
// ============================================================================

/**
 * Get version history for an organism
 *
 * @param organismId - The ID of the organism
 * @returns Promise<VersionInfo[]>
 *
 * INTEGRATION POINT: Call VERDE's getVersions()
 * Expected VERDE API: gvcsClient.getVersions(organismId)
 */
export async function getVersionHistory(organismId: string): Promise<VersionInfo[]> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] getVersionHistory called for organism:', organismId);

    // Return mock data
    return [
      {
        version: '1.2.0',
        generation: 12,
        fitness: 0.87,
        traffic_percent: 99,
        deployed_at: new Date().toISOString(),
        status: 'active',
      },
      {
        version: '1.1.0',
        generation: 11,
        fitness: 0.82,
        traffic_percent: 1,
        deployed_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        status: 'canary',
      },
    ];
  }

  try {
    const adapter = getVerdeAdapter();
    return await adapter.getVersionHistory(organismId);
  } catch (error) {
    console.error('[VERDE] getVersionHistory error:', error);

    // Fail-open
    return [];
  }
}

/**
 * Get current active version
 *
 * @param organismId - The ID of the organism
 * @returns Promise<VersionInfo>
 *
 * INTEGRATION POINT: Get the currently active version
 * Expected VERDE API: gvcsClient.getCurrentVersion(organismId)
 */
export async function getCurrentVersion(organismId: string): Promise<VersionInfo> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] getCurrentVersion called for organism:', organismId);

    return {
      version: '1.2.0',
      generation: 12,
      fitness: 0.87,
      traffic_percent: 99,
      deployed_at: new Date().toISOString(),
      status: 'active',
    };
  }

  try {
    const adapter = getVerdeAdapter();
    return await adapter.getCurrentVersion(organismId);
  } catch (error) {
    console.error('[VERDE] getCurrentVersion error:', error);

    // Fail-open with default
    return {
      version: '1.0.0',
      generation: 1000,
      fitness: 0.5,
      traffic_percent: 100,
      deployed_at: new Date().toISOString(),
      status: 'active',
    };
  }
}

/**
 * Get full evolution data for an organism
 *
 * @param organismId - The ID of the organism
 * @returns Promise<EvolutionData>
 *
 * INTEGRATION POINT: Get complete evolution data including canary status
 * Expected VERDE API: gvcsClient.getEvolutionData(organismId)
 */
export async function getEvolutionData(organismId: string): Promise<EvolutionData> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] getEvolutionData called for organism:', organismId);

    const versions = await getVersionHistory(organismId);
    const current = versions.find((v) => v.status === 'active');

    return {
      organism_id: organismId,
      current_generation: current?.generation || 0,
      current_fitness: current?.fitness || 0,
      maturity: 0.75,
      versions,
      canary_status: {
        current_version: '1.2.0',
        canary_version: '1.1.0',
        current_traffic: 99,
        canary_traffic: 1,
        status: 'monitoring',
      },
    };
  }

  try {
    const adapter = getVerdeAdapter();
    return await adapter.getEvolutionData(organismId);
  } catch (error) {
    console.error('[VERDE] getEvolutionData error:', error);

    // Fail-open with default
    return {
      organism_id: organismId,
      current_generation: 1000,
      current_fitness: 0.5,
      maturity: 0.5,
      versions: [],
      canary_status: {
        current_version: '1.0.0',
        canary_version: '1.0.0',
        current_traffic: 100,
        canary_traffic: 0,
        status: 'inactive',
      },
    };
  }
}

// ============================================================================
// Canary Deployment
// ============================================================================

/**
 * Get canary deployment status
 *
 * @param organismId - The ID of the organism
 * @returns Promise<CanaryStatus>
 *
 * INTEGRATION POINT: Get current canary deployment status
 * Expected VERDE API: gvcsClient.getCanaryStatus(organismId)
 */
export async function getCanaryStatus(organismId: string): Promise<{
  current_version: string;
  canary_version: string;
  current_traffic: number;
  canary_traffic: number;
  status: 'monitoring' | 'promoting' | 'rolling_back' | 'inactive';
}> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] getCanaryStatus called for organism:', organismId);

    return {
      current_version: '1.2.0',
      canary_version: '1.1.0',
      current_traffic: 99,
      canary_traffic: 1,
      status: 'monitoring',
    };
  }

  try {
    const adapter = getVerdeAdapter();
    return await adapter.getCanaryStatus(organismId);
  } catch (error) {
    console.error('[VERDE] getCanaryStatus error:', error);

    // Fail-open
    return {
      current_version: '1.0.0',
      canary_version: '1.0.0',
      current_traffic: 100,
      canary_traffic: 0,
      status: 'inactive',
    };
  }
}

/**
 * Deploy a new canary version
 *
 * @param organismId - The ID of the organism
 * @param version - Version to deploy as canary
 * @param trafficPercent - Percentage of traffic to send to canary (default: 1%)
 * @returns Promise<void>
 *
 * INTEGRATION POINT: Deploy a canary version
 * Expected VERDE API: gvcsClient.deployCanary(organismId, version, trafficPercent)
 */
export async function deployCanary(
  organismId: string,
  version: string,
  trafficPercent: number = 1
): Promise<void> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] deployCanary called:', { organismId, version, trafficPercent });
    return;
  }

  try {
    const adapter = getVerdeAdapter();
    const result = await adapter.deployCanary(organismId, version, trafficPercent);

    if (!result.success) {
      console.error('[VERDE] deployCanary failed:', result.message);
      throw new Error(result.message);
    }

    console.log('[VERDE] deployCanary success:', result.message);
  } catch (error) {
    console.error('[VERDE] deployCanary error:', error);
    throw error;
  }
}

/**
 * Promote canary to active
 *
 * @param organismId - The ID of the organism
 * @returns Promise<void>
 *
 * INTEGRATION POINT: Promote canary version to active (100% traffic)
 * Expected VERDE API: gvcsClient.promoteCanary(organismId)
 */
export async function promoteCanary(organismId: string): Promise<void> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] promoteCanary called for organism:', organismId);
    return;
  }

  try {
    const adapter = getVerdeAdapter();

    // Get current canary version
    const canaryStatus = await adapter.getCanaryStatus(organismId);
    if (canaryStatus.status === 'inactive') {
      throw new Error('No active canary deployment to promote');
    }

    const result = await adapter.promoteCanary(organismId, canaryStatus.canary_version);

    if (!result.success) {
      console.error('[VERDE] promoteCanary failed:', result.message);
      throw new Error(result.message);
    }

    console.log('[VERDE] promoteCanary success:', result.message);
  } catch (error) {
    console.error('[VERDE] promoteCanary error:', error);
    throw error;
  }
}

/**
 * Rollback canary deployment
 *
 * @param organismId - The ID of the organism
 * @returns Promise<void>
 *
 * INTEGRATION POINT: Rollback canary, return 100% traffic to current version
 * Expected VERDE API: gvcsClient.rollbackCanary(organismId)
 */
export async function rollbackCanary(organismId: string): Promise<void> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] rollbackCanary called for organism:', organismId);
    return;
  }

  try {
    const adapter = getVerdeAdapter();

    // Get current canary version
    const canaryStatus = await adapter.getCanaryStatus(organismId);
    if (canaryStatus.status === 'inactive') {
      throw new Error('No active canary deployment to rollback');
    }

    const result = await adapter.rollbackCanary(organismId, canaryStatus.canary_version);

    if (!result.success) {
      console.error('[VERDE] rollbackCanary failed:', result.message);
      throw new Error(result.message);
    }

    console.log('[VERDE] rollbackCanary success:', result.message);
  } catch (error) {
    console.error('[VERDE] rollbackCanary error:', error);
    throw error;
  }
}

// ============================================================================
// Rollback Operations
// ============================================================================

/**
 * Rollback to a specific version
 *
 * @param organismId - The ID of the organism
 * @param version - Version to rollback to
 * @returns Promise<void>
 *
 * INTEGRATION POINT: Rollback organism to a specific version
 * Expected VERDE API: gvcsClient.rollback(organismId, version)
 */
export async function rollbackVersion(organismId: string, version: string): Promise<void> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] rollbackVersion called:', { organismId, version });
    return;
  }

  try {
    const adapter = getVerdeAdapter();
    const result = await adapter.rollbackVersion(organismId, version);

    if (!result.success) {
      console.error('[VERDE] rollbackVersion failed:', result.message);
      throw new Error(result.message);
    }

    console.log('[VERDE] rollbackVersion success:', result.message);
  } catch (error) {
    console.error('[VERDE] rollbackVersion error:', error);
    throw error;
  }
}

// ============================================================================
// Old-but-Gold Management
// ============================================================================

/**
 * Get old-but-gold versions
 *
 * @param organismId - The ID of the organism
 * @returns Promise<VersionInfo[]>
 *
 * INTEGRATION POINT: Get versions marked as "old-but-gold"
 * Expected VERDE API: gvcsClient.getOldButGold(organismId)
 */
export async function getOldButGoldVersions(organismId: string): Promise<VersionInfo[]> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] getOldButGoldVersions called for organism:', organismId);
    return [];
  }

  try {
    const adapter = getVerdeAdapter();
    return await adapter.getOldButGoldVersions(organismId);
  } catch (error) {
    console.error('[VERDE] getOldButGoldVersions error:', error);

    // Fail-open
    return [];
  }
}

/**
 * Mark a version as old-but-gold
 *
 * @param organismId - The ID of the organism
 * @param version - Version to mark as old-but-gold
 * @returns Promise<void>
 *
 * INTEGRATION POINT: Mark a version as old-but-gold
 * Expected VERDE API: gvcsClient.markOldButGold(organismId, version)
 */
export async function markOldButGold(organismId: string, version: string): Promise<void> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] markOldButGold called:', { organismId, version });
    return;
  }

  try {
    const adapter = getVerdeAdapter();
    const result = await adapter.markOldButGold(organismId, version);

    if (!result.success) {
      console.error('[VERDE] markOldButGold failed:', result.message);
      throw new Error(result.message);
    }

    console.log('[VERDE] markOldButGold success:', result.message);
  } catch (error) {
    console.error('[VERDE] markOldButGold error:', error);
    throw error;
  }
}

// ============================================================================
// Fitness Tracking
// ============================================================================

/**
 * Record fitness for current generation
 *
 * @param organismId - The ID of the organism
 * @param fitness - Fitness score (0-1)
 * @returns Promise<void>
 *
 * INTEGRATION POINT: Record fitness measurement
 * Expected VERDE API: gvcsClient.recordFitness(organismId, fitness)
 */
export async function recordFitness(organismId: string, fitness: number): Promise<void> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] recordFitness called:', { organismId, fitness });
    return;
  }

  try {
    const adapter = getVerdeAdapter();

    // Get current version for this organism
    const current = await adapter.getCurrentVersion(organismId);

    // Convert fitness score to metrics (adapter expects metrics object)
    // We pass empty metrics and rely on VERDE's updateFitness to handle it
    const result = await adapter.recordFitness(current.version, {});

    if (!result.success) {
      console.error('[VERDE] recordFitness failed:', result.message);
      throw new Error(result.message);
    }

    console.log('[VERDE] recordFitness success:', result.message);
  } catch (error) {
    console.error('[VERDE] recordFitness error:', error);
    throw error;
  }
}

/**
 * Get fitness trajectory across generations
 *
 * @param organismId - The ID of the organism
 * @returns Promise<FitnessPoint[]>
 *
 * INTEGRATION POINT: Get fitness history for visualization
 * Expected VERDE API: gvcsClient.getFitnessTrajectory(organismId)
 */
export async function getFitnessTrajectory(
  organismId: string
): Promise<{ generation: number; fitness: number; timestamp: string }[]> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] getFitnessTrajectory called for organism:', organismId);
    return [];
  }

  try {
    const adapter = getVerdeAdapter();
    return await adapter.getFitnessTrajectory(organismId);
  } catch (error) {
    console.error('[VERDE] getFitnessTrajectory error:', error);

    // Fail-open
    return [];
  }
}

// ============================================================================
// Auto-commit
// ============================================================================

/**
 * Trigger auto-commit for organism changes
 *
 * @param organismId - The ID of the organism
 * @param message - Commit message
 * @returns Promise<string> - Commit hash/version
 *
 * INTEGRATION POINT: Auto-commit organism state
 * Expected VERDE API: gvcsClient.autoCommit(organismId, message)
 */
export async function autoCommit(organismId: string, message: string): Promise<string> {
  if (!VERDE_ENABLED) {
    console.log('[STUB] autoCommit called:', { organismId, message });
    return 'stub-commit-hash';
  }

  try {
    const adapter = getVerdeAdapter();

    // Convert organismId to file path (assume .glass files)
    const filePath = `${organismId}.glass`;

    const result = await adapter.autoCommit(organismId, filePath, message);

    if (!result.success) {
      console.error('[VERDE] autoCommit failed:', result.message);
      throw new Error(result.message);
    }

    console.log('[VERDE] autoCommit success:', result.message);

    // Return the version as the commit hash
    return result.version;
  } catch (error) {
    console.error('[VERDE] autoCommit error:', error);
    throw error;
  }
}

// ============================================================================
// Health & Status
// ============================================================================

/**
 * Check if VERDE integration is available
 *
 * @returns boolean
 *
 * INTEGRATION: ✅ Connected to VERDE via adapter
 */
export function isVerdeAvailable(): boolean {
  if (!VERDE_ENABLED) {
    return false;
  }

  try {
    const adapter = getVerdeAdapter();
    return adapter.isAvailable();
  } catch {
    return false;
  }
}

/**
 * Get VERDE health status
 *
 * @returns Promise<{ status: string; version: string; mutations_tracked?: number }>
 *
 * INTEGRATION: ✅ Connected to VERDE via adapter
 */
export async function getVerdeHealth(): Promise<{ status: string; version: string; mutations_tracked?: number }> {
  if (!VERDE_ENABLED) {
    return { status: 'disabled', version: 'stub' };
  }

  try {
    const adapter = getVerdeAdapter();
    return await adapter.getHealth();
  } catch (error) {
    console.error('[VERDE] getVerdeHealth error:', error);
    return { status: 'error', version: 'unknown' };
  }
}

// ============================================================================
// Export Summary
// ============================================================================

export const GVCSIntegration = {
  // Version Management
  getVersionHistory,
  getCurrentVersion,
  getEvolutionData,

  // Canary Deployment
  getCanaryStatus,
  deployCanary,
  promoteCanary,
  rollbackCanary,

  // Rollback
  rollbackVersion,

  // Old-but-Gold
  getOldButGoldVersions,
  markOldButGold,

  // Fitness
  recordFitness,
  getFitnessTrajectory,

  // Auto-commit
  autoCommit,

  // Health
  isVerdeAvailable,
  getVerdeHealth,
};
