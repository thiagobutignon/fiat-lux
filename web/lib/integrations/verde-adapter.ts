/**
 * VERDE Adapter - Bridge between AMARELO and VERDE Core
 *
 * This adapter provides a bridge between AMARELO's web dashboard types
 * and VERDE's core genetic versioning system.
 *
 * Architecture:
 * AMARELO → verde-adapter.ts → VERDE Core (genetic-versioning.ts)
 *
 * Type Conversions:
 * - Mutation → VersionInfo
 * - Version → string format
 * - Canary traffic management
 */

import {
  createMutation,
  getFileVersion,
  getMutation,
  updateFitness,
  selectWinner,
  getRankedMutations,
  exportGeneticPool,
} from '../../../src/grammar-lang/vcs/genetic-versioning';

import type { EvolutionData, VersionInfo } from '../types';

// ============================================================================
// Types (VERDE Core)
// ============================================================================

interface Version {
  major: number;
  minor: number;
  patch: number;
}

interface Mutation {
  original: string;
  mutated: string;
  originalVersion: Version;
  mutatedVersion: Version;
  timestamp: number;
  author: 'human' | 'agi';
  fitness: number;
  traffic: number;
}

// ============================================================================
// Adapter Class
// ============================================================================

export class VerdeAdapter {
  private versionCache: Map<string, { versions: VersionInfo[]; timestamp: number }>;
  private cacheTTL: number = 5 * 60 * 1000; // 5 minutes

  constructor() {
    this.versionCache = new Map();
  }

  // ==========================================================================
  // Helper Functions
  // ==========================================================================

  /**
   * Convert VERDE Version to string
   */
  private versionToString(version: Version): string {
    return `${version.major}.${version.minor}.${version.patch}`;
  }

  /**
   * Parse version string to Version object
   */
  private parseVersion(versionStr: string): Version {
    const parts = versionStr.split('.');
    return {
      major: parseInt(parts[0]) || 0,
      minor: parseInt(parts[1]) || 0,
      patch: parseInt(parts[2]) || 0,
    };
  }

  /**
   * Convert VERDE Mutation to AMARELO VersionInfo
   */
  private convertToVersionInfo(mutation: Mutation, status: 'active' | 'canary' | 'old'): VersionInfo {
    const version = this.versionToString(mutation.mutatedVersion);
    const generation = this.calculateGeneration(mutation.mutatedVersion);

    return {
      version,
      generation,
      fitness: mutation.fitness,
      traffic_percent: mutation.traffic,
      deployed_at: new Date(mutation.timestamp).toISOString(),
      status,
    };
  }

  /**
   * Calculate generation number from version
   * Generation = major * 1000 + minor * 100 + patch
   */
  private calculateGeneration(version: Version): number {
    return version.major * 1000 + version.minor * 100 + version.patch;
  }

  // ==========================================================================
  // Version Management
  // ==========================================================================

  /**
   * Get version history for an organism
   */
  async getVersionHistory(organismId: string): Promise<VersionInfo[]> {
    // Check cache
    const cached = this.versionCache.get(organismId);
    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return cached.versions;
    }

    // Get genetic pool from VERDE
    const pool = exportGeneticPool();

    // Convert mutations to version info
    const versions: VersionInfo[] = pool.mutations.map((mutation) => {
      // Determine status based on traffic
      let status: 'active' | 'canary' | 'old' = 'old';
      if (mutation.traffic >= 90) {
        status = 'active';
      } else if (mutation.traffic > 0) {
        status = 'canary';
      }

      return this.convertToVersionInfo(mutation, status);
    });

    // Sort by generation (newest first)
    versions.sort((a, b) => b.generation - a.generation);

    // Cache result
    this.versionCache.set(organismId, {
      versions,
      timestamp: Date.now(),
    });

    return versions;
  }

  /**
   * Get current active version
   */
  async getCurrentVersion(organismId: string): Promise<VersionInfo> {
    const versions = await this.getVersionHistory(organismId);
    const active = versions.find((v) => v.status === 'active');

    if (active) {
      return active;
    }

    // If no active version, return newest version
    if (versions.length > 0) {
      return versions[0];
    }

    // Default version if no mutations exist
    return {
      version: '1.0.0',
      generation: 1000,
      fitness: 0.5,
      traffic_percent: 100,
      deployed_at: new Date().toISOString(),
      status: 'active',
    };
  }

  /**
   * Get evolution data for an organism
   */
  async getEvolutionData(organismId: string): Promise<EvolutionData> {
    const versions = await this.getVersionHistory(organismId);
    const pool = exportGeneticPool();
    const current = await this.getCurrentVersion(organismId);
    const canary = versions.find((v) => v.status === 'canary');

    return {
      organism_id: organismId,
      current_generation: current.generation,
      current_fitness: current.fitness,
      maturity: this.calculateMaturity(pool.stats.avgFitness, versions.length),
      versions,
      canary_status: {
        current_version: current.version,
        canary_version: canary?.version || current.version,
        current_traffic: current.traffic_percent,
        canary_traffic: canary?.traffic_percent || 0,
        status: canary ? 'monitoring' : 'inactive',
      },
    };
  }

  /**
   * Calculate maturity based on average fitness and version count
   */
  private calculateMaturity(avgFitness: number, versionCount: number): number {
    // Maturity = (avgFitness * 0.7) + (min(versionCount / 20, 1) * 0.3)
    const fitnessComponent = avgFitness * 0.7;
    const experienceComponent = Math.min(versionCount / 20, 1) * 0.3;
    return Math.min(1, fitnessComponent + experienceComponent);
  }

  // ==========================================================================
  // Canary Deployment
  // ==========================================================================

  /**
   * Get canary deployment status
   */
  async getCanaryStatus(organismId: string): Promise<{
    current_version: string;
    canary_version: string;
    current_traffic: number;
    canary_traffic: number;
    status: 'monitoring' | 'promoting' | 'rolling_back' | 'inactive';
  }> {
    const versions = await this.getVersionHistory(organismId);
    const active = versions.find((v) => v.status === 'active');
    const canary = versions.find((v) => v.status === 'canary');

    if (!canary) {
      return {
        current_version: active?.version || '1.0.0',
        canary_version: active?.version || '1.0.0',
        current_traffic: 100,
        canary_traffic: 0,
        status: 'inactive',
      };
    }

    // Determine status based on traffic trends
    let status: 'monitoring' | 'promoting' | 'rolling_back' = 'monitoring';
    if (canary.traffic_percent > 50) {
      status = 'promoting';
    } else if (canary.fitness < (active?.fitness || 0.5) * 0.8) {
      status = 'rolling_back';
    }

    return {
      current_version: active?.version || '1.0.0',
      canary_version: canary.version,
      current_traffic: active?.traffic_percent || 100,
      canary_traffic: canary.traffic_percent,
      status,
    };
  }

  /**
   * Deploy a new canary version
   */
  async deployCanary(
    organismId: string,
    filePath: string,
    trafficPercent: number = 1
  ): Promise<{ success: boolean; version: string; message: string }> {
    try {
      // Create mutation (new version)
      const mutation = await createMutation(filePath, 'human', 'patch');

      if (!mutation) {
        return {
          success: false,
          version: '',
          message: 'Failed to create mutation - file not found or validation failed',
        };
      }

      const version = this.versionToString(mutation.mutatedVersion);

      // Update traffic to canary percentage
      mutation.traffic = Math.max(1, Math.min(100, trafficPercent));

      // Invalidate cache
      this.versionCache.delete(organismId);

      return {
        success: true,
        version,
        message: `Canary v${version} deployed with ${trafficPercent}% traffic`,
      };
    } catch (error) {
      return {
        success: false,
        version: '',
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Promote canary to active (100% traffic)
   */
  async promoteCanary(
    organismId: string,
    canaryVersion: string
  ): Promise<{ success: boolean; message: string }> {
    try {
      const mutation = getMutation(canaryVersion);

      if (!mutation) {
        return {
          success: false,
          message: `Canary version ${canaryVersion} not found`,
        };
      }

      // Set canary to 100% traffic (becomes active)
      mutation.traffic = 100;

      // Reduce traffic on other versions
      const pool = exportGeneticPool();
      pool.mutations.forEach((m) => {
        const v = this.versionToString(m.mutatedVersion);
        if (v !== canaryVersion) {
          m.traffic = 0;
        }
      });

      // Invalidate cache
      this.versionCache.delete(organismId);

      return {
        success: true,
        message: `Canary v${canaryVersion} promoted to active (100% traffic)`,
      };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Rollback canary deployment
   */
  async rollbackCanary(
    organismId: string,
    canaryVersion: string
  ): Promise<{ success: boolean; message: string }> {
    try {
      const mutation = getMutation(canaryVersion);

      if (!mutation) {
        return {
          success: false,
          message: `Canary version ${canaryVersion} not found`,
        };
      }

      // Set canary traffic to 0
      mutation.traffic = 0;

      // Invalidate cache
      this.versionCache.delete(organismId);

      return {
        success: true,
        message: `Canary v${canaryVersion} rolled back (traffic set to 0%)`,
      };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // ==========================================================================
  // Rollback Operations
  // ==========================================================================

  /**
   * Rollback to a specific version
   */
  async rollbackVersion(
    organismId: string,
    version: string
  ): Promise<{ success: boolean; message: string }> {
    try {
      const mutation = getMutation(version);

      if (!mutation) {
        return {
          success: false,
          message: `Version ${version} not found`,
        };
      }

      // Set this version to 100% traffic
      mutation.traffic = 100;

      // Reduce traffic on all other versions
      const pool = exportGeneticPool();
      pool.mutations.forEach((m) => {
        const v = this.versionToString(m.mutatedVersion);
        if (v !== version) {
          m.traffic = 0;
        }
      });

      // Invalidate cache
      this.versionCache.delete(organismId);

      return {
        success: true,
        message: `Rolled back to v${version} (100% traffic)`,
      };
    } catch (error) {
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // ==========================================================================
  // Old-but-Gold Management
  // ==========================================================================

  /**
   * Get old-but-gold versions (high fitness but old)
   */
  async getOldButGoldVersions(organismId: string): Promise<VersionInfo[]> {
    const versions = await this.getVersionHistory(organismId);

    // Old-but-gold: status = 'old' but fitness > 0.7
    return versions.filter((v) => v.status === 'old' && v.fitness > 0.7);
  }

  // ==========================================================================
  // Fitness Tracking
  // ==========================================================================

  /**
   * Record fitness for a version
   */
  async recordFitness(
    version: string,
    metrics: {
      latency?: number;
      throughput?: number;
      errorRate?: number;
      crashRate?: number;
    }
  ): Promise<{ success: boolean; fitness: number; message: string }> {
    try {
      const success = updateFitness(version, metrics);

      if (!success) {
        return {
          success: false,
          fitness: 0,
          message: `Version ${version} not found`,
        };
      }

      const mutation = getMutation(version);
      const fitness = mutation?.fitness || 0;

      return {
        success: true,
        fitness,
        message: `Fitness updated: v${version} = ${fitness.toFixed(3)}`,
      };
    } catch (error) {
      return {
        success: false,
        fitness: 0,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Get fitness trajectory across generations
   */
  async getFitnessTrajectory(
    organismId: string
  ): Promise<{ generation: number; fitness: number; timestamp: string }[]> {
    const versions = await this.getVersionHistory(organismId);

    return versions.map((v) => ({
      generation: v.generation,
      fitness: v.fitness,
      timestamp: v.deployed_at,
    }));
  }

  // ==========================================================================
  // Health & Status
  // ==========================================================================

  /**
   * Check if VERDE is available
   */
  isAvailable(): boolean {
    try {
      // Try to get genetic pool
      const pool = exportGeneticPool();
      return pool !== undefined;
    } catch {
      return false;
    }
  }

  /**
   * Get VERDE health status
   */
  async getHealth(): Promise<{ status: string; version: string; mutations_tracked?: number }> {
    try {
      const pool = exportGeneticPool();

      return {
        status: 'healthy',
        version: '1.0.0',
        mutations_tracked: pool.stats.totalMutations,
      };
    } catch (error) {
      return {
        status: 'error',
        version: 'unknown',
        mutations_tracked: 0,
      };
    }
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let adapterInstance: VerdeAdapter | null = null;

export function getVerdeAdapter(): VerdeAdapter {
  if (!adapterInstance) {
    adapterInstance = new VerdeAdapter();
  }
  return adapterInstance;
}
