/**
 * Canary Deployment System - O(1) Gradual Rollout
 *
 * Deploys new versions gradually based on fitness:
 * - 99%/1% traffic split (canary)
 * - Collect metrics in real-time
 * - Gradual rollout if better: 1% → 2% → 5% → 10% → 25% → 50% → 75% → 100%
 * - Automatic rollback if worse
 *
 * Philosophy:
 * - Glass box (100% transparent)
 * - O(1) routing decision (hash-based)
 * - Biological A/B testing (natural selection)
 * - Safe deployment (automatic rollback)
 */

import * as crypto from 'crypto';
import {
  getMutation,
  updateFitness,
  selectWinner,
  getRankedMutations
} from './genetic-versioning';

// ===== TYPES =====

interface CanaryConfig {
  originalVersion: string;
  canaryVersion: string;
  trafficPercentage: number; // 0-100
  rampUpSpeed: 'slow' | 'medium' | 'fast';
  autoRollback: boolean;
  minSampleSize: number; // Minimum requests before decision
}

interface Metrics {
  version: string;
  requestCount: number;
  latency: number[];       // ms per request
  errorCount: number;
  crashCount: number;
  timestamp: number;
}

interface RoutingDecision {
  version: string;
  reason: 'traffic-split' | 'rollback' | 'rollout-complete';
}

// ===== STATE =====

// Active canary deployments
const canaryDeployments = new Map<string, CanaryConfig>();

// Metrics storage (O(1) lookup)
const metricsStore = new Map<string, Metrics>();

// Ramp-up schedules
const RAMP_SCHEDULES = {
  slow: [1, 2, 5, 10, 15, 25, 35, 50, 65, 80, 90, 100],
  medium: [1, 5, 10, 25, 50, 75, 100],
  fast: [1, 10, 50, 100]
};

// ===== TRAFFIC ROUTING =====

/**
 * Route request to version based on traffic split
 * O(1) - hash-based routing
 *
 * Uses consistent hashing to ensure same user gets same version
 */
export function routeRequest(
  deploymentId: string,
  userId: string
): RoutingDecision {
  const canary = canaryDeployments.get(deploymentId);

  if (!canary) {
    return {
      version: 'unknown',
      reason: 'traffic-split'
    };
  }

  // Check if rollout complete
  if (canary.trafficPercentage >= 100) {
    return {
      version: canary.canaryVersion,
      reason: 'rollout-complete'
    };
  }

  // Consistent hashing (O(1))
  // Same user ID always gets same version
  const hash = crypto
    .createHash('md5')
    .update(userId + deploymentId)
    .digest('hex');

  // Convert hash to 0-100 range
  const hashValue = parseInt(hash.substring(0, 8), 16) % 100;

  // Route based on traffic percentage
  if (hashValue < canary.trafficPercentage) {
    return {
      version: canary.canaryVersion,
      reason: 'traffic-split'
    };
  } else {
    return {
      version: canary.originalVersion,
      reason: 'traffic-split'
    };
  }
}

// ===== METRICS COLLECTION =====

/**
 * Record request metrics
 * O(1) - hash map update
 */
export function recordMetrics(
  version: string,
  latency: number,
  isError: boolean = false,
  isCrash: boolean = false
): void {
  let metrics = metricsStore.get(version);

  if (!metrics) {
    metrics = {
      version,
      requestCount: 0,
      latency: [],
      errorCount: 0,
      crashCount: 0,
      timestamp: Date.now()
    };
  }

  metrics.requestCount += 1;
  metrics.latency.push(latency);
  if (isError) metrics.errorCount += 1;
  if (isCrash) metrics.crashCount += 1;

  metricsStore.set(version, metrics);
}

/**
 * Get aggregated metrics
 * O(n) where n = number of requests (bounded by window)
 */
export function getAggregatedMetrics(version: string): {
  avgLatency: number;
  p95Latency: number;
  errorRate: number;
  crashRate: number;
  throughput: number;
} | null {
  const metrics = metricsStore.get(version);

  if (!metrics || metrics.requestCount === 0) {
    return null;
  }

  // Calculate average latency
  const avgLatency = metrics.latency.reduce((a, b) => a + b, 0) / metrics.latency.length;

  // Calculate p95 latency
  const sortedLatency = [...metrics.latency].sort((a, b) => a - b);
  const p95Index = Math.floor(sortedLatency.length * 0.95);
  const p95Latency = sortedLatency[p95Index] || avgLatency;

  // Calculate rates
  const errorRate = metrics.errorCount / metrics.requestCount;
  const crashRate = metrics.crashCount / metrics.requestCount;

  // Calculate throughput (req/s)
  const timeElapsed = (Date.now() - metrics.timestamp) / 1000; // seconds
  const throughput = metrics.requestCount / Math.max(timeElapsed, 1);

  return {
    avgLatency,
    p95Latency,
    errorRate,
    crashRate,
    throughput
  };
}

// ===== CANARY DEPLOYMENT =====

/**
 * Start canary deployment
 * O(1) - create config and store
 */
export function startCanary(
  deploymentId: string,
  originalVersion: string,
  canaryVersion: string,
  options: {
    rampUpSpeed?: 'slow' | 'medium' | 'fast';
    autoRollback?: boolean;
    minSampleSize?: number;
  } = {}
): boolean {
  const config: CanaryConfig = {
    originalVersion,
    canaryVersion,
    trafficPercentage: 1, // Start with 1%
    rampUpSpeed: options.rampUpSpeed || 'medium',
    autoRollback: options.autoRollback !== false,
    minSampleSize: options.minSampleSize || 100
  };

  canaryDeployments.set(deploymentId, config);

  console.log(`🐤 Canary deployment started:`);
  console.log(`   Deployment ID: ${deploymentId}`);
  console.log(`   Original: v${originalVersion} (99% traffic)`);
  console.log(`   Canary: v${canaryVersion} (1% traffic)`);
  console.log(`   Ramp-up: ${config.rampUpSpeed}`);
  console.log(`   Auto-rollback: ${config.autoRollback ? 'enabled' : 'disabled'}`);

  return true;
}

/**
 * Evaluate canary and decide next step
 * O(1) - fitness comparison
 */
export function evaluateCanary(deploymentId: string): {
  decision: 'increase' | 'rollback' | 'maintain' | 'complete';
  currentTraffic: number;
  newTraffic: number;
  reason: string;
} | null {
  const canary = canaryDeployments.get(deploymentId);

  if (!canary) {
    return null;
  }

  // Get metrics for both versions
  const originalMetrics = getAggregatedMetrics(canary.originalVersion);
  const canaryMetrics = getAggregatedMetrics(canary.canaryVersion);

  // Check if we have enough samples
  const canaryStore = metricsStore.get(canary.canaryVersion);
  const canaryRequests = canaryStore?.requestCount || 0;

  if (canaryRequests < canary.minSampleSize) {
    return {
      decision: 'maintain',
      currentTraffic: canary.trafficPercentage,
      newTraffic: canary.trafficPercentage,
      reason: `Waiting for more samples (${canaryRequests}/${canary.minSampleSize})`
    };
  }

  // Update fitness based on metrics
  if (originalMetrics) {
    updateFitness(canary.originalVersion, {
      latency: originalMetrics.avgLatency,
      throughput: originalMetrics.throughput,
      errorRate: originalMetrics.errorRate,
      crashRate: originalMetrics.crashRate
    });
  }

  if (canaryMetrics) {
    updateFitness(canary.canaryVersion, {
      latency: canaryMetrics.avgLatency,
      throughput: canaryMetrics.throughput,
      errorRate: canaryMetrics.errorRate,
      crashRate: canaryMetrics.crashRate
    });
  }

  // Compare versions
  const winner = selectWinner(canary.originalVersion, canary.canaryVersion);

  // Decision logic
  if (winner === canary.canaryVersion) {
    // Canary is better - increase traffic
    const schedule = RAMP_SCHEDULES[canary.rampUpSpeed];
    const currentIndex = schedule.indexOf(canary.trafficPercentage);
    const nextIndex = currentIndex + 1;

    if (nextIndex >= schedule.length) {
      // Rollout complete
      canary.trafficPercentage = 100;
      canaryDeployments.set(deploymentId, canary);

      return {
        decision: 'complete',
        currentTraffic: schedule[currentIndex],
        newTraffic: 100,
        reason: 'Canary outperformed original - rollout complete'
      };
    }

    const newTraffic = schedule[nextIndex];
    canary.trafficPercentage = newTraffic;
    canaryDeployments.set(deploymentId, canary);

    return {
      decision: 'increase',
      currentTraffic: schedule[currentIndex],
      newTraffic,
      reason: `Canary performing better - increasing traffic`
    };
  } else if (winner === canary.originalVersion && canary.autoRollback) {
    // Original is better - rollback
    canary.trafficPercentage = 0;
    canaryDeployments.set(deploymentId, canary);

    return {
      decision: 'rollback',
      currentTraffic: canary.trafficPercentage,
      newTraffic: 0,
      reason: 'Original outperformed canary - rolling back'
    };
  } else {
    // Tie or auto-rollback disabled - maintain
    return {
      decision: 'maintain',
      currentTraffic: canary.trafficPercentage,
      newTraffic: canary.trafficPercentage,
      reason: 'Versions tied or auto-rollback disabled'
    };
  }
}

/**
 * Stop canary deployment
 * O(1) - remove from map
 */
export function stopCanary(deploymentId: string): boolean {
  const deleted = canaryDeployments.delete(deploymentId);

  if (deleted) {
    console.log(`🛑 Canary deployment stopped: ${deploymentId}`);
  }

  return deleted;
}

// ===== MONITORING =====

/**
 * Get canary status
 * O(1) - hash map lookup
 */
export function getCanaryStatus(deploymentId: string): {
  config: CanaryConfig;
  originalMetrics: ReturnType<typeof getAggregatedMetrics>;
  canaryMetrics: ReturnType<typeof getAggregatedMetrics>;
} | null {
  const canary = canaryDeployments.get(deploymentId);

  if (!canary) {
    return null;
  }

  return {
    config: canary,
    originalMetrics: getAggregatedMetrics(canary.originalVersion),
    canaryMetrics: getAggregatedMetrics(canary.canaryVersion)
  };
}

/**
 * Export canary state (glass box)
 */
export function exportCanaryState(): {
  deployments: CanaryConfig[];
  metrics: Record<string, Metrics>;
  summary: {
    activeDeployments: number;
    totalRequests: number;
  };
} {
  const deploymentsArray = Array.from(canaryDeployments.values());
  const metricsObject = Object.fromEntries(metricsStore);

  const totalRequests = Array.from(metricsStore.values()).reduce(
    (sum, m) => sum + m.requestCount,
    0
  );

  return {
    deployments: deploymentsArray,
    metrics: metricsObject,
    summary: {
      activeDeployments: deploymentsArray.length,
      totalRequests
    }
  };
}
