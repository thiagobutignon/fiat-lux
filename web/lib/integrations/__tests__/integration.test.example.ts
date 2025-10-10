/**
 * Integration Testing Examples
 *
 * This file demonstrates how to test the integration layer.
 * Copy this file to integration.test.ts and implement your tests.
 *
 * Run with: npm test (after setting up Jest/Vitest)
 */

import { describe, it, expect, beforeAll } from '@jest/globals'; // or vitest
import {
  checkAllNodesHealth,
  getIntegrationStatus,
  executeQuery,
  getVersionHistory,
  analyzeDuress,
  detectManipulation,
  getOrganism,
} from '../index';

describe('Integration Layer - Health Checks', () => {
  it('should return health status for all nodes', async () => {
    const health = await checkAllNodesHealth();

    expect(health).toHaveProperty('roxo');
    expect(health).toHaveProperty('verde');
    expect(health).toHaveProperty('vermelho');
    expect(health).toHaveProperty('cinza');
    expect(health).toHaveProperty('laranja');

    // Each node should have status and version
    expect(health.roxo).toHaveProperty('available');
    expect(health.roxo).toHaveProperty('status');
    expect(health.roxo).toHaveProperty('version');
  });

  it('should return integration status', () => {
    const status = getIntegrationStatus();

    expect(status).toHaveProperty('nodes');
    expect(status).toHaveProperty('available_count');
    expect(status).toHaveProperty('total_count');
    expect(status).toHaveProperty('progress_percent');
    expect(status).toHaveProperty('ready');

    expect(status.total_count).toBe(5);
    expect(status.nodes).toHaveLength(5);
  });
});

describe('Integration Layer - ROXO (Glass)', () => {
  const TEST_ORGANISM_ID = 'test-organism-123';
  const TEST_QUERY = 'What is the treatment efficacy?';

  it('should execute query and return result', async () => {
    const result = await executeQuery(TEST_ORGANISM_ID, TEST_QUERY);

    expect(result).toHaveProperty('answer');
    expect(result).toHaveProperty('confidence');
    expect(result).toHaveProperty('functions_used');
    expect(result).toHaveProperty('constitutional');
    expect(result).toHaveProperty('cost');
    expect(result).toHaveProperty('time_ms');
    expect(result).toHaveProperty('sources');
    expect(result).toHaveProperty('attention');
    expect(result).toHaveProperty('reasoning');

    expect(typeof result.answer).toBe('string');
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(Array.isArray(result.functions_used)).toBe(true);
  });

  it('should handle invalid organism ID gracefully', async () => {
    await expect(
      executeQuery('invalid-id', TEST_QUERY)
    ).rejects.toThrow();
  });
});

describe('Integration Layer - VERDE (GVCS)', () => {
  const TEST_ORGANISM_ID = 'test-organism-123';

  it('should return version history', async () => {
    const versions = await getVersionHistory(TEST_ORGANISM_ID);

    expect(Array.isArray(versions)).toBe(true);

    if (versions.length > 0) {
      const version = versions[0];
      expect(version).toHaveProperty('version');
      expect(version).toHaveProperty('generation');
      expect(version).toHaveProperty('fitness');
      expect(version).toHaveProperty('traffic_percent');
      expect(version).toHaveProperty('deployed_at');
      expect(version).toHaveProperty('status');
    }
  });

  it('should have at least one active version', async () => {
    const versions = await getVersionHistory(TEST_ORGANISM_ID);
    const activeVersions = versions.filter(v => v.status === 'active');

    expect(activeVersions.length).toBeGreaterThan(0);
  });
});

describe('Integration Layer - VERMELHO (Security)', () => {
  const TEST_USER_ID = 'test-user-123';

  it('should analyze duress in text', async () => {
    const result = await analyzeDuress('Help me please', TEST_USER_ID);

    expect(result).toHaveProperty('is_duress');
    expect(result).toHaveProperty('confidence');
    expect(result).toHaveProperty('indicators');
    expect(result).toHaveProperty('severity');
    expect(result).toHaveProperty('recommended_action');

    expect(typeof result.is_duress).toBe('boolean');
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it('should return low severity for normal text', async () => {
    const result = await analyzeDuress('What is the weather today?', TEST_USER_ID);

    expect(result.severity).toBe('none');
    expect(result.is_duress).toBe(false);
  });
});

describe('Integration Layer - CINZA (Cognitive)', () => {
  it('should detect manipulation techniques', async () => {
    const result = await detectManipulation('You should definitely do this now');

    expect(result).toHaveProperty('detected');
    expect(result).toHaveProperty('confidence');
    expect(result).toHaveProperty('techniques');
    expect(result).toHaveProperty('severity');
    expect(result).toHaveProperty('recommended_action');

    expect(typeof result.detected).toBe('boolean');
    expect(Array.isArray(result.techniques)).toBe(true);
  });

  it('should return clean result for neutral text', async () => {
    const result = await detectManipulation('The sky is blue');

    expect(result.detected).toBe(false);
    expect(result.techniques).toHaveLength(0);
  });
});

describe('Integration Layer - LARANJA (Database)', () => {
  const TEST_ORGANISM_ID = 'test-organism-123';

  it('should retrieve organism from database', async () => {
    // This test requires LARANJA to be enabled
    // Skip if using stub
    const health = await checkAllNodesHealth();
    if (!health.laranja.available) {
      console.log('Skipping: LARANJA not available');
      return;
    }

    const organism = await getOrganism(TEST_ORGANISM_ID);

    expect(organism).toHaveProperty('id');
    expect(organism).toHaveProperty('metadata');
    expect(organism).toHaveProperty('knowledge');
    expect(organism).toHaveProperty('code');
    expect(organism).toHaveProperty('constitutional');
  });

  it('should have O(1) performance', async () => {
    const health = await checkAllNodesHealth();
    if (!health.laranja.available) {
      console.log('Skipping: LARANJA not available');
      return;
    }

    const start = performance.now();
    await getOrganism(TEST_ORGANISM_ID);
    const duration = performance.now() - start;

    // Should be under 2ms (accounting for network latency)
    expect(duration).toBeLessThan(2);
  });
});

describe('Integration Layer - Error Handling', () => {
  it('should throw descriptive errors when node is disabled', async () => {
    // This assumes nodes are disabled by default
    const health = await checkAllNodesHealth();

    if (!health.roxo.available) {
      await expect(
        executeQuery('any-id', 'any query')
      ).rejects.toThrow();
    }
  });

  it('should handle network errors gracefully', async () => {
    // Test error handling
    // Implementation depends on actual error scenarios
  });
});

describe('Integration Layer - Performance', () => {
  it('should complete health check in under 1 second', async () => {
    const start = performance.now();
    await checkAllNodesHealth();
    const duration = performance.now() - start;

    expect(duration).toBeLessThan(1000);
  });

  it('should not block on slow nodes', async () => {
    // All health checks run in parallel
    const start = performance.now();
    await checkAllNodesHealth();
    const duration = performance.now() - start;

    // Even with 5 nodes, should be fast (parallel execution)
    expect(duration).toBeLessThan(1000);
  });
});

/**
 * Example: Testing with Real Node APIs
 *
 * To test with real APIs:
 * 1. Enable the node by setting *_ENABLED = true in integration file
 * 2. Configure *_API_URL in .env.local
 * 3. Ensure the node API is running
 * 4. Run: npm test
 */

describe('Integration Layer - Real API Tests (E2E)', () => {
  beforeAll(async () => {
    const health = await checkAllNodesHealth();
    console.log('Integration Health:', health);
  });

  it.skip('should work with real ROXO API', async () => {
    // Enable this test when ROXO is integrated
    const result = await executeQuery('real-organism-id', 'real query');
    expect(result.answer).toBeTruthy();
  });

  it.skip('should work with real VERDE API', async () => {
    // Enable this test when VERDE is integrated
    const versions = await getVersionHistory('real-organism-id');
    expect(versions.length).toBeGreaterThan(0);
  });
});

/**
 * Coverage Checklist:
 *
 * ROXO (13 functions):
 * - [ ] executeQuery
 * - [ ] validateQuery
 * - [ ] getPatterns
 * - [ ] detectPatterns
 * - [ ] getEmergedFunctions
 * - [ ] synthesizeCode
 * - [ ] ingestKnowledge
 * - [ ] getKnowledgeGraph
 * - [ ] createRuntime
 * - [ ] loadOrganism
 * - [ ] isRoxoAvailable
 * - [ ] getRoxoHealth
 *
 * VERDE (15 functions):
 * - [ ] getVersionHistory
 * - [ ] getCurrentVersion
 * - [ ] getEvolutionData
 * - [ ] getCanaryStatus
 * - [ ] deployCanary
 * - [ ] promoteCanary
 * - [ ] rollbackCanary
 * - [ ] rollbackVersion
 * - [ ] getOldButGoldVersions
 * - [ ] markOldButGold
 * - [ ] recordFitness
 * - [ ] getFitnessTrajectory
 * - [ ] autoCommit
 * - [ ] isVerdeAvailable
 * - [ ] getVerdeHealth
 *
 * ... (similar for VERMELHO, CINZA, LARANJA)
 */
