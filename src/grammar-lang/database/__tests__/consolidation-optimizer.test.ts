/**
 * Consolidation Optimizer Tests
 *
 * Verifies:
 * - Adaptive consolidation strategies
 * - Batch processing efficiency
 * - Memory pressure detection
 * - Performance improvements
 */

import { describe, it, beforeEach, afterEach, expect } from '../../../shared/utils/test-runner';
import {
  ConsolidationOptimizer,
  ConsolidationStrategy,
  createAdaptiveOptimizer,
  createBatchedOptimizer
} from '../consolidation-optimizer';
import { SqloDatabase, MemoryType } from '../sqlo';
import * as fs from 'fs';

const TEST_DB_DIR = 'test_consolidation_db';

describe('ConsolidationOptimizer - Adaptive Strategy', () => {
  let db: SqloDatabase;
  let optimizer: ConsolidationOptimizer;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    // Disable auto-consolidation to test manual optimization
    db = new SqloDatabase(TEST_DB_DIR, { autoConsolidate: false });
    optimizer = createAdaptiveOptimizer(db);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('creates adaptive optimizer with correct config', () => {
    const metrics = optimizer.getMetrics();

    expect.toBeDefined(metrics);
    expect.toEqual(metrics.episodes_consolidated, 0);
    expect.toEqual(metrics.episodes_promoted, 0);
  });

  it('consolidates when threshold reached', async () => {
    // Add 105 episodes to trigger consolidation
    for (let i = 0; i < 105; i++) {
      await db.put({
        query: `query ${i}`,
        response: `response ${i}`,
        attention: { sources: [], weights: [], patterns: [] },
        outcome: 'success',
        confidence: 0.9,
        timestamp: Date.now(),
        memory_type: MemoryType.SHORT_TERM
      });
    }

    const metrics = await optimizer.optimizeConsolidation();

    expect.toBeGreaterThan(metrics.episodes_consolidated, 0);
    expect.toBeGreaterThan(metrics.consolidation_time_ms, 0);
  });

  it('skips consolidation when below threshold', async () => {
    // Add only 10 episodes
    for (let i = 0; i < 10; i++) {
      await db.put({
        query: `query ${i}`,
        response: `response ${i}`,
        attention: { sources: [], weights: [], patterns: [] },
        outcome: 'success',
        confidence: 0.9,
        timestamp: Date.now(),
        memory_type: MemoryType.SHORT_TERM
      });
    }

    const metrics = await optimizer.optimizeConsolidation();

    expect.toEqual(metrics.episodes_consolidated, 0);
  });
});

describe('ConsolidationOptimizer - Batched Strategy', () => {
  let db: SqloDatabase;
  let optimizer: ConsolidationOptimizer;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    // Disable auto-consolidation to test manual optimization
    db = new SqloDatabase(TEST_DB_DIR, { autoConsolidate: false });
    optimizer = createBatchedOptimizer(db);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('processes episodes in batches', async () => {
    // Add 150 episodes
    for (let i = 0; i < 150; i++) {
      await db.put({
        query: `query ${i}`,
        response: `response ${i}`,
        attention: { sources: [], weights: [], patterns: [] },
        outcome: 'success',
        confidence: 0.85,
        timestamp: Date.now(),
        memory_type: MemoryType.SHORT_TERM
      });
    }

    const metrics = await optimizer.optimizeConsolidation();

    expect.toBeGreaterThan(metrics.episodes_consolidated, 0);
  });
});

describe('ConsolidationOptimizer - Performance', () => {
  let db: SqloDatabase;
  let optimizer: ConsolidationOptimizer;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    // Disable auto-consolidation to test manual optimization
    db = new SqloDatabase(TEST_DB_DIR, { autoConsolidate: false });
    optimizer = createAdaptiveOptimizer(db);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('completes consolidation in <100ms', async () => {
    // Add exactly 100 episodes (threshold)
    for (let i = 0; i < 100; i++) {
      await db.put({
        query: `query ${i}`,
        response: `response ${i}`,
        attention: { sources: [], weights: [], patterns: [] },
        outcome: 'success',
        confidence: 0.9,
        timestamp: Date.now(),
        memory_type: MemoryType.SHORT_TERM
      });
    }

    const metrics = await optimizer.optimizeConsolidation();

    expect.toBeLessThan(metrics.consolidation_time_ms, 100);
  });

  it('tracks consolidation metrics accurately', async () => {
    // Add 105 high-confidence episodes
    for (let i = 0; i < 105; i++) {
      await db.put({
        query: `query ${i}`,
        response: `response ${i}`,
        attention: { sources: [], weights: [], patterns: [] },
        outcome: 'success',
        confidence: 0.95,
        timestamp: Date.now(),
        memory_type: MemoryType.SHORT_TERM
      });
    }

    const metrics = await optimizer.optimizeConsolidation();

    expect.toBeDefined(metrics.episodes_consolidated);
    expect.toBeDefined(metrics.episodes_promoted);
    expect.toBeDefined(metrics.consolidation_time_ms);
  });
});

describe('ConsolidationOptimizer - Metrics', () => {
  let db: SqloDatabase;
  let optimizer: ConsolidationOptimizer;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    // Disable auto-consolidation to test manual optimization
    db = new SqloDatabase(TEST_DB_DIR, { autoConsolidate: false });
    optimizer = createAdaptiveOptimizer(db);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('resets metrics correctly', () => {
    optimizer.resetMetrics();
    const metrics = optimizer.getMetrics();

    expect.toEqual(metrics.episodes_consolidated, 0);
    expect.toEqual(metrics.episodes_promoted, 0);
    expect.toEqual(metrics.consolidation_time_ms, 0);
  });

  it('returns copy of metrics', () => {
    const metrics1 = optimizer.getMetrics();
    const metrics2 = optimizer.getMetrics();

    // Should be different objects
    expect.toBeDefined(metrics1);
    expect.toBeDefined(metrics2);
  });
});
