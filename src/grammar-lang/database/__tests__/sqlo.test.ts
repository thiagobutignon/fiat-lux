/**
 * SQLO Database Integration Tests
 *
 * Verifies:
 * - O(1) CRUD operations (put, get, has, delete)
 * - Episodic memory (short-term, long-term, contextual)
 * - Auto-consolidation & auto-cleanup
 * - Performance guarantees (<1ms query)
 */

import { describe, it, beforeEach, afterEach, expect } from '../../../shared/utils/test-runner';
import { SqloDatabase, MemoryType, Episode, AttentionTrace } from '../sqlo';
import * as fs from 'fs';

// Test database directory
const TEST_DB_DIR = 'test_sqlo_db';

describe('SqloDatabase - O(1) CRUD Operations', () => {
  let db: SqloDatabase;

  beforeEach(() => {
    // Clean up test directory
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    db = new SqloDatabase(TEST_DB_DIR);
  });

  afterEach(() => {
    // Clean up after tests
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('put() stores episode and returns hash - O(1)', async () => {
    const episode = createTestEpisode('test query', MemoryType.SHORT_TERM);
    const hash = await db.put(episode);

    expect.toBeDefined(hash);
    expect.toEqual(hash.length, 64); // SHA256 hex
    expect.toBeTruthy(db.has(hash));
  });

  it('get() retrieves episode by hash - O(1)', async () => {
    const episode = createTestEpisode('test query', MemoryType.LONG_TERM);
    const hash = await db.put(episode);

    const retrieved = db.get(hash);
    expect.toBeDefined(retrieved);
    expect.toEqual(retrieved!.query, 'test query');
    expect.toEqual(retrieved!.memory_type, MemoryType.LONG_TERM);
  });

  it('has() checks existence - O(1)', async () => {
    const episode = createTestEpisode('test', MemoryType.CONTEXTUAL);
    const hash = await db.put(episode);

    expect.toBeTruthy(db.has(hash));
    expect.toBeFalsy(db.has('nonexistent'));
  });

  it('delete() removes episode - O(1)', async () => {
    const episode = createTestEpisode('test', MemoryType.SHORT_TERM);
    const hash = await db.put(episode);

    expect.toBeTruthy(db.has(hash));
    const deleted = db.delete(hash);

    expect.toBeTruthy(deleted);
    expect.toBeFalsy(db.has(hash));
    expect.toEqual(db.get(hash), null);
  });

  it('put() returns same hash for identical content (content-addressable)', async () => {
    const episode1 = createTestEpisode('identical', MemoryType.SHORT_TERM);
    const episode2 = createTestEpisode('identical', MemoryType.SHORT_TERM);

    const hash1 = await db.put(episode1);
    const hash2 = await db.put(episode2);

    expect.toEqual(hash1, hash2);
  });
});

describe('SqloDatabase - Episodic Memory', () => {
  let db: SqloDatabase;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    db = new SqloDatabase(TEST_DB_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('listByType() filters by memory type', async () => {
    // Create episodes of different types
    await db.put(createTestEpisode('short1', MemoryType.SHORT_TERM));
    await db.put(createTestEpisode('short2', MemoryType.SHORT_TERM));
    await db.put(createTestEpisode('long1', MemoryType.LONG_TERM));
    await db.put(createTestEpisode('contextual1', MemoryType.CONTEXTUAL));

    const shortTerm = db.listByType(MemoryType.SHORT_TERM);
    const longTerm = db.listByType(MemoryType.LONG_TERM);
    const contextual = db.listByType(MemoryType.CONTEXTUAL);

    expect.toEqual(shortTerm.length, 2);
    expect.toEqual(longTerm.length, 1);
    expect.toEqual(contextual.length, 1);
  });

  it('querySimilar() finds similar episodes', async () => {
    // Create episodes with keywords
    await db.put(createTestEpisode('explain lambda calculus', MemoryType.LONG_TERM));
    await db.put(createTestEpisode('what is lambda expression', MemoryType.LONG_TERM));
    await db.put(createTestEpisode('how to use SQL', MemoryType.LONG_TERM));

    const results = await db.querySimilar('lambda calculus', 5);

    expect.toBeGreaterThan(results.length, 0);
    expect.toBeTruthy(results[0].query.includes('lambda'));
  });

  it('getStatistics() returns accurate counts', async () => {
    await db.put(createTestEpisode('q1', MemoryType.SHORT_TERM));
    await db.put(createTestEpisode('q2', MemoryType.SHORT_TERM));
    await db.put(createTestEpisode('q3', MemoryType.LONG_TERM));
    await db.put(createTestEpisode('q4', MemoryType.CONTEXTUAL));

    const stats = db.getStatistics();

    expect.toEqual(stats.total_episodes, 4);
    expect.toEqual(stats.short_term_count, 2);
    expect.toEqual(stats.long_term_count, 1);
    expect.toEqual(stats.contextual_count, 1);
  });

  it('attention traces are preserved', async () => {
    const attention: AttentionTrace = {
      sources: ['paper1.pdf', 'paper2.pdf'],
      weights: [0.7, 0.3],
      patterns: ['pattern1', 'pattern2']
    };

    const episode = createTestEpisode('test', MemoryType.LONG_TERM, attention);
    const hash = await db.put(episode);

    const retrieved = db.get(hash);
    expect.toDeepEqual(retrieved!.attention.sources, ['paper1.pdf', 'paper2.pdf']);
    expect.toDeepEqual(retrieved!.attention.weights, [0.7, 0.3]);
    expect.toDeepEqual(retrieved!.attention.patterns, ['pattern1', 'pattern2']);
  });
});

describe('SqloDatabase - Auto-Consolidation', () => {
  let db: SqloDatabase;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    db = new SqloDatabase(TEST_DB_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('consolidates short-term to long-term at threshold', async () => {
    // Create 105 successful short-term episodes to trigger consolidation
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

    const stats = db.getStatistics();

    // Some episodes should have been consolidated to long-term
    expect.toBeGreaterThan(stats.long_term_count, 0);
    expect.toBeLessThan(stats.short_term_count, 105);
  });
});

describe('SqloDatabase - Performance Guarantees', () => {
  let db: SqloDatabase;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    db = new SqloDatabase(TEST_DB_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('get() completes in <1ms average (O(1) guarantee)', async () => {
    const episode = createTestEpisode('perf test', MemoryType.LONG_TERM);
    const hash = await db.put(episode);

    const iterations = 100;
    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
      db.get(hash);
    }

    const end = performance.now();
    const avgTime = (end - start) / iterations;

    expect.toBeLessThan(avgTime, 1); // <1ms average
  });

  it('put() completes in <10ms average (O(1) guarantee)', async () => {
    const iterations = 50;
    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
      await db.put(createTestEpisode(`perf ${i}`, MemoryType.LONG_TERM));
    }

    const end = performance.now();
    const avgTime = (end - start) / iterations;

    expect.toBeLessThan(avgTime, 10); // <10ms average (disk write)
  });

  it('has() completes in <0.1ms average (O(1) guarantee)', async () => {
    const episode = createTestEpisode('perf test', MemoryType.LONG_TERM);
    const hash = await db.put(episode);

    const iterations = 1000;
    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
      db.has(hash);
    }

    const end = performance.now();
    const avgTime = (end - start) / iterations;

    expect.toBeLessThan(avgTime, 0.1); // <0.1ms average (index lookup)
  });
});

describe('SqloDatabase - Edge Cases', () => {
  let db: SqloDatabase;

  beforeEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
    db = new SqloDatabase(TEST_DB_DIR);
  });

  afterEach(() => {
    if (fs.existsSync(TEST_DB_DIR)) {
      fs.rmSync(TEST_DB_DIR, { recursive: true });
    }
  });

  it('handles empty database gracefully', () => {
    const stats = db.getStatistics();
    expect.toEqual(stats.total_episodes, 0);
    expect.toEqual(db.get('nonexistent'), null);
    expect.toBeFalsy(db.has('nonexistent'));
  });

  it('handles deletion of non-existent episode', () => {
    const result = db.delete('nonexistent');
    expect.toBeFalsy(result);
  });

  it('handles large responses', async () => {
    const largeResponse = 'x'.repeat(100000); // 100KB
    const episode = createTestEpisode('large', MemoryType.LONG_TERM);
    episode.response = largeResponse;

    const hash = await db.put(episode);
    const retrieved = db.get(hash);

    expect.toEqual(retrieved!.response.length, 100000);
  });

  it('handles special characters in queries', async () => {
    const specialQuery = 'λx.x → λy.y (unicode: 🚀)';
    const episode = createTestEpisode(specialQuery, MemoryType.LONG_TERM);

    const hash = await db.put(episode);
    const retrieved = db.get(hash);

    expect.toEqual(retrieved!.query, specialQuery);
  });
});

// =============================================================================
// Test Helpers
// =============================================================================

function createTestEpisode(
  query: string,
  memoryType: MemoryType,
  attention?: AttentionTrace
): Omit<Episode, 'id'> {
  return {
    query,
    response: `Response to: ${query}`,
    attention: attention || {
      sources: ['test.pdf'],
      weights: [1.0],
      patterns: ['test-pattern']
    },
    outcome: 'success',
    confidence: 0.95,
    timestamp: Date.now(),
    memory_type: memoryType
  };
}
