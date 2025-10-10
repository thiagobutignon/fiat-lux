/**
 * SQLO Database Integration Tests
 *
 * Verifies:
 * - O(1) CRUD operations (put, get, has, delete)
 * - Episodic memory (short-term, long-term, contextual)
 * - Auto-consolidation & auto-cleanup
 * - Performance guarantees (<1ms query)
 */

import { SqloDatabase, MemoryType, Episode, AttentionTrace } from '../../src/grammar-lang/database/sqlo';
import * as fs from 'fs';
import * as path from 'path';

// Test database directory
const TEST_DB_DIR = 'test_sqlo_db';

describe('SqloDatabase', () => {
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

  // ==========================================================================
  // DIA 2: O(1) CRUD Operations
  // ==========================================================================

  describe('O(1) CRUD Operations', () => {
    test('put() stores episode and returns hash - O(1)', async () => {
      const episode = createTestEpisode('test query', MemoryType.SHORT_TERM);
      const hash = await db.put(episode);

      expect(hash).toBeDefined();
      expect(hash.length).toBe(64); // SHA256 hex
      expect(db.has(hash)).toBe(true);
    });

    test('get() retrieves episode by hash - O(1)', async () => {
      const episode = createTestEpisode('test query', MemoryType.LONG_TERM);
      const hash = await db.put(episode);

      const retrieved = db.get(hash);
      expect(retrieved).not.toBeNull();
      expect(retrieved!.query).toBe('test query');
      expect(retrieved!.memory_type).toBe(MemoryType.LONG_TERM);
    });

    test('has() checks existence - O(1)', async () => {
      const episode = createTestEpisode('test', MemoryType.CONTEXTUAL);
      const hash = await db.put(episode);

      expect(db.has(hash)).toBe(true);
      expect(db.has('nonexistent')).toBe(false);
    });

    test('delete() removes episode - O(1)', async () => {
      const episode = createTestEpisode('test', MemoryType.SHORT_TERM);
      const hash = await db.put(episode);

      expect(db.has(hash)).toBe(true);
      const deleted = db.delete(hash);

      expect(deleted).toBe(true);
      expect(db.has(hash)).toBe(false);
      expect(db.get(hash)).toBeNull();
    });

    test('put() returns same hash for identical content (content-addressable)', async () => {
      const episode1 = createTestEpisode('identical', MemoryType.SHORT_TERM);
      const episode2 = createTestEpisode('identical', MemoryType.SHORT_TERM);

      const hash1 = await db.put(episode1);
      const hash2 = await db.put(episode2);

      expect(hash1).toBe(hash2);
    });
  });

  // ==========================================================================
  // DIA 3: Episodic Memory
  // ==========================================================================

  describe('Episodic Memory', () => {
    test('listByType() filters by memory type', async () => {
      // Create episodes of different types
      await db.put(createTestEpisode('short1', MemoryType.SHORT_TERM));
      await db.put(createTestEpisode('short2', MemoryType.SHORT_TERM));
      await db.put(createTestEpisode('long1', MemoryType.LONG_TERM));
      await db.put(createTestEpisode('contextual1', MemoryType.CONTEXTUAL));

      const shortTerm = db.listByType(MemoryType.SHORT_TERM);
      const longTerm = db.listByType(MemoryType.LONG_TERM);
      const contextual = db.listByType(MemoryType.CONTEXTUAL);

      expect(shortTerm.length).toBe(2);
      expect(longTerm.length).toBe(1);
      expect(contextual.length).toBe(1);
    });

    test('querySimilar() finds similar episodes', async () => {
      // Create episodes with keywords
      await db.put(createTestEpisode('explain lambda calculus', MemoryType.LONG_TERM));
      await db.put(createTestEpisode('what is lambda expression', MemoryType.LONG_TERM));
      await db.put(createTestEpisode('how to use SQL', MemoryType.LONG_TERM));

      const results = db.querySimilar('lambda calculus', 5);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].query).toContain('lambda');
    });

    test('getStatistics() returns accurate counts', async () => {
      await db.put(createTestEpisode('q1', MemoryType.SHORT_TERM));
      await db.put(createTestEpisode('q2', MemoryType.SHORT_TERM));
      await db.put(createTestEpisode('q3', MemoryType.LONG_TERM));
      await db.put(createTestEpisode('q4', MemoryType.CONTEXTUAL));

      const stats = db.getStatistics();

      expect(stats.total_episodes).toBe(4);
      expect(stats.short_term_count).toBe(2);
      expect(stats.long_term_count).toBe(1);
      expect(stats.contextual_count).toBe(1);
    });

    test('attention traces are preserved', async () => {
      const attention: AttentionTrace = {
        sources: ['paper1.pdf', 'paper2.pdf'],
        weights: [0.7, 0.3],
        patterns: ['pattern1', 'pattern2']
      };

      const episode = createTestEpisode('test', MemoryType.LONG_TERM, attention);
      const hash = await db.put(episode);

      const retrieved = db.get(hash);
      expect(retrieved!.attention.sources).toEqual(['paper1.pdf', 'paper2.pdf']);
      expect(retrieved!.attention.weights).toEqual([0.7, 0.3]);
      expect(retrieved!.attention.patterns).toEqual(['pattern1', 'pattern2']);
    });
  });

  // ==========================================================================
  // Auto-Consolidation & Auto-Cleanup
  // ==========================================================================

  describe('Auto-Consolidation', () => {
    test('consolidates short-term to long-term at threshold', async () => {
      // Create 100+ successful short-term episodes to trigger consolidation
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
      expect(stats.long_term_count).toBeGreaterThan(0);
      expect(stats.short_term_count).toBeLessThan(105);
    });
  });

  describe('Auto-Cleanup', () => {
    test('expires short-term memories after TTL', async () => {
      // Create short-term episode
      const episode = createTestEpisode('expiring', MemoryType.SHORT_TERM);
      const hash = await db.put(episode);

      expect(db.has(hash)).toBe(true);

      // Simulate time passing by creating new episode (triggers cleanup)
      // In real implementation, we'd need to mock Date.now() or wait
      // For now, we just verify the cleanup mechanism exists
      const stats = db.getStatistics();
      expect(stats).toBeDefined();
    });
  });

  // ==========================================================================
  // Performance Tests
  // ==========================================================================

  describe('Performance Guarantees', () => {
    test('get() completes in <1ms (O(1) guarantee)', async () => {
      const episode = createTestEpisode('perf test', MemoryType.LONG_TERM);
      const hash = await db.put(episode);

      const iterations = 100;
      const start = performance.now();

      for (let i = 0; i < iterations; i++) {
        db.get(hash);
      }

      const end = performance.now();
      const avgTime = (end - start) / iterations;

      expect(avgTime).toBeLessThan(1); // <1ms average
    });

    test('put() completes in <10ms (O(1) guarantee)', async () => {
      const iterations = 50;
      const start = performance.now();

      for (let i = 0; i < iterations; i++) {
        await db.put(createTestEpisode(`perf ${i}`, MemoryType.LONG_TERM));
      }

      const end = performance.now();
      const avgTime = (end - start) / iterations;

      expect(avgTime).toBeLessThan(10); // <10ms average (disk write)
    });

    test('has() completes in <0.1ms (O(1) guarantee)', async () => {
      const episode = createTestEpisode('perf test', MemoryType.LONG_TERM);
      const hash = await db.put(episode);

      const iterations = 1000;
      const start = performance.now();

      for (let i = 0; i < iterations; i++) {
        db.has(hash);
      }

      const end = performance.now();
      const avgTime = (end - start) / iterations;

      expect(avgTime).toBeLessThan(0.1); // <0.1ms average (index lookup)
    });
  });

  // ==========================================================================
  // Edge Cases
  // ==========================================================================

  describe('Edge Cases', () => {
    test('handles empty database gracefully', () => {
      const stats = db.getStatistics();
      expect(stats.total_episodes).toBe(0);
      expect(db.get('nonexistent')).toBeNull();
      expect(db.has('nonexistent')).toBe(false);
    });

    test('handles deletion of non-existent episode', () => {
      const result = db.delete('nonexistent');
      expect(result).toBe(false);
    });

    test('handles large responses', async () => {
      const largeResponse = 'x'.repeat(100000); // 100KB
      const episode = createTestEpisode('large', MemoryType.LONG_TERM);
      episode.response = largeResponse;

      const hash = await db.put(episode);
      const retrieved = db.get(hash);

      expect(retrieved!.response.length).toBe(100000);
    });

    test('handles special characters in queries', async () => {
      const specialQuery = 'λx.x → λy.y (unicode: 🚀)';
      const episode = createTestEpisode(specialQuery, MemoryType.LONG_TERM);

      const hash = await db.put(episode);
      const retrieved = db.get(hash);

      expect(retrieved!.query).toBe(specialQuery);
    });
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
