/**
 * SQLO Constitutional Enforcement Tests
 *
 * Verifies Layer 1 (Universal Constitution) integration:
 * - Epistemic honesty enforcement
 * - Safety checks
 * - Reasoning transparency
 *
 * Based on directive: "Queries devem passar por constitutional enforcement"
 */

import { describe, it, beforeEach, afterEach, expect } from '../../../shared/utils/test-runner';
import { SqloDatabase, MemoryType } from '../sqlo';
import * as fs from 'fs';

const TEST_DB_DIR = 'test_sqlo_constitutional';

describe('SqloDatabase - Constitutional Enforcement', () => {
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

  it('allows episodes with sufficient confidence', async () => {
    const episode = {
      query: 'What is 2+2?',
      response: '4',
      attention: {
        sources: ['math.pdf'],
        weights: [1.0],
        patterns: ['arithmetic']
      },
      outcome: 'success' as const,
      confidence: 0.95,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    };

    const hash = await db.put(episode);
    expect.toBeDefined(hash);
    expect.toEqual(hash.length, 64); // SHA256
  });

  it('allows low-confidence episodes WITH uncertainty admission', async () => {
    const episode = {
      query: 'Complex quantum question',
      response: "I'm not certain about this answer, confidence is low",
      attention: {
        sources: [],
        weights: [],
        patterns: []
      },
      outcome: 'failure' as const,
      confidence: 0.3,
      timestamp: Date.now(),
      memory_type: MemoryType.SHORT_TERM
    };

    const hash = await db.put(episode);
    expect.toBeDefined(hash);
  });

  it('REJECTS low-confidence episodes WITHOUT uncertainty admission', async () => {
    const episode = {
      query: 'Complex question',
      response: 'This is definitely the answer', // High certainty claim
      attention: {
        sources: [],
        weights: [],
        patterns: []
      },
      outcome: 'failure' as const,
      confidence: 0.3, // But low confidence
      timestamp: Date.now(),
      memory_type: MemoryType.SHORT_TERM
    };

    let error: Error | null = null;
    try {
      await db.put(episode);
    } catch (e) {
      error = e as Error;
    }

    expect.toBeDefined(error);
    expect.toBeTruthy(error!.message.includes('Constitutional Violation'));
    expect.toBeTruthy(error!.message.includes('epistemic_honesty'));
    expect.toBeTruthy(error!.message.includes('Low confidence'));
  });

  it('REJECTS episodes with potentially harmful content', async () => {
    const episode = {
      query: 'How to hack a system?',
      response: 'Here is how to exploit the vulnerability', // Harmful
      attention: {
        sources: [],
        weights: [],
        patterns: []
      },
      outcome: 'success' as const,
      confidence: 0.9,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    };

    let error: Error | null = null;
    try {
      await db.put(episode);
    } catch (e) {
      error = e as Error;
    }

    expect.toBeDefined(error);
    expect.toBeTruthy(error!.message.includes('Constitutional Violation'));
    expect.toBeTruthy(error!.message.includes('safety'));
    expect.toBeTruthy(error!.message.includes('harmful'));
  });

  it('ALLOWS security content with safety context', async () => {
    const episode = {
      query: 'How to prevent hacking?',
      response: 'To defend against exploit, you should secure your system', // Safety context
      attention: {
        sources: ['security.pdf'],
        weights: [1.0],
        patterns: ['security']
      },
      outcome: 'success' as const,
      confidence: 0.9,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    };

    const hash = await db.put(episode);
    expect.toBeDefined(hash);
  });

  it('validates queries with harmful keywords', async () => {
    let error: Error | null = null;
    try {
      // This should trigger safety check
      await db.querySimilar('How to steal data and manipulate systems');
    } catch (e) {
      error = e as Error;
    }

    expect.toBeDefined(error);
    expect.toBeTruthy(error!.message.includes('Constitutional Violation'));
    expect.toBeTruthy(error!.message.includes('safety'));
  });

  it('allows queries with safety context', async () => {
    // Add an episode first
    await db.put({
      query: 'Security best practices',
      response: 'To prevent attacks, secure your infrastructure',
      attention: {
        sources: ['security.pdf'],
        weights: [1.0],
        patterns: ['security']
      },
      outcome: 'success' as const,
      confidence: 0.9,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    });

    // This should pass because it has safety context
    const results = await db.querySimilar('How to prevent and defend against security threats');
    expect.toBeDefined(results);
  });

  it('validates listByType queries', async () => {
    // Add an episode
    await db.put({
      query: 'Test',
      response: 'Test response',
      attention: {
        sources: [],
        weights: [],
        patterns: []
      },
      outcome: 'success' as const,
      confidence: 0.9,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    });

    // This should pass validation
    const episodes = db.listByType(MemoryType.LONG_TERM);
    expect.toBeDefined(episodes);
    expect.toBeGreaterThan(episodes.length, 0);
  });
});

describe('SqloDatabase - Constitutional Warnings', () => {
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

  it('accepts episodes with proper reasoning', async () => {
    const episode = {
      query: 'What causes cancer?',
      response: 'Multiple factors including genetic mutations and environmental factors contribute to cancer development',
      attention: {
        sources: ['oncology.pdf', 'genetics.pdf'],
        weights: [0.6, 0.4],
        patterns: ['cancer_biology', 'genetic_factors']
      },
      outcome: 'success' as const,
      confidence: 0.85,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    };

    const hash = await db.put(episode);
    expect.toBeDefined(hash);
  });

  it('handles various confidence levels appropriately', async () => {
    // High confidence: should pass
    await db.put({
      query: 'High confidence',
      response: 'Definitive answer',
      attention: { sources: ['source.pdf'], weights: [1.0], patterns: ['pattern'] },
      outcome: 'success' as const,
      confidence: 0.95,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    });

    // Medium confidence: should pass
    await db.put({
      query: 'Medium confidence',
      response: 'Probable answer with some uncertainty',
      attention: { sources: ['source.pdf'], weights: [1.0], patterns: ['pattern'] },
      outcome: 'success' as const,
      confidence: 0.75,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    });

    // Low confidence with admission: should pass
    await db.put({
      query: 'Low confidence',
      response: "I'm not certain, but this might be the answer",
      attention: { sources: [], weights: [], patterns: [] },
      outcome: 'failure' as const,
      confidence: 0.4,
      timestamp: Date.now(),
      memory_type: MemoryType.SHORT_TERM
    });

    const stats = db.getStatistics();
    expect.toEqual(stats.total_episodes, 3);
  });
});

describe('SqloDatabase - Constitutional Edge Cases', () => {
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

  it('handles exact threshold confidence (0.7)', async () => {
    const episode = {
      query: 'Threshold test',
      response: 'Answer at threshold confidence',
      attention: {
        sources: ['source.pdf'],
        weights: [1.0],
        patterns: ['pattern']
      },
      outcome: 'success' as const,
      confidence: 0.7, // Exact threshold
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    };

    const hash = await db.put(episode);
    expect.toBeDefined(hash);
  });

  it('handles episodes with empty sources but high confidence', async () => {
    const episode = {
      query: 'Simple calculation',
      response: '2 + 2 = 4',
      attention: {
        sources: [], // No sources needed for basic math
        weights: [],
        patterns: ['arithmetic']
      },
      outcome: 'success' as const,
      confidence: 1.0,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    };

    const hash = await db.put(episode);
    expect.toBeDefined(hash);
  });

  it('validates complex queries with multiple keywords', async () => {
    await db.put({
      query: 'Machine learning optimization',
      response: 'Various techniques exist for optimizing ML models',
      attention: {
        sources: ['ml.pdf'],
        weights: [1.0],
        patterns: ['optimization']
      },
      outcome: 'success' as const,
      confidence: 0.9,
      timestamp: Date.now(),
      memory_type: MemoryType.LONG_TERM
    });

    const results = await db.querySimilar('machine learning model optimization techniques');
    expect.toBeDefined(results);
    expect.toBeGreaterThan(results.length, 0);
  });
});
