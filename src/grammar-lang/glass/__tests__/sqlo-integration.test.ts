/**
 * Glass + SQLO Integration Tests
 *
 * Verifies:
 * - Memory embedded in .glass organism
 * - Learning from interactions
 * - Maturity progression (0% → 100%)
 * - Episodic memory recall
 * - Glass box inspection
 */

import { describe, it, beforeEach, afterEach, expect } from '../../../shared/utils/test-runner';
import {
  GlassMemorySystem,
  createGlassWithMemory,
  loadGlassWithMemory,
  LearningInteraction
} from '../sqlo-integration';
import { MemoryType } from '../../database/sqlo';
import * as fs from 'fs';

const TEST_ORGANISMS_DIR = 'test_organisms';

describe('Glass + SQLO Integration - Organism Creation', () => {
  afterEach(() => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
  });

  it('creates new glass organism with embedded memory', async () => {
    const glass = await createGlassWithMemory(
      'test-organism',
      'testing',
      TEST_ORGANISMS_DIR
    );

    const stats = glass.getMemoryStats();

    expect.toBeDefined(glass);
    expect.toEqual(stats.total_episodes, 0);
    expect.toEqual(stats.maturity, 0);
    expect.toEqual(stats.stage, 'nascent');
  });

  it('loads existing glass organism', async () => {
    // Create organism
    const glass1 = await createGlassWithMemory(
      'test-organism',
      'testing',
      TEST_ORGANISMS_DIR
    );

    // Learn something
    await glass1.learn({
      query: 'test query',
      response: 'test response',
      confidence: 0.95,
      sources: ['test.pdf'],
      attention_weights: [1.0],
      outcome: 'success'
    });

    // Load same organism
    const glassPath = `${TEST_ORGANISMS_DIR}/test-organism/test-organism.glass`;
    const glass2 = loadGlassWithMemory(glassPath);

    const stats = glass2.getMemoryStats();
    expect.toBeGreaterThan(stats.total_episodes, 0);
  });
});

describe('Glass + SQLO Integration - Learning', () => {
  let glass: GlassMemorySystem;

  beforeEach(async () => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
    glass = await createGlassWithMemory(
      'learning-test',
      'testing',
      TEST_ORGANISMS_DIR
    );
  });

  afterEach(() => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
  });

  it('learns from successful interaction', async () => {
    const interaction: LearningInteraction = {
      query: 'What is 2+2?',
      response: '4',
      confidence: 0.99,
      sources: ['math.pdf'],
      attention_weights: [1.0],
      outcome: 'success'
    };

    const episodeHash = await glass.learn(interaction);

    expect.toBeDefined(episodeHash);
    expect.toEqual(episodeHash.length, 64); // SHA256 hex

    const stats = glass.getMemoryStats();
    expect.toEqual(stats.total_episodes, 1);
  });

  it('stores high-confidence successes in long-term memory', async () => {
    const interaction: LearningInteraction = {
      query: 'High confidence query',
      response: 'High confidence response',
      confidence: 0.95,
      sources: ['source.pdf'],
      attention_weights: [1.0],
      outcome: 'success'
    };

    await glass.learn(interaction);

    const stats = glass.getMemoryStats();
    expect.toEqual(stats.long_term_count, 1);
    expect.toEqual(stats.short_term_count, 0);
  });

  it('stores failures in short-term memory for learning', async () => {
    const interaction: LearningInteraction = {
      query: 'Failed query',
      response: "I'm not certain about this answer, confidence is low",
      confidence: 0.3,
      sources: [],
      attention_weights: [],
      outcome: 'failure'
    };

    await glass.learn(interaction);

    const stats = glass.getMemoryStats();
    expect.toEqual(stats.short_term_count, 1);
    expect.toEqual(stats.long_term_count, 0);
  });
});

describe('Glass + SQLO Integration - Maturity Progression', () => {
  let glass: GlassMemorySystem;

  beforeEach(async () => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
    glass = await createGlassWithMemory(
      'maturity-test',
      'testing',
      TEST_ORGANISMS_DIR
    );
  });

  afterEach(() => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
  });

  it('starts at 0% maturity (nascent)', async () => {
    const stats = glass.getMemoryStats();

    expect.toEqual(stats.maturity, 0);
    expect.toEqual(stats.stage, 'nascent');
  });

  it('increases maturity with successful learning', async () => {
    const interaction: LearningInteraction = {
      query: 'test',
      response: 'test',
      confidence: 0.95,
      sources: ['test.pdf'],
      attention_weights: [1.0],
      outcome: 'success'
    };

    await glass.learn(interaction);

    const stats = glass.getMemoryStats();
    expect.toBeGreaterThan(stats.maturity, 0);
  });

  it('progresses through lifecycle stages', async () => {
    // Learn 10 successful interactions
    for (let i = 0; i < 10; i++) {
      await glass.learn({
        query: `query ${i}`,
        response: `response ${i}`,
        confidence: 0.95,
        sources: [`source${i}.pdf`],
        attention_weights: [1.0],
        outcome: 'success'
      });
    }

    const stats = glass.getMemoryStats();

    // Should still be nascent/infant with only 10 interactions
    // (maturity increases slowly)
    expect.toBeGreaterThan(stats.maturity, 0);
    expect.toBeTruthy(
      stats.stage === 'nascent' || stats.stage === 'infant'
    );
  });
});

describe('Glass + SQLO Integration - Memory Recall', () => {
  let glass: GlassMemorySystem;

  beforeEach(async () => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
    glass = await createGlassWithMemory(
      'recall-test',
      'testing',
      TEST_ORGANISMS_DIR
    );
  });

  afterEach(() => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
  });

  it('recalls similar experiences', async () => {
    // Learn about cancer treatment
    await glass.learn({
      query: 'Best treatment for lung cancer?',
      response: 'Pembrolizumab is effective',
      confidence: 0.9,
      sources: ['oncology.pdf'],
      attention_weights: [1.0],
      outcome: 'success'
    });

    // Recall similar query
    const similar = await glass.recallSimilar('lung cancer treatment');

    expect.toBeGreaterThan(similar.length, 0);
    expect.toBeTruthy(similar[0].query.includes('lung cancer'));
  });

  it('filters memory by type', async () => {
    // Learn mix of successes and failures
    await glass.learn({
      query: 'success 1',
      response: 'good',
      confidence: 0.95,
      sources: [],
      attention_weights: [],
      outcome: 'success'
    });

    await glass.learn({
      query: 'failure 1',
      response: "I don't know the answer, confidence is very low",
      confidence: 0.2,
      sources: [],
      attention_weights: [],
      outcome: 'failure'
    });

    const longTerm = glass.getMemory(MemoryType.LONG_TERM);
    const shortTerm = glass.getMemory(MemoryType.SHORT_TERM);

    expect.toEqual(longTerm.length, 1); // Success went to long-term
    expect.toEqual(shortTerm.length, 1); // Failure in short-term
  });
});

describe('Glass + SQLO Integration - Glass Box Inspection', () => {
  let glass: GlassMemorySystem;

  beforeEach(async () => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
    glass = await createGlassWithMemory(
      'inspect-test',
      'testing',
      TEST_ORGANISMS_DIR
    );
  });

  afterEach(() => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
  });

  it('provides full glass box inspection', async () => {
    // Learn some things
    await glass.learn({
      query: 'test 1',
      response: 'response 1',
      confidence: 0.9,
      sources: [],
      attention_weights: [],
      outcome: 'success'
    });

    const inspection = glass.inspect();

    expect.toBeDefined(inspection.organism);
    expect.toBeDefined(inspection.memory_stats);
    expect.toBeDefined(inspection.recent_learning);
    expect.toBeDefined(inspection.fitness_trajectory);

    expect.toEqual(inspection.organism.metadata.name, 'inspect-test');
    expect.toBeGreaterThan(inspection.memory_stats.total_episodes, 0);
  });

  it('tracks fitness trajectory over time', async () => {
    // Learn multiple interactions with varying confidence
    for (let i = 0; i < 20; i++) {
      await glass.learn({
        query: `query ${i}`,
        response: `response ${i}`,
        confidence: 0.7 + (i * 0.01), // Gradually increasing confidence
        sources: [],
        attention_weights: [],
        outcome: 'success'
      });
    }

    const inspection = glass.inspect();
    const trajectory = inspection.fitness_trajectory;

    // Trajectory should show improvement (later windows have higher fitness)
    expect.toBeGreaterThan(trajectory.length, 0);
  });
});

describe('Glass + SQLO Integration - Export', () => {
  let glass: GlassMemorySystem;

  beforeEach(async () => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
    glass = await createGlassWithMemory(
      'export-test',
      'testing',
      TEST_ORGANISMS_DIR
    );
  });

  afterEach(() => {
    if (fs.existsSync(TEST_ORGANISMS_DIR)) {
      fs.rmSync(TEST_ORGANISMS_DIR, { recursive: true });
    }
  });

  it('exports organism with memory stats', async () => {
    await glass.learn({
      query: 'test',
      response: 'test',
      confidence: 0.9,
      sources: [],
      attention_weights: [],
      outcome: 'success'
    });

    const exported = await glass.exportGlass();

    expect.toBeDefined(exported.glass);
    expect.toBeDefined(exported.memory_size);
    expect.toBeDefined(exported.total_size);
    expect.toBeGreaterThan(exported.total_size, 0);
  });
});
